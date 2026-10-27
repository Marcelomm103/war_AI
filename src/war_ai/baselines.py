"""Legal WAR 1 policy baselines for sanity checks and training curricula."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from fractions import Fraction
from functools import lru_cache
from typing import TypeAlias

import numpy as np
from numpy.random import Generator
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.combat import enumerate_combat_loss_distribution
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    ClaimTerritoryOperation,
    DiscardTerritoryCardOperation,
    DrawTerritoryCardOperation,
    EndPhaseOperation,
    EndTurnOperation,
    MoveArmiesOperation,
    OccupyTerritoryOperation,
    PlaceArmiesOperation,
    PlayerObservation,
    TerritoryState,
    TradeCardsOperation,
)
from war_game_engine.objectives import (
    OBJECTIVE_CONDITIONS,
    ControlContinentsCondition,
    EliminateColorCondition,
    ObjectiveCondition,
    TerritoryCountCondition,
)

from .envs.war1_aec import War1AECObservation

BaselinePolicy: TypeAlias = Callable[[War1AECObservation, Generator], int]
BASELINE_POLICY_IDS = (
    "random_legal",
    "expansionist",
    "conservative_borders",
    "objective_oriented",
    "continent_control",
    "opportunistic_elimination",
    "exact_probability_planner",
)

_TERRITORY_BY_ID = {
    territory.territory_id: territory for territory in WAR1_BOARD_SPEC.map.territories
}
_CONTINENT_BY_ID = {
    continent.continent_id: continent for continent in WAR1_BOARD_SPEC.map.continents
}
_TERRITORY_NEIGHBORS: dict[str, tuple[str, ...]] = {
    territory_id: () for territory_id in _TERRITORY_BY_ID
}
_neighbor_sets: dict[str, set[str]] = {
    territory_id: set() for territory_id in _TERRITORY_BY_ID
}
for _edge in WAR1_BOARD_SPEC.map.adjacencies:
    _neighbor_sets[_edge.first_territory_id].add(_edge.second_territory_id)
    _neighbor_sets[_edge.second_territory_id].add(_edge.first_territory_id)
_TERRITORY_NEIGHBORS = {
    territory_id: tuple(sorted(neighbors))
    for territory_id, neighbors in _neighbor_sets.items()
}


@dataclass(frozen=True)
class ExactAttackEstimate:
    conquest_probability: Fraction
    expected_attacker_losses: Fraction
    expected_defender_losses: Fraction


@lru_cache(maxsize=9)
def _combat_distribution(
    attacker_dice_count: int,
    defender_dice_count: int,
) -> tuple[tuple[tuple[int, int], Fraction], ...]:
    return tuple(
        enumerate_combat_loss_distribution(
            attacker_dice_count,
            defender_dice_count,
        ).items()
    )


def exact_attack_estimate(
    observation: PlayerObservation,
    operation: AttackOperation,
) -> ExactAttackEstimate:
    if not isinstance(operation, AttackOperation):
        raise TypeError("Exact combat estimates require an AttackOperation.")
    territories = {
        territory.territory_id: territory
        for territory in observation.public_state.territories
    }
    origin = territories.get(operation.origin_territory_id)
    target = territories.get(operation.target_territory_id)
    if (
        origin is None
        or origin.owner_player_id != observation.player_id
        or target is None
        or target.owner_player_id is None
        or target.owner_player_id == observation.player_id
    ):
        raise ValueError("Attack estimate must reference an owned origin and enemy.")

    defender_dice_count = min(target.army_count, 3)
    if defender_dice_count < 1:
        raise ValueError("Combat targets must contain at least one defending army.")
    distribution = _combat_distribution(
        operation.dice_count,
        defender_dice_count,
    )
    return ExactAttackEstimate(
        conquest_probability=sum(
            (
                probability
                for (_, defender_losses), probability in distribution
                if defender_losses >= target.army_count
            ),
            Fraction(),
        ),
        expected_attacker_losses=sum(
            (
                attacker_losses * probability
                for (attacker_losses, _), probability in distribution
            ),
            Fraction(),
        ),
        expected_defender_losses=sum(
            (
                defender_losses * probability
                for (_, defender_losses), probability in distribution
            ),
            Fraction(),
        ),
    )


def random_legal(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    """Choose a legal action for sanity checks and curriculum play."""
    _, actions = _context(observation)
    return int(rng.integers(0, len(actions)))


def expansionist(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    player_observation, actions = _context(observation)
    territories = _territories_by_id(player_observation)
    return _ranked_action(
        actions,
        rng,
        lambda action: _expansion_score(
            action,
            player_observation,
            territories,
        ),
    )


def conservative_borders(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    player_observation, actions = _context(observation)
    territories = _territories_by_id(player_observation)
    return _ranked_action(
        actions,
        rng,
        lambda action: _conservative_score(
            action,
            player_observation,
            territories,
        ),
    )


def objective_oriented(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    player_observation, actions = _context(observation)
    territories = _territories_by_id(player_observation)
    objective = _objective_condition(player_observation)
    return _ranked_action(
        actions,
        rng,
        lambda action: _objective_score(
            action,
            player_observation,
            territories,
            objective,
        ),
    )


def continent_control(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    player_observation, actions = _context(observation)
    territories = _territories_by_id(player_observation)
    return _ranked_action(
        actions,
        rng,
        lambda action: _continent_score(
            action,
            player_observation,
            territories,
        ),
    )


def opportunistic_elimination(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    player_observation, actions = _context(observation)
    territories = _territories_by_id(player_observation)
    objective = _objective_condition(player_observation)
    return _ranked_action(
        actions,
        rng,
        lambda action: _elimination_score(
            action,
            player_observation,
            territories,
            objective,
        ),
    )


def exact_probability_planner(
    observation: War1AECObservation,
    rng: Generator,
) -> int:
    player_observation, actions = _context(observation)
    territories = _territories_by_id(player_observation)
    return _ranked_action(
        actions,
        rng,
        lambda action: _exact_probability_score(
            action,
            player_observation,
            territories,
        ),
    )


BASELINE_POLICIES: dict[str, BaselinePolicy] = {
    "random_legal": random_legal,
    "expansionist": expansionist,
    "conservative_borders": conservative_borders,
    "objective_oriented": objective_oriented,
    "continent_control": continent_control,
    "opportunistic_elimination": opportunistic_elimination,
    "exact_probability_planner": exact_probability_planner,
}


def create_baseline(policy_id: str) -> BaselinePolicy:
    try:
        return BASELINE_POLICIES[policy_id]
    except KeyError as error:
        raise ValueError(f"Unknown WAR 1 baseline {policy_id!r}.") from error


def _context(
    observation: War1AECObservation,
) -> tuple[PlayerObservation, tuple[Action, ...]]:
    player_observation = observation["engine_observation"]
    if player_observation is None:
        raise ValueError("A baseline requires a live player observation.")
    actions = player_observation.legal_actions
    if not actions:
        raise ValueError("A baseline cannot select from an empty legal-action set.")
    mask = observation["action_mask"]
    if mask.shape != (len(actions),) or not np.all(mask == 1):
        raise ValueError("Baseline input has an inconsistent legal-action mask.")
    return player_observation, actions


def _ranked_action(
    actions: tuple[Action, ...],
    rng: Generator,
    score: Callable[[Action], float],
) -> int:
    scores = tuple(score(action) for action in actions)
    best_score = max(scores)
    best_indices = tuple(
        index for index, value in enumerate(scores) if value == best_score
    )
    return best_indices[int(rng.integers(0, len(best_indices)))]


def _territories_by_id(observation: PlayerObservation) -> dict[str, TerritoryState]:
    return {
        territory.territory_id: territory
        for territory in observation.public_state.territories
    }


def _objective_condition(observation: PlayerObservation) -> ObjectiveCondition:
    objective_id = (
        observation.private_state.objective_fallback_id
        or observation.private_state.objective_id
    )
    if objective_id is None:
        raise ValueError("A live baseline player must have an objective.")
    condition = OBJECTIVE_CONDITIONS.get(objective_id)
    if condition is None:
        raise ValueError(f"Unknown objective {objective_id!r}.")
    return condition


def _enemy_pressure(
    territory_id: str,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> int:
    return sum(
        territories[neighbor].army_count
        for neighbor in _TERRITORY_NEIGHBORS[territory_id]
        if territories[neighbor].owner_player_id is not None
        and territories[neighbor].owner_player_id != observation.player_id
    )


def _own_neighbor_count(
    territory_id: str,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> int:
    return sum(
        territories[neighbor].owner_player_id == observation.player_id
        for neighbor in _TERRITORY_NEIGHBORS[territory_id]
    )


def _continent_id(territory_id: str) -> str:
    try:
        return _TERRITORY_BY_ID[territory_id].continent_id
    except KeyError as error:
        raise ValueError(f"Unknown territory ID {territory_id!r}.") from error


def _continent_bonus(territory_id: str) -> int:
    return _CONTINENT_BY_ID[_continent_id(territory_id)].reinforcement_bonus


def _continent_progress(
    observation: PlayerObservation,
    continent_id: str,
) -> tuple[int, int]:
    territory_ids = tuple(
        territory.territory_id
        for territory in WAR1_BOARD_SPEC.map.territories
        if territory.continent_id == continent_id
    )
    if not territory_ids:
        raise ValueError(f"Unknown continent {continent_id!r}.")
    territories = {
        territory.territory_id: territory
        for territory in observation.public_state.territories
    }
    owned = sum(
        territories[territory_id].owner_player_id == observation.player_id
        for territory_id in territory_ids
    )
    return owned, len(territory_ids)


def _controlled_continents(observation: PlayerObservation) -> set[str]:
    return {
        continent_id
        for continent_id in _CONTINENT_BY_ID
        if _continent_progress(observation, continent_id)[0]
        == _continent_progress(observation, continent_id)[1]
    }


def _common_card_score(action: Action) -> float:
    operation = action.operation
    if isinstance(operation, TradeCardsOperation):
        return 2.0
    if isinstance(operation, DrawTerritoryCardOperation):
        return 1.0
    if isinstance(operation, DiscardTerritoryCardOperation):
        return 0.0
    if isinstance(operation, (EndPhaseOperation, EndTurnOperation)):
        return 0.0
    return 0.0


def _attack_value(
    operation: AttackOperation,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> float:
    estimate = exact_attack_estimate(observation, operation)
    target_value = 1.0 + _continent_bonus(operation.target_territory_id) / 10.0
    return (
        float(estimate.conquest_probability) * target_value
        + float(estimate.expected_defender_losses) * 0.1
        - float(estimate.expected_attacker_losses) * 0.2
    )


def _expansion_score(
    action: Action,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> float:
    operation = action.operation
    if isinstance(operation, ClaimTerritoryOperation):
        return (
            2.0 * _own_neighbor_count(operation.territory_id, observation, territories)
            + _continent_bonus(operation.territory_id) / 10.0
        )
    if isinstance(operation, PlaceArmiesOperation):
        territory_id = operation.placements[0].territory_id
        pressure = _enemy_pressure(territory_id, observation, territories)
        army_count = territories[territory_id].army_count
        return 1.0 + pressure / (army_count + 1.0)
    if isinstance(operation, AttackOperation):
        return 2.0 + _attack_value(operation, observation, territories)
    if isinstance(operation, OccupyTerritoryOperation):
        return float(operation.army_count)
    if isinstance(operation, MoveArmiesOperation):
        source_pressure = _enemy_pressure(
            operation.origin_territory_id,
            observation,
            territories,
        )
        target_pressure = _enemy_pressure(
            operation.target_territory_id,
            observation,
            territories,
        )
        return float(target_pressure - source_pressure) + 0.01 * operation.army_count
    return _common_card_score(action)


def _conservative_score(
    action: Action,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> float:
    operation = action.operation
    if isinstance(operation, ClaimTerritoryOperation):
        pressure = _enemy_pressure(operation.territory_id, observation, territories)
        own_neighbors = _own_neighbor_count(
            operation.territory_id,
            observation,
            territories,
        )
        return 0.2 * own_neighbors - pressure
    if isinstance(operation, PlaceArmiesOperation):
        territory_id = operation.placements[0].territory_id
        pressure = _enemy_pressure(territory_id, observation, territories)
        army_count = territories[territory_id].army_count
        return pressure / (army_count + 1.0)
    if isinstance(operation, AttackOperation):
        estimate = exact_attack_estimate(observation, operation)
        probability = float(estimate.conquest_probability)
        if probability < 0.6:
            return -1.0
        return (
            probability
            - 0.5 * float(estimate.expected_attacker_losses)
            + 0.05 * float(estimate.expected_defender_losses)
        )
    if isinstance(operation, OccupyTerritoryOperation):
        return -float(operation.army_count)
    if isinstance(operation, MoveArmiesOperation):
        source_pressure = _enemy_pressure(
            operation.origin_territory_id,
            observation,
            territories,
        )
        target_pressure = _enemy_pressure(
            operation.target_territory_id,
            observation,
            territories,
        )
        return float(target_pressure - source_pressure) - 0.01 * operation.army_count
    return _common_card_score(action)


def _objective_score(
    action: Action,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
    objective: ObjectiveCondition,
) -> float:
    operation = action.operation
    required_continents = (
        set(objective.required_continent_ids)
        if isinstance(objective, ControlContinentsCondition)
        else set()
    )
    target_color = (
        objective.target_color
        if isinstance(objective, EliminateColorCondition)
        else None
    )
    target_player_ids = {
        player.player_id
        for player in observation.public_state.players
        if player.color == target_color
    }
    if isinstance(operation, AttackOperation):
        score = _attack_value(operation, observation, territories)
        if territories[operation.target_territory_id].owner_player_id in target_player_ids:
            score += 5.0
        if _continent_id(operation.target_territory_id) in required_continents:
            score += 3.0
        if isinstance(objective, TerritoryCountCondition):
            score += 1.0
        return score
    if isinstance(operation, ClaimTerritoryOperation):
        score = 0.1 * _own_neighbor_count(
            operation.territory_id,
            observation,
            territories,
        )
        if _continent_id(operation.territory_id) in required_continents:
            score += 4.0
        if isinstance(objective, TerritoryCountCondition):
            score += 2.0
        return score
    if isinstance(operation, PlaceArmiesOperation):
        territory_id = operation.placements[0].territory_id
        territory = territories[territory_id]
        score = 0.0
        if isinstance(objective, TerritoryCountCondition):
            score += max(
                0,
                objective.minimum_armies_per_territory - territory.army_count,
            )
        if _continent_id(territory_id) in required_continents:
            score += 2.0
        if _enemy_pressure(territory_id, observation, territories):
            score += 0.25
        return score
    if isinstance(operation, MoveArmiesOperation):
        score = 0.01 * operation.army_count
        if _continent_id(operation.target_territory_id) in required_continents:
            score += 2.0
        return score
    if isinstance(operation, OccupyTerritoryOperation):
        score = float(operation.army_count)
        if _continent_id(operation.target_territory_id) in required_continents:
            score += 2.0
        return score
    return _common_card_score(action)


def _continent_score(
    action: Action,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> float:
    operation = action.operation
    controlled = _controlled_continents(observation)
    if isinstance(operation, ClaimTerritoryOperation):
        continent_id = _continent_id(operation.territory_id)
        return (
            _continent_bonus(operation.territory_id)
            + (2.0 if continent_id not in controlled else 0.0)
            + 0.1 * _own_neighbor_count(operation.territory_id, observation, territories)
        )
    if isinstance(operation, PlaceArmiesOperation):
        territory_id = operation.placements[0].territory_id
        pressure = _enemy_pressure(territory_id, observation, territories)
        continent_id = _continent_id(territory_id)
        return (
            _continent_bonus(territory_id) / 2.0
            + (1.0 if continent_id in controlled else 0.0)
            + pressure / (territories[territory_id].army_count + 1.0)
        )
    if isinstance(operation, AttackOperation):
        target_continent = _continent_id(operation.target_territory_id)
        return (
            _attack_value(operation, observation, territories)
            + _continent_bonus(operation.target_territory_id) / 2.0
            + (1.0 if target_continent not in controlled else 0.0)
        )
    if isinstance(operation, OccupyTerritoryOperation):
        return (
            float(operation.army_count)
            + _continent_bonus(operation.target_territory_id) / 2.0
        )
    if isinstance(operation, MoveArmiesOperation):
        target_continent = _continent_id(operation.target_territory_id)
        return (
            _continent_bonus(operation.target_territory_id) / 2.0
            + (1.0 if target_continent not in controlled else 0.0)
            + 0.01 * operation.army_count
        )
    return _common_card_score(action)


def _elimination_score(
    action: Action,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
    objective: ObjectiveCondition,
) -> float:
    operation = action.operation
    target_color = (
        objective.target_color
        if isinstance(objective, EliminateColorCondition)
        else None
    )
    elimination_targets = {
        player.player_id: player
        for player in observation.public_state.players
        if not player.is_eliminated
        and (
            player.color == target_color
            or player.territory_count == min(
                candidate.territory_count
                for candidate in observation.public_state.players
                if not candidate.is_eliminated
            )
        )
    }
    if isinstance(operation, AttackOperation):
        estimate = exact_attack_estimate(operation=operation, observation=observation)
        target_player_id = territories[operation.target_territory_id].owner_player_id
        target_player = (
            None
            if target_player_id is None
            else next(
                (
                    player
                    for player in observation.public_state.players
                    if player.player_id == target_player_id
                ),
                None,
            )
        )
        elimination_bonus = (
            5.0
            if target_player is not None
            and target_player.territory_count == 1
            else 0.0
        )
        objective_bonus = (
            2.0
            if target_player_id is not None
            and target_player_id in elimination_targets
            and target_color is not None
            else 0.0
        )
        return (
            2.0 * float(estimate.conquest_probability)
            + elimination_bonus * float(estimate.conquest_probability)
            + objective_bonus
            - 0.2 * float(estimate.expected_attacker_losses)
        )
    if isinstance(operation, OccupyTerritoryOperation):
        return float(operation.army_count)
    return _common_card_score(action)


def _exact_probability_score(
    action: Action,
    observation: PlayerObservation,
    territories: dict[str, TerritoryState],
) -> float:
    operation = action.operation
    if isinstance(operation, AttackOperation):
        return _attack_value(operation, observation, territories)
    if isinstance(operation, OccupyTerritoryOperation):
        return float(operation.army_count)
    return _common_card_score(action)


__all__ = [
    "BASELINE_POLICIES",
    "BASELINE_POLICY_IDS",
    "BaselinePolicy",
    "ExactAttackEstimate",
    "conservative_borders",
    "continent_control",
    "create_baseline",
    "exact_attack_estimate",
    "exact_probability_planner",
    "expansionist",
    "objective_oriented",
    "opportunistic_elimination",
    "random_legal",
]
