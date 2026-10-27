"""Canonical WAR 1 outcomes and optional bounded potential-based shaping."""

from __future__ import annotations

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import PlayerObservation
from war_game_engine.objectives import (
    OBJECTIVE_CONDITIONS,
    ControlContinentsCondition,
    EliminateColorCondition,
    TerritoryCountCondition,
)

EPISODIC_GAMMA = 1.0
MAX_POTENTIAL = 0.05


def official_win_reward(
    focal_player_id: str,
    winner_player_id: str | None,
) -> float:
    if not focal_player_id.strip():
        raise ValueError("The focal player ID must not be blank.")
    if winner_player_id is not None and not winner_player_id.strip():
        raise ValueError("Winner player ID must not be blank.")
    return 1.0 if winner_player_id == focal_player_id else 0.0


class PotentialRewardShaper:
    """Potential based only on public state and the focal player's own objective."""

    def potential(self, observation: PlayerObservation) -> float:
        public_state = observation.public_state
        focal_player = next(
            (
                player
                for player in public_state.players
                if player.player_id == observation.player_id
            ),
            None,
        )
        if focal_player is None:
            raise ValueError("The focal player is absent from the public state.")
        if public_state.is_finished or focal_player.is_eliminated:
            return 0.0

        objective_id = (
            observation.private_state.objective_fallback_id
            or observation.private_state.objective_id
        )
        if objective_id is None or objective_id not in WAR1_BOARD_SPEC.objective_deck.card_ids:
            raise ValueError("The focal observation has no certified objective.")
        condition = OBJECTIVE_CONDITIONS.get(objective_id)
        if condition is None:
            raise ValueError(f"Unknown objective {objective_id!r}.")
        progress = self._objective_progress(
            observation,
            condition,
        )
        if not 0.0 <= progress <= 1.0:
            raise ValueError("Objective progress must be within [0, 1].")
        return MAX_POTENTIAL * progress

    def transition_reward(
        self,
        previous_observation: PlayerObservation,
        next_observation: PlayerObservation | None,
        *,
        terminated: bool,
    ) -> float:
        previous_potential = self.potential(previous_observation)
        next_potential = (
            0.0
            if terminated
            else self.potential(self._require_observation(next_observation))
        )
        return EPISODIC_GAMMA * next_potential - previous_potential

    @staticmethod
    def _require_observation(
        observation: PlayerObservation | None,
    ) -> PlayerObservation:
        if observation is None:
            raise ValueError(
                "A nonterminal shaping transition requires a next observation."
            )
        return observation

    @staticmethod
    def _objective_progress(
        observation: PlayerObservation,
        condition: object,
    ) -> float:
        focal_player_id = observation.player_id
        public_state = observation.public_state
        territories_by_id = {
            territory.territory_id: territory
            for territory in public_state.territories
        }
        board_territories = WAR1_BOARD_SPEC.map.territories
        if set(territories_by_id) != {
            territory.territory_id for territory in board_territories
        }:
            raise ValueError("Potential evaluation requires the complete board state.")

        if isinstance(condition, EliminateColorCondition):
            target = next(
                (
                    player
                    for player in public_state.players
                    if player.color == condition.target_color
                ),
                None,
            )
            if target is None:
                raise ValueError(
                    f"Objective target color {condition.target_color!r} is absent."
                )
            territory_count = len(board_territories)
            if target.territory_count > territory_count:
                raise ValueError("Target territory count exceeds the certified board.")
            return (territory_count - target.territory_count) / territory_count

        if isinstance(condition, TerritoryCountCondition):
            qualified_count = sum(
                territory.owner_player_id == focal_player_id
                and territory.army_count
                >= condition.minimum_armies_per_territory
                for territory in territories_by_id.values()
            )
            return min(qualified_count / condition.minimum_territories, 1.0)

        if isinstance(condition, ControlContinentsCondition):
            continent_territories: dict[str, list[str]] = {
                continent.continent_id: []
                for continent in WAR1_BOARD_SPEC.map.continents
            }
            for territory in board_territories:
                continent_territories[territory.continent_id].append(
                    territory.territory_id
                )
            controlled = {
                continent_id
                for continent_id, territory_ids in continent_territories.items()
                if territory_ids
                and all(
                    territories_by_id[territory_id].owner_player_id
                    == focal_player_id
                    for territory_id in territory_ids
                )
            }
            required = set(condition.required_continent_ids)
            unknown = required - set(continent_territories)
            if unknown:
                raise ValueError(
                    f"Objective references unknown continents: {sorted(unknown)}."
                )
            required_progress = len(required & controlled)
            additional_progress = min(
                len(controlled - required),
                condition.additional_continent_count,
            )
            total_components = (
                len(required) + condition.additional_continent_count
            )
            return (required_progress + additional_progress) / total_components

        raise ValueError(f"Unsupported objective condition {condition!r}.")


__all__ = [
    "EPISODIC_GAMMA",
    "MAX_POTENTIAL",
    "PotentialRewardShaper",
    "official_win_reward",
]
