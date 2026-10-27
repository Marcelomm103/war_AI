"""Finite PPO microtasks built from legal WAR engine actions."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import ClassVar

import torch
from numpy.typing import ArrayLike
from torch import Tensor
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.combat import enumerate_combat_loss_distribution
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    ClaimTerritoryOperation,
    DeckState,
    EndPhaseOperation,
    FullGameState,
    FullPlayerState,
    GamePhaseId,
    MoveArmiesOperation,
    MovementControl,
    OccupyTerritoryOperation,
    PendingOccupation,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    TerritoryState,
    TradeCardsOperation,
)
from war_game_engine.engine import apply_action, legal_actions
from war_game_engine.setup import SetupPlayer, prepare_game

from ..envs.actions import (
    ACTION_CONFIRM_TOKEN,
    ACTION_TOKEN_VOCAB_SIZE,
    StructuredActionCodec,
)
from .losses import joint_action_log_prob
from .ppo import PPOActorCritic, compute_gae, compute_masked_ppo_loss
from .telemetry import TrainingRunJournal, apply_ppo_update

_PLAYER_SPECS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)
_PLAYER_IDS = tuple(player.player_id for player in _PLAYER_SPECS)
_PHASE_IDS = (
    "initial_placement",
    "attack",
    "occupation",
    "movement",
    "reinforcement",
    "territory_claim",
)
_OBJECTIVE_IDS = tuple(
    objective.objective_id for objective in WAR1_BOARD_SPEC.objectives
)
_CODEC = StructuredActionCodec()
INTERMEDIATE_SCENARIO_IDS = (
    "attack",
    "occupation",
    "movement",
    "trade",
    "objective_claim",
)


@dataclass(frozen=True)
class SmallTaskAction:
    label: str
    action: Action
    tokens: tuple[int, ...]
    reward: float

    def __post_init__(self) -> None:
        if not self.label.strip() or not math.isfinite(self.reward):
            raise ValueError("Small-task actions require a label and finite reward.")
        if self.tokens != _CODEC.encode_action(self.action):
            raise ValueError("Small-task tokens must encode their engine action.")
        if self.tokens[-1] != ACTION_CONFIRM_TOKEN:
            raise ValueError("A small-task action must end with confirmation.")


@dataclass(frozen=True)
class FiniteDecisionTask:
    task_id: str
    description: str
    observation: tuple[float, ...]
    actions: tuple[SmallTaskAction, ...]

    _codec: ClassVar[StructuredActionCodec] = _CODEC

    def __post_init__(self) -> None:
        if not self.task_id.strip() or not self.description.strip():
            raise ValueError("Small tasks require an ID and description.")
        if not self.observation or len(self.actions) < 2:
            raise ValueError("Small tasks require features and multiple legal actions.")
        if len({action.label for action in self.actions}) != len(self.actions):
            raise ValueError("Small-task action labels must be unique.")
        if len({action.tokens for action in self.actions}) != len(self.actions):
            raise ValueError("Small-task action tokens must be unique.")
        if not all(math.isfinite(value) for value in self.observation):
            raise ValueError("Small-task observations must be finite.")

    @property
    def exact_return(self) -> float:
        return max(action.reward for action in self.actions)

    @property
    def exact_action_labels(self) -> tuple[str, ...]:
        return tuple(
            action.label
            for action in self.actions
            if action.reward == self.exact_return
        )

    @property
    def engine_actions(self) -> tuple[Action, ...]:
        return tuple(action.action for action in self.actions)

    def prefix_mask(self, prefix: tuple[int, ...]) -> Tensor:
        mask: ArrayLike = self._codec.valid_next_token_mask(
            prefix,
            self.engine_actions,
        )
        return torch.as_tensor(mask, dtype=torch.bool)


@dataclass(frozen=True)
class SmallTaskResult:
    task_id: str
    seed: int
    updates: int
    exact_return: float
    initial_expected_return: float
    learned_expected_return: float
    optimal_action_probability: float
    most_likely_action: str
    exact_action_labels: tuple[str, ...]
    action_probabilities: tuple[float, ...]


def build_small_tasks() -> tuple[FiniteDecisionTask, ...]:
    """Build six fixed-state tasks using engine-generated legal actions."""
    placement_state = _build_state(
        task_id="legal-placement",
        phase_id="initial_placement",
        actor_territories=("T01", "T02"),
        opponent_territories=("T03",),
        army_counts={"T01": 1, "T02": 1},
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player_0:initial",
                player_id="player_0",
                origin="territory_count",
                remaining_armies=1,
            ),
            ReinforcementPool(
                pool_id="player_1:initial",
                player_id="player_1",
                origin="territory_count",
                remaining_armies=1,
            ),
            ReinforcementPool(
                pool_id="player_2:initial",
                player_id="player_2",
                origin="territory_count",
                remaining_armies=1,
            ),
        ),
    )
    placement_task = _task_from_state(
        "legal_placement",
        "Choose a legal initial placement that reinforces the exposed T02 border.",
        placement_state,
        lambda state, action: _placement_reward(state, action),
    )

    attack_state = _build_state(
        task_id="attack-or-stop",
        phase_id="attack",
        actor_territories=("T01", "T03", "T04"),
        opponent_territories=("T02",),
        army_counts={"T01": 3, "T02": 1},
    )
    attack_task = _task_from_state(
        "attack_or_stop",
        "Choose one exact one-round attack option or stop attacking.",
        attack_state,
        _attack_reward,
    )

    occupation_state = _build_state(
        task_id="occupation",
        phase_id="occupation",
        actor_territories=("T01",),
        opponent_territories=("T03",),
        unowned_territories=("T02",),
        army_counts={"T01": 5, "T03": 1},
        pending_occupation=PendingOccupation(
            contract_type="pending_occupation",
            attacker_player_id="player_0",
            defender_player_id="player_1",
            origin_territory_id="T01",
            target_territory_id="T02",
            maximum_army_count=3,
        ),
    )
    occupation_task = _task_from_state(
        "occupation",
        "Choose a legal occupation count for a two-army target garrison.",
        occupation_state,
        _occupation_reward,
    )

    movement_state = _build_state(
        task_id="movement",
        phase_id="movement",
        actor_territories=("T01", "T02"),
        opponent_territories=("T03", "T04"),
        army_counts={"T01": 5, "T02": 1},
    )
    movement_task = _task_from_state(
        "movement",
        "Move armies over a legal direct route to reach a target garrison.",
        movement_state,
        _movement_reward,
    )

    trade_state = _build_state(
        task_id="trades",
        phase_id="reinforcement",
        actor_territories=("T01", "T02"),
        opponent_territories=("T03", "T05", "T07"),
        hand_card_ids=("T01", "T02", "T03", "T05", "T07"),
        trade_count=2,
        reinforcement_pools=(
            ReinforcementPool(
                pool_id="player_0:turn-2:territory-count",
                player_id="player_0",
                origin="territory_count",
                remaining_armies=3,
            ),
        ),
    )
    trade_task = _task_from_state(
        "trades",
        "Choose a legal card trade maximizing base armies and owned-card bonuses.",
        trade_state,
        _trade_reward,
    )

    north_america = {
        territory.territory_id
        for territory in WAR1_BOARD_SPEC.map.territories
        if territory.continent_id == "north_america"
    }
    oceania = {
        territory.territory_id
        for territory in WAR1_BOARD_SPEC.map.territories
        if territory.continent_id == "oceania"
    }
    objective_state = _build_state(
        task_id="objective-completion",
        phase_id="territory_claim",
        actor_territories=tuple(sorted(north_america | (oceania - {"T02"}))),
        opponent_territories=("T20", "T21"),
        unowned_territories=("T02", "T14"),
        objective_ids={
            "player_0": "W1-OBJ-07",
            "player_1": "W1-OBJ-13",
            "player_2": "W1-OBJ-14",
        },
    )
    objective_task = _task_from_state(
        "objective_completion",
        "Claim the missing Oceania territory to complete W1-OBJ-07.",
        objective_state,
        _objective_reward,
    )
    return (
        placement_task,
        attack_task,
        occupation_task,
        movement_task,
        trade_task,
        objective_task,
    )


def build_intermediate_scenario_state(
    scenario_id: str,
    *,
    game_id: str,
    seed: int,
) -> FullGameState:
    """Create one validated mid-game decision state for curriculum rollouts."""
    if scenario_id == "attack":
        return _build_state(
            task_id=f"curriculum-{scenario_id}",
            game_id=game_id,
            seed=seed,
            phase_id="attack",
            actor_territories=("T01", "T03", "T04"),
            opponent_territories=("T02",),
            army_counts={"T01": 3, "T02": 1},
        )
    if scenario_id == "occupation":
        return _build_state(
            task_id=f"curriculum-{scenario_id}",
            game_id=game_id,
            seed=seed,
            phase_id="occupation",
            actor_territories=("T01",),
            opponent_territories=("T03",),
            unowned_territories=("T02",),
            army_counts={"T01": 5, "T03": 1},
            pending_occupation=PendingOccupation(
                contract_type="pending_occupation",
                attacker_player_id="player_0",
                defender_player_id="player_1",
                origin_territory_id="T01",
                target_territory_id="T02",
                maximum_army_count=3,
            ),
        )
    if scenario_id == "movement":
        return _build_state(
            task_id=f"curriculum-{scenario_id}",
            game_id=game_id,
            seed=seed,
            phase_id="movement",
            actor_territories=("T01", "T02"),
            opponent_territories=("T03", "T04"),
            army_counts={"T01": 5, "T02": 1},
        )
    if scenario_id == "trade":
        return _build_state(
            task_id=f"curriculum-{scenario_id}",
            game_id=game_id,
            seed=seed,
            phase_id="reinforcement",
            actor_territories=("T01", "T02"),
            opponent_territories=("T03", "T05", "T07"),
            hand_card_ids=("T01", "T02", "T03", "T05", "T07"),
            trade_count=2,
            reinforcement_pools=(
                ReinforcementPool(
                    pool_id="player_0:turn-2:territory-count",
                    player_id="player_0",
                    origin="territory_count",
                    remaining_armies=3,
                ),
            ),
        )
    if scenario_id == "objective_claim":
        north_america = {
            territory.territory_id
            for territory in WAR1_BOARD_SPEC.map.territories
            if territory.continent_id == "north_america"
        }
        oceania = {
            territory.territory_id
            for territory in WAR1_BOARD_SPEC.map.territories
            if territory.continent_id == "oceania"
        }
        return _build_state(
            task_id=f"curriculum-{scenario_id}",
            game_id=game_id,
            seed=seed,
            phase_id="territory_claim",
            actor_territories=tuple(sorted(north_america | (oceania - {"T02"}))),
            opponent_territories=("T20", "T21"),
            unowned_territories=("T02", "T14"),
            objective_ids={
                "player_0": "W1-OBJ-07",
                "player_1": "W1-OBJ-13",
                "player_2": "W1-OBJ-14",
            },
        )
    raise ValueError(f"Unknown intermediate scenario {scenario_id!r}.")


def train_small_task(
    task: FiniteDecisionTask,
    *,
    seed: int = 2026,
    updates: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    journal: TrainingRunJournal | None = None,
    update_index_offset: int = 0,
    environment_steps_offset: int = 0,
) -> SmallTaskResult:
    """Train one PPO policy and compare it with exact action enumeration."""
    _validate_training_parameters(
        seed=seed,
        updates=updates,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    for label, offset in (
        ("update_index_offset", update_index_offset),
        ("environment_steps_offset", environment_steps_offset),
    ):
        if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
            raise ValueError(f"{label} must be a non-negative integer.")
    candidate_tokens, candidate_token_mask, candidate_legal_masks = (
        _candidate_tensors(task)
    )
    rewards = torch.tensor(
        [action.reward for action in task.actions],
        dtype=torch.float32,
    )
    observation = torch.tensor(task.observation, dtype=torch.float32).unsqueeze(0)
    candidate_observations = observation.expand(len(task.actions), -1)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = PPOActorCritic(
            observation_size=len(task.observation),
            token_vocab_size=ACTION_TOKEN_VOCAB_SIZE,
            hidden_size=16,
            end_token_id=ACTION_CONFIRM_TOKEN,
        )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    sample_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    initial_probabilities = _action_probabilities(
        model,
        candidate_observations,
        candidate_tokens,
        candidate_legal_masks,
        candidate_token_mask,
    )
    initial_expected_return = float(torch.dot(initial_probabilities, rewards))

    model.train()
    for update_offset in range(updates):
        update_started_at = perf_counter()
        with torch.no_grad():
            candidate_output = model(candidate_observations, candidate_tokens)
            candidate_log_probabilities = joint_action_log_prob(
                candidate_output.token_logits,
                candidate_tokens,
                candidate_legal_masks,
                candidate_token_mask,
            )
            candidate_probabilities = _normalized_probabilities(
                candidate_log_probabilities.exp()
            )
            selected_indices = torch.multinomial(
                candidate_probabilities,
                num_samples=batch_size,
                replacement=True,
                generator=sample_generator,
            )
            selected_tokens = candidate_tokens[selected_indices]
            selected_token_mask = candidate_token_mask[selected_indices]
            selected_legal_masks = candidate_legal_masks[selected_indices]
            old_log_probabilities = candidate_log_probabilities[selected_indices]
            old_values = candidate_output.values[0].expand(batch_size).clone()
            sampled_rewards = rewards[selected_indices]

        output = model(
            observation.expand(batch_size, -1),
            selected_tokens,
        )
        advantages, returns = compute_gae(
            rewards=sampled_rewards.unsqueeze(0),
            values=old_values.unsqueeze(0),
            next_values=torch.zeros_like(old_values).unsqueeze(0),
            terminated=torch.ones((1, batch_size), dtype=torch.bool),
            truncated=torch.zeros((1, batch_size), dtype=torch.bool),
            gamma=1.0,
            gae_lambda=1.0,
        )
        loss = compute_masked_ppo_loss(
            new_logits=output.token_logits,
            actions=selected_tokens,
            legal_masks=selected_legal_masks,
            token_mask=selected_token_mask,
            old_log_probabilities=old_log_probabilities,
            advantages=advantages[0],
            value_predictions=output.values,
            returns=returns[0],
            clip_range=0.2,
            value_coefficient=0.5,
            entropy_coefficient=0.01,
        )
        gradient_norm = apply_ppo_update(model, optimizer, loss)
        if journal is not None:
            journal.record_ppo_update(
                update_index=update_index_offset + update_offset + 1,
                environment_steps=(
                    environment_steps_offset + (update_offset + 1) * batch_size
                ),
                loss=loss,
                gradient_norm=gradient_norm,
                learning_rate=learning_rate,
                update_seconds=max(perf_counter() - update_started_at, 1e-9),
            )

    probabilities = _action_probabilities(
        model,
        candidate_observations,
        candidate_tokens,
        candidate_legal_masks,
        candidate_token_mask,
    )
    learned_expected_return = float(torch.dot(probabilities, rewards))
    optimal_mask = torch.tensor(
        [action.reward == task.exact_return for action in task.actions],
        dtype=torch.bool,
    )
    most_likely_index = int(probabilities.argmax().item())
    return SmallTaskResult(
        task_id=task.task_id,
        seed=seed,
        updates=updates,
        exact_return=task.exact_return,
        initial_expected_return=initial_expected_return,
        learned_expected_return=learned_expected_return,
        optimal_action_probability=float(probabilities[optimal_mask].sum()),
        most_likely_action=task.actions[most_likely_index].label,
        exact_action_labels=task.exact_action_labels,
        action_probabilities=tuple(float(probability) for probability in probabilities),
    )


def run_small_task_suite(
    *,
    seed: int = 2026,
    updates: int = 100,
    batch_size: int = 32,
    learning_rate: float = 0.01,
    num_threads: int = 1,
    journal: TrainingRunJournal | None = None,
) -> tuple[SmallTaskResult, ...]:
    """Train all six tasks with deterministic seeds and bounded CPU use."""
    _validate_training_parameters(
        seed=seed,
        updates=updates,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, int)
        or num_threads < 1
    ):
        raise ValueError("num_threads must be a positive integer.")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    update_offset = 0 if journal is None else journal.last_update_index
    environment_steps_offset = 0 if journal is None else journal.environment_steps
    try:
        return tuple(
            train_small_task(
                task,
                seed=seed + index,
                updates=updates,
                batch_size=batch_size,
                learning_rate=learning_rate,
                journal=journal,
                update_index_offset=update_offset + index * updates,
                environment_steps_offset=(
                    environment_steps_offset + index * updates * batch_size
                ),
            )
            for index, task in enumerate(build_small_tasks())
        )
    finally:
        torch.set_num_threads(previous_threads)


def _build_state(
    *,
    task_id: str,
    game_id: str | None = None,
    seed: int = 17,
    phase_id: GamePhaseId,
    actor_territories: tuple[str, ...],
    opponent_territories: tuple[str, ...] = (),
    unowned_territories: tuple[str, ...] = (),
    army_counts: Mapping[str, int] | None = None,
    reinforcement_pools: tuple[ReinforcementPool, ...] = (),
    hand_card_ids: tuple[str, ...] = (),
    trade_count: int = 0,
    objective_ids: Mapping[str, str] | None = None,
    pending_occupation: PendingOccupation | None = None,
) -> FullGameState:
    actor_set = set(actor_territories)
    opponent_set = set(opponent_territories)
    unowned_set = set(unowned_territories)
    if (
        actor_set & opponent_set
        or actor_set & unowned_set
        or opponent_set & unowned_set
    ):
        raise ValueError("Task territory ownership sets must be disjoint.")
    if not actor_set or not opponent_set:
        raise ValueError("Each microtask needs an active actor and opponent.")

    resolved_game_id = (
        f"ppo-microtask-{task_id}" if game_id is None else game_id
    )
    if not resolved_game_id.strip():
        raise ValueError("Microtask game IDs must not be blank.")
    base = prepare_game(
        resolved_game_id,
        _PLAYER_SPECS,
        seed=seed,
    ).state
    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=(
                "player_0"
                if territory.territory_id in actor_set
                else (
                    "player_1"
                    if territory.territory_id in opponent_set
                    else (
                        None
                        if territory.territory_id in unowned_set
                        else "player_2"
                    )
                )
            ),
            army_count=(
                0
                if territory.territory_id in unowned_set
                else (army_counts or {}).get(territory.territory_id, 1)
            ),
        )
        for territory in base.territories
    )
    territory_counts = {
        player_id: sum(
            territory.owner_player_id == player_id for territory in territories
        )
        for player_id in _PLAYER_IDS
    }
    army_totals = {
        player_id: sum(
            territory.army_count
            for territory in territories
            if territory.owner_player_id == player_id
        )
        for player_id in _PLAYER_IDS
    }
    assigned_objectives = {
        "player_0": "W1-OBJ-13",
        "player_1": "W1-OBJ-14",
        "player_2": "W1-OBJ-12",
    }
    assigned_objectives.update(objective_ids or {})
    hands = {"player_0": hand_card_ids}
    held_cards = set(hand_card_ids)
    players = tuple(
        FullPlayerState(
            public=PublicPlayerState(
                player_id=player.player_id,
                color=player.color,
                territory_count=territory_counts[player.player_id],
                army_count=army_totals[player.player_id],
                is_eliminated=territory_counts[player.player_id] == 0,
            ),
            private=PlayerPrivateState(
                player_id=player.player_id,
                objective_id=assigned_objectives[player.player_id],
                objective_fallback_id=None,
                hand_card_ids=hands.get(player.player_id, ()),
            ),
        )
        for player in _PLAYER_SPECS
    )
    deck_data = base.deck.model_dump(mode="python")
    for zone in (
        "draw_pile_card_ids",
        "discard_pile_card_ids",
        "hand_limit_discard_card_ids",
    ):
        deck_data[zone] = tuple(
            card_id
            for card_id in deck_data[zone]
            if card_id not in held_cards
        )
    deck_data["trade_count"] = trade_count
    deck = DeckState.model_validate(deck_data)

    state_data = base.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": phase_id,
            "turn_number": 1 if phase_id in ("initial_placement", "territory_claim") else 2,
            "active_player_id": "player_0",
            "turn_player_id": "player_0",
            "decision_player_id": "player_0",
            "initial_placement_order": _PLAYER_IDS,
            "players": players,
            "territories": territories,
            "reinforcement_pools": reinforcement_pools,
            "movement_control": MovementControl(),
            "pending_combat": None,
            "pending_occupation": pending_occupation,
            "pending_objective_check": None,
            "winner_player_id": None,
            "deck": deck,
            "is_finished": False,
        }
    )
    return FullGameState.model_validate(state_data)


def _task_from_state(
    task_id: str,
    description: str,
    state: FullGameState,
    reward: Callable[[FullGameState, Action], float],
) -> FiniteDecisionTask:
    actions = legal_actions(state)
    if len(actions) < 2:
        raise RuntimeError(f"Microtask {task_id!r} has fewer than two legal actions.")
    task_actions = tuple(
        SmallTaskAction(
            label=_action_label(action),
            action=action,
            tokens=_CODEC.encode_action(action),
            reward=reward(state, action),
        )
        for action in actions
    )
    return FiniteDecisionTask(
        task_id=task_id,
        description=description,
        observation=_local_features(state, actions),
        actions=task_actions,
    )


def _local_features(
    state: FullGameState,
    actions: tuple[Action, ...],
) -> tuple[float, ...]:
    actor_id = state.decision_player_id or state.active_player_id
    if actor_id is None:
        raise RuntimeError("A microtask state requires a decision player.")
    actor = next(
        player for player in state.players if player.public.player_id == actor_id
    )
    phase_features = tuple(
        float(state.phase_id == phase_id) for phase_id in _PHASE_IDS
    )
    objective_features = tuple(
        float(actor.private.objective_id == objective_id)
        for objective_id in _OBJECTIVE_IDS
    )
    pool_total = sum(
        pool.remaining_armies
        for pool in state.reinforcement_pools
        if pool.player_id == actor_id
    )
    return (
        *phase_features,
        actor.public.territory_count / len(WAR1_BOARD_SPEC.map.territories),
        math.log2(actor.public.army_count + 1) / 16.0,
        len(actions) / len(WAR1_BOARD_SPEC.map.territories),
        len(actor.private.hand_card_ids) / 5.0,
        math.log2(pool_total + 1) / 16.0,
        float(state.has_conquered_territory_this_turn),
        *objective_features,
    )


def _placement_reward(state: FullGameState, action: Action) -> float:
    result = apply_action(state, action)
    operation = action.operation
    if not isinstance(operation, PlaceArmiesOperation):
        raise TypeError("The placement task contains a non-placement action.")
    placed_territory = operation.placements[0].territory_id
    if _territory_armies(result, placed_territory) <= _territory_armies(
        state,
        placed_territory,
    ):
        raise RuntimeError("A legal placement did not increase its territory garrison.")
    return 1.0 if placed_territory == "T02" else 0.2


def _attack_reward(state: FullGameState, action: Action) -> float:
    apply_action(state, action)
    operation = action.operation
    if isinstance(operation, EndPhaseOperation):
        return 0.0
    if not isinstance(operation, AttackOperation):
        raise TypeError("The attack task contains an unsupported action.")
    defender_dice = min(
        3,
        _territory_armies(state, operation.target_territory_id),
    )
    distribution = enumerate_combat_loss_distribution(
        operation.dice_count,
        defender_dice,
    )
    return sum(
        float(probability) * (defender_losses - 0.75 * attacker_losses)
        for (attacker_losses, defender_losses), probability in distribution.items()
    )


def _occupation_reward(state: FullGameState, action: Action) -> float:
    result = apply_action(state, action)
    operation = action.operation
    if not isinstance(operation, OccupyTerritoryOperation):
        raise TypeError("The occupation task contains a non-occupation action.")
    target_armies = _territory_armies(result, operation.target_territory_id)
    return 1.0 - 0.45 * abs(target_armies - 2) - 0.1 * max(target_armies - 2, 0)


def _movement_reward(state: FullGameState, action: Action) -> float:
    result = apply_action(state, action)
    operation = action.operation
    if isinstance(operation, EndPhaseOperation):
        return 0.0
    if not isinstance(operation, MoveArmiesOperation):
        raise TypeError("The movement task contains an unsupported action.")
    added_armies = (
        _territory_armies(result, operation.target_territory_id)
        - _territory_armies(state, operation.target_territory_id)
    )
    return min(added_armies, 2) / 2.0 - 0.15 * max(added_armies - 2, 0)


def _trade_reward(state: FullGameState, action: Action) -> float:
    result = apply_action(state, action)
    operation = action.operation
    if isinstance(operation, EndPhaseOperation):
        return 0.0
    if not isinstance(operation, TradeCardsOperation):
        raise TypeError("The trade task contains an unsupported action.")
    return float(
        sum(
            pool.remaining_armies
            for pool in result.reinforcement_pools
            if pool.player_id == action.actor_player_id
            and pool.origin in ("card_trade", "territory_card")
        )
    )


def _objective_reward(state: FullGameState, action: Action) -> float:
    result = apply_action(state, action)
    operation = action.operation
    if not isinstance(operation, ClaimTerritoryOperation):
        raise TypeError("The objective task contains a non-claim action.")
    return float(result.winner_player_id == action.actor_player_id)


def _territory_armies(state: FullGameState, territory_id: str) -> int:
    territory = next(
        (item for item in state.territories if item.territory_id == territory_id),
        None,
    )
    if territory is None:
        raise RuntimeError(f"Task state is missing territory {territory_id!r}.")
    return territory.army_count


def _action_label(action: Action) -> str:
    operation = action.operation
    if isinstance(operation, PlaceArmiesOperation):
        return f"place on {operation.placements[0].territory_id}"
    if isinstance(operation, AttackOperation):
        return (
            f"attack {operation.target_territory_id} "
            f"with {operation.dice_count} dice"
        )
    if isinstance(operation, OccupyTerritoryOperation):
        return f"occupy with {operation.army_count} armies"
    if isinstance(operation, MoveArmiesOperation):
        return f"move {operation.army_count} armies to {operation.target_territory_id}"
    if isinstance(operation, TradeCardsOperation):
        return f"trade {','.join(operation.card_ids)}"
    if isinstance(operation, ClaimTerritoryOperation):
        return f"claim {operation.territory_id}"
    if isinstance(operation, EndPhaseOperation):
        return f"stop {operation.phase_id}"
    raise TypeError(f"Unsupported microtask action {operation.action_type!r}.")


def _candidate_tensors(task: FiniteDecisionTask) -> tuple[Tensor, Tensor, Tensor]:
    action_count = len(task.actions)
    max_length = max(len(action.tokens) for action in task.actions)
    tokens = torch.zeros((action_count, max_length), dtype=torch.long)
    token_mask = torch.zeros((action_count, max_length), dtype=torch.bool)
    legal_masks = torch.zeros(
        (action_count, max_length, ACTION_TOKEN_VOCAB_SIZE),
        dtype=torch.bool,
    )
    for action_index, action in enumerate(task.actions):
        action_tokens = torch.tensor(action.tokens, dtype=torch.long)
        action_length = len(action.tokens)
        tokens[action_index, :action_length] = action_tokens
        token_mask[action_index, :action_length] = True
        for position in range(action_length):
            legal_masks[action_index, position] = task.prefix_mask(
                action.tokens[:position]
            )
    return tokens, token_mask, legal_masks


def _action_probabilities(
    model: PPOActorCritic,
    candidate_observations: Tensor,
    candidate_tokens: Tensor,
    candidate_legal_masks: Tensor,
    candidate_token_mask: Tensor,
) -> Tensor:
    model.eval()
    with torch.no_grad():
        output = model(candidate_observations, candidate_tokens)
        log_probabilities = joint_action_log_prob(
            output.token_logits,
            candidate_tokens,
            candidate_legal_masks,
            candidate_token_mask,
        )
        return _normalized_probabilities(log_probabilities.exp())


def _normalized_probabilities(probabilities: Tensor) -> Tensor:
    if not torch.isfinite(probabilities).all().item():
        raise FloatingPointError("A small-task policy produced non-finite probability.")
    total = probabilities.sum()
    if not torch.isclose(
        total,
        torch.ones_like(total),
        rtol=1e-4,
        atol=1e-4,
    ).item():
        raise RuntimeError(
            "Legal action-sequence probabilities must sum to one; "
            f"observed {float(total)}."
        )
    return probabilities / total


def _validate_training_parameters(
    *,
    seed: int,
    updates: int,
    batch_size: int,
    learning_rate: float,
) -> None:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if isinstance(updates, bool) or not isinstance(updates, int) or updates < 1:
        raise ValueError("updates must be a positive integer.")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 2
    ):
        raise ValueError("batch_size must be an integer of at least two.")
    if (
        isinstance(learning_rate, bool)
        or not math.isfinite(learning_rate)
        or learning_rate <= 0.0
    ):
        raise ValueError("learning_rate must be finite and positive.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train PPO on six finite WAR microtasks and report exact gaps."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--journal-path", type=Path)
    arguments = parser.parse_args()
    journal = (
        None
        if arguments.journal_path is None
        else TrainingRunJournal(
            arguments.journal_path,
            run_id=f"microtask-{arguments.seed}",
        )
    )
    results = run_small_task_suite(
        seed=arguments.seed,
        updates=arguments.updates,
        batch_size=arguments.batch_size,
        learning_rate=arguments.learning_rate,
        num_threads=arguments.threads,
        journal=journal,
    )
    print(
        json.dumps(
            [asdict(result) for result in results],
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()


__all__ = [
    "FiniteDecisionTask",
    "INTERMEDIATE_SCENARIO_IDS",
    "SmallTaskAction",
    "SmallTaskResult",
    "build_intermediate_scenario_state",
    "build_small_tasks",
    "run_small_task_suite",
    "train_small_task",
]
