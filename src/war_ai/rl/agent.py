"""Trainable and exportable full-game PPO agent for WAR 1 classic."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import random
from collections import deque
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor
from war_game_engine.board_spec import WAR1_BOARD_SPEC
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
    TerritoryCountCondition,
)

from ..arena import ArenaPolicy, FrozenPolicySpec, baseline_policy_spec
from ..baselines import _TERRITORY_NEIGHBORS, exact_attack_estimate
from ..envs.gymnasium_env import War1GymEnv
from ..envs.war1_aec import War1AECObservation
from ..reproducibility import (
    HardwareSnapshot,
    ReproducibilityStore,
    RunManifest,
    SourceRevision,
    capture_installed_dependencies,
    capture_platform_snapshot,
)
from ..rewards import PotentialRewardShaper
from .checkpoint import (
    PPOCheckpointReferences,
    PPOResumeState,
    save_ppo_checkpoint,
)
from .curriculum import build_default_curriculum
from .losses import joint_action_log_prob
from .ppo import PPOActorCritic, RecurrentMode, compute_gae, compute_masked_ppo_loss
from .seed_evaluation import (
    PromotionCriteria,
    PromotionDecision,
    SeedComparison,
    assess_promotion,
    run_paired_seed_evaluation,
)
from .telemetry import (
    EpisodeAccumulator,
    TrainingCurve,
    TrainingRunJournal,
    apply_ppo_update,
)

FULL_GAME_INTENT_VOCAB_SIZE = 4
FULL_GAME_INTENT_END_TOKEN = 3
DEFAULT_PROMOTION_SEEDS: tuple[int, ...] = tuple(range(2026, 2036))

_PHASE_IDS = (
    "territory_claim",
    "initial_placement",
    "reinforcement",
    "reinforcement_placement",
    "attack",
    "occupation",
    "movement",
    "card_purchase",
    "card_discard",
    "turn_end",
)
_OBJECTIVE_IDS = tuple(
    objective.objective_id for objective in WAR1_BOARD_SPEC.objectives
)
_CONTINENT_TERRITORIES: dict[str, tuple[str, ...]] = {
    continent.continent_id: tuple(
        territory.territory_id
        for territory in WAR1_BOARD_SPEC.map.territories
        if territory.continent_id == continent.continent_id
    )
    for continent in WAR1_BOARD_SPEC.map.continents
}
FULL_GAME_OBSERVATION_SIZE = len(_PHASE_IDS) + 8 + len(_OBJECTIVE_IDS)


def _build_shortest_paths() -> dict[str, dict[str, int]]:
    distances: dict[str, dict[str, int]] = {}
    for origin_id in _TERRITORY_NEIGHBORS:
        origin_distances = {origin_id: 0}
        queue: deque[str] = deque((origin_id,))
        while queue:
            current = queue.popleft()
            for neighbor in _TERRITORY_NEIGHBORS[current]:
                if neighbor not in origin_distances:
                    origin_distances[neighbor] = origin_distances[current] + 1
                    queue.append(neighbor)
        distances[origin_id] = origin_distances
    return distances


_TERRITORY_DISTANCES = _build_shortest_paths()


@dataclass(frozen=True)
class FullGamePPOConfig:
    """Hyperparameters and architecture settings for full-game PPO."""

    hidden_size: int = 32
    recurrent: RecurrentMode = "none"
    recurrent_layers: int = 1
    learning_rate: float = 0.01
    clip_range: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    gamma: float = 1.0
    gae_lambda: float = 0.95
    rollout_batch_size: int = 32
    ppo_epochs_per_rollout: int = 6
    max_semantic_decisions: int = 2500

    def __post_init__(self) -> None:
        if isinstance(self.hidden_size, bool) or self.hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if self.recurrent not in ("none", "gru", "lstm"):
            raise ValueError("recurrent must be 'none', 'gru', or 'lstm'.")
        if isinstance(self.recurrent_layers, bool) or self.recurrent_layers < 1:
            raise ValueError("recurrent_layers must be a positive integer.")
        if (
            isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive.")
        if (
            isinstance(self.rollout_batch_size, bool)
            or self.rollout_batch_size < 2
        ):
            raise ValueError("rollout_batch_size must be at least two.")
        if (
            isinstance(self.ppo_epochs_per_rollout, bool)
            or self.ppo_epochs_per_rollout < 1
        ):
            raise ValueError("ppo_epochs_per_rollout must be positive.")
        if (
            isinstance(self.max_semantic_decisions, bool)
            or self.max_semantic_decisions < 1
        ):
            raise ValueError("max_semantic_decisions must be positive.")


@dataclass(frozen=True)
class TacticalActionPartition:
    """Player-scoped partitioning of legal actions into structured intents."""

    action_scores: tuple[float, ...]
    intent_by_action: tuple[int, ...]
    action_indices_by_intent: dict[int, tuple[int, ...]]
    available_intents: tuple[int, ...]

    def prefix_mask(self, prefix: tuple[int, ...]) -> Tensor:
        mask = torch.zeros(FULL_GAME_INTENT_VOCAB_SIZE, dtype=torch.bool)
        if not prefix:
            for intent in self.available_intents:
                mask[intent] = True
            return mask
        if len(prefix) == 1 and prefix[0] in self.action_indices_by_intent:
            mask[FULL_GAME_INTENT_END_TOKEN] = True
            return mask
        if (
            len(prefix) == 2
            and prefix[0] in self.action_indices_by_intent
            and prefix[1] == FULL_GAME_INTENT_END_TOKEN
        ):
            return mask
        raise ValueError(f"Unsupported action-intent prefix {prefix!r}.")


@dataclass(frozen=True)
class TrainedFullGamePPOAgent:
    """Summary and artifact references for a full-game trained PPO run."""

    run_id: str
    seed: int
    episodes_completed: int
    official_wins: int
    total_environment_steps: int
    total_ppo_updates: int
    initial_optimal_intent_probability: float
    trained_optimal_intent_probability: float
    checkpoint_references: PPOCheckpointReferences
    manifest: RunManifest
    curves: tuple[TrainingCurve, ...]


@dataclass(frozen=True)
class Phase6PromotionReport:
    """Complete training, checkpointing, and 10-seed evaluation report."""

    run_id: str
    candidate_policy_id: str
    baseline_policy_id: str
    checkpoint_sha256: str
    manifest_sha256: str
    training_episodes: int
    training_official_wins: int
    training_environment_steps: int
    training_ppo_updates: int
    initial_optimal_intent_probability: float
    trained_optimal_intent_probability: float
    comparisons: tuple[SeedComparison, ...]
    promotion_decision: PromotionDecision


def extract_player_observation_features(
    observation: War1AECObservation,
) -> tuple[float, ...]:
    """Build a fixed-width feature vector strictly from the player observation."""
    player_observation = observation["engine_observation"]
    if player_observation is None:
        raise ValueError("A live PPO decision requires a player observation.")
    public_state = player_observation.public_state
    focal_id = player_observation.player_id
    focal_player = next(
        (player for player in public_state.players if player.player_id == focal_id),
        None,
    )
    if focal_player is None:
        raise ValueError(f"Player {focal_id!r} is absent from public state.")

    effective_objective = (
        player_observation.private_state.objective_fallback_id
        or player_observation.private_state.objective_id
    )
    phase_one_hot = tuple(
        float(public_state.phase_id == phase_id) for phase_id in _PHASE_IDS
    )
    objective_one_hot = tuple(
        float(effective_objective == objective_id)
        for objective_id in _OBJECTIVE_IDS
    )
    pool_armies = sum(
        pool.remaining_armies
        for pool in public_state.reinforcement_pools
        if pool.player_id == focal_id
    )
    active_players = sum(
        not player.is_eliminated for player in public_state.players
    )
    return (
        *phase_one_hot,
        focal_player.territory_count / len(WAR1_BOARD_SPEC.map.territories),
        math.log2(focal_player.army_count + 1) / 16.0,
        len(player_observation.legal_actions)
        / len(WAR1_BOARD_SPEC.map.territories),
        len(player_observation.private_state.hand_card_ids) / 5.0,
        math.log2(pool_armies + 1) / 16.0,
        float(public_state.has_conquered_territory_this_turn),
        active_players / 6.0,
        math.log2(public_state.turn_number + 1) / 10.0,
        *objective_one_hot,
    )


def classify_legal_actions(
    observation: War1AECObservation,
) -> TacticalActionPartition:
    """Score legal actions from player-scoped state and group into intents."""
    player_observation = observation["engine_observation"]
    if player_observation is None:
        raise ValueError("Cannot classify actions without a player observation.")
    actions = player_observation.legal_actions
    if not actions:
        raise ValueError("Cannot classify an empty legal-action list.")
    mask = observation["action_mask"]
    if mask.shape != (len(actions),) or not np.all(mask == 1):
        raise ValueError("Observation legal-action mask is inconsistent.")

    territories = {
        territory.territory_id: territory
        for territory in player_observation.public_state.territories
    }
    goals = _goal_territories(player_observation, territories)
    distance_to_goal = _distance_to_goals(goals)
    scores = tuple(
        evaluate_tactical_action(
            player_observation,
            action,
            territories,
            distance_to_goal,
        )
        for action in actions
    )
    best_score = max(scores)
    intents: list[int] = []
    by_intent: dict[int, list[int]] = {}
    for index, score in enumerate(scores):
        if abs(score - best_score) <= 1e-9:
            intent = 0
        elif score >= 0.0:
            intent = 1
        else:
            intent = 2
        intents.append(intent)
        by_intent.setdefault(intent, []).append(index)

    action_indices_by_intent = {
        intent: tuple(indices)
        for intent, indices in sorted(by_intent.items())
    }
    return TacticalActionPartition(
        action_scores=scores,
        intent_by_action=tuple(intents),
        action_indices_by_intent=action_indices_by_intent,
        available_intents=tuple(sorted(action_indices_by_intent)),
    )


def evaluate_tactical_action(
    observation: PlayerObservation,
    action: Action,
    territories: Mapping[str, TerritoryState],
    distance_to_goal: Mapping[str, int],
) -> float:
    """Evaluate one legal action using only player-scoped observation data."""
    operation = action.operation
    focal_id = observation.player_id
    objective_id = (
        observation.private_state.objective_fallback_id
        or observation.private_state.objective_id
    )
    if objective_id is None or objective_id not in OBJECTIVE_CONDITIONS:
        raise ValueError(f"Observation has invalid objective {objective_id!r}.")
    condition = OBJECTIVE_CONDITIONS[objective_id]
    owned_territories = [
        territory
        for territory in territories.values()
        if territory.owner_player_id == focal_id
    ]
    owned_count = len(owned_territories)

    if isinstance(operation, ClaimTerritoryOperation):
        return 10.0 - float(distance_to_goal[operation.territory_id])
    if isinstance(operation, TradeCardsOperation):
        owned_bonus = sum(
            1
            for card_id in operation.card_ids
            if card_id in territories
            and territories[card_id].owner_player_id == focal_id
        )
        return 10.0 + float(owned_bonus)
    if isinstance(operation, PlaceArmiesOperation):
        territory_id = operation.placements[0].territory_id
        territory = territories[territory_id]
        if (
            isinstance(condition, TerritoryCountCondition)
            and condition.minimum_armies_per_territory > 1
            and owned_count >= condition.minimum_territories
            and territory.army_count < condition.minimum_armies_per_territory
        ):
            return 100.0 + float(
                condition.minimum_armies_per_territory - territory.army_count
            )
        enemy_neighbors = [
            neighbor
            for neighbor in _TERRITORY_NEIGHBORS[territory_id]
            if territories[neighbor].owner_player_id not in (None, focal_id)
        ]
        if not enemy_neighbors:
            return -10.0
        min_distance = min(distance_to_goal[neighbor] for neighbor in enemy_neighbors)
        best_enemy_armies = min(
            territories[neighbor].army_count
            for neighbor in enemy_neighbors
            if distance_to_goal[neighbor] == min_distance
        )
        return (
            20.0
            - 4.0 * float(min_distance)
            + 0.5 * float(territory.army_count)
            - 0.2 * float(best_enemy_armies)
        )
    if isinstance(operation, AttackOperation):
        origin = territories[operation.origin_territory_id]
        target = territories[operation.target_territory_id]
        max_dice = min(3, origin.army_count - 1)
        if operation.dice_count < max_dice:
            return -5.0
        if (
            isinstance(condition, TerritoryCountCondition)
            and condition.minimum_armies_per_territory > 1
        ):
            qualified = sum(
                candidate.army_count >= condition.minimum_armies_per_territory
                for candidate in owned_territories
            )
            if (
                qualified >= condition.minimum_territories - 1
                and origin.army_count <= 3
                and target.army_count >= 2
            ):
                return -5.0
        estimate = exact_attack_estimate(observation, operation)
        probability = float(estimate.conquest_probability)
        attacker_losses = float(estimate.expected_attacker_losses)
        defender_losses = float(estimate.expected_defender_losses)
        target_distance = distance_to_goal[operation.target_territory_id]
        origin_distance = distance_to_goal[operation.origin_territory_id]
        ratio = (origin.army_count - 1) / max(1.0, float(target.army_count))
        is_goal_step = target_distance == 0 or target_distance < origin_distance
        if is_goal_step and (probability >= 0.35 or ratio >= 1.2):
            return (
                10.0
                - 2.0 * float(target_distance)
                + 3.0 * probability
                + 0.5 * (defender_losses - attacker_losses)
            )
        if (
            not observation.public_state.has_conquered_territory_this_turn
            and probability >= 0.42
        ):
            return 4.0 + 2.0 * probability
        if isinstance(condition, TerritoryCountCondition) and (
            probability >= 0.42 or ratio >= 1.4
        ):
            return 5.0 + 2.0 * probability - 0.1 * float(target.army_count)
        if probability >= 0.65 and ratio >= 2.0:
            return 2.0 + probability
        return -2.0
    if isinstance(operation, OccupyTerritoryOperation):
        origin = territories[operation.origin_territory_id]
        if (
            isinstance(condition, TerritoryCountCondition)
            and condition.minimum_armies_per_territory > 1
        ):
            remaining = origin.army_count - operation.army_count
            if remaining >= 2 and operation.army_count >= 2:
                return 20.0 + float(operation.army_count)
            if remaining >= 2 and operation.army_count == 1:
                return 10.0
        target_enemies = [
            neighbor
            for neighbor in _TERRITORY_NEIGHBORS[operation.target_territory_id]
            if territories[neighbor].owner_player_id not in (None, focal_id)
        ]
        origin_enemies = [
            neighbor
            for neighbor in _TERRITORY_NEIGHBORS[operation.origin_territory_id]
            if territories[neighbor].owner_player_id not in (None, focal_id)
        ]
        if not target_enemies and origin_enemies:
            return -float(operation.army_count)
        if target_enemies:
            min_target_distance = min(
                distance_to_goal[neighbor] for neighbor in target_enemies
            )
            min_origin_distance = min(
                (distance_to_goal[neighbor] for neighbor in origin_enemies),
                default=99,
            )
            if min_target_distance <= min_origin_distance:
                return float(operation.army_count)
            return -float(operation.army_count)
        return float(operation.army_count)
    if isinstance(operation, MoveArmiesOperation):
        origin = territories[operation.origin_territory_id]
        target = territories[operation.target_territory_id]
        if (
            isinstance(condition, TerritoryCountCondition)
            and condition.minimum_armies_per_territory > 1
        ):
            if (
                origin.army_count - operation.army_count >= 2
                and target.army_count == 1
                and operation.army_count == 1
            ):
                return 50.0
            if origin.army_count - operation.army_count < 2:
                return -10.0
        origin_enemies = [
            neighbor
            for neighbor in _TERRITORY_NEIGHBORS[operation.origin_territory_id]
            if territories[neighbor].owner_player_id not in (None, focal_id)
        ]
        if (
            not origin_enemies
            and distance_to_goal[operation.target_territory_id]
            < distance_to_goal[operation.origin_territory_id]
            and operation.army_count == origin.army_count - 1
        ):
            return 5.0 + 0.1 * float(operation.army_count)
        return -5.0
    if isinstance(
        operation,
        (EndPhaseOperation, EndTurnOperation, DrawTerritoryCardOperation),
    ):
        return 0.0
    if isinstance(operation, DiscardTerritoryCardOperation):
        if (
            operation.card_id in territories
            and territories[operation.card_id].owner_player_id == focal_id
        ):
            return -1.0
        return 1.0
    return 0.0


def select_ppo_action_index(
    model: PPOActorCritic,
    observation: War1AECObservation,
    rng: Generator,
    *,
    deterministic: bool = True,
) -> tuple[int, int, Tensor]:
    """Use PPOActorCritic over structured intent tokens to pick a legal action."""
    partition = classify_legal_actions(observation)
    features = torch.tensor(
        extract_player_observation_features(observation),
        dtype=torch.float32,
    ).unsqueeze(0)
    candidate_intents = partition.available_intents
    candidate_tokens = torch.tensor(
        [[intent, FULL_GAME_INTENT_END_TOKEN] for intent in candidate_intents],
        dtype=torch.long,
    )
    token_mask = torch.ones_like(candidate_tokens, dtype=torch.bool)
    legal_masks = torch.zeros(
        (len(candidate_intents), 2, FULL_GAME_INTENT_VOCAB_SIZE),
        dtype=torch.bool,
    )
    first_mask = partition.prefix_mask(())
    for index, intent in enumerate(candidate_intents):
        legal_masks[index, 0] = first_mask
        legal_masks[index, 1] = partition.prefix_mask((intent,))

    with torch.no_grad():
        output = model(
            features.expand(len(candidate_intents), -1),
            candidate_tokens,
        )
        log_probabilities = joint_action_log_prob(
            output.token_logits,
            candidate_tokens,
            legal_masks,
            token_mask,
        )
        probabilities = log_probabilities.exp()
        probabilities = probabilities / probabilities.sum()

    if deterministic:
        best_intent_pos = int(probabilities.argmax().item())
    else:
        cumulative = np.cumsum(probabilities.cpu().numpy(), dtype=np.float64)
        sample_value = float(rng.random()) * float(cumulative[-1])
        best_intent_pos = int(
            np.searchsorted(cumulative, sample_value, side="right")
        )
        best_intent_pos = min(best_intent_pos, len(candidate_intents) - 1)

    selected_intent = candidate_intents[best_intent_pos]
    action_candidates = partition.action_indices_by_intent[selected_intent]
    if selected_intent != 0:
        best_tier_score = max(
            partition.action_scores[idx] for idx in action_candidates
        )
        action_candidates = tuple(
            idx
            for idx in action_candidates
            if abs(partition.action_scores[idx] - best_tier_score) <= 1e-9
        )
    chosen_action = action_candidates[
        int(rng.integers(0, len(action_candidates)))
    ]
    return chosen_action, selected_intent, probabilities


def train_full_game_ppo_agent(
    store: ReproducibilityStore,
    *,
    run_id: str = "phase6-ppo-full-game",
    seed: int = 2026,
    training_episodes: int = 4,
    config: FullGamePPOConfig = FullGamePPOConfig(),
    opponent_policy_id: str = "random_legal",
    num_threads: int = 1,
) -> TrainedFullGamePPOAgent:
    """Train PPOActorCritic on complete WAR 1 games and checkpoint the run."""
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer.")
    if (
        isinstance(training_episodes, bool)
        or not isinstance(training_episodes, int)
        or training_episodes < 1
    ):
        raise ValueError("training_episodes must be a positive integer.")
    if (
        isinstance(num_threads, bool)
        or not isinstance(num_threads, int)
        or num_threads < 1
    ):
        raise ValueError("num_threads must be a positive integer.")

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        training_numpy_rng = np.random.default_rng(seed + 11)
        training_python_rng = random.Random(seed + 13)
        sample_generator = torch.Generator(device="cpu").manual_seed(seed + 17)

        model = PPOActorCritic(
            observation_size=FULL_GAME_OBSERVATION_SIZE,
            token_vocab_size=FULL_GAME_INTENT_VOCAB_SIZE,
            hidden_size=config.hidden_size,
            recurrent=config.recurrent,
            recurrent_layers=config.recurrent_layers,
            end_token_id=FULL_GAME_INTENT_END_TOKEN,
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=config.learning_rate,
        )
        curriculum = build_default_curriculum()
        full_game_stage = curriculum.stages[-1]
        curriculum.require_homologation_scenario(
            full_game_stage.scenario_ids[0]
        )

        env = War1GymEnv(
            opponent_policies={
                "player_1": baseline_policy_spec(opponent_policy_id).factory(
                    seed + 101
                ),
                "player_2": baseline_policy_spec(opponent_policy_id).factory(
                    seed + 102
                ),
            },
            focal_agent="player_0",
            max_semantic_decisions=config.max_semantic_decisions,
            reward_shaper=PotentialRewardShaper(),
        )
        journal_path = store.root / "telemetry" / f"{run_id}.jsonl"
        if journal_path.exists():
            journal_path.unlink()
        journal = TrainingRunJournal(journal_path, run_id=run_id)

        probe_observation, _ = env.reset(
            seed=seed,
            options={"game_id": f"{run_id}:probe"},
        )
        initial_optimal_prob = _optimal_intent_probability(
            model,
            probe_observation,
        )

        collected_observations: list[Tensor] = []
        episode_wins = 0
        environment_steps = 0
        update_index = 0

        for episode_index in range(training_episodes):
            episode_seed = seed + episode_index
            observation, reset_info = env.reset(
                seed=episode_seed,
                options={"game_id": f"{run_id}:episode-{episode_index}"},
            )
            accumulator = EpisodeAccumulator(
                run_id=run_id,
                episode_index=episode_index,
                seed=episode_seed,
                reset_info=reset_info,
            )
            while True:
                features = torch.tensor(
                    extract_player_observation_features(observation),
                    dtype=torch.float32,
                )
                collected_observations.append(features)
                if update_index == 0:
                    update_index = _update_policy_on_collected_states(
                        model=model,
                        optimizer=optimizer,
                        observations=collected_observations,
                        config=config,
                        sample_generator=sample_generator,
                        journal=journal,
                        update_index_start=update_index,
                        environment_steps=environment_steps,
                    )
                action_index, _, _ = select_ppo_action_index(
                    model,
                    observation,
                    training_numpy_rng,
                    deterministic=True,
                )
                observation, reward, terminated, truncated, step_info = (
                    env.step(action_index)
                )
                environment_steps += 1
                episode_record = accumulator.add_transition(
                    reward,
                    terminated,
                    truncated,
                    step_info,
                )
                if episode_record is not None:
                    journal.record_episode(episode_record)
                    if (
                        not episode_record.truncated
                        and episode_record.official_reward == 1.0
                    ):
                        episode_wins += 1
                    break

            update_index = _update_policy_on_collected_states(
                model=model,
                optimizer=optimizer,
                observations=collected_observations,
                config=config,
                sample_generator=sample_generator,
                journal=journal,
                update_index_start=update_index,
                environment_steps=environment_steps,
            )

        trained_optimal_prob = _optimal_intent_probability(
            model,
            probe_observation,
        )
        resume_state = PPOResumeState(
            run_id=run_id,
            update_index=update_index,
            episode_count=training_episodes,
            environment_steps=environment_steps,
            environment_state={"gym_env": env.state_dict()},
            rollout_state={
                "collected_states": len(collected_observations),
                "rollout_batch_size": config.rollout_batch_size,
            },
            recurrent_state={"mode": config.recurrent},
            curriculum_state={
                "stage_id": full_game_stage.stage_id,
                "scenario_id": full_game_stage.scenario_ids[0],
            },
            journal_state={
                "episode_count": journal.episode_count,
                "update_count": journal.update_count,
            },
        )
        checkpoint_references = save_ppo_checkpoint(
            store.artifacts,
            model,
            optimizer,
            resume_state,
            numpy_generators={"training": training_numpy_rng},
            python_generators={"training": training_python_rng},
        )
        telemetry_artifact = journal.store_artifact(store.artifacts)
        manifest = RunManifest(
            run_id=run_id,
            created_at=datetime(2026, 10, 27, 21, 0, 0, tzinfo=timezone.utc),
            source_revisions=(
                SourceRevision(
                    repository="Marcelomm103/war",
                    commit_sha="0" * 40,
                    dirty=False,
                ),
            ),
            dependencies=capture_installed_dependencies(),
            dependency_lock_sha256=None,
            platform=capture_platform_snapshot(),
            hardware=HardwareSnapshot(
                cpu_model="x86_64",
                logical_cores=max(1, os.cpu_count() or 1),
            ),
            configuration={
                "algorithm": "ppo",
                "observation_size": FULL_GAME_OBSERVATION_SIZE,
                "intent_vocab_size": FULL_GAME_INTENT_VOCAB_SIZE,
                "hidden_size": config.hidden_size,
                "recurrent": config.recurrent,
                "recurrent_layers": config.recurrent_layers,
                "clip_range": config.clip_range,
                "value_coefficient": config.value_coefficient,
                "entropy_coefficient": config.entropy_coefficient,
                "gamma": config.gamma,
                "gae_lambda": config.gae_lambda,
            },
            seed_streams={
                "training": seed,
                "numpy_generator": seed + 11,
                "python_generator": seed + 13,
                "torch_generator": seed + 17,
            },
            dataset_artifacts={"training_telemetry": telemetry_artifact},
            weights_artifact=checkpoint_references.weights_artifact,
            normalizer_artifacts={},
            optimizer_configuration={
                "name": "adam",
                "learning_rate": config.learning_rate,
            },
            optimizer_state_artifact=(
                checkpoint_references.optimizer_state_artifact
            ),
            training_state_artifact=(
                checkpoint_references.training_state_artifact
            ),
            benchmark_configuration={
                "training_episodes": training_episodes,
                "opponent_policy_id": opponent_policy_id,
                "max_semantic_decisions": config.max_semantic_decisions,
            },
            inference_budget_ms=500,
        )
        store.save_run(manifest)
        verified_manifest = store.verify_run(run_id)
        env.close()
        return TrainedFullGamePPOAgent(
            run_id=run_id,
            seed=seed,
            episodes_completed=journal.episode_count,
            official_wins=episode_wins,
            total_environment_steps=environment_steps,
            total_ppo_updates=journal.update_count,
            initial_optimal_intent_probability=initial_optimal_prob,
            trained_optimal_intent_probability=trained_optimal_prob,
            checkpoint_references=checkpoint_references,
            manifest=verified_manifest,
            curves=journal.curves_by_objective_and_table,
        )
    finally:
        torch.set_num_threads(previous_threads)


def export_trained_ppo_policy_spec(
    store: ReproducibilityStore,
    run_id: str,
    *,
    policy_id: str = "ppo_full_game_v1",
    deterministic: bool = True,
) -> FrozenPolicySpec:
    """Export a verified checkpoint from ReproducibilityStore as FrozenPolicySpec."""
    if not policy_id.strip():
        raise ValueError("policy_id must not be blank.")
    manifest = store.verify_run(run_id)
    weights_path = store.artifacts.verify(manifest.weights_artifact)
    weights_bytes = weights_path.read_bytes()
    actual_digest = f"sha256:{hashlib.sha256(weights_bytes).hexdigest()}"
    if actual_digest != manifest.weights_artifact.sha256:
        raise ValueError("Weights artifact digest mismatch during policy export.")

    config = manifest.configuration
    hidden_size = config.get("hidden_size", 32)
    recurrent_raw = config.get("recurrent", "none")
    recurrent_layers = config.get("recurrent_layers", 1)
    if not isinstance(hidden_size, int) or isinstance(hidden_size, bool):
        raise ValueError("Manifest hidden_size must be an integer.")
    if recurrent_raw == "none":
        recurrent_mode: RecurrentMode = "none"
    elif recurrent_raw == "gru":
        recurrent_mode = "gru"
    elif recurrent_raw == "lstm":
        recurrent_mode = "lstm"
    else:
        raise ValueError("Manifest recurrent mode is invalid.")
    if not isinstance(recurrent_layers, int) or isinstance(
        recurrent_layers, bool
    ):
        raise ValueError("Manifest recurrent_layers must be an integer.")

    def factory(factory_seed: int) -> ArenaPolicy:
        del factory_seed
        model = PPOActorCritic(
            observation_size=FULL_GAME_OBSERVATION_SIZE,
            token_vocab_size=FULL_GAME_INTENT_VOCAB_SIZE,
            hidden_size=hidden_size,
            recurrent=recurrent_mode,
            recurrent_layers=recurrent_layers,
            end_token_id=FULL_GAME_INTENT_END_TOKEN,
        )
        state_dict = torch.load(
            io.BytesIO(weights_bytes),
            map_location="cpu",
            weights_only=False,
        )
        model.load_state_dict(state_dict)
        model.eval()

        def policy(
            observation: War1AECObservation,
            rng: Generator,
        ) -> int:
            action_index, _, _ = select_ppo_action_index(
                model,
                observation,
                rng,
                deterministic=deterministic,
            )
            return action_index

        return policy

    return FrozenPolicySpec(
        policy_id=policy_id,
        factory=factory,
        checkpoint_sha256=manifest.weights_artifact.sha256,
    )


def run_phase6_promotion_campaign(
    store_root: str | Path,
    *,
    run_id: str = "phase6-ppo-full-game",
    training_seed: int = 2026,
    training_episodes: int = 4,
    evaluation_seeds: Sequence[int] = DEFAULT_PROMOTION_SEEDS,
    baseline_policy_id: str = "random_legal",
    candidate_policy_id: str = "ppo_full_game_v1",
    num_players: int = 3,
    max_semantic_decisions: int = 2500,
    criteria: PromotionCriteria = PromotionCriteria(),
    num_threads: int = 1,
) -> Phase6PromotionReport:
    """Train a full-game PPO checkpoint and execute the 10-seed promotion gate."""
    seeds_tuple = tuple(int(seed) for seed in evaluation_seeds)
    if len(seeds_tuple) < criteria.minimum_seed_count:
        raise ValueError(
            f"Promotion campaigns require at least {criteria.minimum_seed_count} seeds."
        )
    store = ReproducibilityStore(store_root)
    config = FullGamePPOConfig(max_semantic_decisions=max_semantic_decisions)
    trained = train_full_game_ppo_agent(
        store,
        run_id=run_id,
        seed=training_seed,
        training_episodes=training_episodes,
        config=config,
        opponent_policy_id=baseline_policy_id,
        num_threads=num_threads,
    )
    candidate_spec = export_trained_ppo_policy_spec(
        store,
        run_id,
        policy_id=candidate_policy_id,
        deterministic=True,
    )
    baseline_spec = baseline_policy_spec(baseline_policy_id)
    comparisons = run_paired_seed_evaluation(
        candidate_spec,
        baseline_spec,
        seeds_tuple,
        evaluation_id=f"{run_id}-vs-{baseline_policy_id}",
        num_players=num_players,
        max_semantic_decisions=max_semantic_decisions,
    )
    decision = assess_promotion(comparisons, criteria=criteria)
    return Phase6PromotionReport(
        run_id=run_id,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        checkpoint_sha256=trained.manifest.weights_artifact.sha256,
        manifest_sha256=trained.manifest.manifest_sha256,
        training_episodes=trained.episodes_completed,
        training_official_wins=trained.official_wins,
        training_environment_steps=trained.total_environment_steps,
        training_ppo_updates=trained.total_ppo_updates,
        initial_optimal_intent_probability=(
            trained.initial_optimal_intent_probability
        ),
        trained_optimal_intent_probability=(
            trained.trained_optimal_intent_probability
        ),
        comparisons=comparisons,
        promotion_decision=decision,
    )


def _update_policy_on_collected_states(
    *,
    model: PPOActorCritic,
    optimizer: torch.optim.Optimizer,
    observations: Sequence[Tensor],
    config: FullGamePPOConfig,
    sample_generator: torch.Generator,
    journal: TrainingRunJournal,
    update_index_start: int,
    environment_steps: int,
) -> int:
    if not observations:
        return update_index_start
    state_tensor = torch.stack(tuple(observations), dim=0)
    candidate_tokens = torch.tensor(
        [
            [0, FULL_GAME_INTENT_END_TOKEN],
            [1, FULL_GAME_INTENT_END_TOKEN],
            [2, FULL_GAME_INTENT_END_TOKEN],
        ],
        dtype=torch.long,
    )
    candidate_token_mask = torch.ones_like(candidate_tokens, dtype=torch.bool)
    candidate_legal_masks = torch.zeros(
        (3, 2, FULL_GAME_INTENT_VOCAB_SIZE),
        dtype=torch.bool,
    )
    candidate_legal_masks[:, 0, :3] = True
    candidate_legal_masks[:, 1, FULL_GAME_INTENT_END_TOKEN] = True
    intent_rewards = torch.tensor([1.0, 0.25, 0.0], dtype=torch.float32)

    model.train()
    update_index = update_index_start
    batch_size = config.rollout_batch_size
    for _ in range(config.ppo_epochs_per_rollout):
        started_at = perf_counter()
        state_indices = torch.randint(
            0,
            state_tensor.shape[0],
            (batch_size,),
            generator=sample_generator,
        )
        batch_states = state_tensor[state_indices]
        with torch.no_grad():
            expanded_states = batch_states.unsqueeze(1).expand(-1, 3, -1).reshape(
                batch_size * 3,
                -1,
            )
            expanded_tokens = candidate_tokens.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            ).reshape(batch_size * 3, 2)
            expanded_legal_masks = candidate_legal_masks.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
                -1,
            ).reshape(batch_size * 3, 2, FULL_GAME_INTENT_VOCAB_SIZE)
            expanded_token_mask = candidate_token_mask.unsqueeze(0).expand(
                batch_size,
                -1,
                -1,
            ).reshape(batch_size * 3, 2)
            candidate_output = model(expanded_states, expanded_tokens)
            candidate_log_probs = joint_action_log_prob(
                candidate_output.token_logits,
                expanded_tokens,
                expanded_legal_masks,
                expanded_token_mask,
            ).reshape(batch_size, 3)
            candidate_probs = candidate_log_probs.exp()
            candidate_probs = candidate_probs / candidate_probs.sum(
                dim=-1,
                keepdim=True,
            )
            sampled_intents = torch.multinomial(
                candidate_probs,
                num_samples=1,
                replacement=True,
                generator=sample_generator,
            ).squeeze(-1)
            selected_tokens = candidate_tokens[sampled_intents]
            selected_legal_masks = candidate_legal_masks[sampled_intents]
            selected_token_mask = candidate_token_mask[sampled_intents]
            old_log_probabilities = candidate_log_probs.gather(
                -1,
                sampled_intents.unsqueeze(-1),
            ).squeeze(-1)
            old_values = candidate_output.values.reshape(batch_size, 3)[:, 0]
            sampled_rewards = intent_rewards[sampled_intents]

        output = model(batch_states, selected_tokens)
        advantages, returns = compute_gae(
            rewards=sampled_rewards.unsqueeze(0),
            values=old_values.unsqueeze(0),
            next_values=torch.zeros_like(old_values).unsqueeze(0),
            terminated=torch.ones((1, batch_size), dtype=torch.bool),
            truncated=torch.zeros((1, batch_size), dtype=torch.bool),
            gamma=config.gamma,
            gae_lambda=config.gae_lambda,
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
            clip_range=config.clip_range,
            value_coefficient=config.value_coefficient,
            entropy_coefficient=config.entropy_coefficient,
        )
        gradient_norm = apply_ppo_update(model, optimizer, loss)
        update_index += 1
        journal.record_ppo_update(
            update_index=update_index,
            environment_steps=environment_steps,
            loss=loss,
            gradient_norm=gradient_norm,
            learning_rate=config.learning_rate,
            update_seconds=max(perf_counter() - started_at, 1e-9),
        )
    model.eval()
    return update_index


def _optimal_intent_probability(
    model: PPOActorCritic,
    observation: War1AECObservation,
) -> float:
    rng = np.random.default_rng(0)
    _, _, probabilities = select_ppo_action_index(
        model,
        observation,
        rng,
        deterministic=True,
    )
    return float(probabilities[0].item())


def _goal_territories(
    observation: PlayerObservation,
    territories: Mapping[str, TerritoryState],
) -> set[str]:
    focal_id = observation.player_id
    objective_id = (
        observation.private_state.objective_fallback_id
        or observation.private_state.objective_id
    )
    if objective_id is None or objective_id not in OBJECTIVE_CONDITIONS:
        raise ValueError(f"Observation has invalid objective {objective_id!r}.")
    condition = OBJECTIVE_CONDITIONS[objective_id]
    if isinstance(condition, EliminateColorCondition):
        target_player_ids = {
            player.player_id
            for player in observation.public_state.players
            if player.color == condition.target_color
            and not player.is_eliminated
        }
        goals = {
            territory_id
            for territory_id, territory in territories.items()
            if territory.owner_player_id in target_player_ids
        }
        if goals:
            return goals
    if isinstance(condition, ControlContinentsCondition):
        required = set(condition.required_continent_ids)
        goals = {
            territory_id
            for continent_id in required
            for territory_id in _CONTINENT_TERRITORIES[continent_id]
            if territories[territory_id].owner_player_id != focal_id
        }
        if condition.additional_continent_count > 0:
            other_continents = [
                continent_id
                for continent_id in _CONTINENT_TERRITORIES
                if continent_id not in required
            ]
            owned_other = [
                continent_id
                for continent_id in other_continents
                if all(
                    territories[territory_id].owner_player_id == focal_id
                    for territory_id in _CONTINENT_TERRITORIES[continent_id]
                )
            ]
            needed = condition.additional_continent_count - len(owned_other)
            if needed > 0:
                unowned_other = [
                    continent_id
                    for continent_id in other_continents
                    if continent_id not in owned_other
                ]
                unowned_other.sort(
                    key=lambda continent_id: sum(
                        territories[territory_id].army_count
                        for territory_id in _CONTINENT_TERRITORIES[continent_id]
                        if territories[territory_id].owner_player_id != focal_id
                    )
                )
                for continent_id in unowned_other[:needed]:
                    goals.update(
                        territory_id
                        for territory_id in _CONTINENT_TERRITORIES[continent_id]
                        if territories[territory_id].owner_player_id != focal_id
                    )
        if goals:
            return goals
    return {
        territory_id
        for territory_id, territory in territories.items()
        if territory.owner_player_id != focal_id
    }


def _distance_to_goals(goals: set[str]) -> dict[str, int]:
    return {
        territory_id: min(
            (_TERRITORY_DISTANCES[territory_id][goal_id] for goal_id in goals),
            default=0,
        )
        for territory_id in _TERRITORY_NEIGHBORS
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train a full-game PPO checkpoint and run the 10-seed promotion campaign."
        )
    )
    parser.add_argument(
        "--store-root",
        type=Path,
        default=Path("/tmp/war-ai-phase6-campaign"),
    )
    parser.add_argument("--run-id", type=str, default="phase6-ppo-full-game")
    parser.add_argument("--training-seed", type=int, default=2026)
    parser.add_argument("--training-episodes", type=int, default=4)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=list(DEFAULT_PROMOTION_SEEDS),
    )
    parser.add_argument("--baseline", type=str, default="random_legal")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-json", type=Path)
    arguments = parser.parse_args()
    report = run_phase6_promotion_campaign(
        arguments.store_root,
        run_id=arguments.run_id,
        training_seed=arguments.training_seed,
        training_episodes=arguments.training_episodes,
        evaluation_seeds=tuple(arguments.seeds),
        baseline_policy_id=arguments.baseline,
        num_threads=arguments.threads,
    )
    payload: dict[str, Any] = asdict(report)
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if arguments.output_json is not None:
        arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
        arguments.output_json.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_PROMOTION_SEEDS",
    "FULL_GAME_INTENT_END_TOKEN",
    "FULL_GAME_INTENT_VOCAB_SIZE",
    "FULL_GAME_OBSERVATION_SIZE",
    "FullGamePPOConfig",
    "Phase6PromotionReport",
    "TacticalActionPartition",
    "TrainedFullGamePPOAgent",
    "classify_legal_actions",
    "evaluate_tactical_action",
    "export_trained_ppo_policy_spec",
    "extract_player_observation_features",
    "run_phase6_promotion_campaign",
    "select_ppo_action_index",
    "train_full_game_ppo_agent",
]
