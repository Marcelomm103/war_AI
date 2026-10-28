"""Training configuration sweeps and version-compatibility guards (Phase 7.6)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..action_tokens import ACTION_TOKEN_ENCODING_VERSION
from ..envs.observations import WAR1_RL_OBSERVATION_ENCODING_VERSION
from .algorithms import evaluate_algorithm_branch
from .curriculum import CurriculumScenarioKind
from .experimental_axes import ActionRepresentationAxis, AlgorithmAxis, ComputeBudget

NormalizationMode: TypeAlias = Literal[
    "none",
    "advantage_only",
    "advantage_and_observation",
    "advantage_value_and_observation",
]
AdversarySamplingMode: TypeAlias = Literal[
    "uniform_league",
    "prioritized_pfsp",
    "historical_plus_baselines",
]

_VALID_OBJECTIVE_IDS = tuple(
    objective.objective_id for objective in WAR1_BOARD_SPEC.objectives
)
_DEFAULT_CURRICULUM_PROPORTIONS: tuple[tuple[CurriculumScenarioKind, float], ...] = (
    ("tactical", 0.10),
    ("intermediate", 0.20),
    ("full_game", 0.70),
)
_DEFAULT_TABLE_SIZE_DISTRIBUTION: tuple[tuple[int, float], ...] = (
    (3, 0.25),
    (4, 0.25),
    (5, 0.25),
    (6, 0.25),
)
_DEFAULT_OBJECTIVE_DISTRIBUTION: tuple[tuple[str, float], ...] = tuple(
    (objective_id, 1.0 / len(_VALID_OBJECTIVE_IDS))
    for objective_id in _VALID_OBJECTIVE_IDS
)


class IncompatibleTrajectoryDataError(ValueError):
    """Raised when attempting to mix trajectory data from incompatible versions."""


@dataclass(frozen=True)
class TrajectoryBatchStamp:
    """Version and policy provenance stamp attached to every trajectory batch."""

    batch_id: str
    ruleset_id: str = WAR1_BOARD_SPEC.board_id
    rules_version: str = WAR1_BOARD_SPEC.version
    observation_encoding_version: str = WAR1_RL_OBSERVATION_ENCODING_VERSION
    action_token_encoding_version: str = ACTION_TOKEN_ENCODING_VERSION
    action_representation: ActionRepresentationAxis = "tactical_intent_tokens"
    actor_policy_version: int = 0
    transition_count: int = 32

    def __post_init__(self) -> None:
        for label, text in (
            ("batch_id", self.batch_id),
            ("ruleset_id", self.ruleset_id),
            ("rules_version", self.rules_version),
            ("observation_encoding_version", self.observation_encoding_version),
            ("action_token_encoding_version", self.action_token_encoding_version),
        ):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"{label} must be a non-empty string.")
        if (
            isinstance(self.actor_policy_version, bool)
            or not isinstance(self.actor_policy_version, int)
            or self.actor_policy_version < 0
        ):
            raise ValueError("actor_policy_version must be a non-negative integer.")
        if (
            isinstance(self.transition_count, bool)
            or not isinstance(self.transition_count, int)
            or self.transition_count < 1
        ):
            raise ValueError("transition_count must be a positive integer.")


def _validate_probability_distribution(
    entries: Sequence[tuple[object, float]],
    *,
    label: str,
) -> None:
    if not entries:
        raise ValueError(f"{label} must not be empty.")
    keys = [key for key, _ in entries]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{label} keys must be unique.")
    total = 0.0
    for _, weight in entries:
        if (
            isinstance(weight, bool)
            or not math.isfinite(weight)
            or weight < 0.0
        ):
            raise ValueError(f"{label} weights must be finite and non-negative.")
        total += float(weight)
    if abs(total - 1.0) > 1e-5:
        raise ValueError(f"{label} weights must sum to 1.0 (got {total:.6f}).")


@dataclass(frozen=True)
class TrainingRegimeConfig:
    """Complete training regime specification for Phase 7.6 sweeps."""

    config_id: str
    batch_size: int = 32
    rollout_length: int = 64
    num_workers: int = 4
    burn_in_steps: int = 8
    tbptt_length: int = 16
    replay_to_update_ratio: float = 4.0
    max_policy_lag: int = 2
    normalization_mode: NormalizationMode = "advantage_and_observation"
    adversary_sampling: AdversarySamplingMode = "prioritized_pfsp"
    curriculum_proportions: tuple[tuple[CurriculumScenarioKind, float], ...] = (
        _DEFAULT_CURRICULUM_PROPORTIONS
    )
    table_size_distribution: tuple[tuple[int, float], ...] = (
        _DEFAULT_TABLE_SIZE_DISTRIBUTION
    )
    objective_distribution: tuple[tuple[str, float], ...] = (
        _DEFAULT_OBJECTIVE_DISTRIBUTION
    )

    def __post_init__(self) -> None:
        if not self.config_id.strip():
            raise ValueError("config_id must not be blank.")
        for label, value in (
            ("batch_size", self.batch_size),
            ("rollout_length", self.rollout_length),
            ("num_workers", self.num_workers),
            ("tbptt_length", self.tbptt_length),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")
        if (
            isinstance(self.burn_in_steps, bool)
            or not isinstance(self.burn_in_steps, int)
            or self.burn_in_steps < 0
        ):
            raise ValueError("burn_in_steps must be a non-negative integer.")
        if self.burn_in_steps >= self.tbptt_length:
            raise ValueError(
                "burn_in_steps must be strictly less than tbptt_length."
            )
        if self.tbptt_length > self.rollout_length:
            raise ValueError("tbptt_length cannot exceed rollout_length.")
        if (
            isinstance(self.replay_to_update_ratio, bool)
            or not math.isfinite(self.replay_to_update_ratio)
            or self.replay_to_update_ratio <= 0.0
        ):
            raise ValueError("replay_to_update_ratio must be positive.")
        if (
            isinstance(self.max_policy_lag, bool)
            or not isinstance(self.max_policy_lag, int)
            or self.max_policy_lag < 0
        ):
            raise ValueError("max_policy_lag must be a non-negative integer.")
        if self.normalization_mode not in (
            "none",
            "advantage_only",
            "advantage_and_observation",
            "advantage_value_and_observation",
        ):
            raise ValueError(
                f"Unsupported normalization_mode {self.normalization_mode!r}."
            )
        if self.adversary_sampling not in (
            "uniform_league",
            "prioritized_pfsp",
            "historical_plus_baselines",
        ):
            raise ValueError(
                f"Unsupported adversary_sampling {self.adversary_sampling!r}."
            )

        _validate_probability_distribution(
            self.curriculum_proportions,
            label="curriculum_proportions",
        )
        curriculum_map = dict(self.curriculum_proportions)
        if set(curriculum_map) - {"tactical", "intermediate", "full_game"}:
            raise ValueError("Unknown curriculum stage kind in proportions.")
        if curriculum_map.get("full_game", 0.0) <= 0.0:
            raise ValueError(
                "curriculum_proportions must maintain a positive full_game share."
            )

        _validate_probability_distribution(
            self.table_size_distribution,
            label="table_size_distribution",
        )
        min_players, max_players = PLAYER_COUNT_RANGE
        for table_size, _ in self.table_size_distribution:
            if (
                isinstance(table_size, bool)
                or not isinstance(table_size, int)
                or not min_players <= table_size <= max_players
            ):
                raise ValueError(
                    f"Invalid table size {table_size!r} in table_size_distribution."
                )

        _validate_probability_distribution(
            self.objective_distribution,
            label="objective_distribution",
        )
        unknown_objectives = {
            obj_id for obj_id, _ in self.objective_distribution
        } - set(_VALID_OBJECTIVE_IDS)
        if unknown_objectives:
            raise ValueError(
                f"Unknown objectives in objective_distribution: "
                f"{sorted(unknown_objectives)}."
            )


def verify_trajectory_batch_compatibility(
    expected: TrajectoryBatchStamp,
    incoming: TrajectoryBatchStamp,
    *,
    learner_policy_version: int,
    max_policy_lag: int,
) -> None:
    """Reject incompatible encoding/rules versions or excessive policy lag."""
    for field_name in (
        "ruleset_id",
        "rules_version",
        "observation_encoding_version",
        "action_token_encoding_version",
        "action_representation",
    ):
        expected_val = getattr(expected, field_name)
        incoming_val = getattr(incoming, field_name)
        if expected_val != incoming_val:
            raise IncompatibleTrajectoryDataError(
                f"Cannot mix trajectory batch {incoming.batch_id!r}: "
                f"incompatible {field_name} ({incoming_val!r} != {expected_val!r})."
            )
    if incoming.actor_policy_version > learner_policy_version:
        raise IncompatibleTrajectoryDataError(
            f"Trajectory batch {incoming.batch_id!r} originates from future policy "
            f"version {incoming.actor_policy_version} > {learner_policy_version}."
        )
    policy_lag = learner_policy_version - incoming.actor_policy_version
    if policy_lag > max_policy_lag:
        raise IncompatibleTrajectoryDataError(
            f"Trajectory batch {incoming.batch_id!r} exceeds max_policy_lag "
            f"({policy_lag} > {max_policy_lag})."
        )


class VersionedReplayBuffer:
    """Replay buffer that refuses to mix incompatible trajectory versions."""

    def __init__(
        self,
        reference_stamp: TrajectoryBatchStamp,
        *,
        max_policy_lag: int = 2,
        capacity: int = 256,
    ) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("capacity must be a positive integer.")
        self.reference_stamp = reference_stamp
        self.max_policy_lag = max_policy_lag
        self.capacity = capacity
        self._batches: list[TrajectoryBatchStamp] = []

    @property
    def batches(self) -> tuple[TrajectoryBatchStamp, ...]:
        return tuple(self._batches)

    def add_batch(
        self,
        batch: TrajectoryBatchStamp,
        *,
        learner_policy_version: int,
    ) -> None:
        verify_trajectory_batch_compatibility(
            self.reference_stamp,
            batch,
            learner_policy_version=learner_policy_version,
            max_policy_lag=self.max_policy_lag,
        )
        if len(self._batches) >= self.capacity:
            self._batches.pop(0)
        self._batches.append(batch)

    def collect_compatible_batches(
        self,
        *,
        learner_policy_version: int,
    ) -> tuple[TrajectoryBatchStamp, ...]:
        """Return all stored batches that remain within ``max_policy_lag``."""
        compatible: list[TrajectoryBatchStamp] = []
        for batch in self._batches:
            verify_trajectory_batch_compatibility(
                self.reference_stamp,
                batch,
                learner_policy_version=learner_policy_version,
                max_policy_lag=self.max_policy_lag,
            )
            compatible.append(batch)
        return tuple(compatible)


@dataclass(frozen=True)
class TrainingRegimeSweepResult:
    """Evaluation summary for one training regime configuration."""

    regime: TrainingRegimeConfig
    branch: AlgorithmAxis
    compute_budget: ComputeBudget
    trained_optimal_intent_probability: float
    final_loss: float
    effective_throughput_decisions_per_sec: float


def build_training_regime_sweep(
    baseline: TrainingRegimeConfig = TrainingRegimeConfig(config_id="regime-baseline"),
) -> tuple[TrainingRegimeConfig, ...]:
    """Build a structured sweep across all Phase 7.6 training regime dimensions."""
    return (
        baseline,
        replace(
            baseline,
            config_id="regime-large-batch-rollout",
            batch_size=64,
            rollout_length=128,
        ),
        replace(
            baseline,
            config_id="regime-8-workers",
            num_workers=8,
        ),
        replace(
            baseline,
            config_id="regime-long-burnin-tbptt",
            burn_in_steps=16,
            tbptt_length=32,
        ),
        replace(
            baseline,
            config_id="regime-high-replay-ratio",
            replay_to_update_ratio=8.0,
        ),
        replace(
            baseline,
            config_id="regime-strict-policy-lag",
            max_policy_lag=1,
        ),
        replace(
            baseline,
            config_id="regime-full-normalization",
            normalization_mode="advantage_value_and_observation",
        ),
        replace(
            baseline,
            config_id="regime-historical-plus-baselines-sampling",
            adversary_sampling="historical_plus_baselines",
        ),
        replace(
            baseline,
            config_id="regime-heavy-full-game-curriculum",
            curriculum_proportions=(
                ("tactical", 0.05),
                ("intermediate", 0.10),
                ("full_game", 0.85),
            ),
        ),
        replace(
            baseline,
            config_id="regime-focused-table-distribution",
            table_size_distribution=(
                (3, 0.40),
                (4, 0.30),
                (5, 0.15),
                (6, 0.15),
            ),
        ),
    )


def evaluate_training_regime_sweep(
    regimes: Sequence[TrainingRegimeConfig] | None = None,
    *,
    branch: AlgorithmAxis = "ppo_recurrent",
    compute_budget: ComputeBudget = ComputeBudget(
        semantic_decisions=32,
        learner_updates=6,
        inference_budget_ms=250,
    ),
    seed: int = 2026,
) -> tuple[TrainingRegimeSweepResult, ...]:
    """Evaluate a sweep of training regimes under a controlled compute budget."""
    candidates = (
        build_training_regime_sweep() if regimes is None else tuple(regimes)
    )
    if not candidates:
        raise ValueError("regimes must not be empty.")
    results: list[TrainingRegimeSweepResult] = []
    for idx, regime in enumerate(candidates):
        eval_result = evaluate_algorithm_branch(
            branch,
            compute_budget=compute_budget,
            hidden_size=32,
            seed=seed + idx,
        )
        duration_sec = max(1e-6, eval_result.training_wall_time_ms / 1000.0)
        throughput = compute_budget.semantic_decisions / duration_sec
        results.append(
            TrainingRegimeSweepResult(
                regime=regime,
                branch=branch,
                compute_budget=compute_budget,
                trained_optimal_intent_probability=(
                    eval_result.trained_optimal_intent_probability
                ),
                final_loss=eval_result.final_loss,
                effective_throughput_decisions_per_sec=throughput,
            )
        )
    return tuple(results)


__all__ = [
    "AdversarySamplingMode",
    "IncompatibleTrajectoryDataError",
    "NormalizationMode",
    "TrainingRegimeConfig",
    "TrainingRegimeSweepResult",
    "TrajectoryBatchStamp",
    "VersionedReplayBuffer",
    "build_training_regime_sweep",
    "evaluate_training_regime_sweep",
    "verify_trajectory_batch_compatibility",
]
