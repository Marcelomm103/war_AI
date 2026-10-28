"""Hyperparameter search space and log-scale trial runner for WAR 1 (Phase 7.5)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np

from .algorithms import evaluate_algorithm_branch
from .experimental_axes import AlgorithmAxis, ComputeBudget

INITIAL_LR_RANGE: tuple[float, float] = (1e-5, 3e-4)
INITIAL_PPO_CLIPS: tuple[float, ...] = (0.1, 0.2, 0.3)
INITIAL_GAE_LAMBDAS: tuple[float, ...] = (0.90, 0.95, 0.98, 1.0)
INITIAL_SEMANTIC_DISCOUNTS: tuple[float, ...] = (0.99, 0.995, 0.999, 1.0)
INITIAL_ENTROPY_COEFFICIENTS: tuple[float, ...] = (0.0, 0.01, 0.03, 0.05)
INITIAL_GRAPH_LAYERS: tuple[int, ...] = (3, 6, 12)
INITIAL_HIDDEN_SIZES: tuple[int, ...] = (128, 256, 512, 1024)
INITIAL_HISTORY_EVENTS: tuple[int, ...] = (64, 256, 1024)


@dataclass(frozen=True)
class HyperparameterPoint:
    """One concrete hyperparameter assignment (initial grid or extended)."""

    trial_id: str
    learning_rate: float = 1e-4
    ppo_clip: float = 0.2
    gae_lambda: float = 0.95
    semantic_discount: float = 0.999
    initial_entropy: float = 0.01
    graph_layers: int = 3
    hidden_size: int = 128
    history_events: int = 64

    def __post_init__(self) -> None:
        if not self.trial_id.strip():
            raise ValueError("trial_id must not be blank.")
        if (
            isinstance(self.learning_rate, bool)
            or not math.isfinite(self.learning_rate)
            or self.learning_rate <= 0.0
        ):
            raise ValueError("learning_rate must be finite and positive.")
        if (
            isinstance(self.ppo_clip, bool)
            or not math.isfinite(self.ppo_clip)
            or not 0.0 < self.ppo_clip < 1.0
        ):
            raise ValueError("ppo_clip must be in (0, 1).")
        if (
            isinstance(self.gae_lambda, bool)
            or not math.isfinite(self.gae_lambda)
            or not 0.0 <= self.gae_lambda <= 1.0
        ):
            raise ValueError("gae_lambda must be in [0, 1].")
        if (
            isinstance(self.semantic_discount, bool)
            or not math.isfinite(self.semantic_discount)
            or not 0.0 < self.semantic_discount <= 1.0
        ):
            raise ValueError("semantic_discount must be in (0, 1].")
        if (
            isinstance(self.initial_entropy, bool)
            or not math.isfinite(self.initial_entropy)
            or self.initial_entropy < 0.0
        ):
            raise ValueError("initial_entropy must be non-negative.")
        for label, value in (
            ("graph_layers", self.graph_layers),
            ("hidden_size", self.hidden_size),
            ("history_events", self.history_events),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")


@dataclass(frozen=True)
class HyperparameterSearchSpace:
    """Initial exploration bounds that can be expanded without arbitrary caps."""

    learning_rate_range: tuple[float, float] = INITIAL_LR_RANGE
    learning_rate_log_scale: bool = True
    ppo_clips: tuple[float, ...] = INITIAL_PPO_CLIPS
    gae_lambdas: tuple[float, ...] = INITIAL_GAE_LAMBDAS
    semantic_discounts: tuple[float, ...] = INITIAL_SEMANTIC_DISCOUNTS
    initial_entropies: tuple[float, ...] = INITIAL_ENTROPY_COEFFICIENTS
    graph_layers: tuple[int, ...] = INITIAL_GRAPH_LAYERS
    hidden_sizes: tuple[int, ...] = INITIAL_HIDDEN_SIZES
    history_events: tuple[int, ...] = INITIAL_HISTORY_EVENTS

    def __post_init__(self) -> None:
        lr_min, lr_max = self.learning_rate_range
        if (
            not math.isfinite(lr_min)
            or not math.isfinite(lr_max)
            or lr_min <= 0.0
            or lr_min > lr_max
        ):
            raise ValueError(
                "learning_rate_range must satisfy 0 < lr_min <= lr_max."
            )
        if not self.learning_rate_log_scale:
            raise ValueError("Learning rate exploration requires log scale.")
        for label, values in (
            ("ppo_clips", self.ppo_clips),
            ("gae_lambdas", self.gae_lambdas),
            ("semantic_discounts", self.semantic_discounts),
            ("initial_entropies", self.initial_entropies),
            ("graph_layers", self.graph_layers),
            ("hidden_sizes", self.hidden_sizes),
            ("history_events", self.history_events),
        ):
            if not values:
                raise ValueError(f"{label} must not be empty.")
        for idx, ppo_clip in enumerate(self.ppo_clips):
            HyperparameterPoint(
                trial_id=f"validate-clip-{idx}",
                ppo_clip=ppo_clip,
            )
        for idx, gae_lambda in enumerate(self.gae_lambdas):
            HyperparameterPoint(
                trial_id=f"validate-gae-{idx}",
                gae_lambda=gae_lambda,
            )
        for idx, discount in enumerate(self.semantic_discounts):
            HyperparameterPoint(
                trial_id=f"validate-discount-{idx}",
                semantic_discount=discount,
            )
        for idx, entropy in enumerate(self.initial_entropies):
            HyperparameterPoint(
                trial_id=f"validate-entropy-{idx}",
                initial_entropy=entropy,
            )
        for idx, layers in enumerate(self.graph_layers):
            HyperparameterPoint(
                trial_id=f"validate-layers-{idx}",
                graph_layers=layers,
            )
        for idx, hidden in enumerate(self.hidden_sizes):
            HyperparameterPoint(
                trial_id=f"validate-hidden-{idx}",
                hidden_size=hidden,
            )
        for idx, history in enumerate(self.history_events):
            HyperparameterPoint(
                trial_id=f"validate-history-{idx}",
                history_events=history,
            )

    def expand(
        self,
        *,
        learning_rate_range: tuple[float, float] | None = None,
        extra_ppo_clips: Sequence[float] = (),
        extra_gae_lambdas: Sequence[float] = (),
        extra_semantic_discounts: Sequence[float] = (),
        extra_initial_entropies: Sequence[float] = (),
        extra_graph_layers: Sequence[int] = (),
        extra_hidden_sizes: Sequence[int] = (),
        extra_history_events: Sequence[int] = (),
    ) -> HyperparameterSearchSpace:
        """Return an expanded search space beyond the initial exploration grid."""
        return replace(
            self,
            learning_rate_range=(
                self.learning_rate_range
                if learning_rate_range is None
                else learning_rate_range
            ),
            ppo_clips=tuple(sorted({*self.ppo_clips, *extra_ppo_clips})),
            gae_lambdas=tuple(sorted({*self.gae_lambdas, *extra_gae_lambdas})),
            semantic_discounts=tuple(
                sorted({*self.semantic_discounts, *extra_semantic_discounts})
            ),
            initial_entropies=tuple(
                sorted({*self.initial_entropies, *extra_initial_entropies})
            ),
            graph_layers=tuple(sorted({*self.graph_layers, *extra_graph_layers})),
            hidden_sizes=tuple(sorted({*self.hidden_sizes, *extra_hidden_sizes})),
            history_events=tuple(
                sorted({*self.history_events, *extra_history_events})
            ),
        )


@dataclass(frozen=True)
class HyperparameterTrialResult:
    """Evaluation outcome for a single hyperparameter point."""

    point: HyperparameterPoint
    branch: AlgorithmAxis
    compute_budget: ComputeBudget
    trained_optimal_intent_probability: float
    final_loss: float
    wall_time_ms: float


@dataclass(frozen=True)
class HyperparameterSearchReport:
    """Ranked results of a controlled-budget hyperparameter search."""

    branch: AlgorithmAxis
    search_space: HyperparameterSearchSpace
    compute_budget: ComputeBudget
    trials: tuple[HyperparameterTrialResult, ...]
    best_trial: HyperparameterTrialResult


def sample_log_uniform_learning_rates(
    count: int,
    *,
    lr_range: tuple[float, float] = INITIAL_LR_RANGE,
    seed: int = 2026,
) -> tuple[float, ...]:
    """Sample ``count`` stratified log-uniform learning rates in ``lr_range``."""
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise ValueError("count must be a positive integer.")
    lr_min, lr_max = lr_range
    if lr_min <= 0.0 or lr_min > lr_max:
        raise ValueError("lr_range must satisfy 0 < lr_min <= lr_max.")
    if lr_min == lr_max:
        return tuple(float(lr_min) for _ in range(count))

    rng = np.random.default_rng(seed)
    log_min = math.log10(lr_min)
    log_max = math.log10(lr_max)
    strata = np.linspace(0.0, 1.0, count + 1)
    u = strata[:-1] + rng.random(count) * (strata[1:] - strata[:-1])
    rng.shuffle(u)
    return tuple(float(10.0 ** (log_min + val * (log_max - log_min))) for val in u)


def sample_hyperparameter_trials(
    space: HyperparameterSearchSpace = HyperparameterSearchSpace(),
    *,
    num_trials: int = 8,
    seed: int = 2026,
) -> tuple[HyperparameterPoint, ...]:
    """Sample reproducible hyperparameter configurations from ``space``."""
    if isinstance(num_trials, bool) or not isinstance(num_trials, int) or num_trials < 1:
        raise ValueError("num_trials must be a positive integer.")
    rng = np.random.default_rng(seed)
    lrs = sample_log_uniform_learning_rates(
        num_trials,
        lr_range=space.learning_rate_range,
        seed=seed,
    )
    points: list[HyperparameterPoint] = []
    for idx in range(num_trials):
        points.append(
            HyperparameterPoint(
                trial_id=f"hp-trial-{idx:03d}",
                learning_rate=lrs[idx],
                ppo_clip=float(space.ppo_clips[int(rng.integers(0, len(space.ppo_clips)))]),
                gae_lambda=float(
                    space.gae_lambdas[int(rng.integers(0, len(space.gae_lambdas)))]
                ),
                semantic_discount=float(
                    space.semantic_discounts[
                        int(rng.integers(0, len(space.semantic_discounts)))
                    ]
                ),
                initial_entropy=float(
                    space.initial_entropies[
                        int(rng.integers(0, len(space.initial_entropies)))
                    ]
                ),
                graph_layers=int(
                    space.graph_layers[int(rng.integers(0, len(space.graph_layers)))]
                ),
                hidden_size=int(
                    space.hidden_sizes[int(rng.integers(0, len(space.hidden_sizes)))]
                ),
                history_events=int(
                    space.history_events[
                        int(rng.integers(0, len(space.history_events)))
                    ]
                ),
            )
        )
    return tuple(points)


def run_hyperparameter_search(
    *,
    branch: AlgorithmAxis = "graph_actor_critic",
    space: HyperparameterSearchSpace = HyperparameterSearchSpace(),
    points: Sequence[HyperparameterPoint] | None = None,
    num_trials: int = 4,
    compute_budget: ComputeBudget = ComputeBudget(
        semantic_decisions=32,
        learner_updates=6,
        inference_budget_ms=250,
    ),
    seed: int = 2026,
) -> HyperparameterSearchReport:
    """Evaluate hyperparameter trials under a fixed compute budget and rank them."""
    trial_points = (
        tuple(points)
        if points is not None
        else sample_hyperparameter_trials(
            space,
            num_trials=num_trials,
            seed=seed,
        )
    )
    if not trial_points:
        raise ValueError("Hyperparameter search requires at least one trial point.")

    results: list[HyperparameterTrialResult] = []
    for idx, point in enumerate(trial_points):
        eval_hidden = min(point.hidden_size, 128)
        evaluation = evaluate_algorithm_branch(
            branch,
            compute_budget=compute_budget,
            hidden_size=eval_hidden,
            learning_rate=point.learning_rate,
            seed=seed + idx,
        )
        results.append(
            HyperparameterTrialResult(
                point=point,
                branch=branch,
                compute_budget=compute_budget,
                trained_optimal_intent_probability=(
                    evaluation.trained_optimal_intent_probability
                ),
                final_loss=evaluation.final_loss,
                wall_time_ms=evaluation.training_wall_time_ms,
            )
        )

    ranked = tuple(
        sorted(
            results,
            key=lambda item: (
                -item.trained_optimal_intent_probability,
                item.final_loss,
            ),
        )
    )
    return HyperparameterSearchReport(
        branch=branch,
        search_space=space,
        compute_budget=compute_budget,
        trials=ranked,
        best_trial=ranked[0],
    )


__all__ = [
    "HyperparameterPoint",
    "HyperparameterSearchReport",
    "HyperparameterSearchSpace",
    "HyperparameterTrialResult",
    "INITIAL_ENTROPY_COEFFICIENTS",
    "INITIAL_GAE_LAMBDAS",
    "INITIAL_GRAPH_LAYERS",
    "INITIAL_HIDDEN_SIZES",
    "INITIAL_HISTORY_EVENTS",
    "INITIAL_LR_RANGE",
    "INITIAL_PPO_CLIPS",
    "INITIAL_SEMANTIC_DISCOUNTS",
    "run_hyperparameter_search",
    "sample_hyperparameter_trials",
    "sample_log_uniform_learning_rates",
]
