"""Tests for Phase 7.5 hyperparameter search space and log-scale trials."""

from __future__ import annotations

import math

from war_ai.rl.experimental_axes import ComputeBudget
from war_ai.rl.hyperparameter_search import (
    INITIAL_ENTROPY_COEFFICIENTS,
    INITIAL_GAE_LAMBDAS,
    INITIAL_GRAPH_LAYERS,
    INITIAL_HIDDEN_SIZES,
    INITIAL_HISTORY_EVENTS,
    INITIAL_LR_RANGE,
    INITIAL_PPO_CLIPS,
    INITIAL_SEMANTIC_DISCOUNTS,
    HyperparameterSearchSpace,
    run_hyperparameter_search,
    sample_hyperparameter_trials,
    sample_log_uniform_learning_rates,
)


def test_initial_hyperparameter_space_matches_roadmap_table() -> None:
    space = HyperparameterSearchSpace()
    assert space.learning_rate_range == (1e-5, 3e-4)
    assert space.learning_rate_log_scale is True
    assert space.ppo_clips == (0.1, 0.2, 0.3)
    assert space.gae_lambdas == (0.90, 0.95, 0.98, 1.0)
    assert space.semantic_discounts == (0.99, 0.995, 0.999, 1.0)
    assert space.initial_entropies == (0.0, 0.01, 0.03, 0.05)
    assert space.graph_layers == (3, 6, 12)
    assert space.hidden_sizes == (128, 256, 512, 1024)
    assert space.history_events == (64, 256, 1024)


def test_log_uniform_lr_sampling_and_extensible_search_space() -> None:
    lrs = sample_log_uniform_learning_rates(16, seed=2026)
    assert len(lrs) == 16
    assert all(INITIAL_LR_RANGE[0] <= lr <= INITIAL_LR_RANGE[1] for lr in lrs)
    geometric_mean = 10.0 ** (
        sum(math.log10(lr) for lr in lrs) / len(lrs)
    )
    expected_geo_mean = math.sqrt(INITIAL_LR_RANGE[0] * INITIAL_LR_RANGE[1])
    assert abs(math.log10(geometric_mean) - math.log10(expected_geo_mean)) < 0.08

    space = HyperparameterSearchSpace()
    expanded = space.expand(
        learning_rate_range=(5e-6, 1e-3),
        extra_graph_layers=(16,),
        extra_hidden_sizes=(2048,),
        extra_history_events=(2048,),
    )
    assert 16 in expanded.graph_layers
    assert 2048 in expanded.hidden_sizes
    assert 2048 in expanded.history_events
    assert expanded.learning_rate_range == (5e-6, 1e-3)


def test_sample_and_run_hyperparameter_search_under_controlled_budget() -> None:
    space = HyperparameterSearchSpace()
    trials = sample_hyperparameter_trials(space, num_trials=6, seed=2026)
    assert len(trials) == 6
    for point in trials:
        assert point.ppo_clip in INITIAL_PPO_CLIPS
        assert point.gae_lambda in INITIAL_GAE_LAMBDAS
        assert point.semantic_discount in INITIAL_SEMANTIC_DISCOUNTS
        assert point.initial_entropy in INITIAL_ENTROPY_COEFFICIENTS
        assert point.graph_layers in INITIAL_GRAPH_LAYERS
        assert point.hidden_size in INITIAL_HIDDEN_SIZES
        assert point.history_events in INITIAL_HISTORY_EVENTS

    budget = ComputeBudget(
        semantic_decisions=32,
        learner_updates=4,
        inference_budget_ms=250,
    )
    report = run_hyperparameter_search(
        branch="graph_actor_critic",
        space=space,
        points=trials[:3],
        compute_budget=budget,
        seed=2026,
    )
    assert len(report.trials) == 3
    assert report.best_trial == report.trials[0]
    assert report.best_trial.trained_optimal_intent_probability >= (
        report.trials[-1].trained_optimal_intent_probability
    )
