"""Tests for Phase 7.6 training regime sweeps and version compatibility guards."""

from __future__ import annotations

from dataclasses import replace

import pytest

from war_ai.rl.experimental_axes import ComputeBudget
from war_ai.rl.training_sweeps import (
    IncompatibleTrajectoryDataError,
    TrainingRegimeConfig,
    TrajectoryBatchStamp,
    VersionedReplayBuffer,
    build_training_regime_sweep,
    evaluate_training_regime_sweep,
    verify_trajectory_batch_compatibility,
)


def test_training_regime_sweep_covers_all_phase_7_6_dimensions() -> None:
    regimes = build_training_regime_sweep()
    assert len(regimes) == 10

    budget = ComputeBudget(
        semantic_decisions=32,
        learner_updates=4,
        inference_budget_ms=250,
    )
    results = evaluate_training_regime_sweep(
        regimes[:4],
        branch="ppo_recurrent",
        compute_budget=budget,
        seed=2026,
    )
    assert len(results) == 4
    for item in results:
        assert item.compute_budget == budget
        assert item.effective_throughput_decisions_per_sec > 0.0
        assert 0.0 <= item.trained_optimal_intent_probability <= 1.0


def test_incompatible_trajectory_versions_are_never_mixed_silently() -> None:
    reference = TrajectoryBatchStamp(
        batch_id="batch-ref",
        actor_policy_version=5,
    )
    buffer = VersionedReplayBuffer(reference, max_policy_lag=2, capacity=8)
    buffer.add_batch(
        TrajectoryBatchStamp(batch_id="batch-ok-1", actor_policy_version=4),
        learner_policy_version=5,
    )
    assert len(buffer.collect_compatible_batches(learner_policy_version=5)) == 1

    # Incompatible observation encoding version
    wrong_obs = replace(
        reference,
        batch_id="batch-bad-obs",
        observation_encoding_version="0.9.0",
    )
    with pytest.raises(
        IncompatibleTrajectoryDataError,
        match="observation_encoding_version",
    ):
        buffer.add_batch(wrong_obs, learner_policy_version=5)

    # Incompatible action token encoding version
    wrong_tokens = replace(
        reference,
        batch_id="batch-bad-tokens",
        action_token_encoding_version="2.0.0",
    )
    with pytest.raises(
        IncompatibleTrajectoryDataError,
        match="action_token_encoding_version",
    ):
        buffer.add_batch(wrong_tokens, learner_policy_version=5)

    # Incompatible ruleset version
    wrong_rules = replace(
        reference,
        batch_id="batch-bad-rules",
        rules_version="2.0.0",
    )
    with pytest.raises(
        IncompatibleTrajectoryDataError,
        match="rules_version",
    ):
        buffer.add_batch(wrong_rules, learner_policy_version=5)

    # Excessive policy lag
    stale_batch = replace(
        reference,
        batch_id="batch-stale",
        actor_policy_version=2,
    )
    with pytest.raises(
        IncompatibleTrajectoryDataError,
        match="exceeds max_policy_lag",
    ):
        verify_trajectory_batch_compatibility(
            reference,
            stale_batch,
            learner_policy_version=5,
            max_policy_lag=2,
        )


def test_training_regime_validates_burn_in_and_distributions() -> None:
    with pytest.raises(ValueError, match="burn_in_steps"):
        TrainingRegimeConfig(
            config_id="invalid-burn-in",
            burn_in_steps=16,
            tbptt_length=16,
        )

    with pytest.raises(ValueError, match="sum to 1.0"):
        TrainingRegimeConfig(
            config_id="invalid-curriculum-sum",
            curriculum_proportions=(
                ("tactical", 0.5),
                ("full_game", 0.2),
            ),
        )
