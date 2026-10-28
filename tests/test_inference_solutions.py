"""Tests for inference solution comparisons, portfolio isolation, and distillation (Phase 9.5)."""

from __future__ import annotations

import pytest

from war_ai.envs.war1_aec import War1AECEnv
from war_ai.rl.architectures import ArchitectureConfig, WarArchitectureModel
from war_ai.rl.inference_solutions import (
    INFERENCE_SOLUTION_FAMILIES,
    OpponentIdentityLeakageError,
    SecretBlindPolicyPortfolio,
    compare_inference_solutions,
    validate_portfolio_selector_inputs,
)


def test_portfolio_rejects_opponent_internal_identity_and_privileged_state() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=951)
    active_agent = env.agent_selection
    assert active_agent is not None
    obs = env.observe(active_agent)["engine_observation"]
    assert obs is not None

    m1 = WarArchitectureModel(ArchitectureConfig(hidden_size=32))
    m2 = WarArchitectureModel(ArchitectureConfig(hidden_size=32))
    portfolio = SecretBlindPolicyPortfolio((m1, m2))

    with pytest.raises(OpponentIdentityLeakageError):
        portfolio.select_action(
            obs,
            opponent_policy_ids=("exploiter_alpha", "exploiter_beta"),
        )
    with pytest.raises(OpponentIdentityLeakageError):
        portfolio.compute_context_weights(
            obs,
            opponent_internal_metadata={"opponent_checkpoint_sha": "abc123"},
        )
    with pytest.raises(OpponentIdentityLeakageError):
        validate_portfolio_selector_inputs(env._require_state())

    act, idx, dist = portfolio.select_action(obs)
    assert act in obs.legal_actions
    assert act == obs.legal_actions[idx]
    assert abs(sum(dist.values()) - 1.0) < 1e-6


def test_compare_all_six_inference_solutions_and_distillation() -> None:
    report = compare_inference_solutions(
        num_games=2,
        steps_per_game=24,
        budget_levels=(4, 10, 18),
        seed=952,
    )
    assert report.num_evaluated_decisions >= 4
    assert report.portfolio_identity_blindness_verified
    assert set(report.evaluations_by_solution) == set(INFERENCE_SOLUTION_FAMILIES)
    assert len(report.budget_scaling_curve) == 3

    # Verify distillation reduced loss and preserved or improved agreement with teacher
    distill = report.distillation_summary
    assert distill.final_loss < distill.initial_loss
    assert (
        distill.post_distillation_teacher_agreement
        >= distill.pre_distillation_teacher_agreement
    )
    assert distill.speedup_over_teacher_ratio > 1.0

    # Verify every solution respects opponent identity blindness and finite latency
    for sol_id in INFERENCE_SOLUTION_FAMILIES:
        ev = report.evaluations_by_solution[sol_id]
        assert ev.respects_opponent_identity_blindness
        assert ev.mean_decision_latency_ms > 0.0
        assert ev.p95_decision_latency_ms >= ev.mean_decision_latency_ms
        assert 0.0 <= ev.tactical_quality_score <= 1.0
        assert 0.0 <= ev.expected_win_score <= 1.0
