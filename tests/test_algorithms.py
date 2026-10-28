"""Tests for Phase 7.3 algorithm branches, V-trace, R2D2, and search."""

from __future__ import annotations

import pytest
import torch

from war_ai.rl.algorithms import (
    EXCLUDED_STANDARD_ALGORITHMS,
    MINIMUM_ALGORITHM_BRANCHES,
    compute_vtrace_targets_and_advantages,
    evaluate_all_algorithm_branches,
    signed_parabolic_inverse,
    signed_parabolic_transform,
    validate_algorithm_branch,
)
from war_ai.rl.experimental_axes import ComputeBudget


def test_minimum_algorithm_branches_and_excluded_methods() -> None:
    assert MINIMUM_ALGORITHM_BRANCHES == (
        "ppo_feedforward",
        "ppo_recurrent",
        "graph_actor_critic",
        "impala_vtrace",
        "recurrent_off_policy_r2d2",
        "policy_value_search",
    )
    for branch in MINIMUM_ALGORITHM_BRANCHES:
        assert validate_algorithm_branch(branch) == branch

    for excluded in EXCLUDED_STANDARD_ALGORITHMS:
        with pytest.raises(ValueError, match="excluded for WAR"):
            validate_algorithm_branch(excluded)


def test_vtrace_clips_importance_ratios_and_handles_termination_vs_truncation() -> None:
    behavior_log_probs = torch.tensor([[-2.0, -1.0], [-1.5, -1.0]])
    target_log_probs = torch.tensor([[0.0, -1.0], [-0.5, -1.0]])
    rewards = torch.tensor([[1.0, 1.0], [2.0, 2.0]])
    values = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    next_values = torch.tensor([[0.8, 0.8], [1.2, 1.2]])
    terminated = torch.tensor([[False, False], [True, False]])
    truncated = torch.tensor([[False, False], [False, True]])

    vtrace = compute_vtrace_targets_and_advantages(
        behavior_log_probs=behavior_log_probs,
        target_log_probs=target_log_probs,
        rewards=rewards,
        values=values,
        next_values=next_values,
        terminated=terminated,
        truncated=truncated,
        gamma=1.0,
        clip_rho_threshold=1.0,
        clip_c_threshold=1.0,
    )

    assert torch.all(vtrace.clipped_rhos <= 1.0 + 1e-6)
    assert torch.all(vtrace.clipped_cs <= 1.0 + 1e-6)
    # Step 1 col 0 terminated -> bootstrap is 0, so v_1 = 0.5 + 1.0*(2.0 + 0 - 0.5) = 2.0
    assert torch.isclose(vtrace.vs[1, 0], torch.tensor(2.0), atol=1e-5)
    # Step 1 col 1 truncated -> bootstraps from next_values=1.2, so v_1 = 0.5 + 1.0*(2.0 + 1.2 - 0.5) = 3.2
    assert torch.isclose(vtrace.vs[1, 1], torch.tensor(3.2), atol=1e-5)


def test_r2d2_signed_parabolic_value_rescaling_is_invertible() -> None:
    values = torch.tensor([-50.0, -2.5, -0.1, 0.0, 0.25, 4.0, 100.0])
    transformed = signed_parabolic_transform(values)
    recovered = signed_parabolic_inverse(transformed)
    assert torch.allclose(values, recovered, atol=1e-4)


def test_evaluate_all_six_algorithm_branches_under_controlled_budget() -> None:
    budget = ComputeBudget(
        semantic_decisions=32,
        learner_updates=8,
        inference_budget_ms=250,
    )
    evaluations = evaluate_all_algorithm_branches(
        compute_budget=budget,
        hidden_size=32,
        seed=2026,
    )

    assert len(evaluations) == 6
    for result in evaluations:
        assert result.compute_budget == budget
        assert result.selected_action_legal
        assert result.trained_optimal_intent_probability > (
            result.initial_optimal_intent_probability
        )
