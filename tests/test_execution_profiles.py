"""Tests for turn-budgeted execution profiles and separate evaluation (Phase 9.6)."""

from __future__ import annotations

import pytest
from war_game_engine.contracts import ContractEnvelope, parse_contract_json

from war_ai.rl.execution_profiles import (
    EXECUTION_PROFILE_CATALOG,
    EXECUTION_PROFILE_IDS,
    FAST_EXECUTION_PROFILE,
    RESEARCH_EXECUTION_PROFILE,
    STRONG_EXECUTION_PROFILE,
    ExecutionProfileSpec,
    TokenCountBudgetForbiddenError,
    evaluate_all_execution_profiles,
)


def test_execution_profiles_forbid_token_count_budgets_and_validate_contracts() -> None:
    with pytest.raises(TokenCountBudgetForbiddenError):
        ExecutionProfileSpec(
            profile_id="fast",
            description="Invalid token-budgeted profile",
            uses_search=False,
            whole_turn_budget_ms=1000,
            max_decision_budget_ms=250,
            transport_and_ui_margin_ms=200,
            whole_turn_simulation_budget=0,
            default_decision_simulations=0,
            num_determinizations=0,
            max_search_depth=0,
            rollout_depth=0,
            belief_mode="none",
            published_for_deployment=True,
            token_budget=128,
        )

    for pid in EXECUTION_PROFILE_IDS:
        spec = EXECUTION_PROFILE_CATALOG[pid]
        inf_cfg = spec.to_contract_inference_config()
        env = ContractEnvelope(
            schema_version="1.0",
            api_version="1.0.0",
            payload=inf_cfg,
        )
        assert parse_contract_json(env.model_dump_json()).payload == inf_cfg


def test_evaluate_fast_strong_and_research_profiles_separately() -> None:
    report = evaluate_all_execution_profiles(
        num_games=2,
        steps_per_game=21,
        seed=961,
    )
    assert report.all_profiles_evaluated_separately
    assert report.all_turn_budgets_respected
    assert report.all_legal_action_rates_perfect
    assert set(report.reports_by_profile) == {"fast", "strong", "research"}

    fast_rep = report.reports_by_profile["fast"]
    strong_rep = report.reports_by_profile["strong"]
    research_rep = report.reports_by_profile["research"]

    assert fast_rep.mean_simulations_per_turn == 0.0
    assert fast_rep.p95_decision_latency_ms <= FAST_EXECUTION_PROFILE.max_decision_budget_ms
    assert strong_rep.p95_turn_latency_ms <= STRONG_EXECUTION_PROFILE.whole_turn_budget_ms
    assert research_rep.p95_turn_latency_ms <= RESEARCH_EXECUTION_PROFILE.whole_turn_budget_ms
    assert research_rep.expected_win_rate >= strong_rep.expected_win_rate >= fast_rep.expected_win_rate
