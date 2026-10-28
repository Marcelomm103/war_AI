"""Tests for Phase 8.4 best-response exploiter training and non-collusion contracts."""

from __future__ import annotations

import pytest

from war_ai.rl.best_response import (
    MINIMUM_EXPLOITER_APPROACHES,
    CollusionStressTestScenario,
    ExploiterTrainingConfig,
    run_best_response_campaign,
)
from war_ai.rl.league_population import build_default_league_population


def test_exploiter_training_config_enforces_solo_win_and_non_collusion() -> None:
    with pytest.raises(ValueError, match="at least 2 distinct seeds"):
        ExploiterTrainingConfig(
            approach_id="ppo_recurrent_exploiter",
            target_member_id="main_ppo_recurrent_gen2",
            seeds=(2026,),
        )
    with pytest.raises(ValueError, match="solo_official_victory"):
        ExploiterTrainingConfig(
            approach_id="ppo_recurrent_exploiter",
            target_member_id="main_ppo_recurrent_gen2",
            reward_objective="team_kingmaking",
        )
    with pytest.raises(ValueError, match="Secret sharing is strictly prohibited"):
        ExploiterTrainingConfig(
            approach_id="ppo_recurrent_exploiter",
            target_member_id="main_ppo_recurrent_gen2",
            allow_secret_sharing=True,
        )
    with pytest.raises(ValueError, match="Hidden team coordination"):
        ExploiterTrainingConfig(
            approach_id="ppo_recurrent_exploiter",
            target_member_id="main_ppo_recurrent_gen2",
            allow_hidden_team_coordination=True,
        )


def test_collusion_stress_test_cannot_enter_normal_benchmark_panel() -> None:
    with pytest.raises(ValueError, match="strictly excluded from the normal benchmark panel"):
        CollusionStressTestScenario(
            scenario_id="illegal_collusion_in_panel",
            focal_policy_id="league_main_ppo_recurrent_gen2",
            stress_agent_policy_id="stress_agent",
            eligible_for_normal_benchmark_panel=True,
        )


def test_run_best_response_campaign_covers_three_independent_approaches_and_seeds() -> None:
    population = build_default_league_population(seed=2026)
    with pytest.raises(ValueError, match="at least 3"):
        run_best_response_campaign(
            population,
            approaches=("ppo_recurrent_exploiter", "r2d2_dueling_exploiter"),
        )

    report = run_best_response_campaign(
        population,
        target_member_id="main_ppo_recurrent_gen2",
        seeds=(2026, 2027),
    )
    assert report.independent_approaches_count >= MINIMUM_EXPLOITER_APPROACHES
    assert report.distinct_lineages_count == report.independent_approaches_count
    assert report.all_isolation_audits_passed is True
    assert report.collusion_stress_report.isolated_from_benchmark_panel is True
    assert report.collusion_stress_report.completed_matches == 2

    for app_report in report.approach_reports:
        assert len(app_report.seed_results) == 2
        for seed_res in app_report.seed_results:
            assert (
                seed_res.trained_optimal_intent_probability
                > seed_res.initial_optimal_intent_probability
            )
            assert seed_res.isolation_audit.passed is True
            assert seed_res.matches_evaluated == 3
