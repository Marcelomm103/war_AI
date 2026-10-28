"""Tests for Phase 8.2 full-table lineup sampling and prioritized multi-player training."""

from __future__ import annotations

import math

import pytest

from war_ai.rl.full_table_training import (
    FullTableSamplingConfig,
    PrioritizedFullTableSampler,
    evaluate_lineup_diversity,
    run_full_table_training_campaign,
)
from war_ai.rl.league_population import build_default_league_population


def test_full_table_sampling_rejects_duels_and_logs_probabilities() -> None:
    with pytest.raises(ValueError, match="2-player duels"):
        FullTableSamplingConfig(player_counts=(2, 3))

    population = build_default_league_population(seed=2026)
    sampler = PrioritizedFullTableSampler(
        population,
        focal_member_id="main_ppo_recurrent_gen2",
        seed=2026,
    )
    with pytest.raises(ValueError, match="2-player duels"):
        sampler.sample_lineup(lineup_id="bad-duel", seed=1, num_players=2)

    lineups = sampler.sample_batch(
        batch_prefix="test-batch",
        num_lineups=18,
        base_seed=2026,
    )
    diversity = evaluate_lineup_diversity(lineups)
    assert diversity.duel_lineups_count == 0
    assert diversity.sampled_table_sizes == (3, 4, 5, 6)
    assert diversity.normalized_entropy > 0.85
    assert diversity.effective_support_size >= 8.0
    assert diversity.mean_unique_styles_per_table >= 3.0
    assert diversity.mean_unique_lineages_per_table >= 3.0

    for lineup in lineups:
        assert math.isclose(
            sum(lineup.candidate_distribution.values()),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )
        assert len(lineup.seat_sampling_probabilities) == lineup.num_players - 1
        assert 0.0 < lineup.joint_lineup_probability <= 1.0
        # Held-out generalization models must never appear in training lineups
        for opp_id in lineup.opponent_member_ids_by_seat.values():
            assert not population.get_member(opp_id).held_out_from_training


def test_hard_matchup_priority_shifts_sampling_toward_difficult_opponents() -> None:
    population = build_default_league_population(seed=2026)
    sampler = PrioritizedFullTableSampler(
        population,
        focal_member_id="main_ppo_recurrent_gen2",
        config=FullTableSamplingConfig(
            uniform_history_weight=0.25,
            priority_exponent=2.0,
        ),
        seed=2026,
    )
    initial_dist = sampler.compute_candidate_distribution()
    hard_opp = "exploiter_anti_main_ppo"
    easy_opp = "hist_ppo_recurrent_gen0"

    for _ in range(10):
        sampler.register_matchup_outcome(
            hard_opp,
            focal_won=False,
            opponent_won=True,
        )
        sampler.register_matchup_outcome(
            easy_opp,
            focal_won=True,
            opponent_won=False,
        )

    updated_dist = sampler.compute_candidate_distribution()
    assert updated_dist[hard_opp] > initial_dist[hard_opp]
    assert updated_dist[hard_opp] > updated_dist[easy_opp]
    # Uniform history + floor preserves positive probability for easy opponent too
    assert updated_dist[easy_opp] >= sampler.config.min_opponent_probability_floor


def test_run_full_table_training_campaign_executes_matches_and_improves_policy() -> None:
    population = build_default_league_population(seed=2026)
    report = run_full_table_training_campaign(
        population,
        focal_member_id="main_ppo_recurrent_gen2",
        num_lineups=4,
        learner_updates=6,
        seed=2026,
    )
    assert len(report.sampled_lineups) == 4
    assert len(report.match_results) == 4
    assert all(res.status == "completed" for res in report.match_results)
    assert (
        report.trained_optimal_intent_probability
        > report.initial_optimal_intent_probability
    )
    assert any(
        rec.appearances_against_focal > 0
        for rec in report.difficulty_records.values()
    )
