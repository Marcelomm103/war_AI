"""Tests for Phase 8.3 PSRO and empirical multi-player metagame analysis."""

from __future__ import annotations

import math

import pytest

from war_ai.rl.league_population import build_default_league_population
from war_ai.rl.psro_metagame import (
    EmpiricalMetagameTable,
    canonical_lineup_key,
    detect_intransitive_cycles,
    run_psro_metagame_exploration,
)


def test_empirical_metagame_lineup_payoffs_and_cycle_detection() -> None:
    with pytest.raises(ValueError, match="3..6 players"):
        canonical_lineup_key(("pol_a", "pol_b"))

    metagame = EmpiricalMetagameTable(("pol_a", "pol_b", "pol_c", "pol_d"))
    # Construct rock-paper-scissors cycle: pol_a > pol_b, pol_b > pol_c, pol_c > pol_a
    for _ in range(8):
        metagame.record_lineup_outcome(
            ("pol_a", "pol_b", "pol_d"),
            winner_policy_id="pol_a",
        )
        metagame.record_lineup_outcome(
            ("pol_b", "pol_c", "pol_d"),
            winner_policy_id="pol_b",
        )
        metagame.record_lineup_outcome(
            ("pol_c", "pol_a", "pol_d"),
            winner_policy_id="pol_c",
        )
    # pol_d excels in 4-player tables despite losing 3-player triads
    for _ in range(10):
        metagame.record_lineup_outcome(
            ("pol_a", "pol_b", "pol_c", "pol_d"),
            winner_policy_id="pol_d",
        )

    records = metagame.lineup_records()
    assert len(records) == 4
    for rec in records:
        assert rec.completed_matches > 0
        assert math.isclose(
            sum(
                rec.win_rate_by_policy[pid] * (rec.seat_appearances_by_policy[pid] / rec.completed_matches)
                for pid in rec.win_rate_by_policy
            ),
            1.0,
            rel_tol=1e-5,
            abs_tol=1e-5,
        )

    cycles = detect_intransitive_cycles(metagame, min_margin=0.05)
    assert len(cycles) >= 1
    top_cycle = cycles[0]
    assert set(top_cycle.cycle_policies) == {"pol_a", "pol_b", "pol_c"}
    assert top_cycle.cycle_strength > 0.10


def test_meta_strategies_regret_and_multiplayer_rankings() -> None:
    metagame = EmpiricalMetagameTable(("pol_a", "pol_b", "pol_c", "pol_d"))
    for _ in range(8):
        metagame.record_lineup_outcome(
            ("pol_a", "pol_b", "pol_d"),
            winner_policy_id="pol_a",
        )
        metagame.record_lineup_outcome(
            ("pol_b", "pol_c", "pol_d"),
            winner_policy_id="pol_b",
        )
        metagame.record_lineup_outcome(
            ("pol_c", "pol_a", "pol_d"),
            winner_policy_id="pol_c",
        )

    uniform = metagame.evaluate_meta_strategy(
        {"pol_a": 0.25, "pol_b": 0.25, "pol_c": 0.25, "pol_d": 0.25},
        method="uniform",
    )
    replicator = metagame.solve_replicator_meta_strategy()
    alpharank = metagame.solve_alpharank_meta_strategy()

    assert replicator.nash_conv_regret <= uniform.nash_conv_regret + 1e-6
    assert replicator.probabilities_by_policy["pol_d"] < 0.05
    assert alpharank.probabilities_by_policy["pol_d"] < 0.05

    rankings = metagame.compute_rankings()
    assert len(rankings) == 4
    assert {r.multiplayer_composite_rank for r in rankings} == {1, 2, 3, 4}
    assert {r.pairwise_elo_rank for r in rankings} == {1, 2, 3, 4}


def test_run_psro_metagame_exploration_trains_oracle_and_detects_cycles() -> None:
    population = build_default_league_population(seed=2026)
    report = run_psro_metagame_exploration(population, seed=2026)

    assert len(report.lineup_payoffs) >= 4
    assert len(report.detected_cycles) >= 1
    assert report.elo_alone_insufficient is True
    assert len(report.oracle_results) == 1
    oracle = report.oracle_results[0]
    assert (
        oracle.trained_optimal_intent_probability
        > oracle.initial_optimal_intent_probability
    )
    assert 0.0 <= oracle.empirical_win_rate_vs_mixture <= 1.0
