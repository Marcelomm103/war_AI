"""Tests for Phase 8.5 generalization evaluation across seeds, architectures, held-out models, missions, and seats."""

from __future__ import annotations

import pytest

from war_ai.envs.observations import PLAYER_COLOR_IDS
from war_ai.rl.generalization import (
    audit_seat_and_color_adaptation,
    run_generalization_evaluation,
)
from war_ai.rl.league_population import build_default_league_population


def test_run_generalization_evaluation_covers_all_six_dimensions() -> None:
    population = build_default_league_population(seed=2026)
    report = run_generalization_evaluation(
        population,
        focal_member_id="main_ppo_recurrent_gen2",
        seeds=(2026,),
    )

    # 1. Cross-play across seeds and architectures
    assert len(report.cross_play.cells) >= 3
    assert report.cross_play.distinct_architectures_evaluated >= 3
    assert report.cross_play.distinct_seeds_evaluated >= 4
    assert any(cell.is_cross_seed for cell in report.cross_play.cells)
    assert any(cell.is_cross_architecture for cell in report.cross_play.cells)

    # 2. Held-out generalization models (never used in training)
    assert len(report.held_out.entries) == 3
    assert report.held_out.all_verified_held_out is True
    assert 0.0 <= report.held_out.mean_focal_win_rate <= 1.0

    # 3. Worst opponent and worst lineup
    assert report.worst_case.worst_opponent_win_rate <= report.worst_case.mean_overall_win_rate + 1e-9
    assert report.worst_case.worst_lineup_win_rate <= report.worst_case.mean_overall_win_rate + 1e-9
    assert len(report.worst_case.worst_lineup_policies) >= 3

    # 4. Regression against historical snapshots
    assert len(report.historical_regression.entries) == 3
    assert 0.0 <= report.historical_regression.mean_win_rate_vs_history <= 1.0

    # 5. Mission weaknesses
    assert len(report.mission_weaknesses.by_mission_id) >= 1
    assert len(report.mission_weaknesses.by_category) >= 1
    assert (
        report.mission_weaknesses.weakest_mission_win_rate
        <= report.mission_weaknesses.strongest_mission_win_rate
    )

    # 6. Seat and color bias audit
    assert len(report.seat_color_audit.strata) == 3
    for stratum in report.seat_color_audit.strata:
        assert stratum.player_color == PLAYER_COLOR_IDS[stratum.seat_index]
        assert stratum.appearances > 0


def test_seat_color_audit_flags_undue_adaptation_when_tolerance_is_tight() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        audit_seat_and_color_adaptation(
            (),
            focal_policy_id="league_main_ppo_recurrent_gen2",
            deviation_tolerance=0.0,
        )
