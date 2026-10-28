"""Tests for evidence-based candidate selection and local plateau verification (Phase 9.7)."""

from __future__ import annotations

import pytest

from war_ai.rl.evidence_selection import (
    CONFIRMED_VALIDATION_GAIN_THRESHOLD,
    EVIDENCE_CRITERIA_ORDER,
    MINIMUM_INDEPENDENT_PLATEAU_CYCLES,
    CandidateEvidenceRecord,
    LegalityAndSecrecyGateError,
    LocalPlateauAssessment,
    ResearchCycleRecord,
    rank_candidates_by_evidence,
    run_phase9_evidence_selection,
    verify_local_research_plateau,
)


def test_mandatory_legality_and_secrecy_gates_disqualify_violators() -> None:
    illegal_cand = CandidateEvidenceRecord(
        candidate_id="illegal-cand",
        profile_id="strong",
        description="Emits occasional illegal action",
        legal_action_rate=0.999,
        zero_information_leakage=True,
        win_probability=0.95,
        worst_opponent_or_group_win_rate=0.90,
        exploiter_resistance_score=0.92,
        real_channel_performance_score=0.95,
        operational_cost_ms=20.0,
    )
    leaky_cand = CandidateEvidenceRecord(
        candidate_id="leaky-cand",
        profile_id="strong",
        description="Leaks hidden state during planning",
        legal_action_rate=1.0,
        zero_information_leakage=False,
        win_probability=0.96,
        worst_opponent_or_group_win_rate=0.91,
        exploiter_resistance_score=0.93,
        real_channel_performance_score=0.96,
        operational_cost_ms=15.0,
    )
    with pytest.raises(LegalityAndSecrecyGateError):
        illegal_cand.ensure_mandatory_gates()
    with pytest.raises(LegalityAndSecrecyGateError):
        rank_candidates_by_evidence((illegal_cand, leaky_cand))


def test_five_tier_evidence_hierarchy_and_local_plateau_rules() -> None:
    assert EVIDENCE_CRITERIA_ORDER == (
        "win_probability",
        "worst_opponent_or_group_win_rate",
        "exploiter_resistance_score",
        "real_channel_performance_score",
        "operational_cost_ms",
    )

    cycles = (
        ResearchCycleRecord("c1", 101, "Try A", 0.84, 0.848, 0.008),
        ResearchCycleRecord("c2", 102, "Try B", 0.84, 0.851, 0.011),
        ResearchCycleRecord("c3", 103, "Try C", 0.84, 0.855, 0.015),
    )
    assessment = verify_local_research_plateau("cand-1", cycles)
    assert assessment.local_plateau_reached
    assert assessment.max_confirmed_gain < CONFIRMED_VALIDATION_GAIN_THRESHOLD
    assert len(assessment.independent_cycles) >= MINIMUM_INDEPENDENT_PLATEAU_CYCLES
    assert not assessment.proves_global_optimum

    # Claiming global optimum is strictly rejected
    with pytest.raises(ValueError, match="proves_global_optimum"):
        LocalPlateauAssessment(
            incumbent_candidate_id="cand-1",
            independent_cycles=cycles,
            minimum_required_cycles=3,
            gain_threshold=0.02,
            max_confirmed_gain=0.015,
            local_plateau_reached=True,
            proves_global_optimum=True,
        )

    # Fewer than 3 independent cycles is rejected
    with pytest.raises(ValueError, match="At least 3"):
        verify_local_research_plateau("cand-1", cycles[:2])


def test_run_phase9_evidence_selection_campaign() -> None:
    report = run_phase9_evidence_selection(seed=970)
    assert report.all_mandatory_gates_enforced
    assert "disqualified-naive-clairvoyant-fusion" in report.disqualified_candidate_ids
    assert report.selected_candidate.passes_mandatory_gates
    assert (
        report.selected_candidate.candidate_id
        == "war1-finalist-ismcts-dp-learned-belief-strong"
    )
    assert report.plateau_assessment.local_plateau_reached
    assert not report.plateau_assessment.proves_global_optimum
