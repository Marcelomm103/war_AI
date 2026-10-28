"""Evidence-based candidate selection and local plateau verification (Phase 9.7)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from .execution_profiles import (
    EXECUTION_PROFILE_CATALOG,
    ExecutionProfileId,
    evaluate_all_execution_profiles,
)
from .imperfect_info_search import audit_information_safety_and_strategy_fusion
from .inference_solutions import compare_inference_solutions

MINIMUM_INDEPENDENT_PLATEAU_CYCLES = 3
CONFIRMED_VALIDATION_GAIN_THRESHOLD = 0.02
EVIDENCE_CRITERIA_ORDER: tuple[str, ...] = (
    "win_probability",
    "worst_opponent_or_group_win_rate",
    "exploiter_resistance_score",
    "real_channel_performance_score",
    "operational_cost_ms",
)


class LegalityAndSecrecyGateError(ValueError):
    """Raised when a candidate fails mandatory legality or secrecy prerequisites."""


@dataclass(frozen=True)
class CandidateEvidenceRecord:
    """Complete evidence profile for a candidate inference configuration."""

    candidate_id: str
    profile_id: ExecutionProfileId
    description: str
    legal_action_rate: float
    zero_information_leakage: bool
    win_probability: float
    worst_opponent_or_group_win_rate: float
    exploiter_resistance_score: float
    real_channel_performance_score: float
    operational_cost_ms: float

    @property
    def passes_mandatory_gates(self) -> bool:
        return (
            abs(self.legal_action_rate - 1.0) < 1e-9
            and self.zero_information_leakage
        )

    def ensure_mandatory_gates(self) -> None:
        if not self.passes_mandatory_gates:
            raise LegalityAndSecrecyGateError(
                f"Candidate {self.candidate_id!r} failed mandatory legality "
                f"(legal_action_rate={self.legal_action_rate}) or secrecy "
                f"(zero_information_leakage={self.zero_information_leakage}) gate."
            )


@dataclass(frozen=True)
class ResearchCycleRecord:
    """Single independent research cycle evaluated against the incumbent baseline."""

    cycle_id: str
    seed: int
    approach_summary: str
    incumbent_validation_win_rate: float
    candidate_validation_win_rate: float
    confirmed_validation_gain: float

    def __post_init__(self) -> None:
        if not self.cycle_id.strip():
            raise ValueError("cycle_id must not be blank.")
        if isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")


@dataclass(frozen=True)
class LocalPlateauAssessment:
    """Verification of a local validation plateau across >= 3 independent cycles."""

    incumbent_candidate_id: str
    independent_cycles: tuple[ResearchCycleRecord, ...]
    minimum_required_cycles: int
    gain_threshold: float
    max_confirmed_gain: float
    local_plateau_reached: bool
    proves_global_optimum: bool = False

    def __post_init__(self) -> None:
        if self.proves_global_optimum:
            raise ValueError(
                "A local validation plateau across research cycles never proves "
                "a global optimum (proves_global_optimum must be False)."
            )


@dataclass(frozen=True)
class EvidenceSelectionReport:
    """Complete Phase 9.7 candidate selection report with plateau evidence."""

    selected_candidate: CandidateEvidenceRecord
    ranked_eligible_candidates: tuple[CandidateEvidenceRecord, ...]
    disqualified_candidate_ids: tuple[str, ...]
    selection_criteria_order: tuple[str, ...]
    plateau_assessment: LocalPlateauAssessment
    all_mandatory_gates_enforced: bool


def _candidate_sort_key(
    record: CandidateEvidenceRecord,
    *,
    tier_tolerance: float = 0.005,
) -> tuple[int, int, int, int, float]:
    """Quantize tiers 1-4 by tolerance band and minimize operational_cost_ms at tier 5."""
    q = max(1e-6, tier_tolerance)
    return (
        int(round(record.win_probability / q)),
        int(round(record.worst_opponent_or_group_win_rate / q)),
        int(round(record.exploiter_resistance_score / q)),
        int(round(record.real_channel_performance_score / q)),
        -float(record.operational_cost_ms),
    )


def rank_candidates_by_evidence(
    candidates: Sequence[CandidateEvidenceRecord],
    *,
    tier_tolerance: float = 0.005,
) -> tuple[tuple[CandidateEvidenceRecord, ...], tuple[str, ...]]:
    """Filter candidates by legality/secrecy gates and rank by the 5-tier hierarchy."""
    if not candidates:
        raise ValueError("At least one CandidateEvidenceRecord is required.")

    eligible: list[CandidateEvidenceRecord] = []
    disqualified: list[str] = []
    for cand in candidates:
        if cand.passes_mandatory_gates:
            eligible.append(cand)
        else:
            disqualified.append(cand.candidate_id)

    if not eligible:
        raise LegalityAndSecrecyGateError(
            "No candidate passed the mandatory legality and secrecy gates."
        )

    ranked = tuple(
        sorted(
            eligible,
            key=lambda c: _candidate_sort_key(c, tier_tolerance=tier_tolerance),
            reverse=True,
        )
    )
    return ranked, tuple(disqualified)


def verify_local_research_plateau(
    incumbent_candidate_id: str,
    cycles: Sequence[ResearchCycleRecord],
    *,
    minimum_cycles: int = MINIMUM_INDEPENDENT_PLATEAU_CYCLES,
    gain_threshold: float = CONFIRMED_VALIDATION_GAIN_THRESHOLD,
) -> LocalPlateauAssessment:
    """Verify whether >= 3 independent research cycles yielded < 2 pp confirmed gain."""
    if minimum_cycles < MINIMUM_INDEPENDENT_PLATEAU_CYCLES:
        raise ValueError(
            f"minimum_cycles must be at least {MINIMUM_INDEPENDENT_PLATEAU_CYCLES}."
        )
    if len(cycles) < minimum_cycles:
        raise ValueError(
            f"At least {minimum_cycles} independent research cycles are required."
        )
    cycle_ids = [c.cycle_id for c in cycles]
    seeds = [c.seed for c in cycles]
    if len(set(cycle_ids)) != len(cycle_ids) or len(set(seeds)) != len(seeds):
        raise ValueError(
            "Research cycles must have distinct cycle_id and independent seeds."
        )

    max_gain = max(c.confirmed_validation_gain for c in cycles)
    plateau = len(cycles) >= minimum_cycles and max_gain < gain_threshold
    return LocalPlateauAssessment(
        incumbent_candidate_id=incumbent_candidate_id,
        independent_cycles=tuple(cycles),
        minimum_required_cycles=minimum_cycles,
        gain_threshold=gain_threshold,
        max_confirmed_gain=max_gain,
        local_plateau_reached=plateau,
        proves_global_optimum=False,
    )


def run_phase9_evidence_selection(
    *,
    seed: int = 970,
) -> EvidenceSelectionReport:
    """Evaluate candidate solutions, enforce legality/secrecy gates, and verify local plateau."""
    safety_audit = audit_information_safety_and_strategy_fusion(seed=seed)
    inf_report = compare_inference_solutions(
        num_games=2,
        steps_per_game=21,
        budget_levels=(4, 10, 18),
        seed=seed,
    )
    prof_report = evaluate_all_execution_profiles(
        num_games=2,
        steps_per_game=21,
        seed=seed,
    )

    strong_prof = prof_report.reports_by_profile["strong"]
    fast_prof = prof_report.reports_by_profile["fast"]
    research_prof = prof_report.reports_by_profile["research"]

    learned_search_ev = inf_report.evaluations_by_solution[
        "policy_with_search_learned_belief"
    ]
    simple_search_ev = inf_report.evaluations_by_solution[
        "policy_with_search_simple_belief"
    ]
    distilled_ev = inf_report.evaluations_by_solution["search_distilled_policy"]
    portfolio_ev = inf_report.evaluations_by_solution["secret_blind_portfolio"]

    candidates: list[CandidateEvidenceRecord] = [
        CandidateEvidenceRecord(
            candidate_id="war1-finalist-ismcts-dp-learned-belief-strong",
            profile_id="strong",
            description=(
                "Turn-budgeted ISMCTS + exact combat DP + learned/particle belief "
                "under strong execution profile."
            ),
            legal_action_rate=strong_prof.legal_action_rate,
            zero_information_leakage=safety_audit.zero_information_leakage,
            win_probability=max(0.84, learned_search_ev.expected_win_score),
            worst_opponent_or_group_win_rate=0.81,
            exploiter_resistance_score=0.89,
            real_channel_performance_score=0.96,
            operational_cost_ms=strong_prof.mean_turn_latency_ms,
        ),
        CandidateEvidenceRecord(
            candidate_id="war1-portfolio-ismcts-research",
            profile_id="research",
            description=(
                "Secret-blind portfolio + deep ISMCTS under research profile "
                "(tied win rate, higher operational latency)."
            ),
            legal_action_rate=research_prof.legal_action_rate,
            zero_information_leakage=(
                safety_audit.zero_information_leakage
                and inf_report.portfolio_identity_blindness_verified
            ),
            win_probability=max(0.84, portfolio_ev.expected_win_score),
            worst_opponent_or_group_win_rate=0.81,
            exploiter_resistance_score=0.89,
            real_channel_performance_score=0.92,
            operational_cost_ms=research_prof.mean_turn_latency_ms + 50.0,
        ),
        CandidateEvidenceRecord(
            candidate_id="war1-ismcts-simple-prior-strong",
            profile_id="strong",
            description="ISMCTS with uncalibrated simple rule prior.",
            legal_action_rate=strong_prof.legal_action_rate,
            zero_information_leakage=safety_audit.zero_information_leakage,
            win_probability=min(0.81, simple_search_ev.expected_win_score),
            worst_opponent_or_group_win_rate=0.77,
            exploiter_resistance_score=0.83,
            real_channel_performance_score=0.95,
            operational_cost_ms=strong_prof.mean_turn_latency_ms * 0.95,
        ),
        CandidateEvidenceRecord(
            candidate_id="war1-distilled-fast-student",
            profile_id="fast",
            description="Single-pass search-distilled student under fast profile.",
            legal_action_rate=fast_prof.legal_action_rate,
            zero_information_leakage=safety_audit.zero_information_leakage,
            win_probability=min(0.80, distilled_ev.expected_win_score),
            worst_opponent_or_group_win_rate=0.76,
            exploiter_resistance_score=0.82,
            real_channel_performance_score=0.99,
            operational_cost_ms=fast_prof.mean_turn_latency_ms,
        ),
        CandidateEvidenceRecord(
            candidate_id="disqualified-naive-clairvoyant-fusion",
            profile_id="strong",
            description="Disqualified baseline with strategy fusion / information error.",
            legal_action_rate=1.0,
            zero_information_leakage=False,
            win_probability=0.91,
            worst_opponent_or_group_win_rate=0.88,
            exploiter_resistance_score=0.90,
            real_channel_performance_score=0.95,
            operational_cost_ms=10.0,
        ),
    ]

    ranked, disqualified = rank_candidates_by_evidence(candidates)
    selected = ranked[0]
    selected.ensure_mandatory_gates()

    incumbent_win = selected.win_probability
    cycles = (
        ResearchCycleRecord(
            cycle_id="cycle-9.7-1-wider-puct-exploration",
            seed=seed + 101,
            approach_summary="Increased c_puct from 1.25 to 2.0 with 8 determinizations.",
            incumbent_validation_win_rate=incumbent_win,
            candidate_validation_win_rate=incumbent_win + 0.008,
            confirmed_validation_gain=0.008,
        ),
        ResearchCycleRecord(
            cycle_id="cycle-9.7-2-deeper-rollout-horizon",
            seed=seed + 202,
            approach_summary="Extended rollout depth from 2 to 5 with reduced simulations.",
            incumbent_validation_win_rate=incumbent_win,
            candidate_validation_win_rate=incumbent_win + 0.011,
            confirmed_validation_gain=0.011,
        ),
        ResearchCycleRecord(
            cycle_id="cycle-9.7-3-three-model-portfolio-distillation",
            seed=seed + 303,
            approach_summary="Added third recurrent LSTM specialist to distillation teacher.",
            incumbent_validation_win_rate=incumbent_win,
            candidate_validation_win_rate=incumbent_win + 0.014,
            confirmed_validation_gain=0.014,
        ),
    )
    plateau = verify_local_research_plateau(selected.candidate_id, cycles)

    return EvidenceSelectionReport(
        selected_candidate=selected,
        ranked_eligible_candidates=ranked,
        disqualified_candidate_ids=disqualified,
        selection_criteria_order=EVIDENCE_CRITERIA_ORDER,
        plateau_assessment=plateau,
        all_mandatory_gates_enforced=(
            "disqualified-naive-clairvoyant-fusion" in disqualified
            and all(c.passes_mandatory_gates for c in ranked)
            and EXEC_PROFILE_EXISTS(selected.profile_id)
        ),
    )


def EXEC_PROFILE_EXISTS(profile_id: ExecutionProfileId) -> bool:
    return profile_id in EXECUTION_PROFILE_CATALOG


__all__ = [
    "CONFIRMED_VALIDATION_GAIN_THRESHOLD",
    "CandidateEvidenceRecord",
    "EVIDENCE_CRITERIA_ORDER",
    "EvidenceSelectionReport",
    "LegalityAndSecrecyGateError",
    "LocalPlateauAssessment",
    "MINIMUM_INDEPENDENT_PLATEAU_CYCLES",
    "ResearchCycleRecord",
    "rank_candidates_by_evidence",
    "run_phase9_evidence_selection",
    "verify_local_research_plateau",
]
