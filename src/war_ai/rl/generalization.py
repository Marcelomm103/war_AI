"""Generalization measurement: cross-play, held-out models, worst opponent/lineup, historical regression, mission weaknesses, and seat/color bias auditing (Phase 8.5)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from war_game_engine.objectives import (
    OBJECTIVE_CONDITIONS,
    ControlContinentsCondition,
    EliminateColorCondition,
    PlayerColor,
    TerritoryCountCondition,
)

from ..arena import ArenaMatchConfig, ArenaMatchResult, run_arena
from ..envs.observations import PLAYER_COLOR_IDS
from .league_population import LeaguePopulation
from .psro_metagame import canonical_lineup_key


@dataclass(frozen=True)
class CrossPlayCell:
    """Single cross-play evaluation cell between two (architecture, seed) members."""

    focal_member_id: str
    focal_architecture: str
    focal_seed: int
    opponent_member_id: str
    opponent_architecture: str
    opponent_seed: int
    is_cross_seed: bool
    is_cross_architecture: bool
    matches: int
    focal_wins: int
    focal_win_rate: float


@dataclass(frozen=True)
class CrossPlayMatrixReport:
    """Summary of cross-play across distinct seeds and architectures."""

    cells: tuple[CrossPlayCell, ...]
    mean_cross_seed_win_rate: float
    mean_cross_architecture_win_rate: float
    distinct_architectures_evaluated: int
    distinct_seeds_evaluated: int


@dataclass(frozen=True)
class HeldOutGeneralizationEntry:
    """Performance of the focal agent against one model held out from training."""

    held_out_member_id: str
    held_out_policy_id: str
    held_out_lineage_id: str
    matches: int
    focal_wins: int
    held_out_wins: int
    focal_win_rate: float
    verified_held_out_from_training: bool


@dataclass(frozen=True)
class HeldOutGeneralizationReport:
    """Aggregate generalization against models never used during training."""

    focal_member_id: str
    entries: tuple[HeldOutGeneralizationEntry, ...]
    mean_focal_win_rate: float
    min_focal_win_rate: float
    all_verified_held_out: bool


@dataclass(frozen=True)
class WorstCaseVulnerabilityReport:
    """Identification of the worst opponent and worst full-table lineup for the focal policy."""

    focal_policy_id: str
    worst_opponent_policy_id: str
    worst_opponent_win_rate: float
    worst_lineup_key: str
    worst_lineup_policies: tuple[str, ...]
    worst_lineup_win_rate: float
    mean_overall_win_rate: float
    worst_opponent_gap: float
    worst_lineup_gap: float


@dataclass(frozen=True)
class HistoricalRegressionEntry:
    """Comparison of the focal agent against one earlier historical snapshot."""

    snapshot_member_id: str
    snapshot_policy_id: str
    snapshot_generation: int
    matches: int
    focal_wins: int
    snapshot_wins: int
    focal_win_rate: float
    snapshot_win_rate: float
    regressed: bool


@dataclass(frozen=True)
class HistoricalRegressionReport:
    """Audit verifying that the current agent does not regress against historical snapshots."""

    focal_member_id: str
    focal_generation: int
    entries: tuple[HistoricalRegressionEntry, ...]
    min_win_rate_vs_history: float
    mean_win_rate_vs_history: float
    non_regression_passed: bool


@dataclass(frozen=True)
class MissionStratumStat:
    """Performance stratum for one WAR mission or mission category."""

    stratum_id: str
    stratum_kind: str
    appearances: int
    wins: int
    win_rate: float


@dataclass(frozen=True)
class MissionWeaknessReport:
    """Breakdown of win rates and weaknesses across WAR missions and categories."""

    by_mission_id: tuple[MissionStratumStat, ...]
    by_category: tuple[MissionStratumStat, ...]
    weakest_mission_id: str
    weakest_mission_win_rate: float
    strongest_mission_id: str
    strongest_mission_win_rate: float
    mission_win_rate_spread: float


@dataclass(frozen=True)
class SeatColorStratumStat:
    """Empirical performance for one seat index and its canonical WAR player color."""

    seat_index: int
    player_id: str
    player_color: PlayerColor
    appearances: int
    wins: int
    win_rate: float
    deviation_from_mean: float


@dataclass(frozen=True)
class SeatColorBiasAudit:
    """Audit detecting undue adaptation to player colors or seat positions."""

    focal_policy_id: str
    strata: tuple[SeatColorStratumStat, ...]
    mean_win_rate: float
    max_abs_seat_color_deviation: float
    deviation_tolerance: float
    undue_seat_or_color_adaptation: bool


@dataclass(frozen=True)
class GeneralizationEvaluationReport:
    """Complete Phase 8.5 generalization report across all six vulnerability axes."""

    focal_member_id: str
    focal_policy_id: str
    cross_play: CrossPlayMatrixReport
    held_out: HeldOutGeneralizationReport
    worst_case: WorstCaseVulnerabilityReport
    historical_regression: HistoricalRegressionReport
    mission_weaknesses: MissionWeaknessReport
    seat_color_audit: SeatColorBiasAudit


def _mission_category(mission_id: str) -> str:
    condition = OBJECTIVE_CONDITIONS.get(mission_id)
    if isinstance(condition, ControlContinentsCondition):
        return "continent_control"
    if isinstance(condition, EliminateColorCondition):
        return "eliminate_color"
    if isinstance(condition, TerritoryCountCondition):
        return "territory_count"
    return "other"


def evaluate_cross_play(
    population: LeaguePopulation,
    *,
    focal_member_id: str = "main_ppo_recurrent_gen2",
    seeds: Sequence[int] = (2026,),
) -> tuple[CrossPlayMatrixReport, tuple[ArenaMatchResult, ...]]:
    """Measure cross-play performance across distinct seeds and architectures."""
    focal = population.get_member(focal_member_id)
    leaders = [
        opp
        for opp in population.architecture_seed_leaders.values()
        if opp.member_id != focal.member_id
    ]
    cells: list[CrossPlayCell] = []
    all_matches: list[ArenaMatchResult] = []
    registry = population.policy_registry()

    configs: list[ArenaMatchConfig] = []
    pairings: list[tuple[int, str, str]] = []
    for idx in range(0, len(leaders), 2):
        opp_a = leaders[idx]
        opp_b = leaders[(idx + 1) % len(leaders)]
        for seed in seeds:
            for rot in range(3):
                base_order = (
                    focal.policy_spec.policy_id,
                    opp_a.policy_spec.policy_id,
                    opp_b.policy_spec.policy_id,
                )
                rotated = base_order[rot:] + base_order[:rot]
                configs.append(
                    ArenaMatchConfig(
                        match_id=f"xplay-{idx}-s{seed}-rot{rot}",
                        seed=seed + idx,
                        num_players=3,
                        policy_ids_by_player={
                            f"player_{i}": rotated[i] for i in range(3)
                        },
                        max_semantic_decisions=5000,
                    )
                )
                pairings.append((len(configs) - 1, opp_a.member_id, opp_b.member_id))

    matches = run_arena(configs, registry).matches
    all_matches.extend(matches)

    for opp in leaders:
        opp_pid = opp.policy_spec.policy_id
        opp_matches = [
            m
            for m in matches
            if m.status == "completed" and opp_pid in m.policy_ids_by_player.values()
        ]
        focal_wins = sum(
            1
            for m in opp_matches
            for s in m.seat_results
            if s.policy_id == focal.policy_spec.policy_id and s.is_winner
        )
        denom = max(1, len(opp_matches))
        cells.append(
            CrossPlayCell(
                focal_member_id=focal.member_id,
                focal_architecture=focal.architecture_id,
                focal_seed=focal.seed,
                opponent_member_id=opp.member_id,
                opponent_architecture=opp.architecture_id,
                opponent_seed=opp.seed,
                is_cross_seed=(focal.seed != opp.seed),
                is_cross_architecture=(focal.architecture_id != opp.architecture_id),
                matches=len(opp_matches),
                focal_wins=focal_wins,
                focal_win_rate=focal_wins / denom,
            )
        )

    cross_seed = [c.focal_win_rate for c in cells if c.is_cross_seed]
    cross_arch = [c.focal_win_rate for c in cells if c.is_cross_architecture]
    archs = {focal.architecture_id} | {c.opponent_architecture for c in cells}
    eval_seeds = {focal.seed} | {c.opponent_seed for c in cells}
    return (
        CrossPlayMatrixReport(
            cells=tuple(cells),
            mean_cross_seed_win_rate=(
                sum(cross_seed) / len(cross_seed) if cross_seed else 0.0
            ),
            mean_cross_architecture_win_rate=(
                sum(cross_arch) / len(cross_arch) if cross_arch else 0.0
            ),
            distinct_architectures_evaluated=len(archs),
            distinct_seeds_evaluated=len(eval_seeds),
        ),
        tuple(all_matches),
    )


def evaluate_held_out_generalization(
    population: LeaguePopulation,
    *,
    focal_member_id: str = "main_ppo_recurrent_gen2",
    seeds: Sequence[int] = (2026,),
) -> tuple[HeldOutGeneralizationReport, tuple[ArenaMatchResult, ...]]:
    """Measure performance against independent held-out models never used in training."""
    focal = population.get_member(focal_member_id)
    held_out_pool = population.held_out_generalization_pool()
    if not held_out_pool:
        raise ValueError("Population has no held-out generalization models.")
    trainable_ids = {m.member_id for m in population.trainable_opponent_pool()}
    trainable_lineages = {m.lineage_id for m in population.trainable_opponent_pool()}

    for held_member in held_out_pool:
        if (
            not held_member.held_out_from_training
            or held_member.member_id in trainable_ids
            or held_member.lineage_id in trainable_lineages
        ):
            raise ValueError(
                f"Model {held_member.member_id!r} is contaminated with training pool."
            )

    configs: list[ArenaMatchConfig] = []
    for idx, held_member in enumerate(held_out_pool):
        partner = held_out_pool[(idx + 1) % len(held_out_pool)]
        base_order = (
            focal.policy_spec.policy_id,
            held_member.policy_spec.policy_id,
            partner.policy_spec.policy_id,
        )
        for seed in seeds:
            rot = idx % 3
            rotated = base_order[rot:] + base_order[:rot]
            configs.append(
                ArenaMatchConfig(
                    match_id=f"heldout-{held_member.member_id}-s{seed}-rot{rot}",
                    seed=seed + idx,
                    num_players=3,
                    policy_ids_by_player={
                        f"player_{i}": rotated[i] for i in range(3)
                    },
                    max_semantic_decisions=5000,
                )
            )
    matches = run_arena(configs, population.policy_registry()).matches

    entries: list[HeldOutGeneralizationEntry] = []
    for held_member in held_out_pool:
        held_pid = held_member.policy_spec.policy_id
        completed = [
            m
            for m in matches
            if m.status == "completed" and held_pid in m.policy_ids_by_player.values()
        ]
        focal_wins = sum(
            1
            for m in completed
            for s in m.seat_results
            if s.policy_id == focal.policy_spec.policy_id and s.is_winner
        )
        held_wins = sum(
            1
            for m in completed
            for s in m.seat_results
            if s.policy_id == held_pid and s.is_winner
        )
        denom = max(1, len(completed))
        entries.append(
            HeldOutGeneralizationEntry(
                held_out_member_id=held_member.member_id,
                held_out_policy_id=held_pid,
                held_out_lineage_id=held_member.lineage_id,
                matches=len(completed),
                focal_wins=focal_wins,
                held_out_wins=held_wins,
                focal_win_rate=focal_wins / denom,
                verified_held_out_from_training=True,
            )
        )

    mean_wr = sum(e.focal_win_rate for e in entries) / len(entries)
    min_wr = min(e.focal_win_rate for e in entries)
    return (
        HeldOutGeneralizationReport(
            focal_member_id=focal.member_id,
            entries=tuple(entries),
            mean_focal_win_rate=mean_wr,
            min_focal_win_rate=min_wr,
            all_verified_held_out=all(e.verified_held_out_from_training for e in entries),
        ),
        matches,
    )


def evaluate_historical_regression(
    population: LeaguePopulation,
    *,
    focal_member_id: str = "main_ppo_recurrent_gen2",
    seeds: Sequence[int] = (2026,),
) -> tuple[HistoricalRegressionReport, tuple[ArenaMatchResult, ...]]:
    """Measure performance against historical snapshots to detect training regression."""
    focal = population.get_member(focal_member_id)
    snapshots = population.historical_snapshots
    if not snapshots:
        raise ValueError("Population has no historical snapshots.")

    configs: list[ArenaMatchConfig] = []
    for idx, snap in enumerate(snapshots):
        partner = snapshots[(idx + 1) % len(snapshots)]
        base_order = (
            focal.policy_spec.policy_id,
            snap.policy_spec.policy_id,
            partner.policy_spec.policy_id,
        )
        for seed in seeds:
            rot = idx % 3
            rotated = base_order[rot:] + base_order[:rot]
            configs.append(
                ArenaMatchConfig(
                    match_id=f"histreg-{snap.member_id}-s{seed}-rot{rot}",
                    seed=seed + idx,
                    num_players=3,
                    policy_ids_by_player={
                        f"player_{i}": rotated[i] for i in range(3)
                    },
                    max_semantic_decisions=5000,
                )
            )
    matches = run_arena(configs, population.policy_registry()).matches

    entries: list[HistoricalRegressionEntry] = []
    for snap in snapshots:
        snap_pid = snap.policy_spec.policy_id
        completed = [
            m
            for m in matches
            if m.status == "completed" and snap_pid in m.policy_ids_by_player.values()
        ]
        focal_wins = sum(
            1
            for m in completed
            for s in m.seat_results
            if s.policy_id == focal.policy_spec.policy_id and s.is_winner
        )
        snap_wins = sum(
            1
            for m in completed
            for s in m.seat_results
            if s.policy_id == snap_pid and s.is_winner
        )
        denom = max(1, len(completed))
        focal_wr = focal_wins / denom
        snap_wr = snap_wins / denom
        entries.append(
            HistoricalRegressionEntry(
                snapshot_member_id=snap.member_id,
                snapshot_policy_id=snap_pid,
                snapshot_generation=snap.generation,
                matches=len(completed),
                focal_wins=focal_wins,
                snapshot_wins=snap_wins,
                focal_win_rate=focal_wr,
                snapshot_win_rate=snap_wr,
                regressed=(focal_wr + 0.15 < snap_wr),
            )
        )

    min_wr = min(e.focal_win_rate for e in entries)
    mean_wr = sum(e.focal_win_rate for e in entries) / len(entries)
    return (
        HistoricalRegressionReport(
            focal_member_id=focal.member_id,
            focal_generation=focal.generation,
            entries=tuple(entries),
            min_win_rate_vs_history=min_wr,
            mean_win_rate_vs_history=mean_wr,
            non_regression_passed=not any(e.regressed for e in entries),
        ),
        matches,
    )


def identify_worst_opponent_and_lineup(
    matches: Sequence[ArenaMatchResult],
    *,
    focal_policy_id: str,
) -> WorstCaseVulnerabilityReport:
    """Identify the worst individual opponent and worst multi-player lineup for focal_policy_id."""
    completed = [
        m
        for m in matches
        if m.status == "completed"
        and focal_policy_id in m.policy_ids_by_player.values()
    ]
    if not completed:
        raise ValueError("Require at least one completed match containing focal_policy_id.")

    opp_appearances: dict[str, int] = {}
    opp_focal_wins: dict[str, int] = {}
    lineup_appearances: dict[str, int] = {}
    lineup_focal_wins: dict[str, int] = {}
    lineup_tuples: dict[str, tuple[str, ...]] = {}
    total_focal_wins = 0

    for match in completed:
        seat_pids = [
            match.policy_ids_by_player[f"player_{i}"]
            for i in range(match.num_players)
        ]
        focal_won = any(
            s.policy_id == focal_policy_id and s.is_winner
            for s in match.seat_results
        )
        if focal_won:
            total_focal_wins += 1

        l_key, l_tuple = canonical_lineup_key(seat_pids)
        lineup_tuples[l_key] = l_tuple
        lineup_appearances[l_key] = lineup_appearances.get(l_key, 0) + 1
        if focal_won:
            lineup_focal_wins[l_key] = lineup_focal_wins.get(l_key, 0) + 1

        for opp_pid in set(seat_pids):
            if opp_pid == focal_policy_id:
                continue
            opp_appearances[opp_pid] = opp_appearances.get(opp_pid, 0) + 1
            if focal_won:
                opp_focal_wins[opp_pid] = opp_focal_wins.get(opp_pid, 0) + 1

    opp_win_rates = {
        pid: opp_focal_wins.get(pid, 0) / count
        for pid, count in opp_appearances.items()
    }
    lineup_win_rates = {
        l_key: lineup_focal_wins.get(l_key, 0) / count
        for l_key, count in lineup_appearances.items()
    }
    worst_opp = min(
        sorted(opp_win_rates.keys()),
        key=lambda pid: opp_win_rates[pid],
    )
    worst_l_key = min(
        sorted(lineup_win_rates.keys()),
        key=lambda k: lineup_win_rates[k],
    )
    overall_wr = total_focal_wins / len(completed)
    return WorstCaseVulnerabilityReport(
        focal_policy_id=focal_policy_id,
        worst_opponent_policy_id=worst_opp,
        worst_opponent_win_rate=opp_win_rates[worst_opp],
        worst_lineup_key=worst_l_key,
        worst_lineup_policies=lineup_tuples[worst_l_key],
        worst_lineup_win_rate=lineup_win_rates[worst_l_key],
        mean_overall_win_rate=overall_wr,
        worst_opponent_gap=overall_wr - opp_win_rates[worst_opp],
        worst_lineup_gap=overall_wr - lineup_win_rates[worst_l_key],
    )


def evaluate_mission_weaknesses(
    matches: Sequence[ArenaMatchResult],
    *,
    focal_policy_id: str,
) -> MissionWeaknessReport:
    """Measure per-mission and per-category win rates and identify mission weaknesses."""
    mission_apps: dict[str, int] = {}
    mission_wins: dict[str, int] = {}
    cat_apps: dict[str, int] = {}
    cat_wins: dict[str, int] = {}

    for match in matches:
        if match.status != "completed":
            continue
        for seat in match.seat_results:
            if seat.policy_id != focal_policy_id:
                continue
            m_id = seat.mission_id
            cat = _mission_category(m_id)
            mission_apps[m_id] = mission_apps.get(m_id, 0) + 1
            cat_apps[cat] = cat_apps.get(cat, 0) + 1
            if seat.is_winner:
                mission_wins[m_id] = mission_wins.get(m_id, 0) + 1
                cat_wins[cat] = cat_wins.get(cat, 0) + 1

    if not mission_apps:
        raise ValueError("No completed focal seat appearances found for mission audit.")

    by_mission = tuple(
        MissionStratumStat(
            stratum_id=m_id,
            stratum_kind="mission_id",
            appearances=mission_apps[m_id],
            wins=mission_wins.get(m_id, 0),
            win_rate=mission_wins.get(m_id, 0) / mission_apps[m_id],
        )
        for m_id in sorted(mission_apps.keys())
    )
    by_category = tuple(
        MissionStratumStat(
            stratum_id=cat,
            stratum_kind="mission_category",
            appearances=cat_apps[cat],
            wins=cat_wins.get(cat, 0),
            win_rate=cat_wins.get(cat, 0) / cat_apps[cat],
        )
        for cat in sorted(cat_apps.keys())
    )
    weakest = min(by_mission, key=lambda s: (s.win_rate, s.stratum_id))
    strongest = max(by_mission, key=lambda s: (s.win_rate, s.stratum_id))
    return MissionWeaknessReport(
        by_mission_id=by_mission,
        by_category=by_category,
        weakest_mission_id=weakest.stratum_id,
        weakest_mission_win_rate=weakest.win_rate,
        strongest_mission_id=strongest.stratum_id,
        strongest_mission_win_rate=strongest.win_rate,
        mission_win_rate_spread=strongest.win_rate - weakest.win_rate,
    )


def audit_seat_and_color_adaptation(
    matches: Sequence[ArenaMatchResult],
    *,
    focal_policy_id: str,
    deviation_tolerance: float = 0.55,
) -> SeatColorBiasAudit:
    """Audit focal performance across seats and canonical WAR colors to detect undue bias."""
    if deviation_tolerance <= 0.0:
        raise ValueError("deviation_tolerance must be positive.")
    seat_apps: dict[int, int] = {}
    seat_wins: dict[int, int] = {}

    for match in matches:
        if match.status != "completed":
            continue
        for seat in match.seat_results:
            if seat.policy_id != focal_policy_id:
                continue
            seat_idx = int(seat.player_id.removeprefix("player_"))
            seat_apps[seat_idx] = seat_apps.get(seat_idx, 0) + 1
            if seat.is_winner:
                seat_wins[seat_idx] = seat_wins.get(seat_idx, 0) + 1

    if not seat_apps:
        raise ValueError("No completed focal seat appearances found for seat/color audit.")

    raw_win_rates = {
        idx: seat_wins.get(idx, 0) / count for idx, count in seat_apps.items()
    }
    mean_wr = sum(raw_win_rates.values()) / len(raw_win_rates)
    strata = tuple(
        SeatColorStratumStat(
            seat_index=idx,
            player_id=f"player_{idx}",
            player_color=PLAYER_COLOR_IDS[idx],
            appearances=seat_apps[idx],
            wins=seat_wins.get(idx, 0),
            win_rate=raw_win_rates[idx],
            deviation_from_mean=raw_win_rates[idx] - mean_wr,
        )
        for idx in sorted(seat_apps.keys())
    )
    max_dev = max(abs(s.deviation_from_mean) for s in strata)
    return SeatColorBiasAudit(
        focal_policy_id=focal_policy_id,
        strata=strata,
        mean_win_rate=mean_wr,
        max_abs_seat_color_deviation=max_dev,
        deviation_tolerance=deviation_tolerance,
        undue_seat_or_color_adaptation=(max_dev > deviation_tolerance),
    )


def run_generalization_evaluation(
    population: LeaguePopulation,
    *,
    focal_member_id: str = "main_ppo_recurrent_gen2",
    seeds: Sequence[int] = (2026,),
    seat_deviation_tolerance: float = 0.55,
) -> GeneralizationEvaluationReport:
    """Run all six Phase 8.5 generalization audits on a focal league member."""
    focal = population.get_member(focal_member_id)
    focal_pid = focal.policy_spec.policy_id

    cross_play_report, xplay_matches = evaluate_cross_play(
        population,
        focal_member_id=focal_member_id,
        seeds=seeds,
    )
    held_out_report, held_matches = evaluate_held_out_generalization(
        population,
        focal_member_id=focal_member_id,
        seeds=seeds,
    )
    hist_report, hist_matches = evaluate_historical_regression(
        population,
        focal_member_id=focal_member_id,
        seeds=seeds,
    )
    combined_matches = (*xplay_matches, *held_matches, *hist_matches)

    worst_case = identify_worst_opponent_and_lineup(
        combined_matches,
        focal_policy_id=focal_pid,
    )
    mission_report = evaluate_mission_weaknesses(
        combined_matches,
        focal_policy_id=focal_pid,
    )
    seat_color_audit = audit_seat_and_color_adaptation(
        combined_matches,
        focal_policy_id=focal_pid,
        deviation_tolerance=seat_deviation_tolerance,
    )
    return GeneralizationEvaluationReport(
        focal_member_id=focal_member_id,
        focal_policy_id=focal_pid,
        cross_play=cross_play_report,
        held_out=held_out_report,
        worst_case=worst_case,
        historical_regression=hist_report,
        mission_weaknesses=mission_report,
        seat_color_audit=seat_color_audit,
    )


__all__ = [
    "CrossPlayCell",
    "CrossPlayMatrixReport",
    "GeneralizationEvaluationReport",
    "HeldOutGeneralizationEntry",
    "HeldOutGeneralizationReport",
    "HistoricalRegressionEntry",
    "HistoricalRegressionReport",
    "MissionStratumStat",
    "MissionWeaknessReport",
    "SeatColorBiasAudit",
    "SeatColorStratumStat",
    "WorstCaseVulnerabilityReport",
    "audit_seat_and_color_adaptation",
    "evaluate_cross_play",
    "evaluate_held_out_generalization",
    "evaluate_historical_regression",
    "evaluate_mission_weaknesses",
    "identify_worst_opponent_and_lineup",
    "run_generalization_evaluation",
]
