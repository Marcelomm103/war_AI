"""Frozen benchmark panel formation, diversity/strength gates, and Phase 8 campaign runner (Phase 8.6)."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..arena import ArenaMatchConfig, FrozenPolicySpec, run_arena
from .best_response import BestResponseCampaignReport, run_best_response_campaign
from .full_table_training import (
    FullTableTrainingReport,
    run_full_table_training_campaign,
)
from .generalization import (
    GeneralizationEvaluationReport,
    run_generalization_evaluation,
)
from .league_population import (
    LeaguePopulation,
    LeaguePopulationMember,
    PopulationCompletenessReport,
    build_default_league_population,
)
from .psro_metagame import PSROMetagameReport, run_psro_metagame_exploration

MINIMUM_BENCHMARK_PANEL_POLICIES = 8
MINIMUM_BENCHMARK_APPROACHES = 3

FORBIDDEN_WEAK_OR_RANDOM_TOKENS: tuple[str, ...] = (
    "random",
    "random_legal",
    "dummy",
    "weak",
    "passive_filler",
)


class NearDuplicateCheckpointError(ValueError):
    """Raised when near-identical checkpoints are counted toward panel diversity."""


class WeakOrRandomPanelPolicyError(ValueError):
    """Raised when random or intentionally weak policies are submitted to the frozen panel."""


@dataclass(frozen=True)
class FrozenBenchmarkPanelConfig:
    """Strict criteria governing membership in the Phase 8.6 frozen benchmark panel."""

    min_distinct_policies: int = MINIMUM_BENCHMARK_PANEL_POLICIES
    min_successful_approaches: int = MINIMUM_BENCHMARK_APPROACHES
    min_validation_score: float = 0.75
    min_generation: int = 1
    min_fingerprint_l2_distance: float = 0.005
    max_members_per_lineage: int = 1

    def __post_init__(self) -> None:
        if self.min_distinct_policies < MINIMUM_BENCHMARK_PANEL_POLICIES:
            raise ValueError(
                f"Frozen benchmark panel requires at least "
                f"{MINIMUM_BENCHMARK_PANEL_POLICIES} distinct policies."
            )
        if self.min_successful_approaches < MINIMUM_BENCHMARK_APPROACHES:
            raise ValueError(
                f"Frozen benchmark panel requires at least "
                f"{MINIMUM_BENCHMARK_APPROACHES} successful approaches."
            )
        if not 0.5 <= self.min_validation_score <= 1.0:
            raise ValueError("min_validation_score must be in [0.5, 1.0].")
        if self.min_generation < 1:
            raise ValueError("min_generation must be >= 1 to exclude raw gen-0 snapshots.")
        if self.min_fingerprint_l2_distance <= 0.0:
            raise ValueError("min_fingerprint_l2_distance must be positive.")
        if self.max_members_per_lineage < 1:
            raise ValueError("max_members_per_lineage must be at least 1.")


@dataclass(frozen=True)
class FrozenPanelValidationReport:
    """Validation report for a frozen benchmark panel (Phase 8.6)."""

    passed: bool
    distinct_policies_count: int
    distinct_successful_approaches: tuple[str, ...]
    distinct_lineages_count: int
    includes_leaders: bool
    includes_independent_opponents: bool
    includes_strong_exploiters: bool
    includes_mission_specialists: bool
    min_validation_score_observed: float
    min_pairwise_fingerprint_l2: float
    zero_random_or_weak_fillers: bool
    zero_near_duplicate_checkpoints: bool


def fingerprint_l2_distance(
    member_a: LeaguePopulationMember,
    member_b: LeaguePopulationMember,
) -> float:
    """Compute Euclidean distance between two members' parameter fingerprints."""
    fp_a = member_a.parameter_fingerprint
    fp_b = member_b.parameter_fingerprint
    length = min(len(fp_a), len(fp_b))
    if length == 0:
        return 0.0
    sq_sum = sum((float(fp_a[i]) - float(fp_b[i])) ** 2 for i in range(length))
    return math.sqrt(sq_sum)


@dataclass(frozen=True)
class FrozenBenchmarkPanel:
    """Immutable, validated benchmark panel of >=8 distinct strong frozen policies."""

    panel_id: str
    members: tuple[LeaguePopulationMember, ...]
    config: FrozenBenchmarkPanelConfig
    validation_report: FrozenPanelValidationReport

    def __post_init__(self) -> None:
        if not self.panel_id.strip():
            raise ValueError("panel_id must not be blank.")
        if not self.validation_report.passed:
            raise ValueError("FrozenBenchmarkPanel requires a passing validation_report.")

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return tuple(m.policy_spec.policy_id for m in self.members)

    @property
    def policy_registry(self) -> dict[str, FrozenPolicySpec]:
        return {m.policy_spec.policy_id: m.policy_spec for m in self.members}

    def build_homogeneous_scenario_configs(
        self,
        candidate: FrozenPolicySpec,
        *,
        adversary_policy_id: str,
        num_players: int = 3,
        seeds: Sequence[int] = (2026,),
        max_semantic_decisions: int = 5000,
    ) -> tuple[ArenaMatchConfig, ...]:
        """Build homogeneous scenario: 1 candidate vs N-1 instances of one panel adversary."""
        min_p, max_p = PLAYER_COUNT_RANGE
        if not min_p <= num_players <= max_p:
            raise ValueError(f"num_players must be in {min_p}..{max_p}.")
        if adversary_policy_id not in self.policy_registry:
            raise ValueError(f"Unknown panel adversary {adversary_policy_id!r}.")
        if candidate.policy_id == adversary_policy_id:
            raise ValueError("Candidate policy_id must be distinct from panel adversary.")

        configs: list[ArenaMatchConfig] = []
        for idx, seed in enumerate(seeds):
            cand_seat = idx % num_players
            policy_ids_by_player = {
                f"player_{seat}": (
                    candidate.policy_id
                    if seat == cand_seat
                    else adversary_policy_id
                )
                for seat in range(num_players)
            }
            configs.append(
                ArenaMatchConfig(
                    match_id=(
                        f"{self.panel_id}-homog-{adversary_policy_id}-"
                        f"{num_players}p-i{idx}-s{seed}-seat{cand_seat}"
                    ),
                    seed=seed,
                    num_players=num_players,
                    policy_ids_by_player=policy_ids_by_player,
                    max_semantic_decisions=max_semantic_decisions,
                )
            )
        return tuple(configs)

    def build_mixed_scenario_configs(
        self,
        candidate: FrozenPolicySpec,
        *,
        num_players: int = 3,
        seeds: Sequence[int] = (2026,),
        max_semantic_decisions: int = 5000,
    ) -> tuple[ArenaMatchConfig, ...]:
        """Build mixed scenario: 1 candidate vs N-1 distinct panel adversaries without replacement."""
        min_p, max_p = PLAYER_COUNT_RANGE
        if not min_p <= num_players <= max_p:
            raise ValueError(f"num_players must be in {min_p}..{max_p}.")
        available = [
            pid for pid in self.policy_ids if pid != candidate.policy_id
        ]
        if len(available) < num_players - 1:
            raise ValueError("Not enough distinct panel policies for mixed scenario.")

        configs: list[ArenaMatchConfig] = []
        for idx, seed in enumerate(seeds):
            rng = np.random.default_rng(seed)
            cand_seat = idx % num_players
            chosen_indices = rng.choice(
                len(available),
                size=num_players - 1,
                replace=False,
            )
            chosen_opponents = [available[int(i)] for i in chosen_indices]
            opp_iter = iter(chosen_opponents)
            policy_ids_by_player = {
                f"player_{seat}": (
                    candidate.policy_id if seat == cand_seat else next(opp_iter)
                )
                for seat in range(num_players)
            }
            configs.append(
                ArenaMatchConfig(
                    match_id=(
                        f"{self.panel_id}-mixed-{num_players}p-i{idx}-s{seed}-seat{cand_seat}"
                    ),
                    seed=seed,
                    num_players=num_players,
                    policy_ids_by_player=policy_ids_by_player,
                    max_semantic_decisions=max_semantic_decisions,
                )
            )
        return tuple(configs)


def validate_and_create_frozen_panel(
    panel_id: str,
    members: Sequence[LeaguePopulationMember],
    *,
    config: FrozenBenchmarkPanelConfig = FrozenBenchmarkPanelConfig(),
) -> FrozenBenchmarkPanel:
    """Validate all Phase 8.6 benchmark panel criteria and construct a FrozenBenchmarkPanel."""
    if len(members) < config.min_distinct_policies:
        raise ValueError(
            f"Panel requires at least {config.min_distinct_policies} policies, "
            f"got {len(members)}."
        )

    seen_member_ids: set[str] = set()
    seen_policy_ids: set[str] = set()
    seen_checkpoints: set[str] = set()
    lineage_counts: dict[str, int] = {}

    for member in members:
        text_fields = (
            member.member_id.lower(),
            member.policy_spec.policy_id.lower(),
            member.algorithm_branch.lower(),
            member.style_tag.lower(),
        )
        for token in FORBIDDEN_WEAK_OR_RANDOM_TOKENS:
            if any(token in field for field in text_fields):
                raise WeakOrRandomPanelPolicyError(
                    f"Policy {member.member_id!r} contains forbidden random/weak "
                    f"marker {token!r}."
                )
        if member.generation < config.min_generation:
            raise WeakOrRandomPanelPolicyError(
                f"Policy {member.member_id!r} has generation={member.generation} < "
                f"min_generation={config.min_generation} (early weak checkpoint)."
            )
        if member.validation_score < config.min_validation_score:
            raise WeakOrRandomPanelPolicyError(
                f"Policy {member.member_id!r} has validation_score="
                f"{member.validation_score:.4f} < {config.min_validation_score:.4f}."
            )
        if member.member_id in seen_member_ids or member.policy_spec.policy_id in seen_policy_ids:
            raise NearDuplicateCheckpointError(
                f"Duplicate member or policy ID {member.member_id!r} in panel."
            )
        assert member.policy_spec.checkpoint_sha256 is not None
        if member.policy_spec.checkpoint_sha256 in seen_checkpoints:
            raise NearDuplicateCheckpointError(
                f"Duplicate checkpoint_sha256 for {member.member_id!r} in panel."
            )
        seen_member_ids.add(member.member_id)
        seen_policy_ids.add(member.policy_spec.policy_id)
        seen_checkpoints.add(member.policy_spec.checkpoint_sha256)

        lineage_counts[member.lineage_id] = lineage_counts.get(member.lineage_id, 0) + 1
        if lineage_counts[member.lineage_id] > config.max_members_per_lineage:
            raise NearDuplicateCheckpointError(
                f"Lineage {member.lineage_id!r} appears "
                f"{lineage_counts[member.lineage_id]} times, exceeding "
                f"max_members_per_lineage={config.max_members_per_lineage}."
            )

    min_l2 = float("inf")
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            dist = fingerprint_l2_distance(members[i], members[j])
            min_l2 = min(min_l2, dist)
            if dist < config.min_fingerprint_l2_distance:
                raise NearDuplicateCheckpointError(
                    f"Checkpoints {members[i].member_id!r} and "
                    f"{members[j].member_id!r} are near-duplicates "
                    f"(fingerprint L2 distance {dist:.6f} < "
                    f"{config.min_fingerprint_l2_distance:.6f})."
                )

    approaches = tuple(sorted({m.algorithm_branch for m in members}))
    if len(approaches) < config.min_successful_approaches:
        raise ValueError(
            f"Panel requires at least {config.min_successful_approaches} distinct "
            f"successful approaches, got {len(approaches)}: {approaches}."
        )

    roles = {m.role for m in members}
    includes_leaders = bool(
        roles & {"architecture_seed_leader", "main_agent"}
    )
    includes_independent = "independent_generalization_model" in roles
    includes_exploiters = "weakness_exploiter" in roles
    includes_specialists = "mission_style_specialist" in roles

    if not includes_leaders:
        raise ValueError("Frozen benchmark panel must include relevant leaders.")
    if not includes_independent:
        raise ValueError("Frozen benchmark panel must include independent opponents.")
    if not includes_exploiters:
        raise ValueError("Frozen benchmark panel must include strong exploiters.")

    report = FrozenPanelValidationReport(
        passed=True,
        distinct_policies_count=len(members),
        distinct_successful_approaches=approaches,
        distinct_lineages_count=len(lineage_counts),
        includes_leaders=includes_leaders,
        includes_independent_opponents=includes_independent,
        includes_strong_exploiters=includes_exploiters,
        includes_mission_specialists=includes_specialists,
        min_validation_score_observed=min(m.validation_score for m in members),
        min_pairwise_fingerprint_l2=min_l2,
        zero_random_or_weak_fillers=True,
        zero_near_duplicate_checkpoints=True,
    )
    return FrozenBenchmarkPanel(
        panel_id=panel_id,
        members=tuple(members),
        config=config,
        validation_report=report,
    )


def assemble_frozen_benchmark_panel(
    population: LeaguePopulation,
    *,
    panel_id: str = "phase8_frozen_benchmark_panel_v1",
    exclude_member_ids: Sequence[str] = ("main_ppo_recurrent_gen2",),
    config: FrozenBenchmarkPanelConfig = FrozenBenchmarkPanelConfig(),
) -> FrozenBenchmarkPanel:
    """Select >=8 diverse, strong, non-duplicate policies from the league population."""
    excluded = set(exclude_member_ids)
    preferred_order = (
        # Relevant leaders (main + architecture/seed leaders)
        "main_r2d2_recurrent_gen2",
        "leader_gru_seed2026",
        "leader_gnn_seed2027",
        "leader_mlp_seed2028",
        "leader_search_seed2029",
        # Independent generalization opponents
        "heldout_impala_vtrace_seed2090",
        "heldout_exact_planner",
        "heldout_objective_planner",
        # Strong exploiters found
        "exploiter_anti_expansion",
        "exploiter_anti_fortress",
        "exploiter_anti_main_ppo",
        # Mission/style specialists
        "spec_continent_control",
        "spec_elimination_hunter",
        "spec_territory_expansion",
    )

    selected: list[LeaguePopulationMember] = []
    used_lineages: set[str] = set()
    member_map = {m.member_id: m for m in population.members}

    for member_id in preferred_order:
        if member_id in excluded or member_id not in member_map:
            continue
        candidate = member_map[member_id]
        if candidate.generation < config.min_generation:
            continue
        if candidate.validation_score < config.min_validation_score:
            continue
        if candidate.lineage_id in used_lineages:
            continue
        if any(
            fingerprint_l2_distance(candidate, existing)
            < config.min_fingerprint_l2_distance
            for existing in selected
        ):
            continue
        selected.append(candidate)
        used_lineages.add(candidate.lineage_id)
        if len(selected) >= 12:
            break

    return validate_and_create_frozen_panel(
        panel_id,
        selected,
        config=config,
    )


@dataclass(frozen=True)
class PanelBenchmarkSummary:
    """Summary of evaluating the candidate policy against the frozen benchmark panel."""

    candidate_policy_id: str
    homogeneous_matches: int
    homogeneous_wins: int
    homogeneous_win_rate: float
    mixed_matches: int
    mixed_wins: int
    mixed_win_rate: float


@dataclass(frozen=True)
class Phase8LeagueCampaignReport:
    """Complete Phase 8 deliverable report covering 8.1 through 8.6."""

    population_completeness: PopulationCompletenessReport
    full_table_training_summary: dict[str, object]
    psro_metagame_summary: dict[str, object]
    best_response_summary: dict[str, object]
    generalization_summary: dict[str, object]
    frozen_panel_validation: FrozenPanelValidationReport
    frozen_panel_policy_ids: tuple[str, ...]
    panel_benchmark_summary: PanelBenchmarkSummary


def run_phase8_league_campaign(
    *,
    seed: int = 2026,
    output_json_path: Path | None = None,
) -> Phase8LeagueCampaignReport:
    """Execute the complete Phase 8 league, PSRO, exploiter, generalization, and panel campaign."""
    # 8.1 Population
    population = build_default_league_population(seed=seed)
    completeness = population.validate_completeness()

    # 8.2 Full-table lineup sampling and training
    full_table_report: FullTableTrainingReport = run_full_table_training_campaign(
        population,
        focal_member_id="main_ppo_recurrent_gen2",
        num_lineups=4,
        learner_updates=6,
        seed=seed,
    )

    # 8.3 PSRO and empirical metagame
    psro_report: PSROMetagameReport = run_psro_metagame_exploration(
        population,
        seed=seed,
    )

    # 8.4 Best-response exploiters (>=3 independent approaches, multiple seeds, separate collusion test)
    br_report: BestResponseCampaignReport = run_best_response_campaign(
        population,
        target_member_id="main_ppo_recurrent_gen2",
        seeds=(seed, seed + 1),
    )

    # 8.5 Generalization evaluation
    gen_report: GeneralizationEvaluationReport = run_generalization_evaluation(
        population,
        focal_member_id="main_ppo_recurrent_gen2",
        seeds=(seed,),
    )

    # 8.6 Frozen benchmark panel (>=8 distinct policies, >=3 successful approaches)
    panel = assemble_frozen_benchmark_panel(population)
    candidate_spec = population.get_member("main_ppo_recurrent_gen2").policy_spec
    registry = {
        candidate_spec.policy_id: candidate_spec,
        **panel.policy_registry,
    }

    homog_configs = panel.build_homogeneous_scenario_configs(
        candidate_spec,
        adversary_policy_id="league_spec_continent_control",
        num_players=3,
        seeds=(seed, seed, seed),
    )
    mixed_configs = panel.build_mixed_scenario_configs(
        candidate_spec,
        num_players=3,
        seeds=(seed + 4, seed + 4, seed + 4),
    )
    homog_res = run_arena(homog_configs, registry)
    mixed_res = run_arena(mixed_configs, registry)

    h_stats = homog_res.statistics_by_policy_id[candidate_spec.policy_id]
    m_stats = mixed_res.statistics_by_policy_id[candidate_spec.policy_id]
    bench_summary = PanelBenchmarkSummary(
        candidate_policy_id=candidate_spec.policy_id,
        homogeneous_matches=h_stats.completed_matches,
        homogeneous_wins=h_stats.wins,
        homogeneous_win_rate=float(h_stats.win_rate or 0.0),
        mixed_matches=m_stats.completed_matches,
        mixed_wins=m_stats.wins,
        mixed_win_rate=float(m_stats.win_rate or 0.0),
    )

    campaign_report = Phase8LeagueCampaignReport(
        population_completeness=completeness,
        full_table_training_summary={
            "focal_member_id": full_table_report.focal_member_id,
            "num_lineups": len(full_table_report.sampled_lineups),
            "diversity": asdict(full_table_report.diversity_report),
            "initial_optimal_intent_probability": (
                full_table_report.initial_optimal_intent_probability
            ),
            "trained_optimal_intent_probability": (
                full_table_report.trained_optimal_intent_probability
            ),
        },
        psro_metagame_summary={
            "num_lineup_records": len(psro_report.lineup_payoffs),
            "detected_cycles": [asdict(c) for c in psro_report.detected_cycles],
            "rankings": [asdict(r) for r in psro_report.rankings],
            "elo_alone_insufficient": psro_report.elo_alone_insufficient,
            "oracle_trained_optimal_intent_probability": (
                psro_report.oracle_results[0].trained_optimal_intent_probability
            ),
        },
        best_response_summary={
            "target_member_id": br_report.target_member_id,
            "independent_approaches_count": br_report.independent_approaches_count,
            "distinct_lineages_count": br_report.distinct_lineages_count,
            "all_isolation_audits_passed": br_report.all_isolation_audits_passed,
            "empirical_resistance_bound_satisfied": (
                br_report.empirical_resistance_bound_satisfied
            ),
            "collusion_stress_isolated_from_panel": (
                br_report.collusion_stress_report.isolated_from_benchmark_panel
            ),
        },
        generalization_summary={
            "cross_play_mean_cross_seed_win_rate": (
                gen_report.cross_play.mean_cross_seed_win_rate
            ),
            "cross_play_mean_cross_arch_win_rate": (
                gen_report.cross_play.mean_cross_architecture_win_rate
            ),
            "held_out_mean_win_rate": gen_report.held_out.mean_focal_win_rate,
            "held_out_all_verified": gen_report.held_out.all_verified_held_out,
            "worst_opponent_policy_id": (
                gen_report.worst_case.worst_opponent_policy_id
            ),
            "worst_opponent_win_rate": (
                gen_report.worst_case.worst_opponent_win_rate
            ),
            "worst_lineup_key": gen_report.worst_case.worst_lineup_key,
            "worst_lineup_win_rate": gen_report.worst_case.worst_lineup_win_rate,
            "historical_non_regression_passed": (
                gen_report.historical_regression.non_regression_passed
            ),
            "weakest_mission_id": (
                gen_report.mission_weaknesses.weakest_mission_id
            ),
            "undue_seat_or_color_adaptation": (
                gen_report.seat_color_audit.undue_seat_or_color_adaptation
            ),
        },
        frozen_panel_validation=panel.validation_report,
        frozen_panel_policy_ids=panel.policy_ids,
        panel_benchmark_summary=bench_summary,
    )

    if output_json_path is not None:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_json_path.write_text(
            json.dumps(asdict(campaign_report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return campaign_report


def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the Phase 8 league and frozen benchmark panel campaign."
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("lib/war_AI/benchmarks/phase-8.6-league-panel-2026-10-28.json"),
    )
    args = parser.parse_args()
    run_phase8_league_campaign(seed=args.seed, output_json_path=args.output_json)


if __name__ == "__main__":
    _main()


__all__ = [
    "FORBIDDEN_WEAK_OR_RANDOM_TOKENS",
    "FrozenBenchmarkPanel",
    "FrozenBenchmarkPanelConfig",
    "FrozenPanelValidationReport",
    "MINIMUM_BENCHMARK_APPROACHES",
    "MINIMUM_BENCHMARK_PANEL_POLICIES",
    "NearDuplicateCheckpointError",
    "PanelBenchmarkSummary",
    "Phase8LeagueCampaignReport",
    "WeakOrRandomPanelPolicyError",
    "assemble_frozen_benchmark_panel",
    "fingerprint_l2_distance",
    "run_phase8_league_campaign",
    "validate_and_create_frozen_panel",
]
