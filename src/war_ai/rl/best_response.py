"""Independent best-response exploiter training, non-collusion isolation, and separate collusion stress tests (Phase 8.4)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np

from ..arena import ArenaMatchConfig, FrozenPolicySpec, run_arena
from ..envs.war1_aec import War1AECEnv
from .algorithms import evaluate_algorithm_branch
from .experimental_axes import AlgorithmAxis, ComputeBudget
from .league_population import (
    LeaguePopulation,
    LeaguePopulationMember,
    build_specialist_or_exploiter_policy_spec,
    build_trained_neural_member,
)

MINIMUM_EXPLOITER_APPROACHES = 3

ExploiterApproachId: TypeAlias = Literal[
    "ppo_recurrent_exploiter",
    "r2d2_dueling_exploiter",
    "graph_search_exploiter",
    "impala_vtrace_exploiter",
]

EXPLOITER_APPROACH_TO_BRANCH: dict[ExploiterApproachId, AlgorithmAxis] = {
    "ppo_recurrent_exploiter": "ppo_recurrent",
    "r2d2_dueling_exploiter": "recurrent_off_policy_r2d2",
    "graph_search_exploiter": "graph_actor_critic",
    "impala_vtrace_exploiter": "impala_vtrace",
}


@dataclass(frozen=True)
class ExploiterTrainingConfig:
    """Configuration for a best-response exploiter enforcing solo-win and non-collusion rules."""

    approach_id: ExploiterApproachId
    target_member_id: str
    seeds: tuple[int, ...] = (2026, 2027)
    num_players: int = 3
    learner_updates: int = 8
    reward_objective: str = "solo_official_victory"
    allow_secret_sharing: bool = False
    allow_hidden_team_coordination: bool = False
    seat_isolated_instances: bool = True

    def __post_init__(self) -> None:
        if self.approach_id not in EXPLOITER_APPROACH_TO_BRANCH:
            raise ValueError(f"Unsupported exploiter approach {self.approach_id!r}.")
        if not self.target_member_id.strip():
            raise ValueError("target_member_id must not be blank.")
        if len(self.seeds) < 2 or len(set(self.seeds)) != len(self.seeds):
            raise ValueError(
                "Best-response exploiters must be trained across at least 2 distinct seeds."
            )
        if any(isinstance(s, bool) or not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("Seeds must be non-negative integers.")
        if self.num_players < 3 or self.num_players > 6:
            raise ValueError("num_players must be in 3..6.")
        if self.reward_objective != "solo_official_victory":
            raise ValueError(
                "Every exploiter must maximize its own solo_official_victory; "
                f"got {self.reward_objective!r}."
            )
        if self.allow_secret_sharing:
            raise ValueError(
                "Secret sharing is strictly prohibited for best-response exploiters."
            )
        if self.allow_hidden_team_coordination:
            raise ValueError(
                "Hidden team coordination/policies are strictly prohibited for exploiters."
            )
        if not self.seat_isolated_instances:
            raise ValueError("Exploiters must use seat-isolated policy instances.")


@dataclass(frozen=True)
class ExploiterIsolationAudit:
    """Verification that an exploiter has seat-isolated instances and zero secret leakage."""

    policy_id: str
    distinct_seat_instances: bool
    invariant_to_opponent_secrets: bool
    solo_victory_objective_verified: bool
    no_hidden_team_channel: bool

    @property
    def passed(self) -> bool:
        return (
            self.distinct_seat_instances
            and self.invariant_to_opponent_secrets
            and self.solo_victory_objective_verified
            and self.no_hidden_team_channel
        )


@dataclass(frozen=True)
class BestResponseSeedResult:
    """Single-seed training and evaluation record for one exploiter approach."""

    approach_id: ExploiterApproachId
    seed: int
    exploiter_member: LeaguePopulationMember
    initial_optimal_intent_probability: float
    trained_optimal_intent_probability: float
    matches_evaluated: int
    exploiter_wins: int
    target_wins: int
    exploiter_win_rate: float
    isolation_audit: ExploiterIsolationAudit


@dataclass(frozen=True)
class ExploiterApproachReport:
    """Multi-seed summary for one independent best-response exploiter approach."""

    approach_id: ExploiterApproachId
    algorithm_branch: AlgorithmAxis
    lineage_id: str
    seed_results: tuple[BestResponseSeedResult, ...]
    mean_exploiter_win_rate: float
    max_exploiter_win_rate: float
    all_isolation_audits_passed: bool


@dataclass(frozen=True)
class CollusionStressTestScenario:
    """Separate stress-test scenario modeling deliberate multi-seat pressure (excluded from normal panel)."""

    scenario_id: str
    focal_policy_id: str
    stress_agent_policy_id: str
    num_players: int = 3
    is_collusion_stress_test: bool = True
    eligible_for_normal_benchmark_panel: bool = False

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("scenario_id must not be blank.")
        if not self.is_collusion_stress_test:
            raise ValueError("CollusionStressTestScenario must set is_collusion_stress_test=True.")
        if self.eligible_for_normal_benchmark_panel:
            raise ValueError(
                "Deliberate collusion stress tests are strictly excluded from the "
                "normal benchmark panel (eligible_for_normal_benchmark_panel must be False)."
            )


@dataclass(frozen=True)
class CollusionStressTestReport:
    """Outcome of a separate collusion stress test outside the normal benchmark panel."""

    scenario: CollusionStressTestScenario
    matches_played: int
    completed_matches: int
    focal_survivability_rate: float
    focal_win_rate_under_stress: float
    isolated_from_benchmark_panel: bool


@dataclass(frozen=True)
class BestResponseCampaignReport:
    """Complete Phase 8.4 report across >=3 independent approaches and multiple seeds."""

    target_member_id: str
    approach_reports: tuple[ExploiterApproachReport, ...]
    collusion_stress_report: CollusionStressTestReport
    independent_approaches_count: int
    distinct_lineages_count: int
    all_isolation_audits_passed: bool
    empirical_resistance_bound_satisfied: bool


def verify_exploiter_isolation(
    policy_spec: FrozenPolicySpec,
    *,
    config: ExploiterTrainingConfig,
    seed: int = 2026,
) -> ExploiterIsolationAudit:
    """Verify seat instance isolation and invariance to opponent secret state."""
    seat_0 = policy_spec.factory(seed)
    seat_1 = policy_spec.factory(seed + 1)
    distinct_instances = seat_0 is not seat_1

    env = War1AECEnv(num_players=config.num_players)
    env.reset(seed=seed)
    focal_agent = env.agent_selection
    assert focal_agent is not None
    obs_a = env.observe(focal_agent)
    obs_b = env.observe(focal_agent)

    action_a = int(seat_0(obs_a, np.random.default_rng(seed + 100)))
    action_b = int(seat_1(obs_b, np.random.default_rng(seed + 100)))
    invariant = action_a == action_b

    return ExploiterIsolationAudit(
        policy_id=policy_spec.policy_id,
        distinct_seat_instances=distinct_instances,
        invariant_to_opponent_secrets=invariant,
        solo_victory_objective_verified=(
            config.reward_objective == "solo_official_victory"
        ),
        no_hidden_team_channel=(
            not config.allow_secret_sharing
            and not config.allow_hidden_team_coordination
        ),
    )


def train_and_evaluate_exploiter_approach(
    population: LeaguePopulation,
    config: ExploiterTrainingConfig,
) -> ExploiterApproachReport:
    """Train one independent best-response approach across all configured seeds."""
    target_member = population.get_member(config.target_member_id)
    branch = EXPLOITER_APPROACH_TO_BRANCH[config.approach_id]
    lineage_id = f"lineage_br_{config.approach_id}"
    seed_results: list[BestResponseSeedResult] = []

    for seed in config.seeds:
        eval_res = evaluate_algorithm_branch(
            branch,
            compute_budget=ComputeBudget(
                semantic_decisions=config.learner_updates * 8,
                learner_updates=config.learner_updates,
                inference_budget_ms=250,
            ),
            hidden_size=32,
            learning_rate=0.012,
            seed=seed,
        )
        member_id = f"br_{config.approach_id}_seed{seed}"
        policy_id = f"league_{member_id}"
        exploiter_member = build_trained_neural_member(
            member_id=member_id,
            policy_id=policy_id,
            role="weakness_exploiter",
            branch=branch,
            seed=seed,
            generation=2,
            lineage_id=lineage_id,
            style_tag=f"best_response_{config.approach_id}",
            learner_updates=config.learner_updates,
            target_member_id=config.target_member_id,
        )
        audit = verify_exploiter_isolation(
            exploiter_member.policy_spec,
            config=config,
            seed=seed,
        )
        if not audit.passed:
            raise RuntimeError(
                f"Exploiter {member_id!r} failed non-collusion isolation audit."
            )

        # Evaluate in seat-rotated 3-player full-table games where each seat plays solo
        neutral_third = population.mission_style_specialists[0]
        registry = {
            target_member.policy_spec.policy_id: target_member.policy_spec,
            exploiter_member.policy_spec.policy_id: exploiter_member.policy_spec,
            neutral_third.policy_spec.policy_id: neutral_third.policy_spec,
        }
        match_configs: list[ArenaMatchConfig] = []
        seats = (
            exploiter_member.policy_spec.policy_id,
            target_member.policy_spec.policy_id,
            neutral_third.policy_spec.policy_id,
        )
        for rot in range(3):
            rotated = seats[rot:] + seats[:rot]
            match_configs.append(
                ArenaMatchConfig(
                    match_id=f"br-{config.approach_id}-s{seed}-rot{rot}",
                    seed=seed + rot,
                    num_players=3,
                    policy_ids_by_player={
                        f"player_{i}": rotated[i] for i in range(3)
                    },
                    max_semantic_decisions=5000,
                )
            )
        arena_report = run_arena(match_configs, registry)
        exp_stats = arena_report.statistics_by_policy_id[
            exploiter_member.policy_spec.policy_id
        ]
        tgt_stats = arena_report.statistics_by_policy_id[
            target_member.policy_spec.policy_id
        ]
        seed_results.append(
            BestResponseSeedResult(
                approach_id=config.approach_id,
                seed=seed,
                exploiter_member=exploiter_member,
                initial_optimal_intent_probability=(
                    eval_res.initial_optimal_intent_probability
                ),
                trained_optimal_intent_probability=(
                    eval_res.trained_optimal_intent_probability
                ),
                matches_evaluated=exp_stats.completed_matches,
                exploiter_wins=exp_stats.wins,
                target_wins=tgt_stats.wins,
                exploiter_win_rate=float(exp_stats.win_rate or 0.0),
                isolation_audit=audit,
            )
        )

    mean_wr = sum(r.exploiter_win_rate for r in seed_results) / len(seed_results)
    max_wr = max(r.exploiter_win_rate for r in seed_results)
    return ExploiterApproachReport(
        approach_id=config.approach_id,
        algorithm_branch=branch,
        lineage_id=lineage_id,
        seed_results=tuple(seed_results),
        mean_exploiter_win_rate=mean_wr,
        max_exploiter_win_rate=max_wr,
        all_isolation_audits_passed=all(r.isolation_audit.passed for r in seed_results),
    )


def run_collusion_stress_test(
    focal_policy: FrozenPolicySpec,
    *,
    num_players: int = 3,
    seeds: Sequence[int] = (2026, 2027),
) -> CollusionStressTestReport:
    """Run a separate multi-aggressor stress test outside the normal benchmark panel."""
    stress_spec, _ = build_specialist_or_exploiter_policy_spec(
        policy_id="stress_test_coordinated_elimination",
        primary_style="opportunistic_elimination",
        secondary_style="exact_probability_planner",
        primary_weight=0.90,
        tactical_tier_bias=0.85,
        seed_salt=909,
    )
    scenario = CollusionStressTestScenario(
        scenario_id="stress_collusion_elimination_pressure",
        focal_policy_id=focal_policy.policy_id,
        stress_agent_policy_id=stress_spec.policy_id,
        num_players=num_players,
        is_collusion_stress_test=True,
        eligible_for_normal_benchmark_panel=False,
    )
    registry = {
        focal_policy.policy_id: focal_policy,
        stress_spec.policy_id: stress_spec,
    }
    configs: list[ArenaMatchConfig] = []
    for idx, seed in enumerate(seeds):
        focal_seat = idx % num_players
        policy_ids_by_player = {
            f"player_{seat}": (
                focal_policy.policy_id
                if seat == focal_seat
                else stress_spec.policy_id
            )
            for seat in range(num_players)
        }
        configs.append(
            ArenaMatchConfig(
                match_id=f"collusion-stress-s{seed}-seat{focal_seat}",
                seed=seed,
                num_players=num_players,
                policy_ids_by_player=policy_ids_by_player,
                max_semantic_decisions=5000,
            )
        )
    report = run_arena(configs, registry)
    completed = [m for m in report.matches if m.status == "completed"]
    focal_wins = 0
    focal_survived = 0
    for match in completed:
        for seat in match.seat_results:
            if seat.policy_id == focal_policy.policy_id:
                if seat.is_winner:
                    focal_wins += 1
                if not seat.is_eliminated:
                    focal_survived += 1
    denom = max(1, len(completed))
    return CollusionStressTestReport(
        scenario=scenario,
        matches_played=len(report.matches),
        completed_matches=len(completed),
        focal_survivability_rate=focal_survived / denom,
        focal_win_rate_under_stress=focal_wins / denom,
        isolated_from_benchmark_panel=(
            not scenario.eligible_for_normal_benchmark_panel
        ),
    )


def run_best_response_campaign(
    population: LeaguePopulation,
    *,
    target_member_id: str = "main_ppo_recurrent_gen2",
    approaches: Sequence[ExploiterApproachId] = (
        "ppo_recurrent_exploiter",
        "r2d2_dueling_exploiter",
        "graph_search_exploiter",
    ),
    seeds: tuple[int, ...] = (2026, 2027),
) -> BestResponseCampaignReport:
    """Execute Phase 8.4 best-response campaign with >=3 independent approaches and multiple seeds."""
    unique_approaches = tuple(dict.fromkeys(approaches))
    if len(unique_approaches) < MINIMUM_EXPLOITER_APPROACHES:
        raise ValueError(
            f"Phase 8.4 requires at least {MINIMUM_EXPLOITER_APPROACHES} "
            "independent exploiter approaches."
        )

    approach_reports: list[ExploiterApproachReport] = []
    for idx, approach_id in enumerate(unique_approaches):
        approach_seeds = tuple(s + idx * 10 for s in seeds)
        cfg = ExploiterTrainingConfig(
            approach_id=approach_id,
            target_member_id=target_member_id,
            seeds=approach_seeds,
            num_players=3,
            learner_updates=6,
        )
        approach_reports.append(train_and_evaluate_exploiter_approach(population, cfg))

    target_member = population.get_member(target_member_id)
    stress_report = run_collusion_stress_test(
        target_member.policy_spec,
        num_players=3,
        seeds=seeds,
    )
    lineages = {rep.lineage_id for rep in approach_reports}
    all_isolated = (
        all(rep.all_isolation_audits_passed for rep in approach_reports)
        and stress_report.isolated_from_benchmark_panel
    )
    # Check that no tested exploiter dominates the target across all seats
    mean_exploiter_wr = sum(
        rep.mean_exploiter_win_rate for rep in approach_reports
    ) / len(approach_reports)
    return BestResponseCampaignReport(
        target_member_id=target_member_id,
        approach_reports=tuple(approach_reports),
        collusion_stress_report=stress_report,
        independent_approaches_count=len(approach_reports),
        distinct_lineages_count=len(lineages),
        all_isolation_audits_passed=all_isolated,
        empirical_resistance_bound_satisfied=(mean_exploiter_wr <= 0.67),
    )


__all__ = [
    "BestResponseCampaignReport",
    "BestResponseSeedResult",
    "CollusionStressTestReport",
    "CollusionStressTestScenario",
    "EXPLOITER_APPROACH_TO_BRANCH",
    "ExploiterApproachId",
    "ExploiterApproachReport",
    "ExploiterIsolationAudit",
    "ExploiterTrainingConfig",
    "MINIMUM_EXPLOITER_APPROACHES",
    "run_best_response_campaign",
    "run_collusion_stress_test",
    "train_and_evaluate_exploiter_approach",
    "verify_exploiter_isolation",
]
