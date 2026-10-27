"""Paired multi-seed arena evaluation and promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from math import comb
from statistics import fmean, median

from war_game_engine.contracts import OfficialVictoryOutcome
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..arena import (
    ArenaMatchConfig,
    ArenaMatchResult,
    FrozenPolicySpec,
    run_arena,
)


@dataclass(frozen=True)
class SeedComparison:
    seed: int
    num_players: int
    candidate_policy_id: str
    baseline_policy_id: str
    candidate_wins: int
    candidate_appearances: int
    baseline_wins: int
    baseline_appearances: int
    candidate_mission_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Evaluation seeds must be non-negative integers.")
        minimum, maximum = PLAYER_COUNT_RANGE
        if (
            isinstance(self.num_players, bool)
            or not isinstance(self.num_players, int)
            or not minimum <= self.num_players <= maximum
        ):
            raise ValueError("Evaluation table size is outside the WAR player range.")
        if (
            not self.candidate_policy_id.strip()
            or not self.baseline_policy_id.strip()
            or self.candidate_policy_id == self.baseline_policy_id
        ):
            raise ValueError("Candidate and baseline policy IDs must be distinct.")
        for label, wins, appearances in (
            ("candidate", self.candidate_wins, self.candidate_appearances),
            ("baseline", self.baseline_wins, self.baseline_appearances),
        ):
            if (
                isinstance(appearances, bool)
                or not isinstance(appearances, int)
                or appearances < 1
            ):
                raise ValueError(f"{label} appearances must be positive.")
            if (
                isinstance(wins, bool)
                or not isinstance(wins, int)
                or not 0 <= wins <= appearances
            ):
                raise ValueError(f"{label} wins must fit its completed appearances.")
        if not self.candidate_mission_ids or any(
            not mission_id.strip() for mission_id in self.candidate_mission_ids
        ):
            raise ValueError("Each paired seed must record candidate missions.")

    @property
    def candidate_win_rate(self) -> float:
        return self.candidate_wins / self.candidate_appearances

    @property
    def baseline_win_rate(self) -> float:
        return self.baseline_wins / self.baseline_appearances

    @property
    def paired_gain(self) -> float:
        return self.candidate_win_rate - self.baseline_win_rate


@dataclass(frozen=True)
class PromotionCriteria:
    minimum_seed_count: int = 10
    minimum_positive_fraction: float = 0.8
    significance_level: float = 0.05

    def __post_init__(self) -> None:
        if (
            isinstance(self.minimum_seed_count, bool)
            or not isinstance(self.minimum_seed_count, int)
            or self.minimum_seed_count < 10
        ):
            raise ValueError("Promoted configurations require at least ten seeds.")
        if (
            isinstance(self.minimum_positive_fraction, bool)
            or not math.isfinite(self.minimum_positive_fraction)
            or not 0.0 < self.minimum_positive_fraction <= 1.0
        ):
            raise ValueError("minimum_positive_fraction must be in (0, 1].")
        if (
            isinstance(self.significance_level, bool)
            or not math.isfinite(self.significance_level)
            or not 0.0 < self.significance_level < 1.0
        ):
            raise ValueError("significance_level must be in (0, 1).")


@dataclass(frozen=True)
class PromotionDecision:
    promote: bool
    seed_count: int
    positive_seed_count: int
    positive_fraction: float
    mean_paired_gain: float
    median_paired_gain: float
    sign_test_p_value: float
    candidate_policy_id: str | None
    baseline_policy_id: str | None
    num_players: int | None
    evaluated_seeds: tuple[int, ...]
    reasons: tuple[str, ...]


def build_paired_seed_configs(
    *,
    evaluation_id: str,
    candidate_policy_id: str,
    baseline_policy_id: str,
    seed: int,
    num_players: int = 3,
    max_semantic_decisions: int = 10_000,
    decision_budget_ms: int | None = None,
) -> tuple[ArenaMatchConfig, ...]:
    """Create one same-seed full-game match for each candidate seat."""
    if not evaluation_id.strip():
        raise ValueError("evaluation_id must not be blank.")
    if (
        not candidate_policy_id.strip()
        or not baseline_policy_id.strip()
        or candidate_policy_id == baseline_policy_id
    ):
        raise ValueError("Candidate and baseline policy IDs must be distinct.")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("Evaluation seed must be a non-negative integer.")
    minimum, maximum = PLAYER_COUNT_RANGE
    if (
        isinstance(num_players, bool)
        or not isinstance(num_players, int)
        or not minimum <= num_players <= maximum
    ):
        raise ValueError("num_players is outside the WAR player range.")

    configs: list[ArenaMatchConfig] = []
    for candidate_seat in range(num_players):
        policy_ids_by_player = {
            f"player_{seat}": (
                candidate_policy_id
                if seat == candidate_seat
                else baseline_policy_id
            )
            for seat in range(num_players)
        }
        configs.append(
            ArenaMatchConfig(
                match_id=(
                    f"{evaluation_id}-seed-{seed}-seat-{candidate_seat}"
                ),
                seed=seed,
                num_players=num_players,
                policy_ids_by_player=policy_ids_by_player,
                max_semantic_decisions=max_semantic_decisions,
                decision_budget_ms=decision_budget_ms,
            )
        )
    return tuple(configs)


def run_paired_seed_evaluation(
    candidate: FrozenPolicySpec,
    baseline: FrozenPolicySpec,
    seeds: tuple[int, ...],
    *,
    evaluation_id: str,
    num_players: int = 3,
    max_semantic_decisions: int = 10_000,
    decision_budget_ms: int | None = None,
) -> tuple[SeedComparison, ...]:
    """Run complete seat-rotated arena matches for each paired seed."""
    if candidate.policy_id == baseline.policy_id:
        raise ValueError("Candidate and baseline policy IDs must be distinct.")
    if candidate.checkpoint_sha256 is None:
        raise ValueError("Candidate evaluations require a frozen checkpoint hash.")
    if not seeds:
        raise ValueError("At least one evaluation seed is required.")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Evaluation seeds must be unique.")

    registry = {
        candidate.policy_id: candidate,
        baseline.policy_id: baseline,
    }
    comparisons: list[SeedComparison] = []
    for seed in seeds:
        configs = build_paired_seed_configs(
            evaluation_id=evaluation_id,
            candidate_policy_id=candidate.policy_id,
            baseline_policy_id=baseline.policy_id,
            seed=seed,
            num_players=num_players,
            max_semantic_decisions=max_semantic_decisions,
            decision_budget_ms=decision_budget_ms,
        )
        report = run_arena(configs, registry)
        if any(match.status != "completed" for match in report.matches):
            statuses = sorted({match.status for match in report.matches})
            raise RuntimeError(
                f"Seed {seed} produced incomplete arena matches: {statuses}."
            )
        comparisons.append(
            _compare_seed_matches(
                seed,
                report.matches,
                candidate_policy_id=candidate.policy_id,
                baseline_policy_id=baseline.policy_id,
                num_players=num_players,
            )
        )
    return tuple(comparisons)


def assess_promotion(
    comparisons: tuple[SeedComparison, ...],
    *,
    criteria: PromotionCriteria = PromotionCriteria(),
) -> PromotionDecision:
    """Promote only reproducible official-win gains with paired significance."""
    seeds = tuple(result.seed for result in comparisons)
    if len(set(seeds)) != len(seeds):
        raise ValueError("Promotion comparisons must use unique seeds.")
    configurations = {
        (
            result.candidate_policy_id,
            result.baseline_policy_id,
            result.num_players,
        )
        for result in comparisons
    }
    if len(configurations) > 1:
        raise ValueError(
            "A promotion decision compares one candidate/baseline/table config."
        )
    configuration = next(iter(configurations), None)

    gains = tuple(result.paired_gain for result in comparisons)
    positive_count = sum(gain > 0.0 for gain in gains)
    positive_fraction = positive_count / len(gains) if gains else 0.0
    mean_gain = fmean(gains) if gains else 0.0
    median_gain = median(gains) if gains else 0.0
    non_tied_gains = tuple(gain for gain in gains if gain != 0.0)
    positive_non_tied = sum(gain > 0.0 for gain in non_tied_gains)
    sign_p_value = _one_sided_sign_test(
        positive_non_tied,
        len(non_tied_gains),
    )

    reasons: list[str] = []
    if len(gains) < criteria.minimum_seed_count:
        reasons.append(
            f"requires at least {criteria.minimum_seed_count} paired seeds"
        )
    if positive_fraction < criteria.minimum_positive_fraction:
        reasons.append(
            "learning gain was not positive on enough independent seeds"
        )
    if mean_gain <= 0.0:
        reasons.append("mean paired official win-rate gain is not positive")
    if sign_p_value >= criteria.significance_level:
        reasons.append("paired one-sided sign test is not statistically significant")
    return PromotionDecision(
        promote=not reasons,
        seed_count=len(gains),
        positive_seed_count=positive_count,
        positive_fraction=positive_fraction,
        mean_paired_gain=mean_gain,
        median_paired_gain=float(median_gain),
        sign_test_p_value=sign_p_value,
        candidate_policy_id=None if configuration is None else configuration[0],
        baseline_policy_id=None if configuration is None else configuration[1],
        num_players=None if configuration is None else configuration[2],
        evaluated_seeds=seeds,
        reasons=tuple(reasons),
    )


def _compare_seed_matches(
    seed: int,
    matches: tuple[ArenaMatchResult, ...],
    *,
    candidate_policy_id: str,
    baseline_policy_id: str,
    num_players: int,
) -> SeedComparison:
    if len(matches) != num_players:
        raise ValueError("Paired seed must contain one match per candidate seat.")
    expected_seats = {f"player_{index}" for index in range(num_players)}
    candidate_seats: set[str] = set()
    candidate_wins = 0
    baseline_wins = 0
    candidate_missions: list[str] = []
    baseline_appearances = 0

    for match in matches:
        if (
            match.status != "completed"
            or match.seed != seed
            or match.num_players != num_players
            or not isinstance(match.outcome, OfficialVictoryOutcome)
        ):
            raise ValueError("Seed comparison contains an incomplete or mismatched match.")
        if set(match.policy_ids_by_player) != expected_seats:
            raise ValueError("Seed match does not contain every configured seat.")
        candidate_seat_ids = {
            player_id
            for player_id, policy_id in match.policy_ids_by_player.items()
            if policy_id == candidate_policy_id
        }
        if len(candidate_seat_ids) != 1:
            raise ValueError("Each paired match must contain one candidate seat.")
        candidate_seat = next(iter(candidate_seat_ids))
        candidate_seats.add(candidate_seat)
        candidate_wins += sum(
            seat.is_winner
            for seat in match.seat_results
            if seat.policy_id == candidate_policy_id
        )
        baseline_seats = tuple(
            seat for seat in match.seat_results if seat.policy_id == baseline_policy_id
        )
        if len(baseline_seats) != num_players - 1:
            raise ValueError("Every remaining seat must use the paired baseline.")
        baseline_wins += sum(seat.is_winner for seat in baseline_seats)
        baseline_appearances += len(baseline_seats)
        candidate_result = next(
            seat for seat in match.seat_results if seat.player_id == candidate_seat
        )
        candidate_missions.append(candidate_result.mission_id)

    if candidate_seats != expected_seats:
        raise ValueError("Candidate policy did not rotate through every seat.")
    return SeedComparison(
        seed=seed,
        num_players=num_players,
        candidate_policy_id=candidate_policy_id,
        baseline_policy_id=baseline_policy_id,
        candidate_wins=candidate_wins,
        candidate_appearances=num_players,
        baseline_wins=baseline_wins,
        baseline_appearances=baseline_appearances,
        candidate_mission_ids=tuple(candidate_missions),
    )


def _one_sided_sign_test(positive_count: int, non_tied_count: int) -> float:
    if non_tied_count == 0:
        return 1.0
    favorable_tails = sum(
        comb(non_tied_count, count)
        for count in range(positive_count, non_tied_count + 1)
    )
    return float(favorable_tails) / float(2**non_tied_count)


__all__ = [
    "PromotionCriteria",
    "PromotionDecision",
    "SeedComparison",
    "assess_promotion",
    "build_paired_seed_configs",
    "run_paired_seed_evaluation",
]
