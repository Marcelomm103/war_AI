"""Reproducible, replayable evaluation arena for frozen WAR policies."""

from __future__ import annotations

import logging
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter_ns
from typing import Any, Literal, TypeAlias, TypeGuard

import numpy as np
from pydantic import Field, model_validator
from war_game_engine.contracts import (
    ContractModel,
    FullReplay,
    NonNegativeInt,
    OfficialVictoryOutcome,
    OperationalInterruptionOutcome,
    PositiveInt,
    ReplayOutcome,
    Sha256Hash,
    TrainingTruncationOutcome,
)
from war_game_engine.objectives import effective_objective_id
from war_game_engine.setup import PLAYER_COUNT_RANGE

from .baselines import create_baseline
from .envs.war1_aec import War1AECEnv, War1AECObservation

ArenaPolicy: TypeAlias = Callable[
    [War1AECObservation, np.random.Generator],
    int | np.integer[Any],
]
PolicyFactory: TypeAlias = Callable[[int], ArenaPolicy]
ArenaMatchStatus: TypeAlias = Literal["completed", "truncated", "failed"]
ArenaFailureClass: TypeAlias = Literal[
    "policy_factory_error",
    "policy_error",
    "illegal_action",
    "decision_budget_exceeded",
    "engine_error",
]

_LOGGER = logging.getLogger(__name__)
_CHECKPOINT_SHA256_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_ARENA_SCHEMA_VERSION: Literal["1.0"] = "1.0"


@dataclass(frozen=True, slots=True)
class FrozenPolicySpec:
    """Factory and immutable artifact identity for one frozen policy."""

    policy_id: str
    factory: PolicyFactory
    checkpoint_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("Policy IDs must not be blank.")
        if not callable(self.factory):
            raise TypeError("Policy factories must be callable.")
        if self.checkpoint_sha256 is not None and (
            _CHECKPOINT_SHA256_PATTERN.fullmatch(self.checkpoint_sha256) is None
        ):
            raise ValueError(
                "Checkpoint digests must use the 'sha256:<64 lowercase hex>' format."
            )


def baseline_policy_spec(policy_id: str) -> FrozenPolicySpec:
    """Create a seat-isolated factory for a built-in legal-action baseline."""
    baseline = create_baseline(policy_id)

    def factory(_seed: int) -> ArenaPolicy:
        def policy(
            observation: War1AECObservation,
            rng: np.random.Generator,
        ) -> int | np.integer[Any]:
            return baseline(observation, rng)

        return policy

    return FrozenPolicySpec(policy_id=policy_id, factory=factory)


class ArenaMatchConfig(ContractModel):
    match_id: str = Field(min_length=1)
    seed: NonNegativeInt
    num_players: int = Field(
        ge=PLAYER_COUNT_RANGE[0],
        le=PLAYER_COUNT_RANGE[1],
    )
    policy_ids_by_player: dict[str, str]
    max_semantic_decisions: PositiveInt
    decision_budget_ms: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_match_config(self) -> ArenaMatchConfig:
        if not self.match_id.strip():
            raise ValueError("Match IDs must not be blank.")
        expected_players = {
            f"player_{index}" for index in range(self.num_players)
        }
        if set(self.policy_ids_by_player) != expected_players:
            raise ValueError(
                "Assign exactly one policy to each player; "
                f"expected={sorted(expected_players)}, "
                f"actual={sorted(self.policy_ids_by_player)}."
            )
        if any(not policy_id.strip() for policy_id in self.policy_ids_by_player.values()):
            raise ValueError("Policy IDs must not be blank.")
        return self


class ArenaSeatResult(ContractModel):
    player_id: str = Field(min_length=1)
    policy_id: str = Field(min_length=1)
    checkpoint_sha256: Sha256Hash | None
    mission_id: str = Field(min_length=1)
    is_eliminated: bool
    is_winner: bool


class ArenaMatchResult(ContractModel):
    contract_type: Literal["arena_match_result"] = "arena_match_result"
    schema_version: Literal["1.0"] = _ARENA_SCHEMA_VERSION
    match_id: str = Field(min_length=1)
    game_id: str = Field(min_length=1)
    status: ArenaMatchStatus
    seed: NonNegativeInt
    num_players: int = Field(
        ge=PLAYER_COUNT_RANGE[0],
        le=PLAYER_COUNT_RANGE[1],
    )
    policy_ids_by_player: dict[str, str]
    checkpoint_sha256_by_player: dict[str, Sha256Hash | None]
    policy_factory_seeds_by_player: dict[str, NonNegativeInt]
    policy_rng_seeds_by_player: dict[str, NonNegativeInt]
    max_semantic_decisions: PositiveInt
    decision_budget_ms: PositiveInt | None
    max_policy_decision_duration_ms: float = Field(ge=0.0)
    semantic_decisions: NonNegativeInt
    elapsed_ms: NonNegativeInt
    winner_player_id: str | None
    seat_results: tuple[ArenaSeatResult, ...]
    outcome: ReplayOutcome
    failure_classification: ArenaFailureClass | None
    failure_exception_type: str | None
    failure_message: str | None
    replay: FullReplay

    @model_validator(mode="after")
    def validate_match_result(self) -> ArenaMatchResult:
        expected_players = {
            f"player_{index}" for index in range(self.num_players)
        }
        mapped_players = (
            self.policy_ids_by_player,
            self.checkpoint_sha256_by_player,
            self.policy_factory_seeds_by_player,
            self.policy_rng_seeds_by_player,
        )
        if any(set(mapping) != expected_players for mapping in mapped_players):
            raise ValueError("Arena result maps must include exactly every player.")
        if {seat.player_id for seat in self.seat_results} != expected_players:
            raise ValueError("Arena results must include exactly one row per player.")
        if len(self.seat_results) != self.num_players:
            raise ValueError("Arena results contain duplicate player rows.")
        for seat in self.seat_results:
            if self.policy_ids_by_player[seat.player_id] != seat.policy_id:
                raise ValueError("Seat result policy does not match the match config.")
            if (
                self.checkpoint_sha256_by_player[seat.player_id]
                != seat.checkpoint_sha256
            ):
                raise ValueError("Seat result checkpoint does not match the match config.")
        if self.semantic_decisions > self.max_semantic_decisions:
            raise ValueError("Semantic decisions exceed the configured match budget.")
        if self.replay.game_id != self.game_id:
            raise ValueError("Arena replay game ID does not match its result.")
        if self.replay.replay_id != f"{self.match_id}:full-replay":
            raise ValueError("Arena replay ID does not match its result.")
        if self.replay.outcome != self.outcome:
            raise ValueError("Arena replay outcome does not match its result.")

        if self.status == "completed":
            if (
                not isinstance(self.outcome, OfficialVictoryOutcome)
                or self.winner_player_id != self.outcome.winner_player_id
            ):
                raise ValueError("Completed matches require their official winner.")
            if self.failure_classification is not None:
                raise ValueError("Completed matches cannot have failure categories.")
            if self.failure_exception_type is not None or self.failure_message is not None:
                raise ValueError("Completed matches cannot contain failure details.")
            winning_seats = {
                seat.player_id for seat in self.seat_results if seat.is_winner
            }
            if winning_seats != {self.winner_player_id}:
                raise ValueError("Exactly the official winner must be marked as winner.")
        elif self.status == "truncated":
            if (
                not isinstance(self.outcome, TrainingTruncationOutcome)
                or self.outcome.limit_type != "action_count"
                or self.outcome.limit_value != self.max_semantic_decisions
                or self.semantic_decisions != self.max_semantic_decisions
            ):
                raise ValueError("Truncated matches must reach their action budget.")
            if self.winner_player_id is not None:
                raise ValueError("Truncated matches cannot declare a winner.")
            if (
                self.failure_classification is not None
                or self.failure_exception_type is not None
                or self.failure_message is not None
            ):
                raise ValueError("Truncated matches cannot contain failure details.")
        else:
            if (
                not isinstance(self.outcome, OperationalInterruptionOutcome)
                or self.failure_classification is None
                or self.outcome.cause_code != self.failure_classification
            ):
                raise ValueError("Failed matches require a classified interruption.")
            if (
                self.failure_exception_type is None
                or not self.failure_exception_type.strip()
                or self.failure_message is None
                or not self.failure_message.strip()
            ):
                raise ValueError("Failed matches require explicit error details.")
            if self.winner_player_id is not None:
                raise ValueError("Failed matches cannot declare an official winner.")
        return self


class ArenaStatistics(ContractModel):
    matches: NonNegativeInt
    completed_matches: NonNegativeInt
    truncated_matches: NonNegativeInt
    failed_matches: NonNegativeInt
    seat_appearances: NonNegativeInt
    completed_seat_appearances: NonNegativeInt
    wins: NonNegativeInt
    win_rate: float | None = Field(ge=0.0, le=1.0)
    completion_rate: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_statistics(self) -> ArenaStatistics:
        if self.matches != (
            self.completed_matches + self.truncated_matches + self.failed_matches
        ):
            raise ValueError("Match status counts must sum to the match total.")
        if self.completed_seat_appearances > self.seat_appearances:
            raise ValueError("Completed seat appearances exceed total appearances.")
        if self.wins > self.completed_seat_appearances:
            raise ValueError("Wins exceed completed seat appearances.")
        expected_completion_rate = (
            self.completed_matches / self.matches if self.matches else 0.0
        )
        if self.completion_rate != expected_completion_rate:
            raise ValueError("Completion rate does not match the match counts.")
        expected_win_rate = (
            self.wins / self.completed_seat_appearances
            if self.completed_seat_appearances
            else None
        )
        if self.win_rate != expected_win_rate:
            raise ValueError("Win rate does not match the completed seat counts.")
        return self


class ArenaReport(ContractModel):
    contract_type: Literal["arena_report"] = "arena_report"
    schema_version: Literal["1.0"] = _ARENA_SCHEMA_VERSION
    matches: tuple[ArenaMatchResult, ...]
    statistics_by_policy_id: dict[str, ArenaStatistics]
    statistics_by_mission_id: dict[str, ArenaStatistics]
    statistics_by_table_size: dict[str, ArenaStatistics]

    @model_validator(mode="after")
    def validate_report(self) -> ArenaReport:
        match_ids = tuple(match.match_id for match in self.matches)
        if len(set(match_ids)) != len(match_ids):
            raise ValueError("Arena reports cannot contain duplicate match IDs.")
        return self


@dataclass(slots=True)
class _StatisticsCounter:
    matches: int = 0
    completed_matches: int = 0
    truncated_matches: int = 0
    failed_matches: int = 0
    seat_appearances: int = 0
    completed_seat_appearances: int = 0
    wins: int = 0

    def add(
        self,
        match: ArenaMatchResult,
        seat_results: Sequence[ArenaSeatResult],
    ) -> None:
        self.matches += 1
        if match.status == "completed":
            self.completed_matches += 1
            self.completed_seat_appearances += len(seat_results)
            self.wins += sum(seat.is_winner for seat in seat_results)
        elif match.status == "truncated":
            self.truncated_matches += 1
        else:
            self.failed_matches += 1
        self.seat_appearances += len(seat_results)

    def to_model(self) -> ArenaStatistics:
        return ArenaStatistics(
            matches=self.matches,
            completed_matches=self.completed_matches,
            truncated_matches=self.truncated_matches,
            failed_matches=self.failed_matches,
            seat_appearances=self.seat_appearances,
            completed_seat_appearances=self.completed_seat_appearances,
            wins=self.wins,
            win_rate=(
                self.wins / self.completed_seat_appearances
                if self.completed_seat_appearances
                else None
            ),
            completion_rate=(
                self.completed_matches / self.matches if self.matches else 0.0
            ),
        )


@dataclass(frozen=True, slots=True)
class _MatchFailure:
    classification: ArenaFailureClass
    exception_type: str
    message: str


def run_arena(
    configs: Sequence[ArenaMatchConfig],
    policy_registry: Mapping[str, FrozenPolicySpec],
) -> ArenaReport:
    """Execute a campaign after validating every policy assignment."""
    if not configs:
        raise ValueError("An arena campaign must contain at least one match.")
    match_ids = tuple(config.match_id for config in configs)
    if len(set(match_ids)) != len(match_ids):
        raise ValueError("Arena match IDs must be unique within a campaign.")
    for config in configs:
        _resolve_policy_specs(config, policy_registry)
    results = tuple(
        run_arena_match(config, policy_registry)
        for config in configs
    )
    return aggregate_arena_results(results)


def run_arena_match(
    config: ArenaMatchConfig,
    policy_registry: Mapping[str, FrozenPolicySpec],
) -> ArenaMatchResult:
    """Run one game with independent policy instances and RNGs per seat."""
    policy_specs = _resolve_policy_specs(config, policy_registry)
    started_at_ns = perf_counter_ns()
    game_id = f"arena:{config.match_id}"
    env = War1AECEnv(num_players=config.num_players)
    env.reset(seed=config.seed, options={"game_id": game_id})

    factory_seeds_by_player, rng_seeds_by_player = _derive_seat_seeds(config)
    policy_rngs = {
        player_id: np.random.default_rng(rng_seed)
        for player_id, rng_seed in rng_seeds_by_player.items()
    }
    policies_by_player: dict[str, ArenaPolicy] = {}
    policy_instance_ids: set[int] = set()
    max_policy_decision_duration_ms = 0.0
    semantic_decisions = 0

    def fail(
        player_id: str | None,
        failure: _MatchFailure,
    ) -> ArenaMatchResult:
        return _failed_match_result(
            env=env,
            config=config,
            policy_specs=policy_specs,
            factory_seeds_by_player=factory_seeds_by_player,
            rng_seeds_by_player=rng_seeds_by_player,
            semantic_decisions=semantic_decisions,
            max_policy_decision_duration_ms=max_policy_decision_duration_ms,
            started_at_ns=started_at_ns,
            player_id=player_id,
            failure=failure,
        )

    for seat_player_id in env.possible_agents:
        policy_spec = policy_specs[seat_player_id]
        try:
            candidate: object = policy_spec.factory(
                factory_seeds_by_player[seat_player_id]
            )
        except Exception as error:
            return fail(
                seat_player_id,
                _failure_from_exception("policy_factory_error", error),
            )
        if not _is_arena_policy(candidate):
            return fail(
                seat_player_id,
                _MatchFailure(
                    classification="policy_factory_error",
                    exception_type="TypeError",
                    message=(
                        f"Factory for policy {policy_spec.policy_id!r} did not "
                        "return a callable policy."
                    ),
                ),
            )
        if id(candidate) in policy_instance_ids:
            return fail(
                seat_player_id,
                _MatchFailure(
                    classification="policy_factory_error",
                    exception_type="ValueError",
                    message=(
                        f"Factory for policy {policy_spec.policy_id!r} reused a "
                        "policy instance across seats."
                    ),
                ),
            )
        policy_instance_ids.add(id(candidate))
        policies_by_player[seat_player_id] = candidate

    while True:
        state = env._require_state()
        if state.is_finished:
            winner_player_id = state.winner_player_id
            if winner_player_id is None:
                raise RuntimeError("A finished arena game has no official winner.")
            outcome: ReplayOutcome = OfficialVictoryOutcome(
                outcome_type="official_victory",
                winner_player_id=winner_player_id,
            )
            return _build_match_result(
                env=env,
                config=config,
                policy_specs=policy_specs,
                factory_seeds_by_player=factory_seeds_by_player,
                rng_seeds_by_player=rng_seeds_by_player,
                semantic_decisions=semantic_decisions,
                max_policy_decision_duration_ms=max_policy_decision_duration_ms,
                started_at_ns=started_at_ns,
                status="completed",
                outcome=outcome,
                failure=None,
            )
        if semantic_decisions >= config.max_semantic_decisions:
            outcome = TrainingTruncationOutcome(
                outcome_type="training_truncation",
                limit_type="action_count",
                limit_value=config.max_semantic_decisions,
            )
            return _build_match_result(
                env=env,
                config=config,
                policy_specs=policy_specs,
                factory_seeds_by_player=factory_seeds_by_player,
                rng_seeds_by_player=rng_seeds_by_player,
                semantic_decisions=semantic_decisions,
                max_policy_decision_duration_ms=max_policy_decision_duration_ms,
                started_at_ns=started_at_ns,
                status="truncated",
                outcome=outcome,
                failure=None,
            )

        player_id = env.agent_selection
        if player_id is None:
            return fail(
                None,
                _MatchFailure(
                    classification="engine_error",
                    exception_type="RuntimeError",
                    message="A live arena game has no selected player.",
                ),
            )
        if env.terminations.get(player_id, False) or env.truncations.get(
            player_id, False
        ):
            try:
                env.step(None)
            except Exception as error:
                return fail(
                    player_id,
                    _failure_from_exception("engine_error", error),
                )
            continue

        try:
            observation = env.observe(player_id)
        except Exception as error:
            return fail(
                player_id,
                _failure_from_exception("engine_error", error),
            )

        decision_started_at_ns = perf_counter_ns()
        try:
            action = policies_by_player[player_id](
                observation,
                policy_rngs[player_id],
            )
        except Exception as error:
            max_policy_decision_duration_ms = max(
                max_policy_decision_duration_ms,
                (perf_counter_ns() - decision_started_at_ns) / 1_000_000,
            )
            return fail(
                player_id,
                _failure_from_exception("policy_error", error),
            )
        decision_duration_ns = perf_counter_ns() - decision_started_at_ns
        max_policy_decision_duration_ms = max(
            max_policy_decision_duration_ms,
            decision_duration_ns / 1_000_000,
        )
        if (
            config.decision_budget_ms is not None
            and decision_duration_ns > config.decision_budget_ms * 1_000_000
        ):
            duration_ms = decision_duration_ns / 1_000_000
            return fail(
                player_id,
                _MatchFailure(
                    classification="decision_budget_exceeded",
                    exception_type="TimeoutError",
                    message=(
                        f"Policy decision took {duration_ms:.3f} ms; configured "
                        f"budget is {config.decision_budget_ms} ms."
                    ),
                ),
            )

        try:
            action_index = _validate_action_index(env, player_id, observation, action)
        except (TypeError, ValueError) as error:
            return fail(
                player_id,
                _failure_from_exception("illegal_action", error),
            )
        except Exception as error:
            return fail(
                player_id,
                _failure_from_exception("engine_error", error),
            )

        try:
            env.step(action_index)
        except Exception as error:
            return fail(
                player_id,
                _failure_from_exception("engine_error", error),
            )
        semantic_decisions += 1


def aggregate_arena_results(
    match_results: Sequence[ArenaMatchResult],
) -> ArenaReport:
    """Aggregate match results by policy, effective mission, and table size."""
    if not match_results:
        raise ValueError("At least one arena match result is required.")
    match_ids = tuple(result.match_id for result in match_results)
    if len(set(match_ids)) != len(match_ids):
        raise ValueError("Arena match IDs must be unique when aggregating results.")

    return ArenaReport(
        matches=tuple(match_results),
        statistics_by_policy_id=_aggregate_seat_group(
            match_results,
            lambda seat: seat.policy_id,
        ),
        statistics_by_mission_id=_aggregate_seat_group(
            match_results,
            lambda seat: seat.mission_id,
        ),
        statistics_by_table_size=_aggregate_table_sizes(match_results),
    )


def _resolve_policy_specs(
    config: ArenaMatchConfig,
    policy_registry: Mapping[str, FrozenPolicySpec],
) -> dict[str, FrozenPolicySpec]:
    result: dict[str, FrozenPolicySpec] = {}
    for player_id, policy_id in config.policy_ids_by_player.items():
        policy_spec = policy_registry.get(policy_id)
        if policy_spec is None:
            raise ValueError(
                f"No frozen policy spec is registered for {policy_id!r}."
            )
        if policy_spec.policy_id != policy_id:
            raise ValueError(
                f"Registry key {policy_id!r} does not match policy spec "
                f"{policy_spec.policy_id!r}."
            )
        result[player_id] = policy_spec
    return result


def _derive_seat_seeds(
    config: ArenaMatchConfig,
) -> tuple[dict[str, int], dict[str, int]]:
    factory_seeds: dict[str, int] = {}
    rng_seeds: dict[str, int] = {}
    seat_sequences = np.random.SeedSequence(config.seed).spawn(
        config.num_players
    )
    for index, seat_sequence in enumerate(seat_sequences):
        player_id = f"player_{index}"
        factory_sequence, rng_sequence = seat_sequence.spawn(2)
        factory_seeds[player_id] = int(
            factory_sequence.generate_state(1, dtype=np.uint32)[0]
        )
        rng_seeds[player_id] = int(
            rng_sequence.generate_state(1, dtype=np.uint32)[0]
        )
    return factory_seeds, rng_seeds


def _is_arena_policy(candidate: object) -> TypeGuard[ArenaPolicy]:
    return callable(candidate)


def _validate_action_index(
    env: War1AECEnv,
    player_id: str,
    observation: War1AECObservation,
    action: int | np.integer[Any],
) -> int:
    if isinstance(action, (bool, np.bool_)) or not isinstance(
        action, (int, np.integer)
    ):
        raise TypeError("Policies must return integer legal-action indices.")
    action_index = int(action)
    mask = observation["action_mask"]
    if (
        not 0 <= action_index < mask.size
        or mask[action_index] != 1
        or not env.action_space(player_id).contains(action_index)
    ):
        raise ValueError(
            f"Policy returned illegal action index {action_index} for "
            f"{player_id!r}."
        )
    return action_index


def _failure_from_exception(
    classification: ArenaFailureClass,
    error: Exception,
) -> _MatchFailure:
    message = str(error) or type(error).__name__
    return _MatchFailure(
        classification=classification,
        exception_type=type(error).__name__,
        message=message,
    )


def _failed_match_result(
    *,
    env: War1AECEnv,
    config: ArenaMatchConfig,
    policy_specs: Mapping[str, FrozenPolicySpec],
    factory_seeds_by_player: Mapping[str, int],
    rng_seeds_by_player: Mapping[str, int],
    semantic_decisions: int,
    max_policy_decision_duration_ms: float,
    started_at_ns: int,
    player_id: str | None,
    failure: _MatchFailure,
) -> ArenaMatchResult:
    _LOGGER.error(
        "Arena match %s failed for %s (%s): %s",
        config.match_id,
        player_id or "unknown seat",
        failure.classification,
        failure.message,
    )
    outcome = OperationalInterruptionOutcome(
        outcome_type="operational_interruption",
        cause_code=failure.classification,
    )
    return _build_match_result(
        env=env,
        config=config,
        policy_specs=policy_specs,
        factory_seeds_by_player=factory_seeds_by_player,
        rng_seeds_by_player=rng_seeds_by_player,
        semantic_decisions=semantic_decisions,
        max_policy_decision_duration_ms=max_policy_decision_duration_ms,
        started_at_ns=started_at_ns,
        status="failed",
        outcome=outcome,
        failure=failure,
    )


def _build_match_result(
    *,
    env: War1AECEnv,
    config: ArenaMatchConfig,
    policy_specs: Mapping[str, FrozenPolicySpec],
    factory_seeds_by_player: Mapping[str, int],
    rng_seeds_by_player: Mapping[str, int],
    semantic_decisions: int,
    max_policy_decision_duration_ms: float,
    started_at_ns: int,
    status: ArenaMatchStatus,
    outcome: ReplayOutcome,
    failure: _MatchFailure | None,
) -> ArenaMatchResult:
    state = env._require_state()
    policy_ids_by_player = dict(config.policy_ids_by_player)
    checkpoint_digests = {
        player_id: policy_specs[player_id].checkpoint_sha256
        for player_id in env.possible_agents
    }
    players_by_id = {
        player.public.player_id: player
        for player in state.players
    }
    seat_results = tuple(
        ArenaSeatResult(
            player_id=player_id,
            policy_id=policy_ids_by_player[player_id],
            checkpoint_sha256=checkpoint_digests[player_id],
            mission_id=effective_objective_id(players_by_id[player_id]),
            is_eliminated=players_by_id[player_id].public.is_eliminated,
            is_winner=state.winner_player_id == player_id,
        )
        for player_id in env.possible_agents
    )
    replay = env.create_full_replay(
        f"{config.match_id}:full-replay",
        outcome=outcome,
    )
    return ArenaMatchResult(
        match_id=config.match_id,
        game_id=state.game_id,
        status=status,
        seed=config.seed,
        num_players=config.num_players,
        policy_ids_by_player=policy_ids_by_player,
        checkpoint_sha256_by_player=checkpoint_digests,
        policy_factory_seeds_by_player=dict(factory_seeds_by_player),
        policy_rng_seeds_by_player=dict(rng_seeds_by_player),
        max_semantic_decisions=config.max_semantic_decisions,
        decision_budget_ms=config.decision_budget_ms,
        max_policy_decision_duration_ms=max_policy_decision_duration_ms,
        semantic_decisions=semantic_decisions,
        elapsed_ms=(perf_counter_ns() - started_at_ns) // 1_000_000,
        winner_player_id=state.winner_player_id,
        seat_results=seat_results,
        outcome=outcome,
        failure_classification=(
            None if failure is None else failure.classification
        ),
        failure_exception_type=None if failure is None else failure.exception_type,
        failure_message=None if failure is None else failure.message,
        replay=replay,
    )


def _aggregate_seat_group(
    matches: Sequence[ArenaMatchResult],
    group_key: Callable[[ArenaSeatResult], str],
) -> dict[str, ArenaStatistics]:
    counters: dict[str, _StatisticsCounter] = {}
    for match in matches:
        seats_by_key: dict[str, list[ArenaSeatResult]] = {}
        for seat in match.seat_results:
            seats_by_key.setdefault(group_key(seat), []).append(seat)
        for key, seats in seats_by_key.items():
            counters.setdefault(key, _StatisticsCounter()).add(match, seats)
    return {
        key: counters[key].to_model()
        for key in sorted(counters)
    }


def _aggregate_table_sizes(
    matches: Sequence[ArenaMatchResult],
) -> dict[str, ArenaStatistics]:
    counters: dict[str, _StatisticsCounter] = {}
    for match in matches:
        key = f"{match.num_players}_players"
        counters.setdefault(key, _StatisticsCounter()).add(
            match,
            match.seat_results,
        )
    return {
        key: counters[key].to_model()
        for key in sorted(counters)
    }


__all__ = [
    "ArenaFailureClass",
    "ArenaMatchConfig",
    "ArenaMatchResult",
    "ArenaMatchStatus",
    "ArenaPolicy",
    "ArenaReport",
    "ArenaSeatResult",
    "ArenaStatistics",
    "FrozenPolicySpec",
    "PolicyFactory",
    "aggregate_arena_results",
    "baseline_policy_spec",
    "run_arena",
    "run_arena_match",
]
