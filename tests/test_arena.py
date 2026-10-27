from __future__ import annotations

import time
from typing import Any

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    ChanceOutcome,
    FullGameState,
    OfficialVictoryOutcome,
    OperationalInterruptionOutcome,
    TrainingTruncationOutcome,
)
from war_game_engine.engine import (
    apply_action,
    apply_combat_outcome,
    simulate_pending_combat_outcome,
)
from war_game_engine.objectives import effective_objective_id
from war_game_engine.randomness import RandomStreams
from war_game_engine.replay import (
    create_full_replay,
    create_snapshot,
    replay_full_replay,
)
from war_game_engine.serialization import canonical_sha256
from war_game_engine.setup import SetupPlayer, prepare_game

from war_ai.arena import (
    ArenaMatchConfig,
    ArenaMatchResult,
    ArenaPolicy,
    ArenaSeatResult,
    FrozenPolicySpec,
    aggregate_arena_results,
    baseline_policy_spec,
    run_arena,
    run_arena_match,
)
from war_ai.envs import War1AECEnv, War1AECObservation

PLAYERS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)


def _match_config(
    match_id: str,
    *,
    max_semantic_decisions: int = 3,
    decision_budget_ms: int | None = None,
) -> ArenaMatchConfig:
    return ArenaMatchConfig(
        match_id=match_id,
        seed=17,
        num_players=3,
        policy_ids_by_player={
            "player_0": "fixed-policy",
            "player_1": "fixed-policy",
            "player_2": "fixed-policy",
        },
        max_semantic_decisions=max_semantic_decisions,
        decision_budget_ms=decision_budget_ms,
    )


def _first_legal_action(
    _observation: War1AECObservation,
    _rng: np.random.Generator,
) -> int:
    return 0


def _replay_result(match: ArenaMatchResult) -> FullGameState:
    def replay_action(
        state: FullGameState,
        action: Action,
        _streams: RandomStreams,
    ) -> FullGameState:
        return apply_action(state, action, board=WAR1_BOARD_SPEC)

    def replay_chance(
        state: FullGameState,
        outcome: ChanceOutcome,
        streams: RandomStreams,
    ) -> FullGameState:
        pending = state.pending_combat
        if pending is None:
            raise AssertionError("A replayed chance event requires pending combat.")
        expected_outcome, _ = simulate_pending_combat_outcome(state)
        if expected_outcome != outcome:
            raise AssertionError("Replay combat outcome differs from the seeded roll.")
        for _ in range(pending.attacker_dice_count + pending.defender_dice_count):
            streams.die("combat")
        return apply_combat_outcome(
            state,
            outcome,
            random_streams_state=streams.snapshot(),
            board=WAR1_BOARD_SPEC,
        )

    execution = replay_full_replay(
        match.replay,
        apply_action=replay_action,
        apply_chance_outcome=replay_chance,
    )
    assert canonical_sha256(execution.state) == match.replay.final_state_hash
    assert execution.outcome == match.outcome
    assert isinstance(execution.state, FullGameState)
    assert execution.random_streams is not None
    assert execution.state.random_streams_state == execution.random_streams.snapshot()
    return execution.state


def _known_completed_match() -> ArenaMatchResult:
    initial_state = prepare_game("known-arena-win", PLAYERS, seed=29).state
    finished_data = initial_state.model_dump(mode="python")
    finished_data.update(
        {
            "phase_id": "finished",
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "winner_player_id": "player_0",
            "is_finished": True,
        }
    )
    finished_state = FullGameState.model_validate(finished_data)
    snapshot = create_snapshot(
        "known-arena-win:initial",
        finished_state,
        visibility="full",
    )
    outcome = OfficialVictoryOutcome(
        outcome_type="official_victory",
        winner_player_id="player_0",
    )
    replay = create_full_replay(
        "known-arena-win:full-replay",
        snapshot,
        (),
        finished_state,
        outcome=outcome,
    )
    policy_ids = {
        "player_0": "winner-policy",
        "player_1": "opponent-policy",
        "player_2": "opponent-policy",
    }
    return ArenaMatchResult(
        match_id="known-arena-win",
        game_id=finished_state.game_id,
        status="completed",
        seed=29,
        num_players=3,
        policy_ids_by_player=policy_ids,
        checkpoint_sha256_by_player={
            "player_0": f"sha256:{'a' * 64}",
            "player_1": None,
            "player_2": None,
        },
        policy_factory_seeds_by_player={
            "player_0": 101,
            "player_1": 102,
            "player_2": 103,
        },
        policy_rng_seeds_by_player={
            "player_0": 201,
            "player_1": 202,
            "player_2": 203,
        },
        max_semantic_decisions=5,
        decision_budget_ms=None,
        max_policy_decision_duration_ms=0.0,
        semantic_decisions=0,
        elapsed_ms=0,
        winner_player_id="player_0",
        seat_results=tuple(
            ArenaSeatResult(
                player_id=player.public.player_id,
                policy_id=policy_ids[player.public.player_id],
                checkpoint_sha256=(
                    f"sha256:{'a' * 64}"
                    if player.public.player_id == "player_0"
                    else None
                ),
                mission_id=effective_objective_id(player),
                is_eliminated=player.public.is_eliminated,
                is_winner=player.public.player_id == "player_0",
            )
            for player in finished_state.players
        ),
        outcome=outcome,
        failure_classification=None,
        failure_exception_type=None,
        failure_message=None,
        replay=replay,
    )


def test_arena_records_seat_seeds_and_replayable_truncation() -> None:
    factory_seeds: list[int] = []

    def factory(seed: int) -> ArenaPolicy:
        factory_seeds.append(seed)

        def policy(
            observation: War1AECObservation,
            rng: np.random.Generator,
        ) -> int:
            return _first_legal_action(observation, rng)

        return policy

    policy = FrozenPolicySpec(
        policy_id="fixed-policy",
        factory=factory,
        checkpoint_sha256=f"sha256:{'b' * 64}",
    )
    report = run_arena(
        (_match_config("truncated-match"),),
        {"fixed-policy": policy},
    )

    match = report.matches[0]
    assert match.status == "truncated"
    assert match.semantic_decisions == 3
    assert match.winner_player_id is None
    assert isinstance(match.outcome, TrainingTruncationOutcome)
    assert match.replay.initial_snapshot.visibility == "full"
    assert [entry.sequence_number for entry in match.replay.entries] == [1, 2, 3]
    assert factory_seeds == list(match.policy_factory_seeds_by_player.values())
    assert len(set(factory_seeds)) == 3
    assert len(set(match.policy_rng_seeds_by_player.values())) == 3
    assert all(
        seat.checkpoint_sha256 == f"sha256:{'b' * 64}"
        for seat in match.seat_results
    )
    _replay_result(match)

    policy_stats = report.statistics_by_policy_id["fixed-policy"]
    assert policy_stats.matches == 1
    assert policy_stats.truncated_matches == 1
    assert policy_stats.seat_appearances == 3
    assert policy_stats.completed_seat_appearances == 0
    assert policy_stats.win_rate is None
    assert report.statistics_by_table_size["3_players"].seat_appearances == 3
    assert sum(
        stats.seat_appearances
        for stats in report.statistics_by_mission_id.values()
    ) == 3


def test_arena_classifies_illegal_policy_actions_without_fallback() -> None:
    def illegal_policy(
        observation: War1AECObservation,
        _rng: np.random.Generator,
    ) -> int:
        return int(observation["action_mask"].size)

    def factory(_seed: int) -> ArenaPolicy:
        def policy(
            observation: War1AECObservation,
            rng: np.random.Generator,
        ) -> int:
            return illegal_policy(observation, rng)

        return policy

    config = _match_config("illegal-action-match", max_semantic_decisions=2)
    result = run_arena_match(
        config,
        {
            "fixed-policy": FrozenPolicySpec(
                policy_id="fixed-policy",
                factory=factory,
            )
        },
    )

    assert result.status == "failed"
    assert result.failure_classification == "illegal_action"
    assert result.failure_exception_type == "ValueError"
    assert result.failure_message is not None
    assert isinstance(result.outcome, OperationalInterruptionOutcome)
    assert result.outcome.cause_code == "illegal_action"
    assert result.replay.entries == ()


def test_arena_rejects_a_policy_instance_shared_between_seats() -> None:
    baseline_config = _match_config(
        "baseline-instance-match",
        max_semantic_decisions=2,
    ).model_copy(
        update={
            "policy_ids_by_player": {
                player_id: "random_legal"
                for player_id in ("player_0", "player_1", "player_2")
            }
        }
    )
    result = run_arena_match(
        baseline_config,
        {"random_legal": baseline_policy_spec("random_legal")},
    )
    assert result.status == "truncated"

    config = _match_config("shared-instance-failure", max_semantic_decisions=2)
    shared_policy = _first_legal_action
    result = run_arena_match(
        config,
        {
            "fixed-policy": FrozenPolicySpec(
                policy_id="fixed-policy",
                factory=lambda _seed: shared_policy,
            )
        },
    )

    assert result.status == "failed"
    assert result.failure_classification == "policy_factory_error"
    assert result.failure_message is not None
    assert "reused a policy instance" in result.failure_message
    assert result.replay.entries == ()


def test_arena_classifies_policy_decision_budget_overrun() -> None:
    def slow_policy(
        _observation: War1AECObservation,
        _rng: np.random.Generator,
    ) -> int:
        time.sleep(0.01)
        return 0

    def factory(_seed: int) -> ArenaPolicy:
        def policy(
            observation: War1AECObservation,
            rng: np.random.Generator,
        ) -> int:
            return slow_policy(observation, rng)

        return policy

    result = run_arena_match(
        _match_config(
            "slow-policy-match",
            max_semantic_decisions=2,
            decision_budget_ms=1,
        ),
        {
            "fixed-policy": FrozenPolicySpec(
                policy_id="fixed-policy",
                factory=factory,
            )
        },
    )

    assert result.status == "failed"
    assert result.failure_classification == "decision_budget_exceeded"
    assert result.max_policy_decision_duration_ms > 1.0
    assert isinstance(result.outcome, OperationalInterruptionOutcome)


def test_arena_statistics_use_known_official_winner_and_seat_counts() -> None:
    report = aggregate_arena_results((_known_completed_match(),))

    winner_stats = report.statistics_by_policy_id["winner-policy"]
    assert winner_stats.matches == 1
    assert winner_stats.completed_matches == 1
    assert winner_stats.seat_appearances == 1
    assert winner_stats.wins == 1
    assert winner_stats.win_rate == 1.0
    assert winner_stats.completion_rate == 1.0

    opponent_stats = report.statistics_by_policy_id["opponent-policy"]
    assert opponent_stats.matches == 1
    assert opponent_stats.seat_appearances == 2
    assert opponent_stats.wins == 0
    assert opponent_stats.win_rate == 0.0

    table_stats = report.statistics_by_table_size["3_players"]
    assert table_stats.completed_matches == 1
    assert table_stats.completed_seat_appearances == 3
    assert table_stats.wins == 1
    assert table_stats.win_rate == pytest.approx(1 / 3)


def test_arena_runner_records_official_victory_and_replay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import war_ai.arena as arena_module

    class ImmediateVictoryEnv(War1AECEnv):
        def reset(
            self,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
        ) -> None:
            super().reset(seed=seed, options=options)
            initial_state = self._require_state()
            finished_data = initial_state.model_dump(mode="python")
            finished_data.update(
                {
                    "phase_id": "finished",
                    "active_player_id": None,
                    "turn_player_id": None,
                    "decision_player_id": None,
                    "winner_player_id": "player_0",
                    "is_finished": True,
                }
            )
            finished_state = FullGameState.model_validate(finished_data)
            self._game_state = finished_state
            self._initial_snapshot = create_snapshot(
                f"{finished_state.game_id}:initial",
                finished_state,
                visibility="full",
            )
            self._replay_entries = []
            self.agent_selection = None

    def factory(_seed: int) -> ArenaPolicy:
        def policy(
            _observation: War1AECObservation,
            _rng: np.random.Generator,
        ) -> int:
            return 0

        return policy

    monkeypatch.setattr(arena_module, "War1AECEnv", ImmediateVictoryEnv)
    result = run_arena_match(
        _match_config("official-victory-match", max_semantic_decisions=1),
        {
            "fixed-policy": FrozenPolicySpec(
                policy_id="fixed-policy",
                factory=factory,
            )
        },
    )

    assert result.status == "completed"
    assert result.winner_player_id == "player_0"
    assert isinstance(result.outcome, OfficialVictoryOutcome)
    assert result.replay.entries == ()
    assert result.seat_results[0].is_winner
    _replay_result(result)
