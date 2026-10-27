from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    FullGameState,
    FullPlayerState,
    OfficialVictoryOutcome,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    ReinforcementPool,
    ReplayActionEntry,
    TerritoryState,
)
from war_game_engine.engine import apply_action, legal_actions
from war_game_engine.objectives import effective_objective_id
from war_game_engine.replay import create_full_replay, create_snapshot
from war_game_engine.setup import SetupPlayer, prepare_game

from war_ai.arena import (
    ArenaMatchConfig,
    ArenaMatchResult,
    ArenaSeatResult,
    baseline_policy_spec,
    run_arena_match,
)
from war_ai.envs import War1AECEnv
from war_ai.rl.curriculum import (
    CurriculumScheduler,
    FullGameReplayReservoir,
    build_default_curriculum,
    make_curriculum_aec_env,
    make_curriculum_gym_env,
    make_stage_opponent_policies,
    resolve_tactical_task,
)
from war_ai.rl.small_tasks import build_intermediate_scenario_state

_PLAYERS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)


def _one_action_complete_match() -> ArenaMatchResult:
    initial = prepare_game(
        "curriculum-replay-test",
        _PLAYERS,
        seed=37,
    ).state
    territory_ids = tuple(
        territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories
    )
    actor_ids = set(territory_ids[:18])
    opponent_ids = set(territory_ids[18:30])
    target_id = territory_ids[17]
    owner_by_id = {
        territory_id: (
            "player_0"
            if territory_id in actor_ids
            else ("player_1" if territory_id in opponent_ids else "player_2")
        )
        for territory_id in territory_ids
    }
    territories = tuple(
        TerritoryState(
            territory_id=territory_id,
            owner_player_id=owner_by_id[territory_id],
            army_count=(
                1
                if territory_id == target_id or owner_by_id[territory_id] != "player_0"
                else 2
            ),
        )
        for territory_id in territory_ids
    )
    territory_counts = {
        player_id: sum(
            territory.owner_player_id == player_id for territory in territories
        )
        for player_id in ("player_0", "player_1", "player_2")
    }
    army_counts = {
        player_id: sum(
            territory.army_count
            for territory in territories
            if territory.owner_player_id == player_id
        )
        for player_id in ("player_0", "player_1", "player_2")
    }
    objective_ids = {
        "player_0": "W1-OBJ-14",
        "player_1": "W1-OBJ-13",
        "player_2": "W1-OBJ-12",
    }
    players = tuple(
        FullPlayerState(
            public=PublicPlayerState(
                player_id=player.player_id,
                color=player.color,
                territory_count=territory_counts[player.player_id],
                army_count=army_counts[player.player_id],
                is_eliminated=False,
            ),
            private=PlayerPrivateState(
                player_id=player.player_id,
                objective_id=objective_ids[player.player_id],
                hand_card_ids=(),
            ),
        )
        for player in _PLAYERS
    )
    state_data = initial.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "initial_placement",
            "turn_number": 1,
            "active_player_id": "player_0",
            "turn_player_id": "player_0",
            "decision_player_id": "player_0",
            "initial_placement_order": ("player_0", "player_1", "player_2"),
            "players": players,
            "territories": territories,
            "reinforcement_pools": tuple(
                ReinforcementPool(
                    pool_id=f"{player_id}:initial",
                    player_id=player_id,
                    origin="territory_count",
                    remaining_armies=1,
                )
                for player_id in ("player_0", "player_1", "player_2")
            ),
            "winner_player_id": None,
            "is_finished": False,
        }
    )
    initial_state = FullGameState.model_validate(state_data)
    action = next(
        candidate
        for candidate in legal_actions(initial_state)
        if isinstance(candidate.operation, PlaceArmiesOperation)
        and candidate.operation.placements[0].territory_id == target_id
    )
    final_state = apply_action(initial_state, action)
    assert final_state.is_finished
    assert final_state.winner_player_id == "player_0"

    match_id = "curriculum-replay-test"
    outcome = OfficialVictoryOutcome(
        outcome_type="official_victory",
        winner_player_id="player_0",
    )
    initial_snapshot = create_snapshot(
        f"{match_id}:initial",
        initial_state,
        visibility="full",
    )
    replay = create_full_replay(
        f"{match_id}:full-replay",
        initial_snapshot,
        (
            ReplayActionEntry(
                entry_type="action",
                sequence_number=1,
                action=action,
            ),
        ),
        final_state,
        outcome=outcome,
    )
    policy_ids = {
        player_id: f"policy-{player_id}" for player_id in ("player_0", "player_1", "player_2")
    }
    return ArenaMatchResult(
        match_id=match_id,
        game_id=final_state.game_id,
        status="completed",
        seed=37,
        num_players=3,
        policy_ids_by_player=policy_ids,
        checkpoint_sha256_by_player=dict.fromkeys(policy_ids),
        policy_factory_seeds_by_player={
            player_id: 100 + index
            for index, player_id in enumerate(policy_ids)
        },
        policy_rng_seeds_by_player={
            player_id: 200 + index
            for index, player_id in enumerate(policy_ids)
        },
        max_semantic_decisions=1,
        decision_budget_ms=None,
        max_policy_decision_duration_ms=0.0,
        semantic_decisions=1,
        elapsed_ms=0,
        winner_player_id="player_0",
        seat_results=tuple(
            ArenaSeatResult(
                player_id=player.public.player_id,
                policy_id=policy_ids[player.public.player_id],
                checkpoint_sha256=None,
                mission_id=effective_objective_id(player),
                is_eliminated=player.public.is_eliminated,
                is_winner=player.public.player_id == "player_0",
            )
            for player in final_state.players
        ),
        outcome=outcome,
        failure_classification=None,
        failure_exception_type=None,
        failure_message=None,
        replay=replay,
    )


def test_default_curriculum_increases_complexity_and_excludes_training_scenarios() -> None:
    plan = build_default_curriculum()
    scheduler = CurriculumScheduler(plan)

    assert tuple(stage.scenario_kind for stage in plan.stages) == (
        "tactical",
        "intermediate",
        "full_game",
    )
    assert tuple(stage.complexity for stage in plan.stages) == (0, 1, 2)
    assert plan.homologation_scenario_ids == ("full:war1_classic",)
    with pytest.raises(ValueError, match="training-only"):
        plan.require_homologation_scenario(plan.stages[0].scenario_ids[0])

    rng = np.random.default_rng(42)
    tactical_scenario = scheduler.choose_scenario(rng)
    assert tactical_scenario in plan.stages[0].scenario_ids
    assert resolve_tactical_task(tactical_scenario).task_id == (
        tactical_scenario.removeprefix("tactical:")
    )
    assert not scheduler.record_evaluation(0.99, episodes=63)
    assert not scheduler.record_evaluation(0.94, episodes=64)
    assert scheduler.record_evaluation(0.95, episodes=64)
    assert scheduler.current_stage.stage_id == "valid-intermediate-states"
    assert scheduler.record_evaluation(0.6, episodes=128)
    assert scheduler.is_complete
    assert not scheduler.record_evaluation(1.0, episodes=1000)


def test_opponent_pool_adds_strategies_and_assigns_each_non_focal_seat() -> None:
    plan = build_default_curriculum()
    intermediate, full_game = plan.stages[1:]
    assert set(intermediate.opponent_policy_ids).issubset(
        full_game.opponent_policy_ids
    )
    assert "exact_probability_planner" in full_game.opponent_policy_ids

    opponents = make_stage_opponent_policies(
        intermediate,
        num_players=3,
        focal_player_id="player_0",
        opponent_offset=1,
    )

    assert set(opponents) == {"player_1", "player_2"}
    assert all(callable(policy) for policy in opponents.values())


def test_intermediate_states_start_with_real_legal_actions() -> None:
    plan = build_default_curriculum()
    stage = plan.stages[1]

    for scenario_id in stage.scenario_ids:
        env = make_curriculum_aec_env(stage, scenario_id)
        env.reset(
            seed=19,
            options={"game_id": f"curriculum-test-{scenario_id}"},
        )
        agent = env.agent_selection
        assert agent == "player_0"
        observation = env.observe(agent)
        engine_observation = observation["engine_observation"]
        assert engine_observation is not None
        assert engine_observation.legal_actions
        assert observation["action_mask"].size == len(
            engine_observation.legal_actions
        )


def test_aec_rejects_an_intermediate_state_with_a_different_game_id() -> None:
    env = War1AECEnv(
        initial_state_factory=lambda game_id, seed: build_intermediate_scenario_state(
            "attack",
            game_id=f"{game_id}-wrong",
            seed=seed,
        )
    )

    with pytest.raises(ValueError, match="game ID"):
        env.reset(seed=19, options={"game_id": "expected-curriculum-game"})


def test_gym_curriculum_uses_intermediate_state_and_stage_adversaries() -> None:
    stage = build_default_curriculum().stages[1]
    env = make_curriculum_gym_env(
        stage,
        stage.scenario_ids[0],
        opponent_offset=1,
    )
    observation, info = env.reset(
        seed=11,
        options={"game_id": "curriculum-gym-reset"},
    )

    assert info["game_id"] == "curriculum-gym-reset"
    assert observation["engine_observation"] is not None
    assert observation["engine_observation"].legal_actions
    env.close()


def test_full_game_reservoir_keeps_only_complete_arena_wins() -> None:
    plan = build_default_curriculum()
    reservoir = FullGameReplayReservoir(capacity=1, seed=7)
    completed_match = _one_action_complete_match()

    assert reservoir.add(completed_match, stage=plan.stages[-1])
    assert len(reservoir) == 1
    assert reservoir.seen_count == 1
    assert reservoir.sample(1) == (completed_match,)

    with pytest.raises(ValueError, match="Simplified curriculum scenarios"):
        reservoir.add(completed_match, stage=plan.stages[1])

    truncated = run_arena_match(
        ArenaMatchConfig(
            match_id="curriculum-truncated",
            seed=23,
            num_players=3,
            policy_ids_by_player={
                "player_0": "random_legal",
                "player_1": "random_legal",
                "player_2": "random_legal",
            },
            max_semantic_decisions=1,
        ),
        {"random_legal": baseline_policy_spec("random_legal")},
    )
    assert truncated.status == "truncated"
    with pytest.raises(ValueError, match="completed official victories"):
        reservoir.add(truncated, stage=plan.stages[-1])
