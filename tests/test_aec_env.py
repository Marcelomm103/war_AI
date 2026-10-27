from __future__ import annotations

import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from pettingzoo.test import api_test, seed_test  # type: ignore[import-untyped]
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    ChanceOutcome,
    FullGameState,
    FullPlayerState,
    PlayerPrivateState,
    PublicPlayerState,
    TerritoryState,
    TradeCardsOperation,
    TrainingTruncationOutcome,
)
from war_game_engine.engine import (
    apply_action,
    apply_combat_outcome,
    simulate_pending_combat_outcome,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.replay import create_snapshot, replay_full_replay
from war_game_engine.setup import SetupPlayer, prepare_game
from war_game_engine.versioning import ACTION_SCHEMA_VERSION

from war_ai.envs import War1AECEnv
from war_ai.envs.actions import ACTION_CONFIRM_TOKEN, ACTION_TOKEN_VOCAB_SIZE
from war_ai.envs.observations import decode_integer_records, encode_integer_records

PLAYERS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)


def _combat_state(
    *,
    attacker_territory_count: int,
    attacker_objective_id: str,
) -> FullGameState:
    state = prepare_game("aec-test", PLAYERS, seed=1).state
    adjacency = WAR1_BOARD_SPEC.map.adjacencies[0]
    origin_id = adjacency.first_territory_id
    target_id = adjacency.second_territory_id
    owners = {origin_id: "player_0", target_id: "player_1"}
    for territory in state.territories:
        if territory.territory_id in owners:
            continue
        owners[territory.territory_id] = (
            "player_0"
            if sum(owner == "player_0" for owner in owners.values())
            < attacker_territory_count
            else "player_2"
        )

    territories = tuple(
        TerritoryState(
            territory_id=territory.territory_id,
            owner_player_id=owners[territory.territory_id],
            army_count=4 if territory.territory_id == origin_id else 1,
        )
        for territory in state.territories
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
        "player_0": attacker_objective_id,
        "player_1": (
            "W1-OBJ-14" if attacker_objective_id == "W1-OBJ-13" else "W1-OBJ-13"
        ),
        "player_2": "W1-OBJ-03",
    }
    players = tuple(
        FullPlayerState(
            public=PublicPlayerState(
                player_id=player.public.player_id,
                color=player.public.color,
                territory_count=territory_counts[player.public.player_id],
                army_count=army_counts[player.public.player_id],
                is_eliminated=False,
            ),
            private=PlayerPrivateState(
                player_id=player.public.player_id,
                objective_id=objective_ids[player.public.player_id],
                hand_card_ids=player.private.hand_card_ids,
            ),
        )
        for player in state.players
    )
    state_data = state.model_dump(mode="python")
    state_data.update(
        {
            "phase_id": "attack",
            "turn_number": 2,
            "active_player_id": "player_0",
            "turn_player_id": "player_0",
            "decision_player_id": "player_0",
            "players": players,
            "territories": territories,
            "reinforcement_pools": (),
            "has_conquered_territory_this_turn": False,
            "is_finished": False,
            "winner_player_id": None,
        }
    )
    return FullGameState.model_validate(state_data)


def _env_with_combat_state(
    *,
    attacker_territory_count: int,
    attacker_objective_id: str,
) -> War1AECEnv:
    env = War1AECEnv()
    env.reset(seed=1)
    env._game_state = _combat_state(
        attacker_territory_count=attacker_territory_count,
        attacker_objective_id=attacker_objective_id,
    )
    env._initial_snapshot = create_snapshot(
        "aec-combat-initial",
        env._game_state,
        visibility="full",
    )
    env._replay_entries = []
    env.agent_selection = "player_0"
    env._legal_actions_cache_state = None
    env._legal_actions_cache = ()
    return env


def _attack_index(env: War1AECEnv) -> int:
    observation = env.observe("player_0")["engine_observation"]
    assert observation is not None
    return next(
        index
        for index, action in enumerate(observation.legal_actions)
        if isinstance(action.operation, AttackOperation)
    )


def test_full_replay_records_and_reexecutes_combat_chance_outcomes() -> None:
    env = _env_with_combat_state(
        attacker_territory_count=4,
        attacker_objective_id="W1-OBJ-14",
    )
    env.step(_attack_index(env))
    replay = env.create_full_replay(
        "aec-combat-replay",
        outcome=TrainingTruncationOutcome(
            outcome_type="training_truncation",
            limit_type="action_count",
            limit_value=1,
        ),
    )

    assert tuple(entry.entry_type for entry in replay.entries) == (
        "action",
        "chance_outcome",
    )
    assert tuple(entry.sequence_number for entry in replay.entries) == (1, 2)

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
        assert pending is not None
        expected_outcome, expected_random_state = simulate_pending_combat_outcome(
            state
        )
        for _ in range(pending.attacker_dice_count + pending.defender_dice_count):
            streams.die("combat")
        assert outcome == expected_outcome
        assert streams.snapshot() == expected_random_state
        return apply_combat_outcome(
            state,
            outcome,
            random_streams_state=streams.snapshot(),
            board=WAR1_BOARD_SPEC,
        )

    execution = replay_full_replay(
        replay,
        apply_action=replay_action,
        apply_chance_outcome=replay_chance,
    )
    assert execution.state == env._require_state()
    assert execution.outcome == replay.outcome


def test_observation_and_mask_are_player_scoped() -> None:
    env = War1AECEnv()
    env.reset(seed=23)
    agent = env.agent_selection
    assert agent is not None

    observation = env.observe(agent)
    player_observation = observation["engine_observation"]
    assert player_observation is not None
    assert player_observation.player_id == agent
    assert player_observation.legal_actions
    numeric = observation["observation"]
    assert numeric["adjacency"].shape == (42, 42)
    assert np.array_equal(numeric["adjacency"], numeric["adjacency"].T)
    assert numeric["territory_owner_mask"].shape == (42, 7)
    assert np.all(numeric["territory_owner_mask"].sum(axis=1) == 1)
    assert np.array_equal(numeric["player_mask"][:3], np.ones(3, dtype=np.int8))
    assert np.array_equal(numeric["player_mask"][3:], np.zeros(3, dtype=np.int8))
    assert numeric["phase_one_hot"].dtype == np.int8
    assert np.count_nonzero(
        numeric["phase_one_hot"]
    ) == 1
    assert np.array_equal(
        decode_integer_records(
            numeric["reinforcement_pool_bytes"],
            numeric["reinforcement_pool_offsets"],
        ),
        tuple(
            pool.remaining_armies
            for pool in player_observation.public_state.reinforcement_pools
        ),
    )
    territory_by_id = {
        territory.territory_id: territory
        for territory in player_observation.public_state.territories
    }
    assert decode_integer_records(
        numeric["territory_troop_bytes"],
        numeric["territory_troop_offsets"],
    ) == tuple(
        territory_by_id[territory.territory_id].army_count
        for territory in WAR1_BOARD_SPEC.map.territories
    )
    assert int(numeric["own_objective_index"][0]) == next(
        index
        for index, objective in enumerate(WAR1_BOARD_SPEC.objectives)
        if objective.objective_id == player_observation.private_state.objective_id
    )
    assert observation["action_mask"].dtype == np.int8
    assert observation["action_mask"].shape == (
        len(player_observation.legal_actions),
    )
    assert np.all(observation["action_mask"] == 1)
    assert observation["legal_action_token_values"].size > 0
    assert observation["legal_action_token_offsets"].size == (
        len(player_observation.legal_actions) + 1
    )
    assert observation["action_token_mask"].shape == (ACTION_TOKEN_VOCAB_SIZE,)
    assert env.action_codec.decode_table(
        observation["legal_action_token_values"],
        observation["legal_action_token_offsets"],
        player_observation.legal_actions,
    ) == player_observation.legal_actions
    assert env.observation_space(agent).contains(observation)
    assert all(
        not hasattr(player, "private")
        for player in player_observation.public_state.players
    )

    other_agent = next(item for item in env.agents if item != agent)
    other_observation = env.observe(other_agent)
    assert other_observation["engine_observation"] is not None
    assert other_observation["engine_observation"].legal_actions == ()
    assert np.array_equal(
        other_observation["action_mask"],
        np.zeros(1, dtype=np.int8),
    )
    assert env.action_space(other_agent).n == 1


def test_invalid_action_index_does_not_advance_the_game() -> None:
    env = War1AECEnv()
    env.reset(seed=7)
    agent = env.agent_selection
    assert agent is not None
    state = env._game_state
    rewards = env.rewards.copy()

    with pytest.raises(ValueError, match="currently legal actions"):
        env.step(env.action_space(agent).n)

    assert env._game_state is state
    assert env.agent_selection == agent
    assert env.rewards == rewards


def test_unbounded_integer_encoding_is_lossless() -> None:
    values = (0, 1, 255, 256, (1 << 256) + 123)

    encoded, offsets = encode_integer_records(values)

    assert decode_integer_records(encoded, offsets) == values


def test_observation_player_table_supports_three_to_six_seats() -> None:
    for player_count in (3, 6):
        env = War1AECEnv(num_players=player_count)
        env.reset(seed=player_count)
        agent = env.agent_selection
        assert agent is not None
        numeric = env.observe(agent)["observation"]

        assert numeric["player_mask"].shape == (6,)
        assert int(numeric["player_mask"].sum()) == player_count
        assert numeric["territory_owner_mask"].shape == (42, 7)
        assert env.observation_space(agent).contains(env.observe(agent))


def test_structured_action_is_applied_only_after_confirmation() -> None:
    env = War1AECEnv()
    env.reset(seed=19)
    agent = env.agent_selection
    assert agent is not None
    observation = env.observe(agent)
    player_observation = observation["engine_observation"]
    assert player_observation is not None
    action = player_observation.legal_actions[0]
    tokens = env.action_codec.encode_action(action)
    state = env._game_state

    next_token_mask = env.action_prefix_mask(agent, tokens[:-1])
    assert next_token_mask[ACTION_CONFIRM_TOKEN] == 1
    with pytest.raises(ValueError, match="confirmation"):
        env.step_tokens(tokens[:-1])
    assert env._game_state is state

    env.step_tokens(tokens)

    assert env._game_state is not state


def test_public_card_action_history_does_not_reveal_card_ids() -> None:
    env = War1AECEnv()
    env.reset(seed=23)
    state = env._game_state
    assert state is not None
    actor = state.decision_player_id
    assert actor is not None
    private_card_ids = ("private-card-a", "private-card-b", "private-card-c")
    action = Action(
        contract_type="action",
        action_id="trade-history-test",
        actor_player_id=actor,
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=state.state_version,
        operation=TradeCardsOperation(
            action_type="trade_cards",
            card_ids=private_card_ids,
        ),
    )

    event = env._action_event(
        state,
        action,
        state,
        sequence_number=0,
    )

    assert event.event_type == "trade_cards"
    assert event.data["cards_traded"] == 3
    serialized = json.dumps(event.data, sort_keys=True)
    assert all(card_id not in serialized for card_id in private_card_ids)


def test_eliminated_agent_is_terminated_and_cleaned_up_out_of_turn() -> None:
    env = _env_with_combat_state(
        attacker_territory_count=23,
        attacker_objective_id="W1-OBJ-14",
    )
    off_turn_observation = env.observe("player_1")
    assert off_turn_observation["engine_observation"] is not None
    assert off_turn_observation["engine_observation"].legal_actions == ()
    assert not np.any(off_turn_observation["action_mask"])

    env.step(_attack_index(env))

    assert env._game_state is not None
    assert env._game_state.phase_id == "occupation"
    assert env._game_state.players[1].public.is_eliminated
    assert env.terminations["player_1"]
    history_observation = env.observe("player_0")["engine_observation"]
    assert history_observation is not None
    assert tuple(event.event_type for event in history_observation.recent_events) == (
        "attack",
        "combat_outcome",
    )
    assert tuple(
        event.sequence_number for event in history_observation.recent_events
    ) == (0, 1)
    encoded_history = env.observe("player_0")["observation"]
    history_offsets = encoded_history["public_history_offsets"]
    assert history_offsets.size == len(history_observation.recent_events) + 1
    for index, event in enumerate(history_observation.recent_events):
        start = int(history_offsets[index])
        end = int(history_offsets[index + 1])
        assert json.loads(
            encoded_history["public_history_bytes"][start:end].tobytes()
        ) == (
            event.model_dump(mode="json")
        )
    assert env.agent_selection == "player_1"
    observation, reward, terminated, truncated, _ = env.last()
    assert observation is not None
    assert observation["engine_observation"] is None
    assert reward == 0.0
    assert terminated
    assert not truncated

    env.step(None)

    assert "player_1" not in env.agents
    assert "player_1" not in env.terminations
    assert env.agent_selection == "player_0"
    assert env._game_state.decision_player_id == "player_0"


def test_official_victory_terminates_all_agents_with_winner_reward() -> None:
    env = _env_with_combat_state(
        attacker_territory_count=24,
        attacker_objective_id="W1-OBJ-13",
    )

    env.step(_attack_index(env))

    assert env._game_state is not None
    assert env._game_state.is_finished
    assert env._game_state.winner_player_id == "player_0"
    assert env.rewards == {
        "player_0": 1.0,
        "player_1": 0.0,
        "player_2": 0.0,
    }
    assert all(env.terminations.values())
    observation, reward, terminated, truncated, _ = env.last()
    assert observation is not None
    assert reward == 1.0
    assert terminated
    assert not truncated

    while env.agents:
        env.step(None)
    assert env.agent_selection is None


def test_environment_passes_official_pettingzoo_api_test() -> None:
    api_test(War1AECEnv(), num_cycles=40)


def test_environment_passes_official_pettingzoo_seed_test() -> None:
    seed_test(War1AECEnv, num_cycles=40)
