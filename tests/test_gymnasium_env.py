from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from gymnasium.utils.env_checker import check_env
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    AttackOperation,
    EndPhaseOperation,
    FullGameState,
    FullPlayerState,
    PlaceArmiesOperation,
    PlayerPrivateState,
    PublicPlayerState,
    TerritoryState,
)
from war_game_engine.setup import SetupPlayer, prepare_game

from war_ai.envs import WAR1_RL_OBSERVATION_ENCODING_VERSION, War1AECEnv, War1GymEnv
from war_ai.envs.actions import ACTION_CONFIRM_TOKEN
from war_ai.envs.gymnasium_env import OpponentPolicy
from war_ai.envs.war1_aec import War1AECObservation
from war_ai.rewards import PotentialRewardShaper


def _first_legal(
    observation: War1AECObservation,
    rng: np.random.Generator,
) -> int:
    del observation, rng
    return 0


def _gym_env(
    *,
    focal_agent: str = "player_0",
    max_semantic_decisions: int | None = None,
    policy: OpponentPolicy = _first_legal,
    reward_shaper: PotentialRewardShaper | None = None,
) -> War1GymEnv:
    aec_env = War1AECEnv()
    opponent_policies = {
        agent: policy
        for agent in aec_env.possible_agents
        if agent != focal_agent
    }
    return War1GymEnv(
        aec_env=aec_env,
        focal_agent=focal_agent,
        opponent_policies=opponent_policies,
        max_semantic_decisions=max_semantic_decisions,
        reward_shaper=reward_shaper,
    )


def _turn_end_fixture() -> tuple[FullGameState, str, str]:
    players = (
        SetupPlayer(player_id="player_0", color="white"),
        SetupPlayer(player_id="player_1", color="red"),
        SetupPlayer(player_id="player_2", color="black"),
    )
    state = prepare_game("focal-elimination", players, seed=1).state
    adjacency = WAR1_BOARD_SPEC.map.adjacencies[0]
    origin_id = adjacency.first_territory_id
    target_id = adjacency.second_territory_id
    owners = {origin_id: "player_1", target_id: "player_0"}
    for territory in state.territories:
        if territory.territory_id in owners:
            continue
        owners[territory.territory_id] = (
            "player_1"
            if sum(owner == "player_1" for owner in owners.values()) < 23
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
        "player_0": "W1-OBJ-07",
        "player_1": "W1-OBJ-14",
        "player_2": "W1-OBJ-13",
    }
    full_players = tuple(
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
            "phase_id": "turn_end",
            "turn_number": 2,
            "active_player_id": "player_0",
            "turn_player_id": "player_0",
            "decision_player_id": "player_0",
            "players": full_players,
            "territories": territories,
            "reinforcement_pools": (),
            "has_conquered_territory_this_turn": False,
            "is_finished": False,
            "winner_player_id": None,
        }
    )
    return FullGameState.model_validate(state_data), origin_id, target_id


class TurnEndFixtureAECEnv(War1AECEnv):
    def reset(
        self,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        super().reset(seed=seed, options=options)
        self._game_state, _, _ = _turn_end_fixture()
        self.agent_selection = "player_0"
        self._legal_actions_cache_state = None
        self._legal_actions_cache = ()
        for agent in self.agents:
            self.action_space(agent)


def _attack_focal_policy(
    origin_id: str,
    target_id: str,
) -> OpponentPolicy:
    def policy(
        observation: War1AECObservation,
        rng: np.random.Generator,
    ) -> int:
        del rng
        player_observation = observation["engine_observation"]
        assert player_observation is not None
        if player_observation.public_state.phase_id == "reinforcement":
            return next(
                index
                for index, action in enumerate(player_observation.legal_actions)
                if isinstance(action.operation, EndPhaseOperation)
            )
        if player_observation.public_state.phase_id == "reinforcement_placement":
            return next(
                index
                for index, action in enumerate(player_observation.legal_actions)
                if isinstance(action.operation, PlaceArmiesOperation)
            )
        if player_observation.public_state.phase_id == "attack":
            candidates = [
                (action.operation.dice_count, index)
                for index, action in enumerate(player_observation.legal_actions)
                if isinstance(action.operation, AttackOperation)
                and action.operation.origin_territory_id == origin_id
                and action.operation.target_territory_id == target_id
            ]
            return max(candidates)[1]
        raise AssertionError(
            f"Unexpected opponent phase {player_observation.public_state.phase_id!r}."
        )

    return policy


def test_reset_reaches_focal_decision_reproducibly() -> None:
    first = _gym_env(focal_agent="player_2")
    second = _gym_env(focal_agent="player_2")

    observation, info = first.reset(seed=51)
    repeated_observation, repeated_info = second.reset(seed=51)

    assert first._aec_env.agent_selection == "player_2"
    assert second._aec_env.agent_selection == "player_2"
    assert info["game_id"] == repeated_info["game_id"]
    assert (
        info["observation_encoding_version"]
        == WAR1_RL_OBSERVATION_ENCODING_VERSION
    )
    assert info["initial_opponent_decisions"] == repeated_info[
        "initial_opponent_decisions"
    ]
    assert np.array_equal(
        observation["observation"]["adjacency"],
        repeated_observation["observation"]["adjacency"],
    )
    assert np.array_equal(
        observation["observation"]["territory_troop_bytes"],
        repeated_observation["observation"]["territory_troop_bytes"],
    )
    assert np.array_equal(
        observation["observation"]["phase_one_hot"],
        repeated_observation["observation"]["phase_one_hot"],
    )
    assert np.array_equal(
        observation["action_mask"],
        repeated_observation["action_mask"],
    )
    assert observation["engine_observation"] == repeated_observation[
        "engine_observation"
    ]
    assert (
        first._aec_env.action_space("player_2").n
        == observation["action_mask"].size
    )
    assert np.any(observation["action_mask"])


def test_step_advances_opponents_and_reports_semantic_duration() -> None:
    opponent_decisions: list[str] = []

    def tracking_policy(
        observation: War1AECObservation,
        rng: np.random.Generator,
    ) -> int:
        del rng
        player_observation = observation["engine_observation"]
        assert player_observation is not None
        opponent_decisions.append(player_observation.player_id)
        return 0

    env = _gym_env(
        focal_agent="player_2",
        policy=tracking_policy,
    )
    env.reset(seed=17)
    opponent_decisions.clear()
    before = env._semantic_decisions

    observation, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert env._aec_env.agent_selection == "player_2"
    assert info["semantic_duration"] == 1 + len(opponent_decisions)
    assert info["semantic_decisions_total"] - before == info["semantic_duration"]
    assert info["semantic_duration_unit"] == "live_aec_decisions"
    assert all(agent != "player_2" for agent in opponent_decisions)
    assert observation["engine_observation"] is not None
    assert observation["engine_observation"].player_id == "player_2"
    assert np.any(observation["action_mask"])


def test_gymnasium_structured_action_path() -> None:
    env = _gym_env()
    observation, _ = env.reset(seed=29)
    engine_observation = observation["engine_observation"]
    assert engine_observation is not None
    action = engine_observation.legal_actions[0]
    tokens = env.action_codec.encode_action(action)
    state = env._aec_env._game_state

    assert (
        env.action_prefix_mask(tokens[:-1])[ACTION_CONFIRM_TOKEN] == 1
    )
    next_observation, reward, terminated, truncated, _ = env.step_tokens(tokens)

    assert env._aec_env._game_state is not state
    assert not terminated
    assert not truncated
    assert reward == 0.0
    assert env.observation_space.contains(next_observation)


def test_empty_off_turn_mask_is_not_focal_termination() -> None:
    env = _gym_env()
    env.reset(seed=3)
    other_agent = next(
        agent for agent in env._aec_env.possible_agents if agent != env.focal_agent
    )

    off_turn_observation = env._aec_env.observe(other_agent)
    assert not np.any(off_turn_observation["action_mask"])
    assert env._episode_status() == (False, False)

    env._aec_env.terminations[env.focal_agent] = True
    assert env._episode_status() == (True, False)


def test_focal_elimination_terminates_its_episode_before_game_over() -> None:
    _, origin_id, target_id = _turn_end_fixture()
    aec_env = TurnEndFixtureAECEnv()
    env = War1GymEnv(
        aec_env=aec_env,
        focal_agent="player_0",
        opponent_policies={
            "player_1": _attack_focal_policy(origin_id, target_id),
            "player_2": _first_legal,
        },
    )
    env.reset(seed=1)
    assert aec_env._game_state is not None
    assert aec_env._game_state.phase_id == "turn_end"

    _, reward, terminated, truncated, info = env.step(0)

    assert terminated
    assert not truncated
    assert reward == 0.0
    assert info["semantic_duration"] > 1
    assert aec_env._game_state is not None
    assert not aec_env._game_state.is_finished
    assert aec_env._game_state.players[0].public.is_eliminated


def test_semantic_decision_budget_truncates_without_termination() -> None:
    env = _gym_env(max_semantic_decisions=1)
    observation, _ = env.reset(seed=13)

    next_observation, reward, terminated, truncated, info = env.step(0)

    assert not terminated
    assert truncated
    assert reward == 0.0
    assert info["semantic_duration"] == 1
    assert env.observation_space.contains(next_observation)
    with pytest.raises(RuntimeError, match="call reset"):
        env.step(0)
    assert np.any(observation["action_mask"])


def test_potential_shaping_is_optional_and_bounded_against_official_reward() -> None:
    baseline = _gym_env(focal_agent="player_2")
    shaped = _gym_env(
        focal_agent="player_2",
        reward_shaper=PotentialRewardShaper(),
    )
    baseline.reset(seed=71)
    shaped.reset(seed=71)

    _, baseline_reward, baseline_terminated, baseline_truncated, baseline_info = (
        baseline.step(0)
    )
    _, shaped_reward, shaped_terminated, shaped_truncated, shaped_info = shaped.step(0)

    assert baseline_terminated == shaped_terminated
    assert baseline_truncated == shaped_truncated
    assert baseline_info["official_reward"] == shaped_info["official_reward"]
    assert baseline_info["shaping_reward"] == 0.0
    assert baseline_reward == baseline_info["official_reward"]
    assert abs(shaped_info["shaping_reward"]) <= 0.05
    assert shaped_reward == (
        shaped_info["official_reward"] + shaped_info["shaping_reward"]
    )
    assert baseline._aec_env._game_state == shaped._aec_env._game_state


def test_environment_passes_gymnasium_checker() -> None:
    check_env(_gym_env(), skip_render_check=True)
