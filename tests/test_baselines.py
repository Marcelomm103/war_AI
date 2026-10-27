from __future__ import annotations

from fractions import Fraction

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.combat import enumerate_combat_loss_distribution
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    FullGameState,
    FullPlayerState,
    PlayerPrivateState,
    PublicPlayerState,
    TerritoryState,
)
from war_game_engine.engine import legal_actions
from war_game_engine.setup import SetupPlayer, prepare_game
from war_game_engine.state import observe_player

from war_ai.baselines import (
    BASELINE_POLICIES,
    create_baseline,
    exact_attack_estimate,
    exact_probability_planner,
    expansionist,
    objective_oriented,
    random_legal,
)
from war_ai.envs import War1AECEnv, War1GymEnv, War1NumericObservation
from war_ai.envs.actions import StructuredActionCodec
from war_ai.envs.observations import War1ObservationEncoder
from war_ai.envs.war1_aec import War1AECObservation

PLAYERS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)


def _attack_observation() -> tuple[War1AECObservation, tuple[Action, ...]]:
    state = prepare_game("baseline-combat", PLAYERS, seed=3).state
    adjacency = WAR1_BOARD_SPEC.map.adjacencies[0]
    origin_id = adjacency.first_territory_id
    target_id = adjacency.second_territory_id
    owners = {origin_id: "player_0", target_id: "player_1"}
    for territory in state.territories:
        if territory.territory_id in owners:
            continue
        owners[territory.territory_id] = (
            "player_0"
            if sum(owner == "player_0" for owner in owners.values()) < 23
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
        "player_0": "W1-OBJ-01",
        "player_1": "W1-OBJ-14",
        "player_2": "W1-OBJ-13",
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
            "is_finished": False,
            "winner_player_id": None,
        }
    )
    attack_state = FullGameState.model_validate(state_data)
    actions = legal_actions(attack_state)
    player_observation = observe_player(
        attack_state,
        "player_0",
        legal_actions=actions,
    )
    token_values, token_offsets = StructuredActionCodec().encode_table(actions)
    numeric_observation: War1NumericObservation = (
        War1ObservationEncoder(3).encode_player(player_observation)
    )
    return (
        {
            "observation": numeric_observation,
            "action_mask": np.ones(len(actions), dtype=np.int8),
            "action_token_mask": StructuredActionCodec().valid_next_token_mask(
                (),
                actions,
            ),
            "engine_observation": player_observation,
            "legal_action_token_values": token_values,
            "legal_action_token_offsets": token_offsets,
        },
        actions,
    )


def test_every_baseline_selects_only_from_the_legal_action_mask() -> None:
    env = War1AECEnv()
    env.reset(seed=47)
    agent = env.agent_selection
    assert agent is not None
    observation = env.observe(agent)
    assert observation["engine_observation"] is not None

    for policy in BASELINE_POLICIES.values():
        action_index = policy(observation, np.random.default_rng(9))
        assert 0 <= action_index < len(
            observation["engine_observation"].legal_actions
        )
        assert observation["action_mask"][action_index] == 1


def test_random_legal_baseline_uses_its_seeded_policy_stream() -> None:
    env = War1AECEnv()
    env.reset(seed=19)
    agent = env.agent_selection
    assert agent is not None
    observation = env.observe(agent)

    first = random_legal(observation, np.random.default_rng(123))
    repeated = random_legal(observation, np.random.default_rng(123))

    assert first == repeated
    assert observation["action_mask"][first] == 1


def test_exact_attack_estimate_matches_engine_distribution() -> None:
    observation, actions = _attack_observation()
    player_observation = observation["engine_observation"]
    assert player_observation is not None
    operation = next(
        action.operation
        for action in actions
        if isinstance(action.operation, AttackOperation)
    )
    assert isinstance(operation, AttackOperation)
    estimate = exact_attack_estimate(player_observation, operation)
    target_armies = next(
        territory.army_count
        for territory in player_observation.public_state.territories
        if territory.territory_id == operation.target_territory_id
    )
    defender_dice_count = min(target_armies, 3)
    distribution = enumerate_combat_loss_distribution(1, defender_dice_count)

    assert estimate.conquest_probability == sum(
        (
            probability
            for (_, defender_losses), probability in distribution.items()
            if defender_losses >= target_armies
        ),
        Fraction(),
    )
    assert estimate.expected_attacker_losses == sum(
        (
            attacker_losses * probability
            for (attacker_losses, _), probability in distribution.items()
        ),
        Fraction(),
    )
    assert estimate.expected_defender_losses == sum(
        (
            defender_losses * probability
            for (_, defender_losses), probability in distribution.items()
        ),
        Fraction(),
    )


def test_baselines_rank_legal_exact_probability_attack_choices() -> None:
    observation, actions = _attack_observation()
    player_observation = observation["engine_observation"]
    assert player_observation is not None
    action_index = exact_probability_planner(
        observation,
        np.random.default_rng(5),
    )
    action = actions[action_index]

    assert observation["action_mask"][action_index] == 1
    assert isinstance(action.operation, AttackOperation)


def test_baseline_factory_requires_an_explicit_registered_policy() -> None:
    for policy_id, policy in BASELINE_POLICIES.items():
        assert create_baseline(policy_id) is policy

    with pytest.raises(ValueError, match="Unknown WAR 1 baseline"):
        create_baseline("random-for-final-homologation")


def test_baselines_integrate_as_explicit_gymnasium_opponents() -> None:
    env = War1GymEnv(
        focal_agent="player_0",
        opponent_policies={
            "player_1": expansionist,
            "player_2": objective_oriented,
        },
        max_semantic_decisions=60,
    )
    observation, _ = env.reset(seed=43)

    next_observation, reward, terminated, truncated, info = env.step(0)

    assert np.any(observation["action_mask"])
    assert np.all(
        np.logical_or(
            next_observation["action_mask"] == 0,
            next_observation["action_mask"] == 1,
        )
    )
    assert isinstance(reward, float)
    assert info["semantic_duration"] >= 1
    assert not (terminated and truncated)
