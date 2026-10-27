from __future__ import annotations

import pytest

pytest.importorskip("war_game_engine")

from war_game_engine.contracts import (
    PlayerObservation,
    PlayerPrivateState,
    PublicGameState,
)
from war_game_engine.setup import SetupPlayer, prepare_game

from war_ai.rewards import (
    EPISODIC_GAMMA,
    MAX_POTENTIAL,
    PotentialRewardShaper,
    official_win_reward,
)

PLAYERS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)


def _observation_with_objective(
    observation: PlayerObservation,
    objective_id: str,
    *,
    fallback_id: str | None = None,
) -> PlayerObservation:
    private_state = PlayerPrivateState(
        player_id=observation.player_id,
        objective_id=objective_id,
        objective_fallback_id=fallback_id,
        hand_card_ids=observation.private_state.hand_card_ids,
    )
    data = observation.model_dump(mode="python")
    data["private_state"] = private_state
    data["legal_actions"] = ()
    return PlayerObservation.model_validate(data)


def _finished_observation(
    observation: PlayerObservation,
) -> PlayerObservation:
    public_data = observation.public_state.model_dump(mode="python")
    public_data.update(
        {
            "phase_id": "finished",
            "active_player_id": None,
            "turn_player_id": None,
            "decision_player_id": None,
            "winner_player_id": observation.player_id,
            "winner_objective_id": observation.private_state.objective_id,
            "winner_objective_fallback_id": (
                observation.private_state.objective_fallback_id
            ),
            "is_finished": True,
        }
    )
    terminal_public_state = PublicGameState.model_validate(public_data)
    data = observation.model_dump(mode="python")
    data["public_state"] = terminal_public_state
    data["legal_actions"] = ()
    return PlayerObservation.model_validate(data)


def test_official_reward_is_only_the_winner_indicator() -> None:
    assert official_win_reward("player_0", "player_0") == 1.0
    assert official_win_reward("player_0", "player_1") == 0.0
    assert official_win_reward("player_0", None) == 0.0
    assert EPISODIC_GAMMA == 1.0
    with pytest.raises(ValueError, match="must not be blank"):
        official_win_reward("", None)


def test_potential_is_bounded_and_uses_only_public_state_and_own_objective() -> None:
    state = prepare_game("reward-test", PLAYERS, seed=23).state
    observation = state.players[0]
    from war_game_engine.state import observe_player

    player_observation = observe_player(state, observation.public.player_id)
    shaper = PotentialRewardShaper()
    objective_progress = (
        _observation_with_objective(player_observation, "W1-OBJ-01"),
        _observation_with_objective(player_observation, "W1-OBJ-07"),
        _observation_with_objective(player_observation, "W1-OBJ-13"),
    )

    potentials = tuple(shaper.potential(item) for item in objective_progress)

    assert all(0.0 <= value <= MAX_POTENTIAL for value in potentials)
    fallback = _observation_with_objective(
        player_observation,
        "W1-OBJ-01",
        fallback_id="W1-OBJ-13",
    )
    assert shaper.potential(fallback) == shaper.potential(
        objective_progress[2]
    )


def test_terminal_potential_is_zero_and_shaping_telescopes() -> None:
    state = prepare_game("terminal-reward-test", PLAYERS, seed=31).state
    from war_game_engine.state import observe_player

    observation = observe_player(state, "player_0")
    shaper = PotentialRewardShaper()
    starting_potential = shaper.potential(observation)
    terminal = _finished_observation(observation)

    assert shaper.potential(terminal) == 0.0
    assert shaper.transition_reward(
        observation,
        observation,
        terminated=False,
    ) == 0.0
    assert shaper.transition_reward(
        observation,
        terminal,
        terminated=True,
    ) == pytest.approx(-starting_potential)
    assert shaper.transition_reward(
        observation,
        None,
        terminated=True,
    ) == pytest.approx(-starting_potential)
    with pytest.raises(ValueError, match="requires a next observation"):
        shaper.transition_reward(
            observation,
            None,
            terminated=False,
        )
