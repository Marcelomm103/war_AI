"""Single-player Gymnasium view over the sequential WAR AEC environment."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from typing import Any, TypeAlias

import numpy as np
from gymnasium import Env
from numpy.typing import NDArray
from war_game_engine.contracts import PlayerObservation

from ..rewards import PotentialRewardShaper, official_win_reward
from .actions import ACTION_TOKEN_ENCODING_VERSION, StructuredActionCodec
from .observations import WAR1_RL_OBSERVATION_ENCODING_VERSION
from .war1_aec import War1AECEnv, War1AECObservation

OpponentPolicy: TypeAlias = Callable[
    [War1AECObservation, np.random.Generator],
    int | np.integer[Any],
]


class War1GymEnv(Env[object, np.int64]):
    """Expose one player's turns while configured opponent policies advance."""

    metadata: dict[str, Any] = {"name": "war1_focal_v0", "render_modes": []}

    def __init__(
        self,
        *,
        opponent_policies: Mapping[str, OpponentPolicy],
        focal_agent: str = "player_0",
        aec_env: War1AECEnv | None = None,
        max_semantic_decisions: int | None = None,
        reward_shaper: PotentialRewardShaper | None = None,
    ) -> None:
        super().__init__()
        self._aec_env = War1AECEnv() if aec_env is None else aec_env
        if focal_agent not in self._aec_env.possible_agents:
            raise ValueError(f"Unknown focal agent {focal_agent!r}.")

        expected_opponents = set(self._aec_env.possible_agents) - {focal_agent}
        assigned_opponents = set(opponent_policies)
        if assigned_opponents != expected_opponents:
            missing = sorted(expected_opponents - assigned_opponents)
            unexpected = sorted(assigned_opponents - expected_opponents)
            raise ValueError(
                "Assign exactly one opponent policy to each non-focal player; "
                f"missing={missing}, unexpected={unexpected}."
            )
        if any(not callable(policy) for policy in opponent_policies.values()):
            raise TypeError("Opponent policies must be callable.")
        if max_semantic_decisions is not None and (
            isinstance(max_semantic_decisions, bool)
            or not isinstance(max_semantic_decisions, int)
            or max_semantic_decisions < 1
        ):
            raise ValueError("max_semantic_decisions must be a positive integer.")

        self.focal_agent = focal_agent
        self.opponent_policies = dict(opponent_policies)
        self._opponent_agents = tuple(
            agent
            for agent in self._aec_env.possible_agents
            if agent != focal_agent
        )
        self.max_semantic_decisions = max_semantic_decisions
        self.reward_shaper = reward_shaper
        self.action_space = self._aec_env.action_spaces[focal_agent]
        self.observation_space = self._aec_env.observation_space(focal_agent)
        self._opponent_rngs: dict[str, np.random.Generator] = {}
        self._semantic_decisions = 0
        self._has_reset = False
        self._episode_finished = False

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[War1AECObservation, dict[str, Any]]:
        super().reset(seed=seed)
        game_seed = int(
            self.np_random.integers(0, 2**32, dtype=np.uint32)
        )
        self._aec_env.reset(seed=game_seed, options=options)
        self._semantic_decisions = 0
        self._episode_finished = False
        self._opponent_rngs = {
            agent: np.random.default_rng(
                int(self.np_random.integers(0, 2**32, dtype=np.uint32))
            )
            for agent in self._opponent_agents
        }

        initial_opponent_decisions, terminated, truncated = (
            self._advance_until_focal(count_toward_budget=False)
        )
        if terminated or truncated:
            raise RuntimeError(
                "The focal game ended before its first decision."
            )
        self._has_reset = True
        observation = self._focal_observation()
        state = self._aec_env._require_state()
        info = {
            "focal_agent": self.focal_agent,
            "game_id": state.game_id,
            "action_token_encoding_version": ACTION_TOKEN_ENCODING_VERSION,
            "initial_opponent_decisions": initial_opponent_decisions,
            "observation_encoding_version": (
                WAR1_RL_OBSERVATION_ENCODING_VERSION
            ),
            "semantic_duration_unit": "live_aec_decisions",
        }
        return observation, info

    def step(
        self,
        action: int | np.integer[Any],
    ) -> tuple[War1AECObservation, float, bool, bool, dict[str, Any]]:
        self._validate_open_step()
        focal_observation = self._focal_observation()
        previous_engine_observation = focal_observation["engine_observation"]
        if previous_engine_observation is None:
            raise RuntimeError("A live focal decision requires a player observation.")
        self._validate_action(self.focal_agent, focal_observation, action)
        self._aec_env.step(action)
        return self._finish_focal_step(previous_engine_observation)

    def step_tokens(
        self,
        tokens: Iterable[int],
    ) -> tuple[War1AECObservation, float, bool, bool, dict[str, Any]]:
        self._validate_open_step()
        previous_engine_observation = self._focal_observation()[
            "engine_observation"
        ]
        if previous_engine_observation is None:
            raise RuntimeError("A live focal decision requires a player observation.")
        self._aec_env.step_tokens(tokens)
        return self._finish_focal_step(previous_engine_observation)

    def action_prefix_mask(self, prefix: Iterable[int]) -> NDArray[np.int8]:
        return self._aec_env.action_prefix_mask(self.focal_agent, prefix)

    @property
    def action_codec(self) -> StructuredActionCodec:
        return self._aec_env.action_codec

    def _validate_open_step(self) -> None:
        if not self._has_reset:
            raise RuntimeError("Call reset() before stepping the Gymnasium env.")
        if self._episode_finished:
            raise RuntimeError("The episode ended; call reset() before stepping again.")
        if self._aec_env.agent_selection != self.focal_agent:
            raise RuntimeError(
                "The AEC environment is not at a focal-player decision."
            )

    def _finish_focal_step(
        self,
        previous_engine_observation: PlayerObservation,
    ) -> tuple[War1AECObservation, float, bool, bool, dict[str, Any]]:
        semantic_duration = 1
        self._semantic_decisions += 1
        terminated, truncated = self._episode_status()
        if not terminated and not truncated:
            opponent_duration, terminated, truncated = (
                self._advance_until_focal(count_toward_budget=True)
            )
            semantic_duration += opponent_duration

        observation = self._focal_observation()
        self._episode_finished = terminated or truncated
        state = self._aec_env._require_state()
        official_reward = official_win_reward(
            self.focal_agent,
            state.winner_player_id,
        )
        shaping_reward = (
            0.0
            if self.reward_shaper is None
            else self.reward_shaper.transition_reward(
                previous_engine_observation,
                observation["engine_observation"],
                terminated=terminated,
            )
        )
        reward = official_reward + shaping_reward
        info = {
            "focal_agent": self.focal_agent,
            "game_id": state.game_id,
            "action_token_encoding_version": ACTION_TOKEN_ENCODING_VERSION,
            "official_reward": official_reward,
            "observation_encoding_version": (
                WAR1_RL_OBSERVATION_ENCODING_VERSION
            ),
            "shaping_reward": shaping_reward,
            "semantic_decisions_total": self._semantic_decisions,
            "semantic_duration": semantic_duration,
            "semantic_duration_unit": "live_aec_decisions",
            "state_version": state.state_version,
        }
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        self._aec_env.close()

    def _advance_until_focal(
        self,
        *,
        count_toward_budget: bool,
    ) -> tuple[int, bool, bool]:
        semantic_duration = 0
        while True:
            terminated, truncated = self._episode_status()
            if terminated or truncated:
                return semantic_duration, terminated, truncated

            agent = self._aec_env.agent_selection
            if agent is None:
                raise RuntimeError("A live game has no selected AEC agent.")
            if agent == self.focal_agent:
                return semantic_duration, False, False
            if (
                self._aec_env.terminations.get(agent, False)
                or self._aec_env.truncations.get(agent, False)
            ):
                self._aec_env.step(None)
                continue

            observation = self._aec_env.observe(agent)
            policy = self.opponent_policies[agent]
            action = policy(observation, self._opponent_rngs[agent])
            self._validate_action(agent, observation, action)
            self._aec_env.step(action)
            semantic_duration += 1
            if count_toward_budget:
                self._semantic_decisions += 1

    def _episode_status(self) -> tuple[bool, bool]:
        terminations = self._aec_env.terminations
        if terminations.get(self.focal_agent, False) or (
            terminations and all(terminations.values())
        ):
            return True, False
        if self._aec_env.truncations.get(self.focal_agent, False):
            return False, True
        if (
            self.max_semantic_decisions is not None
            and self._semantic_decisions >= self.max_semantic_decisions
        ):
            return False, True
        return False, False

    def _focal_observation(self) -> War1AECObservation:
        self._aec_env.action_space(self.focal_agent)
        return self._aec_env.observe(self.focal_agent)

    def _validate_action(
        self,
        agent: str,
        observation: War1AECObservation,
        action: int | np.integer[Any],
    ) -> None:
        if isinstance(action, (bool, np.bool_)) or not isinstance(
            action, (int, np.integer)
        ):
            raise ValueError("Actions must be integer legal-action indices.")
        action_index = int(action)
        mask = observation["action_mask"]
        if (
            not self._aec_env.action_space(agent).contains(action)
            or not 0 <= action_index < mask.size
            or mask[action_index] != 1
        ):
            raise ValueError(
                f"Action index {action_index} is not legal for {agent!r}."
            )
