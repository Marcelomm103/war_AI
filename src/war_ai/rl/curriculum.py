"""Progressive training curriculum and full-game replay retention."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
from numpy.random import Generator
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import FullGameState, OfficialVictoryOutcome
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..arena import ArenaMatchResult
from ..baselines import BASELINE_POLICY_IDS, create_baseline
from ..envs.gymnasium_env import OpponentPolicy, War1GymEnv
from ..envs.war1_aec import InitialStateFactory, War1AECEnv
from .small_tasks import (
    INTERMEDIATE_SCENARIO_IDS,
    FiniteDecisionTask,
    build_intermediate_scenario_state,
    build_small_tasks,
)

CurriculumScenarioKind: TypeAlias = Literal[
    "tactical",
    "intermediate",
    "full_game",
]

_SCENARIO_KIND_RANK = {
    "tactical": 0,
    "intermediate": 1,
    "full_game": 2,
}
_TACTICAL_SCENARIO_IDS = (
    "tactical:legal_placement",
    "tactical:attack_or_stop",
    "tactical:occupation",
    "tactical:movement",
    "tactical:trades",
    "tactical:objective_completion",
)
_INTERMEDIATE_SCENARIOS = tuple(
    f"intermediate:{scenario_id}" for scenario_id in INTERMEDIATE_SCENARIO_IDS
)
_FULL_GAME_SCENARIOS = ("full:war1_classic",)


@dataclass(frozen=True)
class CurriculumStage:
    stage_id: str
    complexity: int
    scenario_kind: CurriculumScenarioKind
    scenario_ids: tuple[str, ...]
    opponent_policy_ids: tuple[str, ...]
    promotion_threshold: float | None
    minimum_evaluation_episodes: int

    def __post_init__(self) -> None:
        if not self.stage_id.strip():
            raise ValueError("Curriculum stage IDs must not be blank.")
        if (
            isinstance(self.complexity, bool)
            or not isinstance(self.complexity, int)
            or self.complexity < 0
        ):
            raise ValueError("Stage complexity must be a non-negative integer.")
        if self.scenario_kind not in _SCENARIO_KIND_RANK:
            raise ValueError("Unknown curriculum scenario kind.")
        if not self.scenario_ids or any(not item.strip() for item in self.scenario_ids):
            raise ValueError("Each curriculum stage requires named scenarios.")
        if len(set(self.scenario_ids)) != len(self.scenario_ids):
            raise ValueError("Scenario IDs must be unique within a stage.")
        expected_prefix = {
            "tactical": "tactical:",
            "intermediate": "intermediate:",
            "full_game": "full:",
        }[self.scenario_kind]
        if any(not scenario_id.startswith(expected_prefix) for scenario_id in self.scenario_ids):
            raise ValueError(
                f"{self.scenario_kind} scenarios must use the "
                f"{expected_prefix!r} namespace."
            )
        if len(set(self.opponent_policy_ids)) != len(self.opponent_policy_ids):
            raise ValueError("Opponent policy IDs must be unique within a stage.")
        unknown_policies = (
            set(self.opponent_policy_ids) - set(BASELINE_POLICY_IDS)
        )
        if unknown_policies:
            raise ValueError(
                f"Unknown curriculum opponent policies: {sorted(unknown_policies)}."
            )
        if (
            isinstance(self.minimum_evaluation_episodes, bool)
            or not isinstance(self.minimum_evaluation_episodes, int)
            or self.minimum_evaluation_episodes < 1
        ):
            raise ValueError("Minimum evaluation episodes must be positive.")
        if self.promotion_threshold is not None and (
            isinstance(self.promotion_threshold, bool)
            or not math.isfinite(self.promotion_threshold)
            or not 0.0 <= self.promotion_threshold <= 1.0
        ):
            raise ValueError("Promotion thresholds must be between zero and one.")

    @property
    def simplified(self) -> bool:
        return self.scenario_kind != "full_game"

    @property
    def homologation_eligible(self) -> bool:
        return self.scenario_kind == "full_game"


@dataclass(frozen=True)
class CurriculumPlan:
    stages: tuple[CurriculumStage, ...]

    def __post_init__(self) -> None:
        if len(self.stages) < 3:
            raise ValueError(
                "A curriculum needs tactical, intermediate, and full-game stages."
            )
        if self.stages[0].scenario_kind != "tactical":
            raise ValueError("Curriculum must begin with tactical tasks.")
        if self.stages[-1].scenario_kind != "full_game":
            raise ValueError("Curriculum must end with complete WAR games.")
        if not any(
            stage.scenario_kind == "intermediate" for stage in self.stages[1:-1]
        ):
            raise ValueError("Curriculum must include valid intermediate states.")
        if len({stage.stage_id for stage in self.stages}) != len(self.stages):
            raise ValueError("Curriculum stage IDs must be unique.")

        scenario_ids = tuple(
            scenario_id
            for stage in self.stages
            for scenario_id in stage.scenario_ids
        )
        if len(set(scenario_ids)) != len(scenario_ids):
            raise ValueError("Scenario IDs must be unique across the curriculum.")
        ranks = tuple(_SCENARIO_KIND_RANK[stage.scenario_kind] for stage in self.stages)
        if any(later < earlier for earlier, later in zip(ranks, ranks[1:])):
            raise ValueError("Curriculum scenario complexity cannot move backwards.")
        complexities = tuple(stage.complexity for stage in self.stages)
        if any(
            later <= earlier
            for earlier, later in zip(complexities, complexities[1:])
        ):
            raise ValueError("Curriculum complexity must increase at every stage.")
        if any(
            not set(previous.opponent_policy_ids).issubset(stage.opponent_policy_ids)
            for previous, stage in zip(self.stages, self.stages[1:])
        ):
            raise ValueError(
                "Later stages must retain and expand the adversary pool."
            )
        if any(
            stage.promotion_threshold is None for stage in self.stages[:-1]
        ):
            raise ValueError("Every non-final stage needs a promotion threshold.")
        if self.stages[-1].promotion_threshold is not None:
            raise ValueError("The final full-game stage has no promotion threshold.")

    @property
    def homologation_scenario_ids(self) -> tuple[str, ...]:
        return tuple(
            scenario_id
            for stage in self.stages
            if stage.homologation_eligible
            for scenario_id in stage.scenario_ids
        )

    def is_homologation_eligible(self, scenario_id: str) -> bool:
        return scenario_id in self.homologation_scenario_ids

    def require_homologation_scenario(self, scenario_id: str) -> None:
        if not self.is_homologation_eligible(scenario_id):
            raise ValueError(
                f"Curriculum scenario {scenario_id!r} is training-only; "
                "homologation requires a complete WAR game."
            )

    def stage(self, stage_id: str) -> CurriculumStage:
        stage = next(
            (candidate for candidate in self.stages if candidate.stage_id == stage_id),
            None,
        )
        if stage is None:
            raise KeyError(f"Unknown curriculum stage {stage_id!r}.")
        return stage


def build_default_curriculum() -> CurriculumPlan:
    """Create tactical, intermediate, and full-rule training stages."""
    return CurriculumPlan(
        stages=(
            CurriculumStage(
                stage_id="tactical-microtasks",
                complexity=0,
                scenario_kind="tactical",
                scenario_ids=_TACTICAL_SCENARIO_IDS,
                opponent_policy_ids=(),
                promotion_threshold=0.95,
                minimum_evaluation_episodes=64,
            ),
            CurriculumStage(
                stage_id="valid-intermediate-states",
                complexity=1,
                scenario_kind="intermediate",
                scenario_ids=_INTERMEDIATE_SCENARIOS,
                opponent_policy_ids=(
                    "random_legal",
                    "expansionist",
                    "conservative_borders",
                ),
                promotion_threshold=0.6,
                minimum_evaluation_episodes=128,
            ),
            CurriculumStage(
                stage_id="complete-war1-rules",
                complexity=2,
                scenario_kind="full_game",
                scenario_ids=_FULL_GAME_SCENARIOS,
                opponent_policy_ids=(
                    "random_legal",
                    "expansionist",
                    "conservative_borders",
                    "objective_oriented",
                    "continent_control",
                    "opportunistic_elimination",
                    "exact_probability_planner",
                ),
                promotion_threshold=None,
                minimum_evaluation_episodes=128,
            ),
        )
    )


class CurriculumScheduler:
    """Advance stages only after a minimum-size mastery evaluation."""

    def __init__(self, plan: CurriculumPlan) -> None:
        self.plan = plan
        self._active_index = 0

    @property
    def current_stage(self) -> CurriculumStage:
        return self.plan.stages[self._active_index]

    @property
    def is_complete(self) -> bool:
        return self._active_index == len(self.plan.stages) - 1

    def choose_scenario(self, rng: Generator) -> str:
        scenario_ids = self.current_stage.scenario_ids
        return scenario_ids[int(rng.integers(0, len(scenario_ids)))]

    def record_evaluation(self, mastery_score: float, episodes: int) -> bool:
        if isinstance(mastery_score, bool) or not math.isfinite(mastery_score):
            raise ValueError("Mastery score must be finite.")
        if not 0.0 <= mastery_score <= 1.0:
            raise ValueError("Mastery score must be between zero and one.")
        if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes < 0:
            raise ValueError("Evaluation episodes must be a non-negative integer.")
        if self.is_complete:
            return False
        stage = self.current_stage
        if stage.promotion_threshold is None or episodes < (
            stage.minimum_evaluation_episodes
        ):
            return False
        if mastery_score < stage.promotion_threshold:
            return False
        self._active_index += 1
        return True


def resolve_tactical_task(scenario_id: str) -> FiniteDecisionTask:
    """Resolve a tactical curriculum ID to its finite PPO task."""
    prefix = "tactical:"
    if not scenario_id.startswith(prefix):
        raise ValueError("Tactical task IDs must use the 'tactical:' namespace.")
    task_id = scenario_id.removeprefix(prefix)
    task = next(
        (candidate for candidate in build_small_tasks() if candidate.task_id == task_id),
        None,
    )
    if task is None:
        raise KeyError(f"Unknown tactical task {scenario_id!r}.")
    return task


def make_stage_opponent_policies(
    stage: CurriculumStage,
    *,
    num_players: int,
    focal_player_id: str = "player_0",
    opponent_offset: int = 0,
) -> dict[str, OpponentPolicy]:
    """Assign the stage's adversary tier across all non-focal seats."""
    minimum, maximum = PLAYER_COUNT_RANGE
    if isinstance(num_players, bool) or not isinstance(num_players, int):
        raise TypeError("num_players must be an integer.")
    if not minimum <= num_players <= maximum:
        raise ValueError(
            f"WAR 1 classic requires between {minimum} and {maximum} players."
        )
    player_ids = tuple(f"player_{index}" for index in range(num_players))
    if focal_player_id not in player_ids:
        raise ValueError(f"Unknown focal player {focal_player_id!r}.")
    if (
        isinstance(opponent_offset, bool)
        or not isinstance(opponent_offset, int)
        or opponent_offset < 0
    ):
        raise ValueError("opponent_offset must be a non-negative integer.")
    if not stage.opponent_policy_ids:
        raise ValueError("This curriculum stage does not define game opponents.")

    opponents: dict[str, OpponentPolicy] = {}
    opponent_seats = tuple(
        player_id for player_id in player_ids if player_id != focal_player_id
    )
    for seat_index, player_id in enumerate(opponent_seats):
        policy_index = (seat_index + opponent_offset) % len(
            stage.opponent_policy_ids
        )
        opponents[player_id] = create_baseline(
            stage.opponent_policy_ids[policy_index]
        )
    return opponents


def make_curriculum_aec_env(
    stage: CurriculumStage,
    scenario_id: str,
    *,
    num_players: int = 3,
) -> War1AECEnv:
    """Construct an AEC game at an intermediate state or full-game start."""
    if scenario_id not in stage.scenario_ids:
        raise ValueError(
            f"Scenario {scenario_id!r} does not belong to stage {stage.stage_id!r}."
        )
    if stage.scenario_kind == "tactical":
        raise ValueError(
            "Tactical microtasks are not WAR AEC states; train them with the "
            "small-task runner."
        )
    if stage.scenario_kind == "full_game":
        if scenario_id not in _FULL_GAME_SCENARIOS:
            raise ValueError("Unknown full-game curriculum scenario.")
        return War1AECEnv(num_players=num_players)
    if num_players != 3:
        raise ValueError("Current intermediate scenarios are defined for three seats.")
    scenario_key = scenario_id.removeprefix("intermediate:")
    return War1AECEnv(
        num_players=num_players,
        initial_state_factory=_intermediate_state_factory(scenario_key),
    )


def make_curriculum_gym_env(
    stage: CurriculumStage,
    scenario_id: str,
    *,
    num_players: int = 3,
    focal_player_id: str = "player_0",
    opponent_offset: int = 0,
    max_semantic_decisions: int | None = None,
) -> War1GymEnv:
    """Build a focal-player Gymnasium env with the stage's opponent tier."""
    aec_env = make_curriculum_aec_env(
        stage,
        scenario_id,
        num_players=num_players,
    )
    opponent_policies = make_stage_opponent_policies(
        stage,
        num_players=num_players,
        focal_player_id=focal_player_id,
        opponent_offset=opponent_offset,
    )
    return War1GymEnv(
        opponent_policies=opponent_policies,
        focal_agent=focal_player_id,
        aec_env=aec_env,
        max_semantic_decisions=max_semantic_decisions,
    )


def _intermediate_state_factory(scenario_id: str) -> InitialStateFactory:
    if scenario_id not in INTERMEDIATE_SCENARIO_IDS:
        raise ValueError(f"Unknown intermediate scenario {scenario_id!r}.")

    def factory(game_id: str, seed: int) -> FullGameState:
        return build_intermediate_scenario_state(
            scenario_id,
            game_id=game_id,
            seed=seed,
        )

    return factory


class FullGameReplayReservoir:
    """Reservoir of completed full-game arena results and their private replays.

    Keep this buffer local and access-controlled because full replays contain
    objectives, cards, and random-stream state.
    """

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError("Reservoir capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Reservoir seed must be a non-negative integer.")
        self.capacity = capacity
        self._rng = np.random.default_rng(seed)
        self._results: list[ArenaMatchResult] = []
        self._seen_count = 0

    @property
    def seen_count(self) -> int:
        return self._seen_count

    def __len__(self) -> int:
        return len(self._results)

    def add(
        self,
        result: ArenaMatchResult,
        *,
        stage: CurriculumStage,
    ) -> bool:
        """Retain a complete official match only from the full-game stage."""
        if not isinstance(result, ArenaMatchResult):
            raise TypeError("Replay reservoir accepts ArenaMatchResult values only.")
        if stage.scenario_kind != "full_game":
            raise ValueError(
                "Simplified curriculum scenarios cannot enter the full-game "
                "replay reservoir."
            )
        self._validate_full_game_result(result)

        self._seen_count += 1
        if len(self._results) < self.capacity:
            self._results.append(result)
            return True

        replacement_index = int(self._rng.integers(0, self._seen_count))
        if replacement_index >= self.capacity:
            return False
        self._results[replacement_index] = result
        return True

    def sample(self, count: int) -> tuple[ArenaMatchResult, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise ValueError("Sample count must be a positive integer.")
        if count > len(self._results):
            raise ValueError("Cannot sample more matches than the reservoir contains.")
        indices = self._rng.choice(len(self._results), size=count, replace=False)
        return tuple(self._results[int(index)] for index in indices)

    @staticmethod
    def _validate_full_game_result(result: ArenaMatchResult) -> None:
        if result.status != "completed" or not isinstance(
            result.outcome,
            OfficialVictoryOutcome,
        ):
            raise ValueError("Only completed official victories are replay samples.")
        initial_snapshot = result.replay.initial_snapshot
        initial_state = initial_snapshot.state
        if not isinstance(initial_state, FullGameState):
            raise ValueError("Replay sample requires a full initial game state.")
        if (
            initial_snapshot.visibility != "full"
            or initial_state.ruleset_id != WAR1_BOARD_SPEC.board_id
            or initial_state.phase_id != "initial_placement"
            or initial_state.turn_number != 1
            or initial_state.state_version != 0
            or initial_state.is_finished
        ):
            raise ValueError(
                "Replay sample must begin at a complete full-game initial state."
            )
        if result.replay.outcome != result.outcome:
            raise ValueError("Arena result and replay outcomes do not match.")


__all__ = [
    "CurriculumPlan",
    "CurriculumScenarioKind",
    "CurriculumScheduler",
    "CurriculumStage",
    "FullGameReplayReservoir",
    "build_default_curriculum",
    "make_curriculum_aec_env",
    "make_curriculum_gym_env",
    "make_stage_opponent_policies",
    "resolve_tactical_task",
]
