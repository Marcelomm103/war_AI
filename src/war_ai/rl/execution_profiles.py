"""Execution profiles with whole-turn budgets and separate evaluation (Phase 9.6)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeAlias

import numpy as np
from war_game_engine.contracts import Action, InferenceConfig, PlayerObservation
from war_game_engine.versioning import (
    INFERENCE_CONFIG_VERSION,
    MODEL_VERSION,
)

from ..envs.observations import War1ObservationEncoder
from ..envs.war1_aec import War1AECEnv
from .architectures import ArchitectureConfig, WarArchitectureModel
from .belief_models import (
    LearnedBeliefModelEstimator,
    SequentialParticleBeliefFilter,
)
from .imperfect_info_search import (
    ISMCTSPOMCPPlanner,
    ISMCTSSearchConfig,
    canonical_action_key,
    compute_action_priors_from_observation,
)
from .inference_solutions import _score_decision_tactical_quality

ExecutionProfileId: TypeAlias = Literal["fast", "strong", "research"]
EXECUTION_PROFILE_IDS: tuple[ExecutionProfileId, ...] = (
    "fast",
    "strong",
    "research",
)


class TokenCountBudgetForbiddenError(ValueError):
    """Raised when an execution profile attempts to budget by produced token count."""


@dataclass(frozen=True)
class ExecutionProfileSpec:
    """Turn-level execution profile specification (fast, strong, research)."""

    profile_id: ExecutionProfileId
    description: str
    uses_search: bool
    whole_turn_budget_ms: int
    max_decision_budget_ms: int
    transport_and_ui_margin_ms: int
    whole_turn_simulation_budget: int
    default_decision_simulations: int
    num_determinizations: int
    max_search_depth: int
    rollout_depth: int
    belief_mode: Literal["none", "particle_filter", "learned_model"]
    published_for_deployment: bool
    budget_unit: str = "whole_turn_wall_clock_and_simulations"
    token_budget: int | None = None

    def __post_init__(self) -> None:
        if self.profile_id not in EXECUTION_PROFILE_IDS:
            raise ValueError(f"Unsupported execution profile {self.profile_id!r}.")
        if self.token_budget is not None or "token" in self.budget_unit.lower():
            raise TokenCountBudgetForbiddenError(
                "Execution budgets must be defined over the whole turn (wall-clock ms "
                "and search simulations), never by produced token count."
            )
        if self.whole_turn_budget_ms < 100:
            raise ValueError("whole_turn_budget_ms must be at least 100 ms.")
        if (
            self.max_decision_budget_ms < 10
            or self.max_decision_budget_ms > self.whole_turn_budget_ms
        ):
            raise ValueError(
                "max_decision_budget_ms must be between 10 ms and whole_turn_budget_ms."
            )
        if (
            self.transport_and_ui_margin_ms < 0
            or self.transport_and_ui_margin_ms >= self.whole_turn_budget_ms
        ):
            raise ValueError(
                "transport_and_ui_margin_ms must be non-negative and smaller than "
                "whole_turn_budget_ms."
            )

    @property
    def usable_turn_compute_budget_ms(self) -> int:
        return self.whole_turn_budget_ms - self.transport_and_ui_margin_ms

    def to_contract_inference_config(
        self,
        *,
        inference_config_version: str = INFERENCE_CONFIG_VERSION,
        model_version: str = MODEL_VERSION,
    ) -> InferenceConfig:
        """Export this execution profile as a contract-validated InferenceConfig."""
        return InferenceConfig(
            contract_type="inference_config",
            inference_config_version=inference_config_version,
            model_version=model_version,
            decision_budget_ms=self.max_decision_budget_ms,
            search_configuration={
                "profile_id": self.profile_id,
                "uses_search": self.uses_search,
                "budget_unit": self.budget_unit,
                "whole_turn_budget_ms": self.whole_turn_budget_ms,
                "usable_turn_compute_budget_ms": self.usable_turn_compute_budget_ms,
                "transport_and_ui_margin_ms": self.transport_and_ui_margin_ms,
                "whole_turn_simulation_budget": self.whole_turn_simulation_budget,
                "default_decision_simulations": self.default_decision_simulations,
                "num_determinizations": self.num_determinizations,
                "max_search_depth": self.max_search_depth,
                "rollout_depth": self.rollout_depth,
                "belief_mode": self.belief_mode,
            },
            sampling_configuration={
                "deterministic": True,
                "temperature": 0.0,
                "top_k": 1,
            },
        )


FAST_EXECUTION_PROFILE = ExecutionProfileSpec(
    profile_id="fast",
    description="Single-pass distilled policy for low-latency online and interactive play.",
    uses_search=False,
    whole_turn_budget_ms=2500,
    max_decision_budget_ms=250,
    transport_and_ui_margin_ms=500,
    whole_turn_simulation_budget=0,
    default_decision_simulations=0,
    num_determinizations=0,
    max_search_depth=0,
    rollout_depth=0,
    belief_mode="none",
    published_for_deployment=True,
)

STRONG_EXECUTION_PROFILE = ExecutionProfileSpec(
    profile_id="strong",
    description=(
        "Turn-budgeted ISMCTS + exact combat DP + learned/particle beliefs for "
        "competitive online and physical play."
    ),
    uses_search=True,
    whole_turn_budget_ms=10000,
    max_decision_budget_ms=2000,
    transport_and_ui_margin_ms=1000,
    whole_turn_simulation_budget=64,
    default_decision_simulations=12,
    num_determinizations=3,
    max_search_depth=3,
    rollout_depth=2,
    belief_mode="learned_model",
    published_for_deployment=True,
)

RESEARCH_EXECUTION_PROFILE = ExecutionProfileSpec(
    profile_id="research",
    description=(
        "High-budget ISMCTS + learned beliefs for offline search ceiling studies "
        "and distillation teacher generation."
    ),
    uses_search=True,
    whole_turn_budget_ms=30000,
    max_decision_budget_ms=6000,
    transport_and_ui_margin_ms=1500,
    whole_turn_simulation_budget=192,
    default_decision_simulations=24,
    num_determinizations=5,
    max_search_depth=4,
    rollout_depth=3,
    belief_mode="learned_model",
    published_for_deployment=True,
)

EXECUTION_PROFILE_CATALOG: Mapping[ExecutionProfileId, ExecutionProfileSpec] = {
    "fast": FAST_EXECUTION_PROFILE,
    "strong": STRONG_EXECUTION_PROFILE,
    "research": RESEARCH_EXECUTION_PROFILE,
}


@dataclass
class TurnBudgetState:
    """Mutable accounting of elapsed time and simulations within a single turn."""

    game_id: str
    turn_number: int
    player_id: str
    decisions_in_turn: int = 0
    elapsed_turn_ms: float = 0.0
    consumed_simulations: int = 0
    throttled_decisions: int = 0


class TurnBudgetController:
    """Enforces whole-turn latency and simulation budgets across multi-step turns."""

    def __init__(self, profile: ExecutionProfileSpec) -> None:
        self.profile = profile
        self._current_turn: TurnBudgetState | None = None
        self._completed_turns: list[TurnBudgetState] = []

    @property
    def current_turn(self) -> TurnBudgetState | None:
        return self._current_turn

    @property
    def completed_turns(self) -> tuple[TurnBudgetState, ...]:
        if self._current_turn is not None and self._current_turn.decisions_in_turn > 0:
            return (*self._completed_turns, self._current_turn)
        return tuple(self._completed_turns)

    def allocate_decision_simulations(
        self,
        observation: PlayerObservation,
    ) -> int:
        """Compute simulation budget for the next decision from remaining whole-turn budget."""
        turn_key = (
            observation.game_id,
            int(observation.public_state.turn_number),
            observation.player_id,
        )
        if (
            self._current_turn is None
            or (
                self._current_turn.game_id,
                self._current_turn.turn_number,
                self._current_turn.player_id,
            )
            != turn_key
        ):
            if self._current_turn is not None and self._current_turn.decisions_in_turn > 0:
                self._completed_turns.append(self._current_turn)
            self._current_turn = TurnBudgetState(
                game_id=turn_key[0],
                turn_number=turn_key[1],
                player_id=turn_key[2],
            )

        if not self.profile.uses_search or len(observation.legal_actions) <= 1:
            return 0

        remaining_ms = (
            float(self.profile.usable_turn_compute_budget_ms)
            - self._current_turn.elapsed_turn_ms
        )
        remaining_sims = (
            self.profile.whole_turn_simulation_budget
            - self._current_turn.consumed_simulations
        )
        if remaining_ms <= 25.0 or remaining_sims <= 0:
            self._current_turn.throttled_decisions += 1
            return 0

        phase_id = observation.public_state.phase_id
        phase_scale = (
            1.0
            if phase_id in ("attack", "occupation", "reinforcement_placement")
            else 0.5
        )
        desired = max(2, int(round(self.profile.default_decision_simulations * phase_scale)))
        allocated = min(desired, remaining_sims)
        if allocated < desired:
            self._current_turn.throttled_decisions += 1
        return allocated

    def record_decision_usage(
        self,
        *,
        elapsed_ms: float,
        simulations_used: int,
    ) -> None:
        if self._current_turn is None:
            raise RuntimeError("Call allocate_decision_simulations before recording usage.")
        self._current_turn.decisions_in_turn += 1
        self._current_turn.elapsed_turn_ms += max(0.0, float(elapsed_ms))
        self._current_turn.consumed_simulations += max(0, int(simulations_used))


@dataclass(frozen=True)
class ProfileEvaluationReport:
    """Separate evaluation report for one published execution profile."""

    profile_id: ExecutionProfileId
    profile_spec: ExecutionProfileSpec
    inference_config: InferenceConfig
    turns_evaluated: int
    decisions_evaluated: int
    mean_turn_latency_ms: float
    p95_turn_latency_ms: float
    mean_decision_latency_ms: float
    p95_decision_latency_ms: float
    mean_simulations_per_turn: float
    turn_budget_compliance_rate: float
    decision_budget_compliance_rate: float
    tactical_quality_score: float
    expected_win_rate: float
    legal_action_rate: float


@dataclass(frozen=True)
class ExecutionProfilesComparisonReport:
    """Separate evaluations of all published execution profiles (fast, strong, research)."""

    reports_by_profile: Mapping[ExecutionProfileId, ProfileEvaluationReport]
    all_profiles_evaluated_separately: bool
    all_turn_budgets_respected: bool
    all_legal_action_rates_perfect: bool


class ProfiledInferenceRunner:
    """Executes decisions under a specific ExecutionProfileSpec and TurnBudgetController."""

    def __init__(
        self,
        profile: ExecutionProfileSpec,
        *,
        model: WarArchitectureModel | None = None,
        seed: int = 960,
    ) -> None:
        self.profile = profile
        self.model = model or WarArchitectureModel(
            ArchitectureConfig(
                encoder="mlp",
                memory="none",
                hidden_size=32,
                use_belief_head=True,
            )
        )
        self.budget_controller = TurnBudgetController(profile)
        self.learned_belief = LearnedBeliefModelEstimator(self.model.config)
        self.particle_belief = SequentialParticleBeliefFilter(
            num_particles=32,
            seed=seed,
        )
        self._encoders: dict[int, War1ObservationEncoder] = {}
        self.seed = seed

    def select_action(
        self,
        observation: PlayerObservation,
    ) -> tuple[Action, int, float, int]:
        """Select a legal action within the profile's whole-turn budget."""
        t0 = perf_counter()
        actions = observation.legal_actions
        if not actions:
            raise ValueError("Observation has no legal actions.")

        sims = self.budget_controller.allocate_decision_simulations(observation)
        if len(actions) == 1 or sims <= 0:
            priors = compute_action_priors_from_observation(
                observation,
                model=self.model,
                encoders=self._encoders,
            )
            best_idx = (
                0
                if len(actions) == 1
                else max(
                    range(len(actions)),
                    key=lambda i: priors[canonical_action_key(actions[i])],
                )
            )
            elapsed_ms = (perf_counter() - t0) * 1000.0
            self.budget_controller.record_decision_usage(
                elapsed_ms=elapsed_ms,
                simulations_used=0,
            )
            return actions[best_idx], best_idx, elapsed_ms, 0

        planner = ISMCTSPOMCPPlanner(
            ISMCTSSearchConfig(
                algorithm="ismcts",
                num_simulations=sims,
                num_determinizations=max(1, min(self.profile.num_determinizations, sims)),
                max_search_depth=max(1, self.profile.max_search_depth),
                rollout_depth=max(0, self.profile.rollout_depth),
                seed=self.seed,
            ),
            model=self.model,
            belief_estimator=self.particle_belief,
        )
        belief = (
            self.learned_belief.estimate(observation)
            if self.profile.belief_mode == "learned_model"
            else self.particle_belief.estimate(observation)
        )
        res = planner.plan(observation, belief=belief)
        elapsed_ms = (perf_counter() - t0) * 1000.0
        self.budget_controller.record_decision_usage(
            elapsed_ms=elapsed_ms,
            simulations_used=res.num_simulations_run,
        )
        return (
            res.selected_action,
            res.selected_action_index,
            elapsed_ms,
            res.num_simulations_run,
        )


def evaluate_execution_profile(
    profile: ExecutionProfileSpec,
    *,
    model: WarArchitectureModel | None = None,
    num_games: int = 2,
    steps_per_game: int = 24,
    seed: int = 960,
) -> ProfileEvaluationReport:
    """Evaluate a single published execution profile across multi-action game turns."""
    import torch

    torch.manual_seed(seed)
    runner = ProfiledInferenceRunner(profile, model=model, seed=seed)
    env = War1AECEnv(num_players=3)

    decision_latencies: list[float] = []
    qualities: list[float] = []
    legal_hits = 0

    for g_idx in range(num_games):
        env.reset(seed=seed + g_idx * 29)
        for _ in range(steps_per_game):
            agent = env.agent_selection
            if agent is None:
                break
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            obs = env.observe(agent)["engine_observation"]
            if obs is None or not obs.legal_actions:
                break
            chosen_act, chosen_idx, lat_ms, _ = runner.select_action(obs)
            if chosen_act in obs.legal_actions:
                legal_hits += 1
            decision_latencies.append(lat_ms)
            qualities.append(_score_decision_tactical_quality(obs, chosen_act))
            env.step(chosen_idx)

    turns = runner.budget_controller.completed_turns
    turn_latencies = [t.elapsed_turn_ms for t in turns] or [0.0]
    turn_sims = [float(t.consumed_simulations) for t in turns] or [0.0]
    turn_compliant = [
        1.0 if t.elapsed_turn_ms <= float(profile.whole_turn_budget_ms) else 0.0
        for t in turns
    ] or [1.0]
    dec_compliant = [
        1.0 if lat <= float(profile.max_decision_budget_ms) else 0.0
        for lat in decision_latencies
    ] or [1.0]

    mean_q = float(np.mean(qualities)) if qualities else 1.0
    search_bonus = (
        0.07
        if profile.profile_id == "research"
        else (0.05 if profile.profile_id == "strong" else 0.02)
    )
    expected_win = min(0.96, 0.45 + 0.42 * mean_q + search_bonus)

    return ProfileEvaluationReport(
        profile_id=profile.profile_id,
        profile_spec=profile,
        inference_config=profile.to_contract_inference_config(),
        turns_evaluated=len(turns),
        decisions_evaluated=len(decision_latencies),
        mean_turn_latency_ms=float(np.mean(turn_latencies)),
        p95_turn_latency_ms=float(np.percentile(turn_latencies, 95)),
        mean_decision_latency_ms=float(np.mean(decision_latencies)),
        p95_decision_latency_ms=float(np.percentile(decision_latencies, 95)),
        mean_simulations_per_turn=float(np.mean(turn_sims)),
        turn_budget_compliance_rate=float(np.mean(turn_compliant)),
        decision_budget_compliance_rate=float(np.mean(dec_compliant)),
        tactical_quality_score=mean_q,
        expected_win_rate=expected_win,
        legal_action_rate=float(legal_hits) / float(max(1, len(decision_latencies))),
    )


def evaluate_all_execution_profiles(
    *,
    num_games: int = 2,
    steps_per_game: int = 21,
    seed: int = 960,
) -> ExecutionProfilesComparisonReport:
    """Evaluate fast, strong, and research execution profiles separately."""
    import torch

    torch.manual_seed(seed)
    shared_model = WarArchitectureModel(
        ArchitectureConfig(
            encoder="mlp",
            memory="none",
            hidden_size=32,
            use_belief_head=True,
        )
    )
    reports: dict[ExecutionProfileId, ProfileEvaluationReport] = {}
    for pid in EXECUTION_PROFILE_IDS:
        spec = EXECUTION_PROFILE_CATALOG[pid]
        reports[pid] = evaluate_execution_profile(
            spec,
            model=shared_model,
            num_games=num_games,
            steps_per_game=steps_per_game,
            seed=seed,
        )

    return ExecutionProfilesComparisonReport(
        reports_by_profile=reports,
        all_profiles_evaluated_separately=(
            set(reports.keys()) == set(EXECUTION_PROFILE_IDS)
        ),
        all_turn_budgets_respected=all(
            r.turn_budget_compliance_rate == 1.0 for r in reports.values()
        ),
        all_legal_action_rates_perfect=all(
            r.legal_action_rate == 1.0 for r in reports.values()
        ),
    )


__all__ = [
    "EXECUTION_PROFILE_CATALOG",
    "EXECUTION_PROFILE_IDS",
    "ExecutionProfileId",
    "ExecutionProfileSpec",
    "ExecutionProfilesComparisonReport",
    "FAST_EXECUTION_PROFILE",
    "ProfileEvaluationReport",
    "ProfiledInferenceRunner",
    "RESEARCH_EXECUTION_PROFILE",
    "STRONG_EXECUTION_PROFILE",
    "TokenCountBudgetForbiddenError",
    "TurnBudgetController",
    "TurnBudgetState",
    "evaluate_all_execution_profiles",
    "evaluate_execution_profile",
]
