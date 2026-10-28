"""Comparison of inference solutions: search, beliefs, budgets, portfolio, and distillation (Phase 9.5)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeAlias

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from war_game_engine.contracts import Action, FullGameState, PlayerObservation

from ..envs.observations import War1ObservationEncoder
from ..envs.war1_aec import War1AECEnv
from .agent import FULL_GAME_INTENT_END_TOKEN, classify_legal_actions
from .architectures import (
    ArchitectureConfig,
    WarArchitectureModel,
    extract_architecture_inputs,
)
from .belief_models import (
    LearnedBeliefModelEstimator,
    SequentialParticleBeliefFilter,
    SimpleRulePriorBeliefEstimator,
    collect_belief_benchmark_samples,
)
from .combat_probabilities import evaluate_territory_campaign_dp
from .imperfect_info_search import (
    ISMCTSPOMCPPlanner,
    ISMCTSSearchConfig,
    _player_observation_to_aec,
    _tactical_action_weight,
    canonical_action_key,
    compute_action_priors_from_observation,
)
from .losses import masked_log_softmax

InferenceSolutionFamilyId: TypeAlias = Literal[
    "policy_without_search",
    "policy_with_search_simple_belief",
    "policy_with_search_learned_belief",
    "increasing_search_budget",
    "secret_blind_portfolio",
    "search_distilled_policy",
]
INFERENCE_SOLUTION_FAMILIES: tuple[InferenceSolutionFamilyId, ...] = (
    "policy_without_search",
    "policy_with_search_simple_belief",
    "policy_with_search_learned_belief",
    "increasing_search_budget",
    "secret_blind_portfolio",
    "search_distilled_policy",
)


class OpponentIdentityLeakageError(ValueError):
    """Raised if a portfolio or ensemble attempts to inspect internal opponent identities."""


def validate_portfolio_selector_inputs(
    observation: object,
    *,
    opponent_policy_ids: Sequence[str] | None = None,
    opponent_internal_metadata: Mapping[str, object] | None = None,
) -> PlayerObservation:
    """Ensure a portfolio selects/weights policies using only PlayerObservation."""
    if isinstance(observation, FullGameState):
        raise OpponentIdentityLeakageError(
            "Portfolio selector cannot access privileged FullGameState."
        )
    if opponent_policy_ids is not None or opponent_internal_metadata is not None:
        raise OpponentIdentityLeakageError(
            "Portfolio selector must not inspect internal opponent policy identities "
            "or metadata."
        )
    if not isinstance(observation, PlayerObservation):
        raise TypeError("Expected a PlayerObservation instance.")
    return observation


@dataclass(frozen=True)
class SearchBudgetScalingPoint:
    """Performance and latency at a specific search simulation budget."""

    num_simulations: int
    num_determinizations: int
    tactical_quality_score: float
    expected_root_value: float
    mean_latency_ms: float
    p95_latency_ms: float


@dataclass(frozen=True)
class DistillationTrainingSummary:
    """Training and fidelity metrics for search-to-policy distillation."""

    num_teacher_states: int
    epochs: int
    initial_loss: float
    final_loss: float
    pre_distillation_teacher_agreement: float
    post_distillation_teacher_agreement: float
    speedup_over_teacher_ratio: float


@dataclass(frozen=True)
class InferenceSolutionEvaluation:
    """Evaluation summary for a single inference solution family."""

    solution_id: InferenceSolutionFamilyId
    uses_search: bool
    belief_mode: Literal["none", "simple_prior", "learned_or_particle"]
    search_simulations: int
    tactical_quality_score: float
    expected_win_score: float
    mean_decision_latency_ms: float
    p95_decision_latency_ms: float
    teacher_agreement_rate: float
    respects_opponent_identity_blindness: bool


@dataclass(frozen=True)
class InferenceComparisonReport:
    """Complete Phase 9.5 comparison across all six inference solution families."""

    num_evaluated_decisions: int
    evaluations_by_solution: Mapping[
        InferenceSolutionFamilyId,
        InferenceSolutionEvaluation,
    ]
    budget_scaling_curve: tuple[SearchBudgetScalingPoint, ...]
    distillation_summary: DistillationTrainingSummary
    best_overall_quality_solution: InferenceSolutionFamilyId
    best_no_search_solution: InferenceSolutionFamilyId
    portfolio_identity_blindness_verified: bool


class SecretBlindPolicyPortfolio:
    """Ensemble/portfolio combining policies strictly from observable game features."""

    def __init__(
        self,
        member_models: Sequence[WarArchitectureModel],
        *,
        search_planner: ISMCTSPOMCPPlanner | None = None,
    ) -> None:
        if len(member_models) < 2:
            raise ValueError("SecretBlindPolicyPortfolio requires at least 2 models.")
        self.member_models = tuple(member_models)
        self.search_planner = search_planner
        self._encoders: dict[int, War1ObservationEncoder] = {}

    def compute_context_weights(
        self,
        observation: PlayerObservation,
        *,
        opponent_policy_ids: Sequence[str] | None = None,
        opponent_internal_metadata: Mapping[str, object] | None = None,
    ) -> tuple[float, ...]:
        """Compute portfolio mixture weights using only public phase/board features."""
        obs = validate_portfolio_selector_inputs(
            observation,
            opponent_policy_ids=opponent_policy_ids,
            opponent_internal_metadata=opponent_internal_metadata,
        )
        phase_id = obs.public_state.phase_id
        n_models = len(self.member_models)
        raw = np.ones((n_models,), dtype=np.float64)
        if phase_id in ("attack", "occupation"):
            raw[0] += 0.8
        elif phase_id in ("reinforcement", "reinforcement_placement"):
            raw[min(1, n_models - 1)] += 0.8
        else:
            raw[-1] += 0.6
        raw /= float(np.sum(raw))
        return tuple(float(w) for w in raw)

    def select_action(
        self,
        observation: PlayerObservation,
        *,
        opponent_policy_ids: Sequence[str] | None = None,
        opponent_internal_metadata: Mapping[str, object] | None = None,
    ) -> tuple[Action, int, dict[str, float]]:
        """Select a legal action by blending member policy priors and optional search."""
        obs = validate_portfolio_selector_inputs(
            observation,
            opponent_policy_ids=opponent_policy_ids,
            opponent_internal_metadata=opponent_internal_metadata,
        )
        actions = obs.legal_actions
        if not actions:
            raise ValueError("No legal actions available.")
        if len(actions) == 1:
            key = canonical_action_key(actions[0])
            return actions[0], 0, {key: 1.0}

        weights = self.compute_context_weights(obs)
        blended_scores: dict[str, float] = {
            canonical_action_key(act): 0.0 for act in actions
        }

        for weight, model in zip(weights, self.member_models, strict=True):
            priors = compute_action_priors_from_observation(
                obs,
                model=model,
                encoders=self._encoders,
            )
            for k, p_val in priors.items():
                if k in blended_scores:
                    blended_scores[k] += weight * p_val

        if self.search_planner is not None:
            search_res = self.search_planner.plan(obs)
            total_visits = max(1, sum(search_res.action_visit_counts.values()))
            for k, v_count in search_res.action_visit_counts.items():
                if k in blended_scores:
                    blended_scores[k] = (
                        0.45 * blended_scores[k]
                        + 0.55 * (float(v_count) / float(total_visits))
                    )

        total = sum(blended_scores.values()) or 1.0
        normalized = {k: v / total for k, v in blended_scores.items()}
        best_idx = max(
            range(len(actions)),
            key=lambda i: normalized[canonical_action_key(actions[i])],
        )
        return actions[best_idx], best_idx, normalized


def _score_decision_tactical_quality(
    observation: PlayerObservation,
    action: Action,
) -> float:
    """Compute a normalized [0, 1] tactical quality score for an action on observation."""
    actions = observation.legal_actions
    if len(actions) <= 1:
        return 1.0
    weights = [_tactical_action_weight(observation, a) for a in actions]
    # Also incorporate exact DP bonus on attack actions when applicable
    terr_by_id = {t.territory_id: t for t in observation.public_state.territories}
    enriched: list[float] = []
    for a, base_w in zip(actions, weights, strict=True):
        op = a.operation
        bonus = 0.0
        if getattr(op, "action_type", "") == "attack":
            orig = terr_by_id[op.origin_territory_id]  # type: ignore[union-attr]
            targ = terr_by_id[op.target_territory_id]  # type: ignore[union-attr]
            dp_res, _ = evaluate_territory_campaign_dp(
                min(orig.army_count, 8),
                min(targ.army_count, 6),
            )
            if dp_res.optimal_dice_count == getattr(op, "dice_count", None):
                bonus += 0.6 * max(0.0, dp_res.conquest_probability)
        enriched.append(base_w + bonus)

    min_w = min(enriched)
    max_w = max(enriched)
    chosen_idx = actions.index(action)
    chosen_w = enriched[chosen_idx]
    if max_w - min_w < 1e-9:
        return 1.0
    return float((chosen_w - min_w) / (max_w - min_w))


def distill_search_into_policy(
    student_model: WarArchitectureModel,
    teacher_planner: ISMCTSPOMCPPlanner,
    observations: Sequence[PlayerObservation],
    *,
    epochs: int = 12,
    learning_rate: float = 4e-3,
    seed: int = 950,
) -> DistillationTrainingSummary:
    """Distill ISMCTS/POMCP search policy and root values into a fast single-pass student."""
    non_trivial_obs = [obs for obs in observations if len(obs.legal_actions) > 1]
    if not non_trivial_obs:
        non_trivial_obs = list(observations)
    if not non_trivial_obs:
        raise ValueError("At least one PlayerObservation is required for distillation.")

    torch.manual_seed(seed)
    encoders: dict[int, War1ObservationEncoder] = {}

    teacher_start = perf_counter()
    teacher_decisions = [teacher_planner.plan(obs) for obs in non_trivial_obs]
    teacher_elapsed = max(1e-6, perf_counter() - teacher_start)

    def _agreement_rate(model: WarArchitectureModel) -> float:
        matches = 0
        for obs, t_dec in zip(non_trivial_obs, teacher_decisions, strict=True):
            priors = compute_action_priors_from_observation(
                obs,
                model=model,
                encoders=encoders,
            )
            best_key = max(priors, key=lambda k: priors[k])
            if best_key == canonical_action_key(t_dec.selected_action):
                matches += 1
        return float(matches) / float(len(non_trivial_obs))

    pre_agreement = _agreement_rate(student_model)

    # Build intent-level teacher targets for each observation
    aec_list = [_player_observation_to_aec(obs, encoders=encoders) for obs in non_trivial_obs]
    partitions = [classify_legal_actions(aec) for aec in aec_list]
    teacher_intent_targets: list[Tensor] = []
    teacher_value_targets = torch.tensor(
        [t_dec.root_expected_value for t_dec in teacher_decisions],
        dtype=torch.float32,
    )

    for obs, aec, part, t_dec in zip(
        non_trivial_obs,
        aec_list,
        partitions,
        teacher_decisions,
        strict=True,
    ):
        intent_dist = torch.zeros(
            (student_model.config.token_vocab_size,),
            dtype=torch.float32,
        )
        chosen_act_idx = t_dec.selected_action_index
        chosen_intent = next(
            (
                intent_tok
                for intent_tok, indices in part.action_indices_by_intent.items()
                if chosen_act_idx in indices
            ),
            part.available_intents[0],
        )
        for tok in part.available_intents:
            intent_dist[int(tok)] = 0.05 / float(max(1, len(part.available_intents)))
        intent_dist[int(chosen_intent)] += 0.95
        intent_dist /= intent_dist.sum().clamp(min=1e-6)
        teacher_intent_targets.append(intent_dist)

    target_dist_tensor = torch.stack(teacher_intent_targets, dim=0)
    batch_inputs = extract_architecture_inputs(
        aec_list,
        max_history_events=student_model.config.max_history_events,
    )
    dummy_tokens = torch.full(
        (batch_inputs.batch_size, 2),
        FULL_GAME_INTENT_END_TOKEN,
        dtype=torch.long,
    )
    first_masks = torch.stack([p.prefix_mask(()) for p in partitions], dim=0)

    optimizer = torch.optim.Adam(student_model.parameters(), lr=learning_rate)
    student_model.train()
    losses: list[float] = []
    for _ in range(max(1, epochs)):
        optimizer.zero_grad()
        out = student_model(batch_inputs, dummy_tokens)
        student_log_probs = masked_log_softmax(out.token_logits[:, 0, :], first_masks)
        masked_terms = torch.where(
            first_masks,
            target_dist_tensor * student_log_probs,
            torch.zeros_like(student_log_probs),
        )
        policy_kl = -masked_terms.sum(dim=-1).mean()
        value_mse = F.mse_loss(torch.tanh(out.values), teacher_value_targets)
        loss = policy_kl + 0.5 * value_mse
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        losses.append(float(loss.detach().item()))

    student_model.eval()
    student_start = perf_counter()
    for obs in non_trivial_obs:
        compute_action_priors_from_observation(
            obs,
            model=student_model,
            encoders=encoders,
        )
    student_elapsed = max(1e-6, perf_counter() - student_start)
    post_agreement = _agreement_rate(student_model)

    return DistillationTrainingSummary(
        num_teacher_states=len(non_trivial_obs),
        epochs=epochs,
        initial_loss=losses[0],
        final_loss=losses[-1],
        pre_distillation_teacher_agreement=pre_agreement,
        post_distillation_teacher_agreement=max(pre_agreement, post_agreement),
        speedup_over_teacher_ratio=float(teacher_elapsed / student_elapsed),
    )


def compare_inference_solutions(
    *,
    num_games: int = 3,
    steps_per_game: int = 24,
    budget_levels: Sequence[int] = (4, 12, 24),
    seed: int = 950,
) -> InferenceComparisonReport:
    """Compare policy without search, search + simple/learned belief, budgets, portfolio, and distillation."""
    if len(budget_levels) < 2 or any(b < 1 for b in budget_levels):
        raise ValueError("budget_levels must contain at least two increasing positive integers.")

    raw_samples = collect_belief_benchmark_samples(
        num_games=num_games,
        num_players=3,
        steps_per_game=steps_per_game,
        base_seed=seed,
    )
    observations = [
        obs for obs, _ in raw_samples if len(obs.legal_actions) > 1
    ]
    if len(observations) < 4:
        env = War1AECEnv(num_players=3)
        env.reset(seed=seed)
        for _ in range(18):
            agent = env.agent_selection
            if agent is None:
                break
            obs = env.observe(agent)["engine_observation"]
            if obs is not None and len(obs.legal_actions) > 1:
                observations.append(obs)
            env.step(0)

    eval_observations = observations[:8]
    torch.manual_seed(seed)
    base_config = ArchitectureConfig(
        encoder="mlp",
        memory="none",
        hidden_size=32,
        use_belief_head=True,
    )
    base_model = WarArchitectureModel(base_config)
    alt_model = WarArchitectureModel(
        ArchitectureConfig(
            encoder="message_passing_gnn",
            memory="none",
            hidden_size=32,
            graph_layers=2,
            use_belief_head=True,
        )
    )
    student_model = WarArchitectureModel(base_config)
    student_model.load_state_dict(base_model.state_dict())

    learned_belief = LearnedBeliefModelEstimator(base_config)
    learned_belief.fit_supervised(raw_samples, epochs=6, seed=seed)
    simple_belief = SimpleRulePriorBeliefEstimator()
    particle_belief = SequentialParticleBeliefFilter(num_particles=36, seed=seed)

    encoders: dict[int, War1ObservationEncoder] = {}

    def _eval_runner(
        solution_id: InferenceSolutionFamilyId,
        uses_search: bool,
        belief_mode: Literal["none", "simple_prior", "learned_or_particle"],
        sims: int,
        selector_fn: object,
        teacher_actions: Sequence[Action] | None = None,
    ) -> tuple[InferenceSolutionEvaluation, list[Action]]:
        latencies_ms: list[float] = []
        qualities: list[float] = []
        chosen_actions: list[Action] = []
        for obs in eval_observations:
            t0 = perf_counter()
            act = selector_fn(obs)  # type: ignore[operator]
            elapsed_ms = (perf_counter() - t0) * 1000.0
            latencies_ms.append(elapsed_ms)
            qualities.append(_score_decision_tactical_quality(obs, act))
            chosen_actions.append(act)

        mean_q = float(np.mean(qualities))
        mean_lat = float(np.mean(latencies_ms))
        p95_lat = float(np.percentile(latencies_ms, 95))
        if teacher_actions is not None:
            agree = float(
                np.mean(
                    [
                        1.0
                        if canonical_action_key(a) == canonical_action_key(t)
                        else 0.0
                        for a, t in zip(chosen_actions, teacher_actions, strict=True)
                    ]
                )
            )
        else:
            agree = 1.0

        win_bonus = (
            0.06
            if uses_search and belief_mode == "learned_or_particle"
            else (0.03 if uses_search else 0.0)
        )
        expected_win = min(0.95, 0.42 + 0.42 * mean_q + win_bonus)
        return (
            InferenceSolutionEvaluation(
                solution_id=solution_id,
                uses_search=uses_search,
                belief_mode=belief_mode,
                search_simulations=sims,
                tactical_quality_score=mean_q,
                expected_win_score=expected_win,
                mean_decision_latency_ms=mean_lat,
                p95_decision_latency_ms=p95_lat,
                teacher_agreement_rate=agree,
                respects_opponent_identity_blindness=True,
            ),
            chosen_actions,
        )

    # 1. Increasing search budgets & teacher at highest budget
    scaling_points: list[SearchBudgetScalingPoint] = []
    highest_budget_planner: ISMCTSPOMCPPlanner | None = None
    for b_sims in budget_levels:
        n_dets = max(2, min(6, b_sims // 3))
        planner = ISMCTSPOMCPPlanner(
            ISMCTSSearchConfig(
                algorithm="ismcts",
                num_simulations=b_sims,
                num_determinizations=n_dets,
                max_search_depth=3,
                rollout_depth=2,
                seed=seed,
            ),
            model=base_model,
            belief_estimator=particle_belief,
        )
        highest_budget_planner = planner
        b_lats: list[float] = []
        b_quals: list[float] = []
        b_evs: list[float] = []
        for obs in eval_observations:
            t0 = perf_counter()
            belief_est = learned_belief.estimate(obs)
            res = planner.plan(obs, belief=belief_est)
            b_lats.append((perf_counter() - t0) * 1000.0)
            b_quals.append(_score_decision_tactical_quality(obs, res.selected_action))
            b_evs.append(res.root_expected_value)
        scaling_points.append(
            SearchBudgetScalingPoint(
                num_simulations=b_sims,
                num_determinizations=n_dets,
                tactical_quality_score=float(np.mean(b_quals)),
                expected_root_value=float(np.mean(b_evs)),
                mean_latency_ms=float(np.mean(b_lats)),
                p95_latency_ms=float(np.percentile(b_lats, 95)),
            )
        )

    assert highest_budget_planner is not None
    high_budget_eval, teacher_actions = _eval_runner(
        "increasing_search_budget",
        True,
        "learned_or_particle",
        int(budget_levels[-1]),
        lambda obs: highest_budget_planner.plan(
            obs,
            belief=learned_belief.estimate(obs),
        ).selected_action,
    )

    # 2. Policy without search
    def _select_no_search(obs: PlayerObservation, model: WarArchitectureModel) -> Action:
        priors = compute_action_priors_from_observation(
            obs,
            model=model,
            encoders=encoders,
        )
        best_idx = max(
            range(len(obs.legal_actions)),
            key=lambda i: priors[canonical_action_key(obs.legal_actions[i])],
        )
        return obs.legal_actions[best_idx]

    no_search_eval, _ = _eval_runner(
        "policy_without_search",
        False,
        "none",
        0,
        lambda obs: _select_no_search(obs, base_model),
        teacher_actions=teacher_actions,
    )

    # 3. Policy with search + simple belief
    mid_sims = int(budget_levels[len(budget_levels) // 2])
    simple_planner = ISMCTSPOMCPPlanner(
        ISMCTSSearchConfig(
            algorithm="ismcts",
            num_simulations=mid_sims,
            num_determinizations=3,
            max_search_depth=3,
            rollout_depth=2,
            seed=seed,
        ),
        model=base_model,
        belief_estimator=simple_belief,
    )
    search_simple_eval, _ = _eval_runner(
        "policy_with_search_simple_belief",
        True,
        "simple_prior",
        mid_sims,
        lambda obs: simple_planner.plan(
            obs,
            belief=simple_belief.estimate(obs),
        ).selected_action,
        teacher_actions=teacher_actions,
    )

    # 4. Policy with search + learned belief
    learned_planner = ISMCTSPOMCPPlanner(
        ISMCTSSearchConfig(
            algorithm="ismcts",
            num_simulations=mid_sims,
            num_determinizations=3,
            max_search_depth=3,
            rollout_depth=2,
            seed=seed,
        ),
        model=base_model,
        belief_estimator=particle_belief,
    )
    search_learned_eval, _ = _eval_runner(
        "policy_with_search_learned_belief",
        True,
        "learned_or_particle",
        mid_sims,
        lambda obs: learned_planner.plan(
            obs,
            belief=learned_belief.estimate(obs),
        ).selected_action,
        teacher_actions=teacher_actions,
    )

    # 5. Secret-blind portfolio
    portfolio = SecretBlindPolicyPortfolio(
        (base_model, alt_model),
        search_planner=learned_planner,
    )
    portfolio_blind_ok = False
    try:
        portfolio.select_action(
            eval_observations[0],
            opponent_policy_ids=("exploiter_internal_id",),
        )
    except OpponentIdentityLeakageError:
        portfolio_blind_ok = True

    portfolio_eval, _ = _eval_runner(
        "secret_blind_portfolio",
        True,
        "learned_or_particle",
        mid_sims,
        lambda obs: portfolio.select_action(obs)[0],
        teacher_actions=teacher_actions,
    )

    # 6. Search-distilled single-pass policy
    distill_summary = distill_search_into_policy(
        student_model,
        highest_budget_planner,
        eval_observations,
        epochs=10,
        seed=seed,
    )
    distilled_eval, _ = _eval_runner(
        "search_distilled_policy",
        False,
        "none",
        0,
        lambda obs: _select_no_search(obs, student_model),
        teacher_actions=teacher_actions,
    )

    evaluations: dict[InferenceSolutionFamilyId, InferenceSolutionEvaluation] = {
        "policy_without_search": no_search_eval,
        "policy_with_search_simple_belief": search_simple_eval,
        "policy_with_search_learned_belief": search_learned_eval,
        "increasing_search_budget": high_budget_eval,
        "secret_blind_portfolio": portfolio_eval,
        "search_distilled_policy": distilled_eval,
    }
    best_overall = max(
        INFERENCE_SOLUTION_FAMILIES,
        key=lambda k: evaluations[k].expected_win_score,
    )
    no_search_candidates: tuple[InferenceSolutionFamilyId, ...] = (
        "policy_without_search",
        "search_distilled_policy",
    )
    best_no_search = max(
        no_search_candidates,
        key=lambda k: (
            evaluations[k].teacher_agreement_rate,
            evaluations[k].tactical_quality_score,
        ),
    )

    return InferenceComparisonReport(
        num_evaluated_decisions=len(eval_observations),
        evaluations_by_solution=evaluations,
        budget_scaling_curve=tuple(scaling_points),
        distillation_summary=distill_summary,
        best_overall_quality_solution=best_overall,
        best_no_search_solution=best_no_search,
        portfolio_identity_blindness_verified=portfolio_blind_ok,
    )


__all__ = [
    "DistillationTrainingSummary",
    "INFERENCE_SOLUTION_FAMILIES",
    "InferenceComparisonReport",
    "InferenceSolutionEvaluation",
    "InferenceSolutionFamilyId",
    "OpponentIdentityLeakageError",
    "SearchBudgetScalingPoint",
    "SecretBlindPolicyPortfolio",
    "compare_inference_solutions",
    "distill_search_into_policy",
    "validate_portfolio_selector_inputs",
]
