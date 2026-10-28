"""Algorithmic branches and comparative evaluators for WAR 1 RL (Phase 7.3)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter

import numpy as np
import torch
from numpy.random import Generator
from torch import Tensor
from torch.nn import functional as F

from ..arena import ArenaPolicy, FrozenPolicySpec
from ..envs.war1_aec import War1AECEnv, War1AECObservation
from .agent import (
    FULL_GAME_INTENT_END_TOKEN,
    FULL_GAME_INTENT_VOCAB_SIZE,
    classify_legal_actions,
)
from .architectures import (
    ArchitectureConfig,
    StructuredArchitectureInputs,
    WarArchitectureModel,
    extract_architecture_inputs,
)
from .experimental_axes import AlgorithmAxis, ComputeBudget
from .losses import (
    joint_action_entropy,
    joint_action_log_prob,
    masked_log_softmax,
)
from .ppo import compute_gae, compute_masked_ppo_loss

MINIMUM_ALGORITHM_BRANCHES: tuple[AlgorithmAxis, ...] = (
    "ppo_feedforward",
    "ppo_recurrent",
    "graph_actor_critic",
    "impala_vtrace",
    "recurrent_off_policy_r2d2",
    "policy_value_search",
)

EXCLUDED_STANDARD_ALGORITHMS: dict[str, str] = {
    "sac_continuous": (
        "Continuous-action SAC is not a standard choice for discrete, "
        "prefix-masked structured WAR actions."
    ),
    "ddpg": (
        "Deterministic continuous policy gradients (DDPG) are incompatible "
        "with discrete masked multi-token WAR operations."
    ),
    "qmix": (
        "Cooperative monotonic value factorization (QMIX) is unsuitable for "
        "non-cooperative, hidden-objective multiplayer WAR."
    ),
}


@dataclass(frozen=True)
class VTraceOutput:
    """Targets, policy-gradient advantages, and clipped ratios for IMPALA/V-trace."""

    vs: Tensor
    pg_advantages: Tensor
    clipped_rhos: Tensor
    clipped_cs: Tensor


@dataclass(frozen=True)
class ImpalaVTraceLoss:
    """Loss terms and diagnostics for an IMPALA/V-trace update."""

    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    mean_clipped_rho: Tensor


@dataclass(frozen=True)
class R2D2StructuredLoss:
    """Loss terms and TD errors for recurrent off-policy R2D2 updates."""

    total_loss: Tensor
    td_errors: Tensor
    mean_q_value: Tensor
    burn_in_steps: int
    learning_steps: int


@dataclass(frozen=True)
class AlgorithmBranchEvaluation:
    """Controlled-budget training and evaluation summary for an algorithm branch."""

    branch: AlgorithmAxis
    architecture_config: ArchitectureConfig
    compute_budget: ComputeBudget
    initial_optimal_intent_probability: float
    trained_optimal_intent_probability: float
    final_loss: float
    training_wall_time_ms: float
    mean_inference_ms: float
    selected_action_legal: bool


def validate_algorithm_branch(branch: str) -> AlgorithmAxis:
    """Validate that ``branch`` is one of the supported WAR algorithm families."""
    normalized = branch.strip().lower()
    if normalized in EXCLUDED_STANDARD_ALGORITHMS:
        reason = EXCLUDED_STANDARD_ALGORITHMS[normalized]
        raise ValueError(
            f"Algorithm {branch!r} is excluded for WAR: {reason}"
        )
    if normalized not in MINIMUM_ALGORITHM_BRANCHES:
        raise ValueError(
            f"Unknown algorithm branch {branch!r}; expected one of "
            f"{MINIMUM_ALGORITHM_BRANCHES}."
        )
    return normalized


def default_architecture_for_branch(
    branch: AlgorithmAxis,
    *,
    hidden_size: int = 64,
    graph_layers: int = 3,
    max_history_events: int = 64,
) -> ArchitectureConfig:
    """Return the canonical architecture configuration paired with ``branch``."""
    validated = validate_algorithm_branch(branch)
    if validated == "ppo_feedforward":
        return ArchitectureConfig(
            encoder="mlp",
            memory="none",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=False,
        )
    if validated == "ppo_recurrent":
        return ArchitectureConfig(
            encoder="mlp",
            memory="gru",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        )
    if validated == "graph_actor_critic":
        return ArchitectureConfig(
            encoder="message_passing_gnn",
            memory="none",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        )
    if validated == "impala_vtrace":
        return ArchitectureConfig(
            encoder="message_passing_gnn",
            memory="gru",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        )
    if validated == "recurrent_off_policy_r2d2":
        return ArchitectureConfig(
            encoder="mlp",
            memory="lstm",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=False,
        )
    return ArchitectureConfig(
        encoder="graph_transformer",
        memory="causal_event_transformer",
        hidden_size=hidden_size,
        graph_layers=graph_layers,
        max_history_events=max_history_events,
        phase_conditioned_heads=True,
        use_objective_and_card_encoder=True,
        use_belief_head=True,
    )


def compute_vtrace_targets_and_advantages(
    *,
    behavior_log_probs: Tensor,
    target_log_probs: Tensor,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
) -> VTraceOutput:
    """Compute IMPALA V-trace targets and advantages over ``(time, batch)``."""
    if behavior_log_probs.ndim != 2:
        raise ValueError("V-trace tensors must have shape (time, batch).")
    shape = behavior_log_probs.shape
    for name, tensor in (
        ("target_log_probs", target_log_probs),
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
        ("terminated", terminated),
        ("truncated", truncated),
    ):
        if tensor.shape != shape:
            raise ValueError(f"{name} must match shape {shape}.")
    if clip_rho_threshold <= 0.0 or clip_c_threshold <= 0.0:
        raise ValueError("V-trace clipping thresholds must be positive.")
    if not 0.0 <= gamma <= 1.0:
        raise ValueError("gamma must be in [0, 1].")

    log_rhos = target_log_probs - behavior_log_probs
    rhos = torch.exp(log_rhos)
    clipped_rhos = torch.clamp(rhos, max=clip_rho_threshold)
    clipped_cs = torch.clamp(rhos, max=clip_c_threshold)

    not_terminated = (~terminated).to(dtype=values.dtype)
    discounts = gamma * not_terminated
    bootstrap_values = torch.where(
        terminated,
        torch.zeros_like(next_values),
        next_values,
    )
    deltas = clipped_rhos * (rewards + discounts * bootstrap_values - values)

    time_steps, batch_size = shape
    vs_minus_v = torch.zeros_like(values)
    acc = torch.zeros((batch_size,), dtype=values.dtype, device=values.device)
    episode_boundary = terminated | truncated
    for t in range(time_steps - 1, -1, -1):
        acc = torch.where(
            episode_boundary[t],
            torch.zeros_like(acc),
            acc,
        )
        acc = deltas[t] + discounts[t] * clipped_cs[t] * acc
        vs_minus_v[t] = acc

    vs = vs_minus_v + values
    next_vs = torch.zeros_like(vs)
    if time_steps > 1:
        next_vs[:-1] = torch.where(
            episode_boundary[:-1],
            bootstrap_values[:-1],
            vs[1:],
        )
    next_vs[-1] = bootstrap_values[-1]
    pg_advantages = clipped_rhos * (rewards + discounts * next_vs - values)
    return VTraceOutput(
        vs=vs.detach(),
        pg_advantages=pg_advantages.detach(),
        clipped_rhos=clipped_rhos.detach(),
        clipped_cs=clipped_cs.detach(),
    )


def compute_impala_vtrace_loss(
    *,
    new_logits: Tensor,
    actions: Tensor,
    legal_masks: Tensor,
    token_mask: Tensor,
    behavior_log_probs: Tensor,
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    gamma: float = 1.0,
    clip_rho_threshold: float = 1.0,
    clip_c_threshold: float = 1.0,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
) -> ImpalaVTraceLoss:
    """Compute IMPALA V-trace actor-critic loss for structured actions."""
    target_log_probs = joint_action_log_prob(
        new_logits,
        actions,
        legal_masks,
        token_mask,
    )
    entropy = joint_action_entropy(new_logits, legal_masks, token_mask).mean()
    vtrace = compute_vtrace_targets_and_advantages(
        behavior_log_probs=behavior_log_probs,
        target_log_probs=target_log_probs.detach(),
        rewards=rewards,
        values=values.detach(),
        next_values=next_values.detach(),
        terminated=terminated,
        truncated=truncated,
        gamma=gamma,
        clip_rho_threshold=clip_rho_threshold,
        clip_c_threshold=clip_c_threshold,
    )
    policy_loss = -(target_log_probs * vtrace.pg_advantages).mean()
    value_loss = 0.5 * F.mse_loss(values, vtrace.vs)
    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    return ImpalaVTraceLoss(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        mean_clipped_rho=vtrace.clipped_rhos.mean(),
    )


def signed_parabolic_transform(x: Tensor, *, eps: float = 1e-3) -> Tensor:
    """Invertible value rescaling h(x) used in R2D2 recurrent Q-learning."""
    return torch.sign(x) * (torch.sqrt(torch.abs(x) + 1.0) - 1.0) + eps * x


def signed_parabolic_inverse(y: Tensor, *, eps: float = 1e-3) -> Tensor:
    """Exact inverse h^-1(y) of :func:`signed_parabolic_transform`."""
    abs_y = torch.abs(y)
    numerator = torch.sqrt(1.0 + 4.0 * eps * (abs_y + 1.0 + eps)) - 1.0
    root = numerator / (2.0 * eps)
    return torch.sign(y) * (torch.square(root) - 1.0)


def compute_r2d2_structured_q_loss(
    model: WarArchitectureModel,
    target_model: WarArchitectureModel,
    *,
    sequence_inputs: Sequence[StructuredArchitectureInputs],
    candidate_tokens: Tensor,
    taken_intent_indices: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    burn_in_steps: int = 1,
    gamma: float = 0.99,
) -> R2D2StructuredLoss:
    """Run R2D2 burn-in and compute rescaled double-Q loss on structured intents."""
    total_steps = len(sequence_inputs)
    if total_steps < 2:
        raise ValueError("R2D2 sequence requires at least 2 steps.")
    if (
        isinstance(burn_in_steps, bool)
        or burn_in_steps < 0
        or burn_in_steps >= total_steps - 1
    ):
        raise ValueError(
            "burn_in_steps must leave at least one transition for learning."
        )
    num_intents = candidate_tokens.shape[0]
    batch_size = sequence_inputs[0].batch_size
    if taken_intent_indices.shape != (total_steps - 1, batch_size):
        raise ValueError(
            "taken_intent_indices must have shape (total_steps - 1, batch)."
        )
    if rewards.shape != (total_steps - 1, batch_size):
        raise ValueError("rewards must have shape (total_steps - 1, batch).")
    if terminated.shape != (total_steps - 1, batch_size):
        raise ValueError("terminated must have shape (total_steps - 1, batch).")

    online_state = None
    target_state = None
    with torch.no_grad():
        for step_idx in range(burn_in_steps):
            _, online_state = model.encode_context(
                sequence_inputs[step_idx],
                recurrent_state=online_state,
            )
            _, target_state = target_model.encode_context(
                sequence_inputs[step_idx],
                recurrent_state=target_state,
            )

    q_predictions: list[Tensor] = []
    q_targets: list[Tensor] = []
    learning_steps = total_steps - 1 - burn_in_steps

    for step_idx in range(burn_in_steps, total_steps - 1):
        current_in = sequence_inputs[step_idx]
        next_in = sequence_inputs[step_idx + 1]
        context, online_state = model.encode_context(
            current_in,
            recurrent_state=online_state,
        )
        expanded_context = context.repeat_interleave(num_intents, dim=0)
        expanded_phases = current_in.phase_indices.repeat_interleave(
            num_intents,
            dim=0,
        )
        tiled_tokens = candidate_tokens.repeat(batch_size, 1)
        token_logits = model.action_head(
            expanded_context,
            tiled_tokens,
            expanded_phases,
        )
        intent_scores = token_logits[:, 0, :num_intents].diagonal(
            dim1=-2,
            dim2=-1,
        ) if False else token_logits[:, 0, :num_intents].reshape(
            batch_size,
            num_intents,
            num_intents,
        )[:, 0, :]
        state_values = model.value_head(context)
        q_all = state_values + (
            intent_scores - intent_scores.mean(dim=-1, keepdim=True)
        )
        chosen_q = q_all.gather(
            -1,
            taken_intent_indices[step_idx].unsqueeze(-1),
        ).squeeze(-1)
        q_predictions.append(chosen_q)

        with torch.no_grad():
            next_online_ctx, _ = model.encode_context(
                next_in,
                recurrent_state=online_state,
            )
            next_exp_ctx = next_online_ctx.repeat_interleave(num_intents, dim=0)
            next_exp_phases = next_in.phase_indices.repeat_interleave(
                num_intents,
                dim=0,
            )
            next_online_logits = model.action_head(
                next_exp_ctx,
                tiled_tokens,
                next_exp_phases,
            )
            next_online_scores = next_online_logits[:, 0, :num_intents].reshape(
                batch_size,
                num_intents,
                num_intents,
            )[:, 0, :]
            best_next_intent = next_online_scores.argmax(dim=-1, keepdim=True)

            next_target_ctx, target_state = target_model.encode_context(
                next_in,
                recurrent_state=target_state,
            )
            next_target_exp_ctx = next_target_ctx.repeat_interleave(
                num_intents,
                dim=0,
            )
            next_target_logits = target_model.action_head(
                next_target_exp_ctx,
                tiled_tokens,
                next_exp_phases,
            )
            next_target_scores = next_target_logits[:, 0, :num_intents].reshape(
                batch_size,
                num_intents,
                num_intents,
            )[:, 0, :]
            next_target_v = target_model.value_head(next_target_ctx)
            next_target_q = next_target_v + (
                next_target_scores
                - next_target_scores.mean(dim=-1, keepdim=True)
            )
            chosen_next_q = next_target_q.gather(-1, best_next_intent).squeeze(-1)
            unscaled_next_q = signed_parabolic_inverse(chosen_next_q)
            not_done = (~terminated[step_idx]).to(dtype=rewards.dtype)
            bellman = rewards[step_idx] + gamma * not_done * unscaled_next_q
            q_targets.append(signed_parabolic_transform(bellman))

    pred_tensor = torch.stack(q_predictions, dim=0)
    target_tensor = torch.stack(q_targets, dim=0)
    td_errors = (target_tensor - pred_tensor).detach()
    total_loss = F.smooth_l1_loss(pred_tensor, target_tensor)
    return R2D2StructuredLoss(
        total_loss=total_loss,
        td_errors=td_errors,
        mean_q_value=pred_tensor.detach().mean(),
        burn_in_steps=burn_in_steps,
        learning_steps=learning_steps,
    )


@torch.no_grad()
def select_action_with_policy_value_search(
    model: WarArchitectureModel,
    observation: War1AECObservation,
    *,
    puct_c: float = 1.25,
    max_candidates_evaluated: int = 16,
    rng: Generator | None = None,
) -> tuple[int, int, Tensor]:
    """Combine policy priors, value estimates, and tactical lookahead search."""
    if isinstance(max_candidates_evaluated, bool) or max_candidates_evaluated < 1:
        raise ValueError("max_candidates_evaluated must be a positive integer.")
    partition = classify_legal_actions(observation)
    candidate_intents = partition.available_intents
    if len(candidate_intents) == 1:
        chosen_intent = candidate_intents[0]
        action_candidates = partition.action_indices_by_intent[chosen_intent]
        best_action = (
            action_candidates[int(rng.integers(0, len(action_candidates)))]
            if rng is not None and len(action_candidates) > 1
            else action_candidates[0]
        )
        return (
            best_action,
            chosen_intent,
            torch.ones((1,), dtype=torch.float32),
        )
    single_inputs = extract_architecture_inputs(
        observation,
        max_history_events=model.config.max_history_events,
    )
    context, _ = model.encode_context(single_inputs)
    root_value = float(torch.tanh(model.value_head(context)).item())
    num_candidates = len(candidate_intents)
    candidate_tokens = torch.tensor(
        [[intent, model.config.end_token_id] for intent in candidate_intents],
        dtype=torch.long,
    )
    token_mask = torch.ones_like(candidate_tokens, dtype=torch.bool)
    legal_masks = torch.zeros(
        (num_candidates, 2, model.config.token_vocab_size),
        dtype=torch.bool,
    )
    first_mask = partition.prefix_mask(())
    for idx, intent in enumerate(candidate_intents):
        legal_masks[idx, 0] = first_mask
        legal_masks[idx, 1] = partition.prefix_mask((intent,))
    token_logits = model.action_head(
        context.expand(num_candidates, -1),
        candidate_tokens,
        single_inputs.phase_indices.expand(num_candidates),
    )
    log_probs = joint_action_log_prob(
        token_logits,
        candidate_tokens,
        legal_masks,
        token_mask,
    )
    prior_probs = masked_log_softmax(
        log_probs.unsqueeze(0),
        torch.ones((1, num_candidates), dtype=torch.bool),
    ).squeeze(0).exp()

    search_scores: list[float] = []
    visit_total = float(min(len(partition.action_scores), max_candidates_evaluated))
    for idx, intent in enumerate(candidate_intents):
        prior = float(prior_probs[idx].item())
        indices = partition.action_indices_by_intent[intent][:max_candidates_evaluated]
        best_tactical = max(partition.action_scores[action_i] for action_i in indices)
        normalized_tactical = math.tanh(best_tactical / 20.0)
        q_estimate = 0.7 * normalized_tactical + 0.3 * root_value
        exploration_bonus = puct_c * prior * math.sqrt(visit_total) / (1.0 + len(indices))
        search_scores.append(q_estimate + exploration_bonus)

    score_tensor = torch.tensor(search_scores, dtype=torch.float32)
    best_pos = int(score_tensor.argmax().item())
    selected_intent = candidate_intents[best_pos]
    action_candidates = partition.action_indices_by_intent[selected_intent]
    best_tier_score = max(
        partition.action_scores[idx] for idx in action_candidates
    )
    top_candidates = tuple(
        idx
        for idx in action_candidates
        if abs(partition.action_scores[idx] - best_tier_score) <= 1e-9
    )
    best_action = (
        top_candidates[int(rng.integers(0, len(top_candidates)))]
        if rng is not None and len(top_candidates) > 1
        else top_candidates[0]
    )
    posterior = torch.softmax(score_tensor * 4.0, dim=0)
    return best_action, selected_intent, posterior


def _collect_reference_observations(
    count: int = 8,
    *,
    seed: int = 2026,
) -> list[War1AECObservation]:
    env = War1AECEnv()
    env.reset(seed=seed)
    observations: list[War1AECObservation] = []
    for _ in range(count):
        agent_id = env.agent_selection
        assert agent_id is not None
        observations.append(env.observe(agent_id))
        env.step(0)
    return observations


def _optimal_intent_probability(
    model: WarArchitectureModel,
    observation: War1AECObservation,
    *,
    branch: AlgorithmAxis,
) -> float:
    if branch == "policy_value_search":
        _, _, probs = select_action_with_policy_value_search(model, observation)
        return float(probs[0].item())
    if branch == "recurrent_off_policy_r2d2":
        _, _, probs = select_r2d2_q_action(model, observation)
        return float(probs[0].item())
    _, _, probs, _ = model.select_legal_action(observation, deterministic=True)
    return float(probs[0].item())


@torch.no_grad()
def select_r2d2_q_action(
    model: WarArchitectureModel,
    observation: War1AECObservation,
    *,
    temperature: float = 0.2,
    rng: Generator | None = None,
) -> tuple[int, int, Tensor]:
    """Select a legal action from unscaled R2D2 dueling Q-values."""
    if temperature <= 0.0:
        raise ValueError("temperature must be positive.")
    partition = classify_legal_actions(observation)
    candidate_intents = partition.available_intents
    if len(candidate_intents) == 1:
        chosen_intent = candidate_intents[0]
        action_candidates = partition.action_indices_by_intent[chosen_intent]
        chosen_action = (
            action_candidates[int(rng.integers(0, len(action_candidates)))]
            if rng is not None
            else action_candidates[0]
        )
        return (
            chosen_action,
            chosen_intent,
            torch.ones((1,), dtype=torch.float32),
        )
    single_inputs = extract_architecture_inputs(
        observation,
        max_history_events=model.config.max_history_events,
    )
    candidate_tokens = torch.tensor(
        [
            [0, FULL_GAME_INTENT_END_TOKEN],
            [1, FULL_GAME_INTENT_END_TOKEN],
            [2, FULL_GAME_INTENT_END_TOKEN],
        ],
        dtype=torch.long,
    )
    context, _ = model.encode_context(single_inputs)
    expanded_ctx = context.repeat_interleave(3, dim=0)
    expanded_phases = single_inputs.phase_indices.repeat_interleave(3, dim=0)
    token_logits = model.action_head(
        expanded_ctx,
        candidate_tokens,
        expanded_phases,
    )
    intent_scores = token_logits[:, 0, :3].reshape(1, 3, 3)[:, 0, :]
    state_value = model.value_head(context)
    rescaled_q = (
        state_value
        + (intent_scores - intent_scores.mean(dim=-1, keepdim=True))
    ).squeeze(0)
    unscaled_q = signed_parabolic_inverse(rescaled_q)
    available_indices = torch.tensor(
        partition.available_intents,
        dtype=torch.long,
    )
    available_q = unscaled_q[available_indices]
    probs = torch.softmax(available_q / temperature, dim=0)
    best_pos = int(probs.argmax().item())
    chosen_intent = partition.available_intents[best_pos]
    action_candidates = partition.action_indices_by_intent[chosen_intent]
    if chosen_intent != 0:
        best_tier_score = max(
            partition.action_scores[idx] for idx in action_candidates
        )
        action_candidates = tuple(
            idx
            for idx in action_candidates
            if abs(partition.action_scores[idx] - best_tier_score) <= 1e-9
        )
    if rng is not None:
        chosen_action = action_candidates[
            int(rng.integers(0, len(action_candidates)))
        ]
    else:
        chosen_action = action_candidates[0]
    return chosen_action, chosen_intent, probs


def evaluate_algorithm_branch(
    branch: AlgorithmAxis | str,
    *,
    compute_budget: ComputeBudget = ComputeBudget(
        semantic_decisions=64,
        learner_updates=12,
        inference_budget_ms=250,
    ),
    hidden_size: int = 32,
    learning_rate: float = 0.008,
    seed: int = 2026,
) -> AlgorithmBranchEvaluation:
    """Train and evaluate one of the six Phase 7.3 algorithm branches."""
    validated_branch = validate_algorithm_branch(str(branch))
    torch.manual_seed(seed)
    np.random.seed(seed)

    arch_config = default_architecture_for_branch(
        validated_branch,
        hidden_size=hidden_size,
        graph_layers=3,
        max_history_events=32,
    )
    model = WarArchitectureModel(arch_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    observations = _collect_reference_observations(count=6, seed=seed)
    base_inputs = extract_architecture_inputs(
        observations,
        max_history_events=arch_config.max_history_events,
    )
    num_obs = base_inputs.batch_size
    candidate_tokens = torch.tensor(
        [
            [0, FULL_GAME_INTENT_END_TOKEN],
            [1, FULL_GAME_INTENT_END_TOKEN],
            [2, FULL_GAME_INTENT_END_TOKEN],
        ],
        dtype=torch.long,
    )
    batch_inputs = base_inputs.repeat_batch(3)
    batch_size = batch_inputs.batch_size
    all_intent_tokens = candidate_tokens.repeat(num_obs, 1)
    intent_rewards_1d = torch.tensor(
        [1.0, 0.25, 0.0] * num_obs,
        dtype=torch.float32,
    )
    taken_intents_1d = torch.tensor(
        [0, 1, 2] * num_obs,
        dtype=torch.long,
    )
    token_mask = torch.ones_like(all_intent_tokens, dtype=torch.bool)
    legal_masks = torch.zeros(
        (batch_size, 2, FULL_GAME_INTENT_VOCAB_SIZE),
        dtype=torch.bool,
    )
    legal_masks[:, 0, :3] = True
    legal_masks[:, 1, FULL_GAME_INTENT_END_TOKEN] = True

    initial_prob = _optimal_intent_probability(
        model,
        observations[0],
        branch=validated_branch,
    )

    started_train = perf_counter()
    final_loss_value = 0.0
    target_model = WarArchitectureModel(arch_config)
    target_model.load_state_dict(model.state_dict())

    for _ in range(compute_budget.learner_updates):
        optimizer.zero_grad()
        if validated_branch in (
            "ppo_feedforward",
            "ppo_recurrent",
            "graph_actor_critic",
            "policy_value_search",
        ):
            with torch.no_grad():
                old_out = model(batch_inputs, all_intent_tokens)
                old_log_probs = joint_action_log_prob(
                    old_out.token_logits,
                    all_intent_tokens,
                    legal_masks,
                    token_mask,
                )
                rewards = intent_rewards_1d.unsqueeze(0)
                advantages, returns = compute_gae(
                    rewards=rewards,
                    values=old_out.values.unsqueeze(0),
                    next_values=torch.zeros((1, batch_size), dtype=torch.float32),
                    terminated=torch.ones((1, batch_size), dtype=torch.bool),
                    truncated=torch.zeros((1, batch_size), dtype=torch.bool),
                )
            new_out = model(batch_inputs, all_intent_tokens)
            ppo_loss = compute_masked_ppo_loss(
                new_logits=new_out.token_logits,
                actions=all_intent_tokens,
                legal_masks=legal_masks,
                token_mask=token_mask,
                old_log_probabilities=old_log_probs,
                advantages=advantages[0],
                value_predictions=new_out.values,
                returns=returns[0],
            )
            loss_tensor = ppo_loss.total_loss
        elif validated_branch == "impala_vtrace":
            with torch.no_grad():
                beh_out = model(batch_inputs, all_intent_tokens)
                beh_log_probs = joint_action_log_prob(
                    beh_out.token_logits,
                    all_intent_tokens,
                    legal_masks,
                    token_mask,
                ).unsqueeze(0)
            new_out = model(batch_inputs, all_intent_tokens)
            vtrace_loss = compute_impala_vtrace_loss(
                new_logits=new_out.token_logits.unsqueeze(0),
                actions=all_intent_tokens.unsqueeze(0),
                legal_masks=legal_masks.unsqueeze(0),
                token_mask=token_mask.unsqueeze(0),
                behavior_log_probs=beh_log_probs,
                rewards=intent_rewards_1d.unsqueeze(0),
                values=new_out.values.unsqueeze(0),
                next_values=torch.zeros((1, batch_size), dtype=torch.float32),
                terminated=torch.ones((1, batch_size), dtype=torch.bool),
                truncated=torch.zeros((1, batch_size), dtype=torch.bool),
            )
            loss_tensor = vtrace_loss.total_loss
        else:
            seq_inputs = [batch_inputs, batch_inputs, batch_inputs]
            taken = taken_intents_1d.unsqueeze(0).expand(2, -1)
            rewards = intent_rewards_1d.unsqueeze(0).expand(2, -1)
            done = torch.ones((2, batch_size), dtype=torch.bool)
            r2d2_loss = compute_r2d2_structured_q_loss(
                model,
                target_model,
                sequence_inputs=seq_inputs,
                candidate_tokens=candidate_tokens,
                taken_intent_indices=taken,
                rewards=rewards,
                terminated=done,
                burn_in_steps=1,
            )
            loss_tensor = r2d2_loss.total_loss

        loss_tensor.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        if validated_branch == "recurrent_off_policy_r2d2":
            target_model.load_state_dict(model.state_dict())
        final_loss_value = float(loss_tensor.detach().item())

    training_wall_ms = (perf_counter() - started_train) * 1000.0
    started_infer = perf_counter()
    if validated_branch == "policy_value_search":
        action_idx, _, _ = select_action_with_policy_value_search(
            model,
            observations[0],
        )
    elif validated_branch == "recurrent_off_policy_r2d2":
        action_idx, _, _ = select_r2d2_q_action(
            model,
            observations[0],
        )
    else:
        action_idx, _, _, _ = model.select_legal_action(
            observations[0],
            deterministic=True,
        )
    infer_ms = (perf_counter() - started_infer) * 1000.0
    trained_prob = _optimal_intent_probability(
        model,
        observations[0],
        branch=validated_branch,
    )
    action_mask = observations[0]["action_mask"]
    is_legal = 0 <= action_idx < int(action_mask.shape[0]) and bool(
        action_mask[action_idx] == 1
    )
    return AlgorithmBranchEvaluation(
        branch=validated_branch,
        architecture_config=arch_config,
        compute_budget=compute_budget,
        initial_optimal_intent_probability=initial_prob,
        trained_optimal_intent_probability=trained_prob,
        final_loss=final_loss_value,
        training_wall_time_ms=training_wall_ms,
        mean_inference_ms=infer_ms,
        selected_action_legal=is_legal,
    )


def evaluate_all_algorithm_branches(
    *,
    compute_budget: ComputeBudget = ComputeBudget(
        semantic_decisions=64,
        learner_updates=12,
        inference_budget_ms=250,
    ),
    hidden_size: int = 32,
    seed: int = 2026,
) -> tuple[AlgorithmBranchEvaluation, ...]:
    """Evaluate all six Phase 7.3 algorithm branches under a shared compute budget."""
    return tuple(
        evaluate_algorithm_branch(
            branch,
            compute_budget=compute_budget,
            hidden_size=hidden_size,
            seed=seed,
        )
        for branch in MINIMUM_ALGORITHM_BRANCHES
    )


def build_branch_policy_spec(
    branch: AlgorithmAxis,
    model: WarArchitectureModel,
    *,
    policy_id: str,
    checkpoint_sha256: str,
) -> FrozenPolicySpec:
    """Wrap a trained ``WarArchitectureModel`` into a seat-isolated arena policy."""
    validated = validate_algorithm_branch(branch)
    state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    config = model.config

    def factory(factory_seed: int) -> ArenaPolicy:
        del factory_seed
        seat_model = WarArchitectureModel(config)
        seat_model.load_state_dict(state_copy)
        seat_model.eval()

        def policy(observation: War1AECObservation, rng: Generator) -> int:
            if validated == "policy_value_search":
                action_idx, _, _ = select_action_with_policy_value_search(
                    seat_model,
                    observation,
                    rng=rng,
                )
                return action_idx
            if validated == "recurrent_off_policy_r2d2":
                action_idx, _, _ = select_r2d2_q_action(
                    seat_model,
                    observation,
                    rng=rng,
                )
                return action_idx
            action_idx, _, _, _ = seat_model.select_legal_action(
                observation,
                deterministic=True,
                rng=rng,
            )
            return action_idx

        return policy

    return FrozenPolicySpec(
        policy_id=policy_id,
        factory=factory,
        checkpoint_sha256=checkpoint_sha256,
    )


__all__ = [
    "AlgorithmBranchEvaluation",
    "EXCLUDED_STANDARD_ALGORITHMS",
    "ImpalaVTraceLoss",
    "MINIMUM_ALGORITHM_BRANCHES",
    "R2D2StructuredLoss",
    "VTraceOutput",
    "build_branch_policy_spec",
    "compute_impala_vtrace_loss",
    "compute_r2d2_structured_q_loss",
    "compute_vtrace_targets_and_advantages",
    "default_architecture_for_branch",
    "evaluate_algorithm_branch",
    "evaluate_all_algorithm_branches",
    "masked_log_softmax",
    "select_action_with_policy_value_search",
    "select_r2d2_q_action",
    "signed_parabolic_inverse",
    "signed_parabolic_transform",
    "validate_algorithm_branch",
]
