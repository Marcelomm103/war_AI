"""Reference masked PPO components for structured WAR actions."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, TypeAlias

import torch
from numpy.typing import ArrayLike
from torch import Tensor, nn

from ..action_tokens import ACTION_CONFIRM_TOKEN, ACTION_TOKEN_VOCAB_SIZE
from .losses import (
    joint_action_entropy,
    joint_action_log_prob,
    masked_categorical_entropy,
    masked_log_softmax,
)
from .memory import RecurrentState

RecurrentMode: TypeAlias = Literal["none", "gru", "lstm"]
PrefixMask: TypeAlias = Callable[[tuple[int, ...]], Tensor | ArrayLike]


@dataclass(frozen=True)
class PPOActorCriticOutput:
    """Teacher-forced policy logits, local values, and next recurrent state."""

    token_logits: Tensor
    values: Tensor
    recurrent_state: RecurrentState | None


@dataclass(frozen=True)
class PPOActionSample:
    """A complete legal action sampled with its rollout statistics."""

    action_tokens: tuple[int, ...]
    legal_masks: Tensor
    token_mask: Tensor
    old_log_probability: Tensor
    entropy: Tensor
    value: Tensor
    recurrent_state: RecurrentState | None


@dataclass(frozen=True)
class PPOLoss:
    """PPO loss and per-update diagnostics."""

    total_loss: Tensor
    policy_loss: Tensor
    value_loss: Tensor
    entropy: Tensor
    approximate_kl: Tensor
    clip_fraction: Tensor
    importance_ratios: Tensor


class PPOActorCritic(nn.Module):
    """Local actor-critic with autoregressive, prefix-masked action tokens.

    ``observations`` are fixed-width floating-point features derived solely
    from the acting player's observation. Sequences use ``(time, batch, dim)``
    layout so their recurrent state matches :class:`PerPlayerRecurrentMemory`.
    The action-token decoder is autoregressive in every mode; ``recurrent``
    selects whether observations also carry GRU/LSTM state between decisions.
    """

    def __init__(
        self,
        observation_size: int,
        *,
        token_vocab_size: int = ACTION_TOKEN_VOCAB_SIZE,
        hidden_size: int = 128,
        recurrent: RecurrentMode = "none",
        recurrent_layers: int = 1,
        end_token_id: int = ACTION_CONFIRM_TOKEN,
    ) -> None:
        super().__init__()
        for label, size in (
            ("observation_size", observation_size),
            ("token_vocab_size", token_vocab_size),
            ("hidden_size", hidden_size),
            ("recurrent_layers", recurrent_layers),
        ):
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                raise ValueError(f"{label} must be a positive integer.")
        if recurrent not in ("none", "gru", "lstm"):
            raise ValueError("recurrent must be 'none', 'gru', or 'lstm'.")
        if (
            isinstance(end_token_id, bool)
            or not isinstance(end_token_id, int)
            or not 0 <= end_token_id < token_vocab_size
        ):
            raise ValueError("end_token_id must be inside the token vocabulary.")

        self.observation_size = observation_size
        self.token_vocab_size = token_vocab_size
        self.hidden_size = hidden_size
        self.recurrent_mode = recurrent
        self.recurrent_layers = recurrent_layers
        self.end_token_id = end_token_id

        self.observation_encoder = nn.Sequential(
            nn.Linear(observation_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self._state_gru: nn.GRU | None = None
        self._state_lstm: nn.LSTM | None = None
        if recurrent == "gru":
            self._state_gru = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=recurrent_layers,
            )
        elif recurrent == "lstm":
            self._state_lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=recurrent_layers,
            )

        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.token_embedding = nn.Embedding(token_vocab_size + 1, hidden_size)
        self.action_decoder = nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.policy_head = nn.Linear(hidden_size, token_vocab_size)

    def forward(
        self,
        observations: Tensor,
        action_tokens: Tensor,
        *,
        recurrent_state: RecurrentState | None = None,
        episode_start_mask: Tensor | None = None,
    ) -> PPOActorCriticOutput:
        """Evaluate recorded actions using teacher forcing.

        ``observations`` and ``action_tokens`` may be ``(batch, ...)`` for a
        single decision per batch item or ``(time, batch, ...)`` for recurrent
        rollouts. Padded action tokens must still contain in-vocabulary IDs;
        exclude padding with the ``token_mask`` passed to the PPO loss.
        """
        if observations.ndim not in (2, 3):
            raise ValueError("Observations must have batch or time/batch axes.")
        is_sequence = observations.ndim == 3
        expected_token_rank = 3 if is_sequence else 2
        if action_tokens.ndim != expected_token_rank:
            raise ValueError("Action-token axes must match observation batch axes.")
        if action_tokens.shape[:-1] != observations.shape[:-1]:
            raise ValueError("Action-token batch axes must match observations.")
        if action_tokens.shape[-1] < 1:
            raise ValueError("Each recorded action must contain at least one token.")
        if action_tokens.dtype not in (
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise TypeError("Action tokens must use an integer tensor dtype.")
        if torch.any(
            (action_tokens < 0) | (action_tokens >= self.token_vocab_size)
        ).item():
            raise ValueError("An action token is outside the configured vocabulary.")
        if action_tokens.device != observations.device:
            raise ValueError("Action tokens and observations must share a device.")

        single_step = not is_sequence
        observation_sequence = (
            observations.unsqueeze(0) if single_step else observations
        )
        start_mask = self._episode_start_mask(
            episode_start_mask,
            single_step=single_step,
            time_size=observation_sequence.shape[0],
            batch_size=observation_sequence.shape[1],
            device=observations.device,
        )
        contexts, next_recurrent_state = self._encode_observations(
            observation_sequence,
            recurrent_state=recurrent_state,
            episode_start_mask=start_mask,
        )

        time_size, batch_size = contexts.shape[:2]
        action_length = action_tokens.shape[-1]
        flat_contexts = contexts.reshape(-1, self.hidden_size)
        flat_actions = action_tokens.to(dtype=torch.long).reshape(
            time_size * batch_size,
            action_length,
        )
        decoder_tokens = torch.full_like(flat_actions, self.token_vocab_size)
        decoder_tokens[:, 1:] = flat_actions[:, :-1]
        token_embeddings = self.token_embedding(decoder_tokens)
        repeated_contexts = flat_contexts.unsqueeze(1).expand(
            -1,
            action_length,
            -1,
        )
        decoder_inputs = torch.cat(
            (token_embeddings, repeated_contexts),
            dim=-1,
        )
        decoder_state = flat_contexts.unsqueeze(0)
        decoded, _ = self.action_decoder(decoder_inputs, decoder_state)
        flat_logits = self.policy_head(decoded)
        values = self.value_head(contexts).squeeze(-1)

        if single_step:
            token_logits = flat_logits.reshape(
                batch_size,
                action_length,
                self.token_vocab_size,
            )
            values = values.squeeze(0)
        else:
            token_logits = flat_logits.reshape(
                time_size,
                batch_size,
                action_length,
                self.token_vocab_size,
            )
        return PPOActorCriticOutput(
            token_logits=token_logits,
            values=values,
            recurrent_state=next_recurrent_state,
        )

    @torch.no_grad()
    def sample_action(
        self,
        observation: Tensor,
        prefix_mask: PrefixMask,
        *,
        recurrent_state: RecurrentState | None = None,
        episode_start: bool = False,
        deterministic: bool = False,
        generator: torch.Generator | None = None,
        max_action_tokens: int | None = None,
    ) -> PPOActionSample:
        """Sample one complete action, querying a legal mask for every prefix.

        ``max_action_tokens`` is an optional caller-selected safety guard; by
        default, action length is limited only by the legal-prefix provider.
        """
        if observation.ndim == 1:
            batch_observation = observation.unsqueeze(0)
        elif observation.ndim == 2 and observation.shape[0] == 1:
            batch_observation = observation
        else:
            raise ValueError("Sampling requires exactly one observation.")
        if (
            max_action_tokens is not None
            and (
                isinstance(max_action_tokens, bool)
                or not isinstance(max_action_tokens, int)
                or max_action_tokens < 1
            )
        ):
            raise ValueError("max_action_tokens must be a positive integer.")
        if not isinstance(episode_start, bool):
            raise TypeError("episode_start must be a boolean.")

        start_mask = torch.tensor(
            [[episode_start]],
            dtype=torch.bool,
            device=batch_observation.device,
        )
        contexts, next_recurrent_state = self._encode_observations(
            batch_observation.unsqueeze(0),
            recurrent_state=recurrent_state,
            episode_start_mask=start_mask,
        )
        context = contexts[0]
        value = self.value_head(context).squeeze(-1)[0]
        decoder_state = context.unsqueeze(0)
        previous_token = torch.full(
            (1,),
            self.token_vocab_size,
            dtype=torch.long,
            device=observation.device,
        )
        tokens: list[int] = []
        legal_masks: list[Tensor] = []
        selected_log_probabilities: list[Tensor] = []
        entropies: list[Tensor] = []

        while True:
            if (
                max_action_tokens is not None
                and len(tokens) >= max_action_tokens
            ):
                raise RuntimeError(
                    "The legal-prefix provider did not produce an end token "
                    "within max_action_tokens."
                )
            legal_mask = torch.as_tensor(
                prefix_mask(tuple(tokens)),
                device=observation.device,
            )
            if legal_mask.shape != (self.token_vocab_size,):
                raise ValueError("A prefix mask must match the token vocabulary.")
            legal_mask = _binary_mask(
                legal_mask,
                "prefix_mask",
                device=observation.device,
            )

            token_embedding = self.token_embedding(previous_token)
            decoder_input = torch.cat(
                (token_embedding, context),
                dim=-1,
            ).unsqueeze(1)
            decoded, decoder_state = self.action_decoder(
                decoder_input,
                decoder_state,
            )
            logits = self.policy_head(decoded[:, 0, :])
            log_probabilities = masked_log_softmax(
                logits,
                legal_mask.unsqueeze(0),
            )
            if deterministic:
                next_token = log_probabilities.argmax(dim=-1)
            else:
                next_token = torch.multinomial(
                    log_probabilities.exp(),
                    num_samples=1,
                    generator=generator,
                ).squeeze(-1)
            selected_log_probabilities.append(
                log_probabilities.gather(
                    -1,
                    next_token.unsqueeze(-1),
                ).squeeze(-1)[0]
            )
            entropies.append(
                masked_categorical_entropy(
                    logits,
                    legal_mask.unsqueeze(0),
                )[0]
            )
            legal_masks.append(legal_mask)
            token = int(next_token.item())
            tokens.append(token)
            previous_token = next_token
            if token == self.end_token_id:
                break

        token_mask = torch.ones(
            (1, len(tokens)),
            dtype=torch.bool,
            device=observation.device,
        )
        return PPOActionSample(
            action_tokens=tuple(tokens),
            legal_masks=torch.stack(legal_masks).unsqueeze(0),
            token_mask=token_mask,
            old_log_probability=torch.stack(selected_log_probabilities).sum(),
            entropy=torch.stack(entropies).sum(),
            value=value,
            recurrent_state=next_recurrent_state,
        )

    def _encode_observations(
        self,
        observation_sequence: Tensor,
        *,
        recurrent_state: RecurrentState | None,
        episode_start_mask: Tensor | None,
    ) -> tuple[Tensor, RecurrentState | None]:
        if observation_sequence.ndim != 3:
            raise ValueError("Internal observations must have time/batch axes.")
        if observation_sequence.shape[-1] != self.observation_size:
            raise ValueError("Observation features do not match observation_size.")
        if not observation_sequence.is_floating_point():
            raise TypeError("Observation features must be floating point.")
        if not torch.isfinite(observation_sequence).all().item():
            raise ValueError("Observation features must be finite.")

        encoded = self.observation_encoder(observation_sequence)
        if self.recurrent_mode == "none":
            if recurrent_state is not None:
                raise ValueError("Feedforward policies do not accept recurrent state.")
            return encoded, None

        state = self._validate_recurrent_state(
            recurrent_state,
            batch_size=observation_sequence.shape[1],
            device=encoded.device,
            dtype=encoded.dtype,
        )
        if episode_start_mask is None:
            return self._run_recurrent_encoder(encoded, state)

        recurrent_outputs: list[Tensor] = []
        current_state = state
        for time_index in range(encoded.shape[0]):
            current_state = _reset_recurrent_state(
                current_state,
                episode_start_mask[time_index],
            )
            recurrent_output, current_state = self._run_recurrent_encoder(
                encoded[time_index : time_index + 1],
                current_state,
            )
            recurrent_outputs.append(recurrent_output)
        return torch.cat(recurrent_outputs, dim=0), current_state

    def _run_recurrent_encoder(
        self,
        encoded: Tensor,
        recurrent_state: RecurrentState | None,
    ) -> tuple[Tensor, RecurrentState]:
        if self.recurrent_mode == "gru":
            if self._state_gru is None:
                raise RuntimeError("The configured GRU encoder is unavailable.")
            output, next_state = self._state_gru(encoded, recurrent_state)
            return output, next_state
        if self._state_lstm is None:
            raise RuntimeError("The configured LSTM encoder is unavailable.")
        if recurrent_state is not None and not isinstance(recurrent_state, tuple):
            raise TypeError("LSTM policies require hidden and cell recurrent state.")
        output, next_state = self._state_lstm(encoded, recurrent_state)
        return output, next_state

    def _validate_recurrent_state(
        self,
        recurrent_state: RecurrentState | None,
        *,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> RecurrentState | None:
        if recurrent_state is None:
            return None
        expected_shape = (self.recurrent_layers, batch_size, self.hidden_size)
        states = (
            recurrent_state
            if isinstance(recurrent_state, tuple)
            else (recurrent_state,)
        )
        if self.recurrent_mode == "gru" and isinstance(recurrent_state, tuple):
            raise TypeError("GRU policies accept hidden state only.")
        if self.recurrent_mode == "lstm" and not isinstance(recurrent_state, tuple):
            raise TypeError("LSTM policies require hidden and cell recurrent state.")
        if len(states) != (2 if self.recurrent_mode == "lstm" else 1):
            raise ValueError("Recurrent state does not match the configured policy.")
        for state in states:
            if state.shape != expected_shape:
                raise ValueError(
                    f"Recurrent state must have shape {expected_shape}, "
                    f"got {tuple(state.shape)}."
                )
            if state.device != device or state.dtype != dtype:
                raise ValueError(
                    "Recurrent state must match observation device and dtype."
                )
        return recurrent_state

    @staticmethod
    def _episode_start_mask(
        mask: Tensor | None,
        *,
        single_step: bool,
        time_size: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor | None:
        if mask is None:
            return None
        expected_shape = (batch_size,) if single_step else (time_size, batch_size)
        if mask.shape != expected_shape:
            raise ValueError(
                f"episode_start_mask must have shape {expected_shape}."
            )
        validated = _binary_mask(
            mask,
            "episode_start_mask",
            device=device,
        )
        return validated.unsqueeze(0) if single_step else validated


def compute_masked_ppo_loss(
    *,
    new_logits: Tensor,
    actions: Tensor,
    legal_masks: Tensor,
    token_mask: Tensor,
    old_log_probabilities: Tensor,
    advantages: Tensor,
    value_predictions: Tensor,
    returns: Tensor,
    clip_range: float = 0.2,
    value_coefficient: float = 0.5,
    entropy_coefficient: float = 0.01,
    sample_mask: Tensor | None = None,
    normalize_advs: bool = True,
    normalization_epsilon: float = 1e-8,
) -> PPOLoss:
    """Compute clipped PPO losses from legal-prefix token distributions."""
    _validate_probability(clip_range, "clip_range", allow_zero=False)
    if clip_range >= 1.0:
        raise ValueError("clip_range must be less than one.")
    _validate_nonnegative(value_coefficient, "value_coefficient")
    _validate_nonnegative(entropy_coefficient, "entropy_coefficient")
    _validate_probability(
        normalization_epsilon,
        "normalization_epsilon",
        allow_zero=False,
    )
    if new_logits.ndim < 3:
        raise ValueError("PPO logits require sample, token, and vocabulary axes.")
    sample_shape = new_logits.shape[:-2]
    if not sample_shape or any(size < 1 for size in sample_shape):
        raise ValueError("PPO logits must contain at least one sample.")
    if actions.shape != new_logits.shape[:-1]:
        raise ValueError("Action-token shape must match the PPO logits.")
    if legal_masks.shape != new_logits.shape:
        raise ValueError("Legal masks must match the PPO logits.")
    if token_mask.shape != actions.shape:
        raise ValueError("Token mask shape must match the action-token shape.")
    for label, tensor in (
        ("old_log_probabilities", old_log_probabilities),
        ("advantages", advantages),
        ("value_predictions", value_predictions),
        ("returns", returns),
    ):
        if tensor.shape != sample_shape:
            raise ValueError(f"{label} shape must match PPO sample axes.")
        if tensor.device != new_logits.device:
            raise ValueError(f"{label} and new_logits must share a device.")
    if actions.device != new_logits.device or legal_masks.device != new_logits.device:
        raise ValueError("Actions, masks, and logits must share a device.")
    if token_mask.device != new_logits.device:
        raise ValueError("Token mask and logits must share a device.")
    if not isinstance(normalize_advs, bool):
        raise TypeError("normalize_advs must be a boolean.")

    valid_samples = _binary_mask(
        sample_mask,
        "sample_mask",
        device=new_logits.device,
        expected_shape=sample_shape,
    )
    if not valid_samples.any().item():
        raise ValueError("At least one PPO sample must be active.")
    for label, tensor in (
        ("old_log_probabilities", old_log_probabilities),
        ("advantages", advantages),
        ("value_predictions", value_predictions),
        ("returns", returns),
    ):
        if not torch.isfinite(tensor[valid_samples]).all().item():
            raise ValueError(f"{label} must be finite for active samples.")

    effective_token_mask = _binary_mask(
        token_mask,
        "token_mask",
        device=new_logits.device,
        expected_shape=actions.shape,
    ) & valid_samples.unsqueeze(-1)
    new_log_probabilities = joint_action_log_prob(
        new_logits,
        actions,
        legal_masks,
        effective_token_mask,
    )
    action_entropies = joint_action_entropy(
        new_logits,
        legal_masks,
        effective_token_mask,
    )
    if not torch.isfinite(new_log_probabilities[valid_samples]).all().item():
        raise FloatingPointError("New policy log-probabilities must be finite.")
    if not torch.isfinite(action_entropies[valid_samples]).all().item():
        raise FloatingPointError("New policy entropies must be finite.")

    if normalize_advs:
        policy_advantages = normalize_advantages(
            advantages,
            sample_mask=valid_samples,
            epsilon=normalization_epsilon,
        )
    else:
        policy_advantages = torch.where(
            valid_samples,
            advantages.detach(),
            torch.zeros_like(advantages),
        )
    safe_old_log_probabilities = torch.where(
        valid_samples,
        old_log_probabilities.detach(),
        torch.zeros_like(old_log_probabilities),
    )
    safe_new_log_probabilities = torch.where(
        valid_samples,
        new_log_probabilities,
        torch.zeros_like(new_log_probabilities),
    )
    log_ratios = safe_new_log_probabilities - safe_old_log_probabilities
    raw_ratios = log_ratios.exp()
    if not torch.isfinite(raw_ratios[valid_samples]).all().item():
        raise FloatingPointError("PPO importance ratios overflowed.")
    importance_ratios = torch.where(
        valid_samples,
        raw_ratios,
        torch.ones_like(raw_ratios),
    )
    clipped_ratios = importance_ratios.clamp(
        min=1.0 - clip_range,
        max=1.0 + clip_range,
    )
    surrogate = torch.minimum(
        importance_ratios * policy_advantages,
        clipped_ratios * policy_advantages,
    )
    policy_loss = -_masked_mean(surrogate, valid_samples)

    safe_values = torch.where(
        valid_samples,
        value_predictions,
        torch.zeros_like(value_predictions),
    )
    safe_returns = torch.where(
        valid_samples,
        returns.detach(),
        torch.zeros_like(returns),
    )
    value_loss = 0.5 * _masked_mean(
        (safe_values - safe_returns).square(),
        valid_samples,
    )
    entropy = _masked_mean(action_entropies, valid_samples)
    approximate_kl = _masked_mean(
        safe_old_log_probabilities - safe_new_log_probabilities,
        valid_samples,
    )
    clip_fraction = _masked_mean(
        (importance_ratios - 1.0).abs().gt(clip_range).to(new_logits.dtype),
        valid_samples,
    )
    total_loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * entropy
    )
    return PPOLoss(
        total_loss=total_loss,
        policy_loss=policy_loss,
        value_loss=value_loss,
        entropy=entropy,
        approximate_kl=approximate_kl,
        clip_fraction=clip_fraction,
        importance_ratios=importance_ratios,
    )


def normalize_advantages(
    advantages: Tensor,
    *,
    sample_mask: Tensor | None = None,
    epsilon: float = 1e-8,
) -> Tensor:
    """Normalize active advantages with population variance; zero padding."""
    _validate_probability(epsilon, "epsilon", allow_zero=False)
    if advantages.ndim < 1:
        raise ValueError("Advantages must have at least one sample axis.")
    if not advantages.is_floating_point():
        raise TypeError("Advantages must be floating point.")
    mask = _binary_mask(
        sample_mask,
        "sample_mask",
        device=advantages.device,
        expected_shape=advantages.shape,
    )
    if not mask.any().item():
        raise ValueError("At least one advantage sample must be active.")
    active_advantages = advantages[mask]
    if not torch.isfinite(active_advantages).all().item():
        raise ValueError("Active advantages must be finite.")
    active_advantages = active_advantages.detach()
    mean = active_advantages.mean()
    standard_deviation = active_advantages.std(unbiased=False)
    normalized = (advantages.detach() - mean) / (standard_deviation + epsilon)
    return torch.where(mask, normalized, torch.zeros_like(normalized))


def compute_gae(
    rewards: Tensor,
    values: Tensor,
    next_values: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    *,
    gamma: float = 1.0,
    gae_lambda: float = 0.95,
    valid_mask: Tensor | None = None,
) -> tuple[Tensor, Tensor]:
    """Compute time-major GAE, bootstrapping truncations but not terminations."""
    _validate_unit_interval(gamma, "gamma")
    _validate_unit_interval(gae_lambda, "gae_lambda")
    if rewards.ndim < 1 or rewards.shape[0] < 1:
        raise ValueError("Rewards must contain at least one time step.")
    if not rewards.is_floating_point():
        raise TypeError("Rewards must be floating point.")
    for label, tensor in (
        ("values", values),
        ("next_values", next_values),
    ):
        if tensor.shape != rewards.shape:
            raise ValueError(f"{label} shape must match rewards.")
        if tensor.device != rewards.device or tensor.dtype != rewards.dtype:
            raise ValueError(f"{label} must match rewards device and dtype.")
    if terminated.shape != rewards.shape or truncated.shape != rewards.shape:
        raise ValueError("Termination flags must match the rewards shape.")
    if terminated.device != rewards.device or truncated.device != rewards.device:
        raise ValueError("Termination flags and rewards must share a device.")

    active = _binary_mask(
        valid_mask,
        "valid_mask",
        device=rewards.device,
        expected_shape=rewards.shape,
    )
    terminal = _binary_mask(
        terminated,
        "terminated",
        device=rewards.device,
        expected_shape=rewards.shape,
    )
    truncation = _binary_mask(
        truncated,
        "truncated",
        device=rewards.device,
        expected_shape=rewards.shape,
    )
    if not active.any().item():
        raise ValueError("At least one GAE transition must be active.")
    for label, tensor in (
        ("rewards", rewards),
        ("values", values),
        ("next_values", next_values),
    ):
        if not torch.isfinite(tensor[active]).all().item():
            raise ValueError(f"{label} must be finite for active transitions.")

    with torch.no_grad():
        detached_rewards = rewards.detach()
        detached_values = values.detach()
        detached_next_values = next_values.detach()
        advantages = torch.zeros_like(rewards)
        advantage_carry = torch.zeros_like(rewards[0])
        for time_index in range(rewards.shape[0] - 1, -1, -1):
            transition_valid = active[time_index]
            can_bootstrap = (~terminal[time_index]).to(rewards.dtype)
            can_continue = (~(terminal[time_index] | truncation[time_index])).to(
                rewards.dtype
            )
            delta = (
                detached_rewards[time_index]
                + gamma * can_bootstrap * detached_next_values[time_index]
                - detached_values[time_index]
            )
            advantage_carry = (
                delta + gamma * gae_lambda * can_continue * advantage_carry
            )
            advantage_carry = torch.where(
                transition_valid,
                advantage_carry,
                torch.zeros_like(advantage_carry),
            )
            advantages[time_index] = advantage_carry
        returns = torch.where(
            active,
            advantages + detached_values,
            torch.zeros_like(values),
        )
    return advantages, returns


def _masked_mean(values: Tensor, mask: Tensor) -> Tensor:
    if values.shape != mask.shape:
        raise ValueError("Values and sample mask must have identical shapes.")
    active_count = mask.sum()
    if active_count.item() == 0:
        raise ValueError("At least one sample must be active.")
    return torch.where(mask, values, torch.zeros_like(values)).sum() / active_count


def _binary_mask(
    mask: Tensor | None,
    label: str,
    *,
    device: torch.device,
    expected_shape: torch.Size | tuple[int, ...] | None = None,
) -> Tensor:
    if mask is None:
        if expected_shape is None:
            raise ValueError(f"{label} is required.")
        return torch.ones(expected_shape, dtype=torch.bool, device=device)
    if expected_shape is not None and mask.shape != expected_shape:
        raise ValueError(f"{label} must have shape {tuple(expected_shape)}.")
    if mask.device != device:
        raise ValueError(f"{label} and its data must share a device.")
    if mask.dtype == torch.bool:
        return mask
    if mask.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.float16,
        torch.float32,
        torch.float64,
        torch.bfloat16,
    ):
        raise TypeError(f"{label} must be a boolean or numeric binary tensor.")
    if torch.any((mask != 0) & (mask != 1)).item():
        raise ValueError(f"{label} may only contain zero and one.")
    return mask.to(dtype=torch.bool)


def _reset_recurrent_state(
    recurrent_state: RecurrentState | None,
    reset_mask: Tensor,
) -> RecurrentState | None:
    if recurrent_state is None or not reset_mask.any().item():
        return recurrent_state
    keep = (~reset_mask).to(dtype=reset_mask.dtype).to(
        dtype=(
            recurrent_state[0].dtype
            if isinstance(recurrent_state, tuple)
            else recurrent_state.dtype
        )
    )
    keep = keep.reshape(1, -1, 1)
    if isinstance(recurrent_state, tuple):
        return (
            recurrent_state[0] * keep,
            recurrent_state[1] * keep,
        )
    return recurrent_state * keep


def _validate_probability(
    value: float,
    label: str,
    *,
    allow_zero: bool,
) -> None:
    if isinstance(value, bool) or not math.isfinite(value):
        raise ValueError(f"{label} must be a finite number.")
    if value < 0.0 or (not allow_zero and value == 0.0):
        requirement = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{label} must be {requirement}.")


def _validate_nonnegative(value: float, label: str) -> None:
    _validate_probability(value, label, allow_zero=True)


def _validate_unit_interval(value: float, label: str) -> None:
    if isinstance(value, bool) or not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{label} must be finite and between zero and one.")


__all__ = [
    "PPOActionSample",
    "PPOActorCritic",
    "PPOActorCriticOutput",
    "PPOLoss",
    "compute_gae",
    "compute_masked_ppo_loss",
    "normalize_advantages",
]
