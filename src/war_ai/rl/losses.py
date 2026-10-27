"""Mask-correct probability, entropy, and KL operations for structured actions."""

from __future__ import annotations

import torch
from torch import Tensor


def masked_log_softmax(
    logits: Tensor,
    legal_mask: Tensor,
    *,
    active_mask: Tensor | None = None,
) -> Tensor:
    legal, active = _validated_distribution(logits, legal_mask, active_mask)
    fallback = torch.zeros_like(legal)
    fallback[..., 0] = True
    safe_legal = torch.where(active.unsqueeze(-1), legal, fallback)
    safe_logits = torch.where(
        active.unsqueeze(-1),
        logits,
        torch.zeros_like(logits),
    )
    masked_logits = safe_logits.masked_fill(~safe_legal, -torch.inf)
    log_probabilities = torch.log_softmax(masked_logits, dim=-1)
    return log_probabilities.masked_fill(~active.unsqueeze(-1), 0.0)


def joint_action_log_prob(
    logits: Tensor,
    actions: Tensor,
    legal_mask: Tensor,
    token_mask: Tensor,
) -> Tensor:
    """Return the exact sum of conditional log-probabilities for each action."""
    if logits.ndim < 2:
        raise ValueError("Structured action logits require token and vocabulary axes.")
    if actions.shape != logits.shape[:-1]:
        raise ValueError("Action-token shape must match the logits' token axes.")
    if actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("Action tokens must use an integer tensor dtype.")
    if token_mask.shape != actions.shape:
        raise ValueError("Token mask shape must match the action-token shape.")

    active = _binary_mask(token_mask, "token_mask")
    log_probabilities = masked_log_softmax(
        logits,
        legal_mask,
        active_mask=active,
    )
    action_indices = actions.to(dtype=torch.long)
    vocabulary_size = logits.shape[-1]
    invalid_indices = (action_indices < 0) | (action_indices >= vocabulary_size)
    if torch.any(active & invalid_indices).item():
        raise ValueError("An active action token is outside the vocabulary.")
    safe_indices = action_indices.clamp(min=0, max=vocabulary_size - 1)
    selected_is_legal = _binary_mask(legal_mask, "legal_mask").gather(
        -1,
        safe_indices.unsqueeze(-1),
    ).squeeze(-1)
    if torch.any(active & ~selected_is_legal).item():
        raise ValueError("An action token is masked as illegal.")
    selected_log_probabilities = log_probabilities.gather(
        -1,
        safe_indices.unsqueeze(-1),
    ).squeeze(-1)
    return torch.where(active, selected_log_probabilities, 0.0).sum(dim=-1)


def masked_categorical_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    *,
    active_mask: Tensor | None = None,
) -> Tensor:
    legal, active = _validated_distribution(logits, legal_mask, active_mask)
    log_probabilities = masked_log_softmax(
        logits,
        legal,
        active_mask=active,
    )
    valid = legal & active.unsqueeze(-1)
    safe_log_probabilities = torch.where(
        valid,
        log_probabilities,
        torch.zeros_like(log_probabilities),
    )
    probabilities = torch.where(
        valid,
        log_probabilities.exp(),
        torch.zeros_like(log_probabilities),
    )
    return -(probabilities * safe_log_probabilities).sum(dim=-1)


def joint_action_entropy(
    logits: Tensor,
    legal_mask: Tensor,
    token_mask: Tensor,
) -> Tensor:
    """Sum masked conditional entropy at the recorded action-prefix positions."""
    if logits.ndim < 2 or token_mask.shape != logits.shape[:-1]:
        raise ValueError("Token mask shape must match the logits' token axes.")
    return masked_categorical_entropy(
        logits,
        legal_mask,
        active_mask=token_mask,
    ).sum(dim=-1)


def masked_categorical_kl_divergence(
    old_logits: Tensor,
    new_logits: Tensor,
    legal_mask: Tensor,
    *,
    active_mask: Tensor | None = None,
) -> Tensor:
    if old_logits.shape != new_logits.shape:
        raise ValueError("Old and new logits must have identical shapes.")
    legal, active = _validated_distribution(old_logits, legal_mask, active_mask)
    if new_logits.device != old_logits.device:
        raise ValueError("Old and new logits must be on the same device.")

    old_log_probabilities = masked_log_softmax(
        old_logits,
        legal,
        active_mask=active,
    )
    new_log_probabilities = masked_log_softmax(
        new_logits,
        legal,
        active_mask=active,
    )
    valid = legal & active.unsqueeze(-1)
    old_probabilities = torch.where(
        valid,
        old_log_probabilities.exp(),
        torch.zeros_like(old_log_probabilities),
    )
    safe_old_log_probabilities = torch.where(
        valid,
        old_log_probabilities,
        torch.zeros_like(old_log_probabilities),
    )
    safe_new_log_probabilities = torch.where(
        valid,
        new_log_probabilities,
        torch.zeros_like(old_log_probabilities),
    )
    safe_log_ratio = safe_old_log_probabilities - safe_new_log_probabilities
    return (old_probabilities * safe_log_ratio).sum(dim=-1)


def joint_action_kl_divergence(
    old_logits: Tensor,
    new_logits: Tensor,
    legal_mask: Tensor,
    token_mask: Tensor,
) -> Tensor:
    """Sum masked conditional KL at the same recorded action prefixes."""
    if old_logits.ndim < 2 or token_mask.shape != old_logits.shape[:-1]:
        raise ValueError("Token mask shape must match the logits' token axes.")
    return masked_categorical_kl_divergence(
        old_logits,
        new_logits,
        legal_mask,
        active_mask=token_mask,
    ).sum(dim=-1)


def _validated_distribution(
    logits: Tensor,
    legal_mask: Tensor,
    active_mask: Tensor | None,
) -> tuple[Tensor, Tensor]:
    if logits.ndim < 1 or logits.shape[-1] == 0:
        raise ValueError("Logits must have a non-empty vocabulary axis.")
    if not logits.is_floating_point():
        raise TypeError("Logits must use a floating-point tensor dtype.")
    if legal_mask.shape != logits.shape:
        raise ValueError("Legal mask shape must match logits.")
    if legal_mask.device != logits.device:
        raise ValueError("Legal mask and logits must be on the same device.")

    legal = _binary_mask(legal_mask, "legal_mask")
    expected_active_shape = logits.shape[:-1]
    if active_mask is None:
        active = torch.ones(
            expected_active_shape,
            dtype=torch.bool,
            device=logits.device,
        )
    else:
        if active_mask.shape != expected_active_shape:
            raise ValueError("Active mask shape must match logits' token axes.")
        if active_mask.device != logits.device:
            raise ValueError("Active mask and logits must be on the same device.")
        active = _binary_mask(active_mask, "active_mask")
    if torch.any(active & ~legal.any(dim=-1)).item():
        raise ValueError("Every active token position must have a legal token.")
    legal_active = legal & active.unsqueeze(-1)
    if torch.any(legal_active & ~torch.isfinite(logits)).item():
        raise ValueError("Logits must be finite for every legal active token.")
    return legal, active


def _binary_mask(mask: Tensor, label: str) -> Tensor:
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


__all__ = [
    "joint_action_entropy",
    "joint_action_kl_divergence",
    "joint_action_log_prob",
    "masked_categorical_entropy",
    "masked_categorical_kl_divergence",
    "masked_log_softmax",
]
