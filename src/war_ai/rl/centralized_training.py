"""Centralized training with decentralized execution (CTDE) for WAR 1 (Phase 7.4)."""

from __future__ import annotations

import hashlib
import io
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

import torch
from numpy.random import Generator
from torch import Tensor, nn
from torch.nn import functional as F
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import FullGameState

from ..arena import ArenaPolicy, FrozenPolicySpec
from ..envs.observations import MAX_PLAYERS
from ..envs.war1_aec import War1AECObservation
from .agent import FULL_GAME_INTENT_END_TOKEN, FULL_GAME_INTENT_VOCAB_SIZE
from .architectures import (
    NUM_CARDS,
    NUM_OBJECTIVES,
    ArchitectureConfig,
    StructuredArchitectureInputs,
    WarArchitectureModel,
    extract_architecture_inputs,
)
from .losses import joint_action_log_prob
from .ppo import compute_gae, compute_masked_ppo_loss

CriticMode: TypeAlias = Literal["local", "privileged"]

_OBJECTIVE_INDEX_BY_ID = {
    objective.objective_id: index
    for index, objective in enumerate(WAR1_BOARD_SPEC.objectives)
}
_CARD_INDEX_BY_ID = {
    card_id: index
    for index, card_id in enumerate(WAR1_BOARD_SPEC.territory_deck.card_ids)
}
PRIVILEGED_SECRET_FEATURE_DIM = MAX_PLAYERS * (NUM_OBJECTIVES + NUM_CARDS + 1)


@dataclass(frozen=True)
class PrivilegedTableSecrets:
    """Training-only hidden opponent objectives and cards (never passed to actor)."""

    opponent_objective_indices: Tensor
    opponent_card_targets: Tensor
    opponent_seat_mask: Tensor

    def __post_init__(self) -> None:
        if self.opponent_objective_indices.ndim != 2 or (
            self.opponent_objective_indices.shape[1] != MAX_PLAYERS
        ):
            raise ValueError(
                "opponent_objective_indices must have shape (batch, MAX_PLAYERS)."
            )
        batch_size = self.opponent_objective_indices.shape[0]
        if self.opponent_card_targets.shape != (
            batch_size,
            MAX_PLAYERS,
            NUM_CARDS,
        ):
            raise ValueError(
                "opponent_card_targets must have shape "
                f"(batch, {MAX_PLAYERS}, {NUM_CARDS})."
            )
        if self.opponent_seat_mask.shape != (batch_size, MAX_PLAYERS):
            raise ValueError(
                f"opponent_seat_mask must have shape (batch, {MAX_PLAYERS})."
            )
        if self.opponent_seat_mask.dtype != torch.bool:
            raise TypeError("opponent_seat_mask must be a boolean tensor.")

    @property
    def batch_size(self) -> int:
        return int(self.opponent_objective_indices.shape[0])

    def as_feature_tensor(self) -> Tensor:
        """Flatten masked opponent secrets into a training-only critic vector."""
        mask_float = self.opponent_seat_mask.to(dtype=torch.float32).unsqueeze(-1)
        one_hot_objectives = F.one_hot(
            self.opponent_objective_indices.clamp(min=0, max=NUM_OBJECTIVES - 1),
            num_classes=NUM_OBJECTIVES,
        ).to(dtype=torch.float32)
        masked_objectives = one_hot_objectives * mask_float
        masked_cards = self.opponent_card_targets * mask_float
        combined = torch.cat(
            (masked_objectives, masked_cards, mask_float),
            dim=-1,
        )
        return combined.reshape(self.batch_size, PRIVILEGED_SECRET_FEATURE_DIM)


@dataclass(frozen=True)
class CentralizedTrainingLoss:
    """Loss breakdown for centralized training with secret-only auxiliary terms."""

    total_loss: Tensor
    policy_loss: Tensor
    critic_value_loss: Tensor
    local_value_loss: Tensor
    belief_objective_loss: Tensor
    belief_card_loss: Tensor
    entropy: Tensor


@dataclass(frozen=True)
class CounterfactualNonInterferenceReport:
    """Proof that mutating opponent secrets does not affect actor decisions."""

    actor_logits_identical: bool
    local_values_identical: bool
    selected_actions_identical: bool
    max_actor_logit_abs_diff: float
    max_local_value_abs_diff: float
    privileged_critic_value_abs_diff: float


@dataclass(frozen=True)
class ProductionPolicyArtifact:
    """Decentralized production artifact stripped of privileged training weights."""

    policy_id: str
    architecture_config: ArchitectureConfig
    state_dict_bytes: bytes
    checkpoint_sha256: str
    excluded_training_parameter_keys: tuple[str, ...]
    policy_spec: FrozenPolicySpec


def extract_privileged_table_secrets(
    full_states: FullGameState | Sequence[FullGameState],
    *,
    focal_player_id: str,
    device: torch.device | str = "cpu",
) -> PrivilegedTableSecrets:
    """Extract training-only opponent secret labels from full engine states."""
    states = (
        [full_states]
        if isinstance(full_states, FullGameState)
        else list(full_states)
    )
    if not states:
        raise ValueError("At least one FullGameState is required.")

    batch_size = len(states)
    objective_indices = torch.zeros(
        (batch_size, MAX_PLAYERS),
        dtype=torch.long,
        device=device,
    )
    card_targets = torch.zeros(
        (batch_size, MAX_PLAYERS, NUM_CARDS),
        dtype=torch.float32,
        device=device,
    )
    seat_mask = torch.zeros(
        (batch_size, MAX_PLAYERS),
        dtype=torch.bool,
        device=device,
    )

    for batch_idx, state in enumerate(states):
        if not any(p.public.player_id == focal_player_id for p in state.players):
            raise ValueError(
                f"Focal player {focal_player_id!r} not found in FullGameState."
            )
        for seat_idx, player in enumerate(state.players):
            if seat_idx >= MAX_PLAYERS or player.public.player_id == focal_player_id:
                continue
            seat_mask[batch_idx, seat_idx] = True
            private = player.private
            effective_obj = (
                private.objective_fallback_id or private.objective_id or ""
            )
            objective_indices[batch_idx, seat_idx] = _OBJECTIVE_INDEX_BY_ID.get(
                effective_obj,
                0,
            )
            for card_id in private.hand_card_ids:
                card_idx = _CARD_INDEX_BY_ID.get(card_id)
                if card_idx is not None:
                    card_targets[batch_idx, seat_idx, card_idx] = 1.0

    return PrivilegedTableSecrets(
        opponent_objective_indices=objective_indices,
        opponent_card_targets=card_targets,
        opponent_seat_mask=seat_mask,
    )


class PrivilegedCritic(nn.Module):
    """Centralized training critic conditioned on player context and table secrets."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        self.secret_encoder = nn.Sequential(
            nn.Linear(PRIVILEGED_SECRET_FEATURE_DIM, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )
        self.value_network = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        player_context: Tensor,
        secrets: PrivilegedTableSecrets,
    ) -> Tensor:
        secret_features = secrets.as_feature_tensor().to(
            device=player_context.device,
            dtype=player_context.dtype,
        )
        encoded_secrets = self.secret_encoder(secret_features)
        combined = torch.cat((player_context, encoded_secrets), dim=-1)
        return cast(Tensor, self.value_network(combined).squeeze(-1))


class CentralizedTrainingHarness(nn.Module):
    """Training harness separating the player-only actor from the privileged critic."""

    def __init__(
        self,
        config: ArchitectureConfig = ArchitectureConfig(),
        *,
        critic_mode: CriticMode = "privileged",
    ) -> None:
        super().__init__()
        if critic_mode not in ("local", "privileged"):
            raise ValueError("critic_mode must be 'local' or 'privileged'.")
        self.config = config
        self.critic_mode = critic_mode
        self.actor_model = WarArchitectureModel(config)
        self.privileged_critic: PrivilegedCritic | None = (
            PrivilegedCritic(config.hidden_size)
            if critic_mode == "privileged"
            else None
        )

    def evaluate_critic_values(
        self,
        player_context: Tensor,
        secrets: PrivilegedTableSecrets | None = None,
    ) -> Tensor:
        """Compute critic values using local or privileged state depending on mode."""
        if self.critic_mode == "local":
            return cast(Tensor, self.actor_model.value_head(player_context).squeeze(-1))
        if secrets is None:
            raise ValueError(
                "Privileged critic mode requires PrivilegedTableSecrets during training."
            )
        assert self.privileged_critic is not None
        return cast(Tensor, self.privileged_critic(player_context, secrets))


def compute_secret_belief_losses(
    *,
    opponent_objective_logits: Tensor,
    opponent_card_logits: Tensor,
    secrets: PrivilegedTableSecrets,
) -> tuple[Tensor, Tensor]:
    """Compute supervised training losses for opponent secret objectives and cards."""
    if not torch.any(secrets.opponent_seat_mask).item():
        raise ValueError("PrivilegedTableSecrets must include at least one opponent seat.")
    active_obj_logits = opponent_objective_logits[secrets.opponent_seat_mask]
    active_obj_targets = secrets.opponent_objective_indices[
        secrets.opponent_seat_mask
    ]
    objective_loss = F.cross_entropy(active_obj_logits, active_obj_targets)

    active_card_logits = opponent_card_logits[secrets.opponent_seat_mask]
    active_card_targets = secrets.opponent_card_targets[
        secrets.opponent_seat_mask
    ]
    card_loss = F.binary_cross_entropy_with_logits(
        active_card_logits,
        active_card_targets,
    )
    return objective_loss, card_loss


def compute_centralized_training_loss(
    harness: CentralizedTrainingHarness,
    *,
    inputs: StructuredArchitectureInputs,
    actions: Tensor,
    legal_masks: Tensor,
    token_mask: Tensor,
    old_log_probabilities: Tensor,
    rewards: Tensor,
    terminated: Tensor,
    truncated: Tensor,
    secrets: PrivilegedTableSecrets | None = None,
    belief_loss_coefficient: float = 0.2,
) -> CentralizedTrainingLoss:
    """Compute CTDE loss using privileged critic and secret labels only in training."""
    actor_out = harness.actor_model(inputs, actions)
    critic_values = harness.evaluate_critic_values(
        actor_out.latent_context,
        secrets=secrets,
    )
    with torch.no_grad():
        advantages, returns = compute_gae(
            rewards=rewards.unsqueeze(0),
            values=critic_values.detach().unsqueeze(0),
            next_values=torch.zeros_like(critic_values).unsqueeze(0),
            terminated=terminated.unsqueeze(0),
            truncated=truncated.unsqueeze(0),
        )
    ppo = compute_masked_ppo_loss(
        new_logits=actor_out.token_logits,
        actions=actions,
        legal_masks=legal_masks,
        token_mask=token_mask,
        old_log_probabilities=old_log_probabilities,
        advantages=advantages[0],
        value_predictions=critic_values,
        returns=returns[0],
    )
    local_value_loss = 0.5 * F.mse_loss(actor_out.values, returns[0].detach())

    belief_obj_loss = torch.tensor(0.0, device=actor_out.values.device)
    belief_card_loss = torch.tensor(0.0, device=actor_out.values.device)
    if (
        secrets is not None
        and actor_out.opponent_objective_logits is not None
        and actor_out.opponent_card_logits is not None
    ):
        belief_obj_loss, belief_card_loss = compute_secret_belief_losses(
            opponent_objective_logits=actor_out.opponent_objective_logits,
            opponent_card_logits=actor_out.opponent_card_logits,
            secrets=secrets,
        )

    total_loss = (
        ppo.total_loss
        + 0.25 * local_value_loss
        + belief_loss_coefficient * (belief_obj_loss + belief_card_loss)
    )
    return CentralizedTrainingLoss(
        total_loss=total_loss,
        policy_loss=ppo.policy_loss,
        critic_value_loss=ppo.value_loss,
        local_value_loss=local_value_loss,
        belief_objective_loss=belief_obj_loss,
        belief_card_loss=belief_card_loss,
        entropy=ppo.entropy,
    )


@torch.no_grad()
def verify_counterfactual_non_interference(
    harness: CentralizedTrainingHarness,
    observation: War1AECObservation,
    *,
    secrets_a: PrivilegedTableSecrets,
    secrets_b: PrivilegedTableSecrets,
) -> CounterfactualNonInterferenceReport:
    """Verify that counterfactual opponent secrets never alter actor outputs."""
    if harness.critic_mode != "privileged" or harness.privileged_critic is None:
        raise ValueError(
            "Counterfactual non-interference test requires a privileged critic harness."
        )
    inputs = extract_architecture_inputs(
        observation,
        max_history_events=harness.config.max_history_events,
    )
    probe_tokens = torch.tensor(
        [[0, FULL_GAME_INTENT_END_TOKEN]],
        dtype=torch.long,
    )
    out_a = harness.actor_model(inputs, probe_tokens)
    val_priv_a = harness.evaluate_critic_values(
        out_a.latent_context,
        secrets=secrets_a,
    )
    action_a, _, _, _ = harness.actor_model.select_legal_action(
        observation,
        deterministic=True,
    )

    out_b = harness.actor_model(inputs, probe_tokens)
    val_priv_b = harness.evaluate_critic_values(
        out_b.latent_context,
        secrets=secrets_b,
    )
    action_b, _, _, _ = harness.actor_model.select_legal_action(
        observation,
        deterministic=True,
    )

    logit_diff = float(
        (out_a.token_logits - out_b.token_logits).abs().max().item()
    )
    local_val_diff = float((out_a.values - out_b.values).abs().max().item())
    priv_val_diff = float((val_priv_a - val_priv_b).abs().max().item())

    return CounterfactualNonInterferenceReport(
        actor_logits_identical=(logit_diff == 0.0),
        local_values_identical=(local_val_diff == 0.0),
        selected_actions_identical=(action_a == action_b),
        max_actor_logit_abs_diff=logit_diff,
        max_local_value_abs_diff=local_val_diff,
        privileged_critic_value_abs_diff=priv_val_diff,
    )


def verify_production_artifact_isolation(
    state_dict: Mapping[str, object],
) -> None:
    """Reject production checkpoints that retain privileged training weights."""
    forbidden = [
        key
        for key in state_dict
        if key.startswith("privileged_critic") or "secret_encoder" in key
    ]
    if forbidden:
        raise ValueError(
            "Production policy artifact contains privileged training parameters: "
            f"{sorted(forbidden)}."
        )


def export_production_policy_artifact(
    harness: CentralizedTrainingHarness,
    *,
    policy_id: str,
) -> ProductionPolicyArtifact:
    """Export a decentralized production policy stripped of privileged critic state."""
    if not policy_id.strip():
        raise ValueError("policy_id must not be blank.")
    full_keys = set(harness.state_dict().keys())
    actor_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in harness.actor_model.state_dict().items()
    }
    verify_production_artifact_isolation(actor_state)
    excluded_keys = tuple(
        sorted(
            key
            for key in full_keys
            if not key.startswith("actor_model.")
        )
    )

    buffer = io.BytesIO()
    torch.save(actor_state, buffer)
    raw_bytes = buffer.getvalue()
    digest = f"sha256:{hashlib.sha256(raw_bytes).hexdigest()}"
    config = harness.config

    def factory(factory_seed: int) -> ArenaPolicy:
        del factory_seed
        loaded = torch.load(
            io.BytesIO(raw_bytes),
            map_location="cpu",
            weights_only=False,
        )
        verify_production_artifact_isolation(loaded)
        model = WarArchitectureModel(config)
        model.load_state_dict(loaded)
        model.eval()

        def policy(observation: War1AECObservation, rng: Generator) -> int:
            del rng
            action_idx, _, _, _ = model.select_legal_action(
                observation,
                deterministic=True,
            )
            return action_idx

        return policy

    spec = FrozenPolicySpec(
        policy_id=policy_id,
        factory=factory,
        checkpoint_sha256=digest,
    )
    return ProductionPolicyArtifact(
        policy_id=policy_id,
        architecture_config=config,
        state_dict_bytes=raw_bytes,
        checkpoint_sha256=digest,
        excluded_training_parameter_keys=excluded_keys,
        policy_spec=spec,
    )


def compare_local_vs_privileged_critic(
    observation: War1AECObservation,
    secrets: PrivilegedTableSecrets,
    *,
    updates: int = 8,
    seed: int = 2026,
) -> dict[CriticMode, float]:
    """Compare value-fit loss convergence for local vs privileged critics."""
    results: dict[CriticMode, float] = {}
    inputs = extract_architecture_inputs(observation, max_history_events=32)
    batch_inputs = inputs.repeat_batch(3)
    batch_secrets = PrivilegedTableSecrets(
        opponent_objective_indices=secrets.opponent_objective_indices.repeat_interleave(
            3, dim=0
        ),
        opponent_card_targets=secrets.opponent_card_targets.repeat_interleave(
            3, dim=0
        ),
        opponent_seat_mask=secrets.opponent_seat_mask.repeat_interleave(
            3, dim=0
        ),
    )
    actions = torch.tensor(
        [
            [0, FULL_GAME_INTENT_END_TOKEN],
            [1, FULL_GAME_INTENT_END_TOKEN],
            [2, FULL_GAME_INTENT_END_TOKEN],
        ],
        dtype=torch.long,
    )
    token_mask = torch.ones_like(actions, dtype=torch.bool)
    legal_masks = torch.zeros(
        (3, 2, FULL_GAME_INTENT_VOCAB_SIZE),
        dtype=torch.bool,
    )
    legal_masks[:, 0, :3] = True
    legal_masks[:, 1, FULL_GAME_INTENT_END_TOKEN] = True
    rewards = torch.tensor([1.0, 0.25, 0.0], dtype=torch.float32)
    terminated = torch.ones((3,), dtype=torch.bool)
    truncated = torch.zeros((3,), dtype=torch.bool)

    for mode in ("local", "privileged"):
        torch.manual_seed(seed)
        harness = CentralizedTrainingHarness(
            ArchitectureConfig(hidden_size=32, use_belief_head=True),
            critic_mode=mode,
        )
        optimizer = torch.optim.Adam(harness.parameters(), lr=0.01)
        final_critic_loss = 0.0
        for _ in range(updates):
            optimizer.zero_grad()
            with torch.no_grad():
                old_out = harness.actor_model(batch_inputs, actions)
                old_log_probs = joint_action_log_prob(
                    old_out.token_logits,
                    actions,
                    legal_masks,
                    token_mask,
                )
            loss = compute_centralized_training_loss(
                harness,
                inputs=batch_inputs,
                actions=actions,
                legal_masks=legal_masks,
                token_mask=token_mask,
                old_log_probabilities=old_log_probs,
                rewards=rewards,
                terminated=terminated,
                truncated=truncated,
                secrets=batch_secrets if mode == "privileged" else None,
            )
            loss.total_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            final_critic_loss = float(loss.critic_value_loss.detach().item())
        results[mode] = final_critic_loss
    return results


__all__ = [
    "CentralizedTrainingHarness",
    "CentralizedTrainingLoss",
    "CounterfactualNonInterferenceReport",
    "CriticMode",
    "PRIVILEGED_SECRET_FEATURE_DIM",
    "PrivilegedCritic",
    "PrivilegedTableSecrets",
    "ProductionPolicyArtifact",
    "compare_local_vs_privileged_critic",
    "compute_centralized_training_loss",
    "compute_secret_belief_losses",
    "export_production_policy_artifact",
    "extract_privileged_table_secrets",
    "verify_counterfactual_non_interference",
    "verify_production_artifact_isolation",
]
