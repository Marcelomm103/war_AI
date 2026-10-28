"""Modular neural architectures for WAR 1 RL (Phase 7.2)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, TypeAlias, cast

import numpy as np
import torch
from numpy.random import Generator
from numpy.typing import NDArray
from torch import Tensor, nn
from war_game_engine.board_spec import WAR1_BOARD_SPEC

from ..action_tokens import ACTION_CONFIRM_TOKEN
from ..envs.observations import (
    MAX_PLAYERS,
    PHASE_IDS,
    PUBLIC_EVENT_TYPES,
    War1ObservationEncoder,
)
from ..envs.war1_aec import War1AECObservation
from .agent import (
    FULL_GAME_INTENT_END_TOKEN,
    FULL_GAME_INTENT_VOCAB_SIZE,
    FULL_GAME_OBSERVATION_SIZE,
    classify_legal_actions,
    extract_player_observation_features,
)
from .experimental_axes import EncoderAxis, MemoryAxis
from .losses import joint_action_log_prob, masked_log_softmax
from .memory import RecurrentState

NUM_TERRITORIES = len(WAR1_BOARD_SPEC.map.territories)
NUM_CONTINENTS = len(WAR1_BOARD_SPEC.map.continents)
NUM_OBJECTIVES = len(WAR1_BOARD_SPEC.objectives)
NUM_CARDS = len(WAR1_BOARD_SPEC.territory_deck.card_ids)
NUM_PHASES = len(PHASE_IDS)
NUM_PUBLIC_EVENT_TYPES = len(PUBLIC_EVENT_TYPES)
TERRITORY_FEATURE_DIM = 18

ArchitectureVariantId: TypeAlias = Literal[
    "mlp_baseline",
    "message_passing_gnn",
    "graph_transformer",
    "recurrent_gru",
    "recurrent_lstm",
    "causal_event_transformer",
    "phase_conditioned_policy_value_belief",
]

_ENCODER_3P = War1ObservationEncoder(3)
_DEFAULT_ADJACENCY_TENSOR = torch.tensor(
    _ENCODER_3P._adjacency,
    dtype=torch.float32,
)
_TERRITORY_INDEX_BY_ID = {
    territory.territory_id: index
    for index, territory in enumerate(WAR1_BOARD_SPEC.map.territories)
}
_CONTINENT_INDEX_BY_ID = {
    continent.continent_id: index
    for index, continent in enumerate(WAR1_BOARD_SPEC.map.continents)
}
_CONTINENT_BONUS_BY_ID = {
    continent.continent_id: float(continent.reinforcement_bonus)
    for continent in WAR1_BOARD_SPEC.map.continents
}
_OBJECTIVE_INDEX_BY_ID = {
    objective.objective_id: index
    for index, objective in enumerate(WAR1_BOARD_SPEC.objectives)
}
_CARD_INDEX_BY_ID = {
    card_id: index
    for index, card_id in enumerate(WAR1_BOARD_SPEC.territory_deck.card_ids)
}
_PHASE_INDEX_BY_ID = {
    phase_id: index for index, phase_id in enumerate(PHASE_IDS)
}
_EVENT_TYPE_INDEX_BY_ID = {
    event_type: index + 1
    for index, event_type in enumerate(PUBLIC_EVENT_TYPES)
}


def _build_static_territory_base() -> NDArray[np.float32]:
    base = np.zeros((NUM_TERRITORIES, TERRITORY_FEATURE_DIM), dtype=np.float32)
    for t_idx, territory in enumerate(WAR1_BOARD_SPEC.map.territories):
        c_idx = _CONTINENT_INDEX_BY_ID[territory.continent_id]
        base[t_idx, 9 + c_idx] = 1.0
        base[t_idx, 15] = _CONTINENT_BONUS_BY_ID[territory.continent_id] / 7.0
    return base


_STATIC_TERRITORY_BASE = _build_static_territory_base()


@dataclass(frozen=True)
class StructuredArchitectureInputs:
    """Batched player-scoped tensors feeding all Phase 7.2 architectures."""

    global_features: Tensor
    territory_features: Tensor
    adjacency: Tensor
    phase_indices: Tensor
    objective_indices: Tensor
    hand_card_features: Tensor
    owned_card_territory_mask: Tensor
    event_type_indices: Tensor
    event_actor_indices: Tensor
    event_mask: Tensor

    def __post_init__(self) -> None:
        batch_size = self.global_features.shape[0]
        if self.global_features.ndim != 2 or (
            self.global_features.shape[1] != FULL_GAME_OBSERVATION_SIZE
        ):
            raise ValueError(
                "global_features must have shape "
                f"(batch, {FULL_GAME_OBSERVATION_SIZE})."
            )
        if self.territory_features.shape != (
            batch_size,
            NUM_TERRITORIES,
            TERRITORY_FEATURE_DIM,
        ):
            raise ValueError(
                "territory_features must have shape "
                f"(batch, {NUM_TERRITORIES}, {TERRITORY_FEATURE_DIM})."
            )
        if self.adjacency.shape != (batch_size, NUM_TERRITORIES, NUM_TERRITORIES):
            raise ValueError(
                "adjacency must have shape "
                f"(batch, {NUM_TERRITORIES}, {NUM_TERRITORIES})."
            )
        if self.phase_indices.shape != (batch_size,):
            raise ValueError("phase_indices must have shape (batch,).")
        if self.objective_indices.shape != (batch_size,):
            raise ValueError("objective_indices must have shape (batch,).")
        if self.hand_card_features.shape != (batch_size, NUM_CARDS):
            raise ValueError(
                f"hand_card_features must have shape (batch, {NUM_CARDS})."
            )
        if self.owned_card_territory_mask.shape != (batch_size, NUM_CARDS):
            raise ValueError(
                f"owned_card_territory_mask must have shape (batch, {NUM_CARDS})."
            )
        if (
            self.event_type_indices.ndim != 2
            or self.event_type_indices.shape[0] != batch_size
            or self.event_actor_indices.shape != self.event_type_indices.shape
            or self.event_mask.shape != self.event_type_indices.shape
        ):
            raise ValueError(
                "Event history tensors must share shape (batch, history_length)."
            )

    @property
    def batch_size(self) -> int:
        return int(self.global_features.shape[0])

    def repeat_batch(self, repeats: int) -> StructuredArchitectureInputs:
        if isinstance(repeats, bool) or not isinstance(repeats, int) or repeats < 1:
            raise ValueError("repeats must be a positive integer.")
        return StructuredArchitectureInputs(
            global_features=self.global_features.repeat_interleave(repeats, dim=0),
            territory_features=self.territory_features.repeat_interleave(
                repeats, dim=0
            ),
            adjacency=self.adjacency.repeat_interleave(repeats, dim=0),
            phase_indices=self.phase_indices.repeat_interleave(repeats, dim=0),
            objective_indices=self.objective_indices.repeat_interleave(
                repeats, dim=0
            ),
            hand_card_features=self.hand_card_features.repeat_interleave(
                repeats, dim=0
            ),
            owned_card_territory_mask=self.owned_card_territory_mask.repeat_interleave(
                repeats, dim=0
            ),
            event_type_indices=self.event_type_indices.repeat_interleave(
                repeats, dim=0
            ),
            event_actor_indices=self.event_actor_indices.repeat_interleave(
                repeats, dim=0
            ),
            event_mask=self.event_mask.repeat_interleave(repeats, dim=0),
        )


@dataclass(frozen=True)
class ArchitectureConfig:
    """Configuration for modular WAR architectures across experimental axes."""

    encoder: EncoderAxis = "mlp"
    memory: MemoryAxis = "none"
    hidden_size: int = 128
    graph_layers: int = 3
    attention_heads: int = 4
    recurrent_layers: int = 1
    max_history_events: int = 64
    phase_conditioned_heads: bool = True
    use_objective_and_card_encoder: bool = True
    use_belief_head: bool = True
    token_vocab_size: int = FULL_GAME_INTENT_VOCAB_SIZE
    end_token_id: int = FULL_GAME_INTENT_END_TOKEN

    def __post_init__(self) -> None:
        if self.encoder not in ("mlp", "message_passing_gnn", "graph_transformer"):
            raise ValueError(f"Unsupported encoder {self.encoder!r}.")
        if self.memory not in (
            "none",
            "gru",
            "lstm",
            "causal_event_transformer",
        ):
            raise ValueError(f"Unsupported memory {self.memory!r}.")
        for label, value in (
            ("hidden_size", self.hidden_size),
            ("graph_layers", self.graph_layers),
            ("attention_heads", self.attention_heads),
            ("recurrent_layers", self.recurrent_layers),
            ("max_history_events", self.max_history_events),
            ("token_vocab_size", self.token_vocab_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")
        if self.hidden_size % self.attention_heads != 0:
            raise ValueError("hidden_size must be divisible by attention_heads.")
        if (
            isinstance(self.end_token_id, bool)
            or not isinstance(self.end_token_id, int)
            or not 0 <= self.end_token_id < self.token_vocab_size
        ):
            raise ValueError("end_token_id must be inside token_vocab_size.")


@dataclass(frozen=True)
class PolicyValueBeliefOutput:
    """Outputs from policy, value, and opponent-belief heads."""

    token_logits: Tensor
    values: Tensor
    opponent_objective_logits: Tensor | None
    opponent_card_logits: Tensor | None
    latent_context: Tensor
    recurrent_state: RecurrentState | None


@dataclass(frozen=True)
class ArchitectureEvaluationSummary:
    """Summary of evaluating a Phase 7.2 architecture on player observations."""

    variant_id: str
    config: ArchitectureConfig
    parameter_count: int
    mean_inference_ms: float
    legal_action_selection_valid: bool
    policy_logits_shape: tuple[int, ...]
    value_shape: tuple[int, ...]
    belief_objective_shape: tuple[int, ...] | None
    belief_card_shape: tuple[int, ...] | None


def extract_architecture_inputs(
    observations: War1AECObservation | Sequence[War1AECObservation],
    *,
    max_history_events: int = 64,
    device: torch.device | str = "cpu",
) -> StructuredArchitectureInputs:
    """Convert one or more player-scoped observations into architecture tensors."""
    if isinstance(max_history_events, bool) or max_history_events < 1:
        raise ValueError("max_history_events must be a positive integer.")
    obs_list: list[War1AECObservation]
    if isinstance(observations, dict):
        obs_list = [observations]
    else:
        obs_list = list(observations)
    if not obs_list:
        raise ValueError("At least one observation is required.")

    batch_size = len(obs_list)
    global_rows: list[tuple[float, ...]] = []
    territory_tensors = np.broadcast_to(
        _STATIC_TERRITORY_BASE,
        (batch_size, NUM_TERRITORIES, TERRITORY_FEATURE_DIM),
    ).copy()
    phase_indices = np.zeros((batch_size,), dtype=np.int64)
    objective_indices = np.zeros((batch_size,), dtype=np.int64)
    hand_cards = np.zeros((batch_size, NUM_CARDS), dtype=np.float32)
    owned_card_mask = np.zeros((batch_size, NUM_CARDS), dtype=np.float32)
    event_types = np.zeros((batch_size, max_history_events), dtype=np.int64)
    event_actors = np.zeros((batch_size, max_history_events), dtype=np.int64)
    event_mask = np.zeros((batch_size, max_history_events), dtype=np.bool_)

    for batch_idx, obs in enumerate(obs_list):
        global_rows.append(extract_player_observation_features(obs))
        engine_obs = obs["engine_observation"]
        if engine_obs is None:
            raise ValueError("Observation requires a non-null engine_observation.")
        public_state = engine_obs.public_state
        private_state = engine_obs.private_state
        focal_id = engine_obs.player_id
        player_index_by_id = {
            player.player_id: idx
            for idx, player in enumerate(public_state.players)
        }
        phase_indices[batch_idx] = _PHASE_INDEX_BY_ID.get(public_state.phase_id, 0)
        effective_obj = (
            private_state.objective_fallback_id
            or private_state.objective_id
            or ""
        )
        objective_indices[batch_idx] = _OBJECTIVE_INDEX_BY_ID.get(effective_obj, 0)

        owned_territories: set[str] = set()
        pending_origin = (
            public_state.pending_occupation.origin_territory_id
            if public_state.pending_occupation is not None
            else None
        )
        pending_target = (
            public_state.pending_occupation.target_territory_id
            if public_state.pending_occupation is not None
            else None
        )
        restricted_allowed: set[str] = set()
        for pool in public_state.reinforcement_pools:
            if pool.player_id == focal_id and pool.allowed_territory_ids:
                restricted_allowed.update(pool.allowed_territory_ids)

        for territory in public_state.territories:
            t_id = territory.territory_id
            t_idx = _TERRITORY_INDEX_BY_ID[t_id]
            owner_id = territory.owner_player_id
            if owner_id == focal_id:
                territory_tensors[batch_idx, t_idx, 0] = 1.0
                owned_territories.add(t_id)
            elif owner_id is not None:
                territory_tensors[batch_idx, t_idx, 1] = 1.0
            if owner_id in player_index_by_id:
                territory_tensors[
                    batch_idx,
                    t_idx,
                    2 + player_index_by_id[owner_id],
                ] = 1.0
            territory_tensors[batch_idx, t_idx, 8] = (
                math.log2(max(1, territory.army_count)) / 10.0
            )
            if t_id == pending_origin:
                territory_tensors[batch_idx, t_idx, 16] = 1.0
            if t_id == pending_target or t_id in restricted_allowed:
                territory_tensors[batch_idx, t_idx, 17] = 1.0

        for card_id in private_state.hand_card_ids:
            card_idx = _CARD_INDEX_BY_ID.get(card_id)
            if card_idx is not None:
                hand_cards[batch_idx, card_idx] += 1.0
                if card_id in owned_territories:
                    owned_card_mask[batch_idx, card_idx] = 1.0

        recent_events = engine_obs.recent_events[-max_history_events:]
        for pos, event in enumerate(recent_events):
            event_types[batch_idx, pos] = _EVENT_TYPE_INDEX_BY_ID.get(
                event.event_type,
                0,
            )
            actor_seat = (
                player_index_by_id.get(event.actor_player_id, -1) + 1
                if event.actor_player_id is not None
                else 0
            )
            event_actors[batch_idx, pos] = max(0, actor_seat)
            event_mask[batch_idx, pos] = True

    adjacency = _DEFAULT_ADJACENCY_TENSOR.unsqueeze(0).expand(
        batch_size,
        -1,
        -1,
    ).to(device=device)
    return StructuredArchitectureInputs(
        global_features=torch.tensor(
            global_rows,
            dtype=torch.float32,
            device=device,
        ),
        territory_features=torch.tensor(
            territory_tensors,
            dtype=torch.float32,
            device=device,
        ),
        adjacency=adjacency,
        phase_indices=torch.tensor(
            phase_indices,
            dtype=torch.long,
            device=device,
        ),
        objective_indices=torch.tensor(
            objective_indices,
            dtype=torch.long,
            device=device,
        ),
        hand_card_features=torch.tensor(
            hand_cards,
            dtype=torch.float32,
            device=device,
        ),
        owned_card_territory_mask=torch.tensor(
            owned_card_mask,
            dtype=torch.float32,
            device=device,
        ),
        event_type_indices=torch.tensor(
            event_types,
            dtype=torch.long,
            device=device,
        ),
        event_actor_indices=torch.tensor(
            event_actors,
            dtype=torch.long,
            device=device,
        ),
        event_mask=torch.tensor(
            event_mask,
            dtype=torch.bool,
            device=device,
        ),
    )


class ObjectiveAndCardEncoder(nn.Module):
    """Joint representation of the acting player's secret objective and hand cards."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        self.hidden_size = hidden_size
        self.objective_embedding = nn.Embedding(NUM_OBJECTIVES, hidden_size)
        nn.init.normal_(self.objective_embedding.weight, mean=0.0, std=0.02)
        self.card_projection = nn.Sequential(
            nn.Linear(2 * NUM_CARDS, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.fusion = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )

    def forward(
        self,
        objective_indices: Tensor,
        hand_card_features: Tensor,
        owned_card_territory_mask: Tensor,
    ) -> Tensor:
        obj_emb = self.objective_embedding(objective_indices)
        card_inputs = torch.cat(
            (hand_card_features, owned_card_territory_mask),
            dim=-1,
        )
        card_emb = self.card_projection(card_inputs)
        return cast(Tensor, self.fusion(torch.cat((obj_emb, card_emb), dim=-1)))


class MLPObservationEncoder(nn.Module):
    """Feedforward MLP encoder over global and territory summary features."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        input_dim = FULL_GAME_OBSERVATION_SIZE + 2 * TERRITORY_FEATURE_DIM
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
        )

    def forward(self, inputs: StructuredArchitectureInputs) -> Tensor:
        territory_mean = inputs.territory_features.mean(dim=1)
        territory_max = inputs.territory_features.amax(dim=1)
        combined = torch.cat(
            (inputs.global_features, territory_mean, territory_max),
            dim=-1,
        )
        return cast(Tensor, self.network(combined))


class _MessagePassingLayer(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.self_linear = nn.Linear(hidden_size, hidden_size)
        self.neighbor_linear = nn.Linear(hidden_size, hidden_size)
        self.norm = nn.LayerNorm(hidden_size)
        self.activation = nn.SiLU()

    def forward(self, node_states: Tensor, normalized_adjacency: Tensor) -> Tensor:
        neighbor_messages = torch.bmm(normalized_adjacency, node_states)
        updated = self.self_linear(node_states) + self.neighbor_linear(
            neighbor_messages
        )
        return cast(Tensor, self.norm(node_states + self.activation(updated)))


class MessagePassingGNNEncoder(nn.Module):
    """Message-passing graph neural network over the 42 WAR territories."""

    def __init__(self, hidden_size: int, *, num_layers: int = 3) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.node_embedding = nn.Linear(TERRITORY_FEATURE_DIM, hidden_size)
        self.layers = nn.ModuleList(
            [_MessagePassingLayer(hidden_size) for _ in range(num_layers)]
        )
        self.global_projection = nn.Linear(
            FULL_GAME_OBSERVATION_SIZE + 2 * hidden_size,
            hidden_size,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: StructuredArchitectureInputs) -> Tensor:
        degree = inputs.adjacency.sum(dim=-1, keepdim=True).clamp(min=1.0)
        normalized_adjacency = inputs.adjacency / degree
        nodes = torch.tanh(self.node_embedding(inputs.territory_features))
        for layer in self.layers:
            nodes = layer(nodes, normalized_adjacency)
        pooled_mean = nodes.mean(dim=1)
        pooled_max = nodes.amax(dim=1)
        combined = torch.cat(
            (inputs.global_features, pooled_mean, pooled_max),
            dim=-1,
        )
        return torch.tanh(self.output_norm(self.global_projection(combined)))


class _GraphTransformerLayer(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.qkv = nn.Linear(hidden_size, 3 * hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        self.edge_bias = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.non_edge_bias = nn.Parameter(torch.tensor(-0.25, dtype=torch.float32))
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def forward(self, nodes: Tensor, adjacency: Tensor) -> Tensor:
        batch_size, num_nodes, hidden_size = nodes.shape
        qkv = self.qkv(nodes).reshape(
            batch_size,
            num_nodes,
            3,
            self.num_heads,
            self.head_dim,
        )
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = 1.0 / math.sqrt(self.head_dim)
        scores = torch.matmul(q, k.transpose(-2, -1)) * scale
        eye = torch.eye(num_nodes, device=adjacency.device, dtype=adjacency.dtype)
        is_neighbor = (adjacency > 0) | (eye.unsqueeze(0) > 0)
        structural_bias = torch.where(
            is_neighbor.unsqueeze(1),
            self.edge_bias,
            self.non_edge_bias,
        )
        weights = torch.softmax(scores + structural_bias, dim=-1)
        attended = torch.matmul(weights, v).transpose(1, 2).reshape(
            batch_size,
            num_nodes,
            hidden_size,
        )
        nodes = self.norm1(nodes + self.out_proj(attended))
        return cast(Tensor, self.norm2(nodes + self.ffn(nodes)))


class GraphTransformerEncoder(nn.Module):
    """Graph Transformer over territory nodes with adjacency structural bias."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_layers: int = 3,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if isinstance(num_layers, bool) or num_layers < 1:
            raise ValueError("num_layers must be a positive integer.")
        if (
            isinstance(num_heads, bool)
            or num_heads < 1
            or hidden_size % num_heads != 0
        ):
            raise ValueError("hidden_size must be divisible by num_heads.")
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.node_embedding = nn.Linear(TERRITORY_FEATURE_DIM, hidden_size)
        self.territory_positional = nn.Embedding(NUM_TERRITORIES, hidden_size)
        self.layers = nn.ModuleList(
            [
                _GraphTransformerLayer(hidden_size, num_heads)
                for _ in range(num_layers)
            ]
        )
        self.readout = nn.Linear(
            FULL_GAME_OBSERVATION_SIZE + 2 * hidden_size,
            hidden_size,
        )
        self.output_norm = nn.LayerNorm(hidden_size)

    def forward(self, inputs: StructuredArchitectureInputs) -> Tensor:
        territory_ids = torch.arange(
            NUM_TERRITORIES,
            device=inputs.territory_features.device,
        ).unsqueeze(0)
        nodes = self.node_embedding(
            inputs.territory_features
        ) + self.territory_positional(territory_ids)
        for layer in self.layers:
            nodes = layer(nodes, inputs.adjacency)
        pooled_mean = nodes.mean(dim=1)
        pooled_max = nodes.amax(dim=1)
        combined = torch.cat(
            (inputs.global_features, pooled_mean, pooled_max),
            dim=-1,
        )
        return torch.tanh(self.output_norm(self.readout(combined)))


class CausalEventTransformer(nn.Module):
    """Causal Transformer encoder over public game event sequences."""

    def __init__(
        self,
        hidden_size: int,
        *,
        num_heads: int = 4,
        max_history_events: int = 64,
    ) -> None:
        super().__init__()
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if (
            isinstance(num_heads, bool)
            or num_heads < 1
            or hidden_size % num_heads != 0
        ):
            raise ValueError("hidden_size must be divisible by num_heads.")
        if isinstance(max_history_events, bool) or max_history_events < 1:
            raise ValueError("max_history_events must be a positive integer.")
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.max_history_events = max_history_events
        self.event_type_embedding = nn.Embedding(
            NUM_PUBLIC_EVENT_TYPES + 1,
            hidden_size,
            padding_idx=0,
        )
        self.actor_embedding = nn.Embedding(
            MAX_PLAYERS + 1,
            hidden_size,
            padding_idx=0,
        )
        self.position_embedding = nn.Embedding(max_history_events, hidden_size)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=num_heads,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.SiLU(),
            nn.Linear(2 * hidden_size, hidden_size),
        )

    def encode_sequence(
        self,
        event_type_indices: Tensor,
        event_actor_indices: Tensor,
        event_mask: Tensor,
    ) -> Tensor:
        """Return causal per-position embeddings ``(batch, seq_len, hidden_size)``."""
        seq_len = event_type_indices.shape[1]
        if seq_len > self.max_history_events:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_history_events "
                f"{self.max_history_events}."
            )
        positions = torch.arange(
            seq_len,
            device=event_type_indices.device,
        ).unsqueeze(0)
        tokens = (
            self.event_type_embedding(event_type_indices)
            + self.actor_embedding(event_actor_indices)
            + self.position_embedding(positions)
        )
        causal_mask = torch.triu(
            torch.ones((seq_len, seq_len), device=tokens.device, dtype=torch.bool),
            diagonal=1,
        )
        attended, _ = self.attention(
            tokens,
            tokens,
            tokens,
            attn_mask=causal_mask,
            need_weights=False,
        )
        hidden = self.norm1(tokens + attended)
        hidden = self.norm2(hidden + self.ffn(hidden))
        return cast(
            Tensor,
            hidden * event_mask.unsqueeze(-1).to(dtype=hidden.dtype),
        )

    def forward(
        self,
        event_type_indices: Tensor,
        event_actor_indices: Tensor,
        event_mask: Tensor,
    ) -> Tensor:
        sequence_hidden = self.encode_sequence(
            event_type_indices,
            event_actor_indices,
            event_mask,
        )
        valid_counts = event_mask.sum(dim=1, keepdim=True).clamp(min=1)
        return sequence_hidden.sum(dim=1) / valid_counts.to(
            dtype=sequence_hidden.dtype
        )


class PhaseConditionedActionHead(nn.Module):
    """Autoregressive action-token decoder modulated by the active game phase."""

    def __init__(
        self,
        hidden_size: int,
        *,
        token_vocab_size: int = FULL_GAME_INTENT_VOCAB_SIZE,
        phase_conditioned: bool = True,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.token_vocab_size = token_vocab_size
        self.phase_conditioned = phase_conditioned
        self.phase_scale = nn.Embedding(NUM_PHASES, hidden_size)
        self.phase_shift = nn.Embedding(NUM_PHASES, hidden_size)
        self.phase_logit_bias = nn.Embedding(NUM_PHASES, token_vocab_size)
        nn.init.normal_(self.phase_scale.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.phase_shift.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.phase_logit_bias.weight, mean=0.0, std=0.02)
        self.token_embedding = nn.Embedding(token_vocab_size + 1, hidden_size)
        self.decoder = nn.GRU(
            input_size=2 * hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.policy_projection = nn.Linear(hidden_size, token_vocab_size)

    def modulate_context(self, context: Tensor, phase_indices: Tensor) -> Tensor:
        if not self.phase_conditioned:
            return context
        scale = 1.0 + torch.tanh(self.phase_scale(phase_indices))
        shift = self.phase_shift(phase_indices)
        return torch.tanh(context * scale + shift)

    def forward(
        self,
        context: Tensor,
        action_tokens: Tensor,
        phase_indices: Tensor,
    ) -> Tensor:
        batch_size, action_length = action_tokens.shape
        conditioned = self.modulate_context(context, phase_indices)
        decoder_tokens = torch.full_like(action_tokens, self.token_vocab_size)
        decoder_tokens[:, 1:] = action_tokens[:, :-1]
        token_emb = self.token_embedding(decoder_tokens)
        repeated = conditioned.unsqueeze(1).expand(-1, action_length, -1)
        decoder_in = torch.cat((token_emb, repeated), dim=-1)
        decoded, _ = self.decoder(decoder_in, conditioned.unsqueeze(0))
        logits = self.policy_projection(decoded)
        if self.phase_conditioned:
            logits = logits + self.phase_logit_bias(phase_indices).unsqueeze(1)
        return cast(
            Tensor,
            logits.reshape(batch_size, action_length, self.token_vocab_size),
        )


class OpponentBeliefHead(nn.Module):
    """Auxiliary head predicting opponent secret objectives and held cards."""

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.objective_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, MAX_PLAYERS * NUM_OBJECTIVES),
        )
        self.card_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, MAX_PLAYERS * NUM_CARDS),
        )

    def forward(self, context: Tensor) -> tuple[Tensor, Tensor]:
        batch_size = context.shape[0]
        obj_logits = self.objective_head(context).reshape(
            batch_size,
            MAX_PLAYERS,
            NUM_OBJECTIVES,
        )
        card_logits = self.card_head(context).reshape(
            batch_size,
            MAX_PLAYERS,
            NUM_CARDS,
        )
        return obj_logits, card_logits


class WarArchitectureModel(nn.Module):
    """Unified WAR actor-critic-belief model combining all Phase 7.2 axes."""

    def __init__(
        self,
        config: ArchitectureConfig = ArchitectureConfig(),
    ) -> None:
        super().__init__()
        self.config = config
        hidden_size = config.hidden_size

        self.mlp_encoder: MLPObservationEncoder | None = None
        self.gnn_encoder: MessagePassingGNNEncoder | None = None
        self.graph_transformer_encoder: GraphTransformerEncoder | None = None
        if config.encoder == "mlp":
            self.mlp_encoder = MLPObservationEncoder(hidden_size)
        elif config.encoder == "message_passing_gnn":
            self.gnn_encoder = MessagePassingGNNEncoder(
                hidden_size,
                num_layers=config.graph_layers,
            )
        elif config.encoder == "graph_transformer":
            self.graph_transformer_encoder = GraphTransformerEncoder(
                hidden_size,
                num_layers=config.graph_layers,
                num_heads=config.attention_heads,
            )

        self.objective_card_encoder: ObjectiveAndCardEncoder | None = (
            ObjectiveAndCardEncoder(hidden_size)
            if config.use_objective_and_card_encoder
            else None
        )
        self._state_gru: nn.GRU | None = None
        self._state_lstm: nn.LSTM | None = None
        self.event_transformer: CausalEventTransformer | None = None
        if config.memory == "gru":
            self._state_gru = nn.GRU(
                hidden_size,
                hidden_size,
                num_layers=config.recurrent_layers,
                batch_first=True,
            )
        elif config.memory == "lstm":
            self._state_lstm = nn.LSTM(
                hidden_size,
                hidden_size,
                num_layers=config.recurrent_layers,
                batch_first=True,
            )
        elif config.memory == "causal_event_transformer":
            self.event_transformer = CausalEventTransformer(
                hidden_size,
                num_heads=config.attention_heads,
                max_history_events=config.max_history_events,
            )

        self.context_fusion = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.Tanh(),
        )
        self.action_head = PhaseConditionedActionHead(
            hidden_size,
            token_vocab_size=config.token_vocab_size,
            phase_conditioned=config.phase_conditioned_heads,
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1),
        )
        self.belief_head: OpponentBeliefHead | None = (
            OpponentBeliefHead(hidden_size) if config.use_belief_head else None
        )

    def encode_context(
        self,
        inputs: StructuredArchitectureInputs,
        *,
        recurrent_state: RecurrentState | None = None,
    ) -> tuple[Tensor, RecurrentState | None]:
        if self.mlp_encoder is not None:
            encoded = self.mlp_encoder(inputs)
        elif self.gnn_encoder is not None:
            encoded = self.gnn_encoder(inputs)
        elif self.graph_transformer_encoder is not None:
            encoded = self.graph_transformer_encoder(inputs)
        else:
            raise RuntimeError("No encoder initialized.")

        if self.objective_card_encoder is not None:
            encoded = encoded + self.objective_card_encoder(
                inputs.objective_indices,
                inputs.hand_card_features,
                inputs.owned_card_territory_mask,
            )

        next_recurrent: RecurrentState | None = None
        if self._state_gru is not None:
            h0 = (
                torch.zeros(
                    (
                        self.config.recurrent_layers,
                        inputs.batch_size,
                        self.config.hidden_size,
                    ),
                    dtype=encoded.dtype,
                    device=encoded.device,
                )
                if recurrent_state is None
                else recurrent_state
            )
            if not isinstance(h0, Tensor):
                raise TypeError("GRU recurrent_state must be a Tensor.")
            gru_out, next_h = self._state_gru(encoded.unsqueeze(1), h0)
            encoded = gru_out.squeeze(1)
            next_recurrent = next_h
        elif self._state_lstm is not None:
            if recurrent_state is None:
                zeros = torch.zeros(
                    (
                        self.config.recurrent_layers,
                        inputs.batch_size,
                        self.config.hidden_size,
                    ),
                    dtype=encoded.dtype,
                    device=encoded.device,
                )
                lstm_state: tuple[Tensor, Tensor] = (zeros, zeros.clone())
            else:
                if not isinstance(recurrent_state, tuple):
                    raise TypeError("LSTM recurrent_state must be a tensor pair.")
                lstm_state = recurrent_state
            lstm_out, next_pair = self._state_lstm(
                encoded.unsqueeze(1),
                lstm_state,
            )
            encoded = lstm_out.squeeze(1)
            next_recurrent = next_pair
        elif self.event_transformer is not None:
            event_context = self.event_transformer(
                inputs.event_type_indices,
                inputs.event_actor_indices,
                inputs.event_mask,
            )
            encoded = encoded + event_context

        return self.context_fusion(encoded), next_recurrent

    def forward(
        self,
        inputs: StructuredArchitectureInputs,
        action_tokens: Tensor,
        *,
        recurrent_state: RecurrentState | None = None,
    ) -> PolicyValueBeliefOutput:
        if action_tokens.ndim != 2 or action_tokens.shape[0] != inputs.batch_size:
            raise ValueError("action_tokens must have shape (batch, action_length).")
        context, next_recurrent = self.encode_context(
            inputs,
            recurrent_state=recurrent_state,
        )
        token_logits = self.action_head(
            context,
            action_tokens.to(dtype=torch.long),
            inputs.phase_indices,
        )
        values = self.value_head(context).squeeze(-1)
        obj_logits: Tensor | None = None
        card_logits: Tensor | None = None
        if self.belief_head is not None:
            obj_logits, card_logits = self.belief_head(context)
        return PolicyValueBeliefOutput(
            token_logits=token_logits,
            values=values,
            opponent_objective_logits=obj_logits,
            opponent_card_logits=card_logits,
            latent_context=context,
            recurrent_state=next_recurrent,
        )

    @torch.no_grad()
    def select_legal_action(
        self,
        observation: War1AECObservation,
        *,
        deterministic: bool = True,
        recurrent_state: RecurrentState | None = None,
        rng: Generator | None = None,
    ) -> tuple[int, int, Tensor, RecurrentState | None]:
        """Select a legal action index strictly from the player observation."""
        partition = classify_legal_actions(observation)
        candidate_intents = partition.available_intents
        num_candidates = len(candidate_intents)
        if num_candidates == 1 and recurrent_state is None:
            chosen_intent = candidate_intents[0]
            action_candidates = partition.action_indices_by_intent[chosen_intent]
            if rng is not None:
                chosen_action = action_candidates[
                    int(rng.integers(0, len(action_candidates)))
                ]
            else:
                chosen_action = action_candidates[0]
            return (
                chosen_action,
                chosen_intent,
                torch.ones((1,), dtype=torch.float32),
                None,
            )

        single_inputs = extract_architecture_inputs(
            observation,
            max_history_events=self.config.max_history_events,
        )
        candidate_tokens = torch.tensor(
            [[intent, self.config.end_token_id] for intent in candidate_intents],
            dtype=torch.long,
        )
        token_mask = torch.ones_like(candidate_tokens, dtype=torch.bool)
        legal_masks = torch.zeros(
            (num_candidates, 2, self.config.token_vocab_size),
            dtype=torch.bool,
        )
        first_mask = partition.prefix_mask(())
        for idx, intent in enumerate(candidate_intents):
            legal_masks[idx, 0] = first_mask
            legal_masks[idx, 1] = partition.prefix_mask((intent,))

        context, next_single_recurrent = self.encode_context(
            single_inputs,
            recurrent_state=recurrent_state,
        )
        token_logits = self.action_head(
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
        probs = masked_log_softmax(
            log_probs.unsqueeze(0),
            torch.ones((1, num_candidates), dtype=torch.bool),
        ).squeeze(0).exp()
        chosen_pos = int(probs.argmax().item()) if deterministic else 0
        chosen_intent = candidate_intents[chosen_pos]
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

        return chosen_action, chosen_intent, probs, next_single_recurrent


def build_phase7_architecture_catalog(
    *,
    hidden_size: int = 64,
    graph_layers: int = 3,
    max_history_events: int = 64,
) -> dict[ArchitectureVariantId, ArchitectureConfig]:
    """Return the canonical Phase 7.2 architecture configurations."""
    return {
        "mlp_baseline": ArchitectureConfig(
            encoder="mlp",
            memory="none",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=False,
            use_objective_and_card_encoder=False,
            use_belief_head=False,
        ),
        "message_passing_gnn": ArchitectureConfig(
            encoder="message_passing_gnn",
            memory="none",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        ),
        "graph_transformer": ArchitectureConfig(
            encoder="graph_transformer",
            memory="none",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        ),
        "recurrent_gru": ArchitectureConfig(
            encoder="mlp",
            memory="gru",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        ),
        "recurrent_lstm": ArchitectureConfig(
            encoder="mlp",
            memory="lstm",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        ),
        "causal_event_transformer": ArchitectureConfig(
            encoder="message_passing_gnn",
            memory="causal_event_transformer",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        ),
        "phase_conditioned_policy_value_belief": ArchitectureConfig(
            encoder="graph_transformer",
            memory="causal_event_transformer",
            hidden_size=hidden_size,
            graph_layers=graph_layers,
            max_history_events=max_history_events,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        ),
    }


def evaluate_architecture_catalog(
    observations: Sequence[War1AECObservation],
    *,
    hidden_size: int = 64,
    graph_layers: int = 3,
    max_history_events: int = 64,
    seed: int = 2026,
) -> tuple[ArchitectureEvaluationSummary, ...]:
    """Benchmark and validate every Phase 7.2 architecture on player observations."""
    if not observations:
        raise ValueError("observations must not be empty.")
    catalog = build_phase7_architecture_catalog(
        hidden_size=hidden_size,
        graph_layers=graph_layers,
        max_history_events=max_history_events,
    )
    inputs = extract_architecture_inputs(
        observations,
        max_history_events=max_history_events,
    )
    sample_tokens = torch.tensor(
        [[0, FULL_GAME_INTENT_END_TOKEN] for _ in range(inputs.batch_size)],
        dtype=torch.long,
    )
    summaries: list[ArchitectureEvaluationSummary] = []

    for variant_id, config in catalog.items():
        torch.manual_seed(seed)
        model = WarArchitectureModel(config)
        model.eval()
        param_count = sum(p.numel() for p in model.parameters())
        started = perf_counter()
        with torch.no_grad():
            output = model(inputs, sample_tokens)
            action_idx, _, _, _ = model.select_legal_action(observations[0])
        elapsed_ms = (perf_counter() - started) * 1000.0
        legal_count = int(observations[0]["action_mask"].shape[0])
        is_valid = 0 <= action_idx < legal_count and bool(
            observations[0]["action_mask"][action_idx] == 1
        )
        summaries.append(
            ArchitectureEvaluationSummary(
                variant_id=variant_id,
                config=config,
                parameter_count=param_count,
                mean_inference_ms=elapsed_ms / max(1, len(observations)),
                legal_action_selection_valid=is_valid,
                policy_logits_shape=tuple(output.token_logits.shape),
                value_shape=tuple(output.values.shape),
                belief_objective_shape=(
                    tuple(output.opponent_objective_logits.shape)
                    if output.opponent_objective_logits is not None
                    else None
                ),
                belief_card_shape=(
                    tuple(output.opponent_card_logits.shape)
                    if output.opponent_card_logits is not None
                    else None
                ),
            )
        )
    return tuple(summaries)


__all__ = [
    "ACTION_CONFIRM_TOKEN",
    "ArchitectureConfig",
    "ArchitectureEvaluationSummary",
    "ArchitectureVariantId",
    "CausalEventTransformer",
    "GraphTransformerEncoder",
    "MLPObservationEncoder",
    "MessagePassingGNNEncoder",
    "NUM_CARDS",
    "NUM_CONTINENTS",
    "NUM_OBJECTIVES",
    "NUM_PHASES",
    "NUM_PUBLIC_EVENT_TYPES",
    "NUM_TERRITORIES",
    "ObjectiveAndCardEncoder",
    "OpponentBeliefHead",
    "PhaseConditionedActionHead",
    "PolicyValueBeliefOutput",
    "StructuredArchitectureInputs",
    "TERRITORY_FEATURE_DIM",
    "WarArchitectureModel",
    "build_phase7_architecture_catalog",
    "evaluate_architecture_catalog",
    "extract_architecture_inputs",
]
