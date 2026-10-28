"""Opponent objective and hand-card belief models and calibration (Phase 9.1)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Literal, TypeAlias

import numpy as np
import torch
from numpy.random import Generator
from numpy.typing import NDArray
from pydantic import JsonValue
from torch import nn
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    BeliefHypothesis,
    BeliefState,
    ClaimTerritoryOperation,
    DiscardTerritoryCardOperation,
    FullGameState,
    KnownFact,
    MoveArmiesOperation,
    OccupyTerritoryOperation,
    PlaceArmiesOperation,
    PlayerKnowledgeState,
    PlayerObservation,
    PublicGameState,
    PublicPlayerState,
    TradeCardsOperation,
)
from war_game_engine.objectives import (
    ELIMINATION_OBJECTIVE_COLORS,
    FALLBACK_OBJECTIVE_ID,
    OBJECTIVE_CONDITIONS,
    ControlContinentsCondition,
    EliminateColorCondition,
    TerritoryCountCondition,
)

from ..envs.observations import MAX_PLAYERS, War1ObservationEncoder
from ..envs.war1_aec import War1AECEnv, War1AECObservation
from .agent import FULL_GAME_INTENT_END_TOKEN
from .architectures import (
    NUM_CARDS,
    NUM_OBJECTIVES,
    ArchitectureConfig,
    WarArchitectureModel,
    extract_architecture_inputs,
)
from .centralized_training import (
    PrivilegedTableSecrets,
    compute_secret_belief_losses,
    extract_privileged_table_secrets,
)

BeliefMethodId: TypeAlias = Literal[
    "simple_prior",
    "particle_filter",
    "learned_model",
]
BELIEF_METHODS: tuple[BeliefMethodId, ...] = (
    "simple_prior",
    "particle_filter",
    "learned_model",
)

OBJECTIVE_IDS: tuple[str, ...] = tuple(
    objective.objective_id for objective in WAR1_BOARD_SPEC.objectives
)
CARD_IDS: tuple[str, ...] = WAR1_BOARD_SPEC.territory_deck.card_ids
_OBJECTIVE_INDEX_BY_ID: dict[str, int] = {
    obj_id: idx for idx, obj_id in enumerate(OBJECTIVE_IDS)
}
_CARD_INDEX_BY_ID: dict[str, int] = {
    card_id: idx for idx, card_id in enumerate(CARD_IDS)
}
_TERRITORY_CONTINENT_BY_ID: dict[str, str] = {
    territory.territory_id: territory.continent_id
    for territory in WAR1_BOARD_SPEC.map.territories
}
_CONTINENT_TERRITORIES: dict[str, frozenset[str]] = {
    continent.continent_id: frozenset(
        territory.territory_id
        for territory in WAR1_BOARD_SPEC.map.territories
        if territory.continent_id == continent.continent_id
    )
    for continent in WAR1_BOARD_SPEC.map.continents
}
_ADJACENT_TERRITORIES: dict[str, frozenset[str]] = {
    territory.territory_id: frozenset(
        [
            *(
                adj.second_territory_id
                for adj in WAR1_BOARD_SPEC.map.adjacencies
                if adj.first_territory_id == territory.territory_id
            ),
            *(
                adj.first_territory_id
                for adj in WAR1_BOARD_SPEC.map.adjacencies
                if adj.second_territory_id == territory.territory_id
            ),
        ]
    )
    for territory in WAR1_BOARD_SPEC.map.territories
}
_DEFAULT_KNOWN_AT = datetime(2026, 10, 28, tzinfo=timezone.utc)


@dataclass(frozen=True)
class OpponentBeliefEstimate:
    """Fact-grounded belief distribution over opponent objectives and hand cards."""

    method_id: BeliefMethodId
    game_id: str
    focal_player_id: str
    state_version: int
    opponent_player_ids: tuple[str, ...]
    objective_probabilities: NDArray[np.float64]
    card_probabilities: NDArray[np.float64]
    estimated_hand_counts: Mapping[str, int]
    feasible_objective_mask: NDArray[np.bool_]
    available_card_mask: NDArray[np.bool_]
    knowledge_state: PlayerKnowledgeState
    contract_belief_state: BeliefState

    def __post_init__(self) -> None:
        if self.method_id not in BELIEF_METHODS:
            raise ValueError(f"Unsupported belief method {self.method_id!r}.")
        if self.objective_probabilities.shape != (MAX_PLAYERS, NUM_OBJECTIVES):
            raise ValueError(
                "objective_probabilities must have shape "
                f"({MAX_PLAYERS}, {NUM_OBJECTIVES})."
            )
        if self.card_probabilities.shape != (MAX_PLAYERS, NUM_CARDS):
            raise ValueError(
                f"card_probabilities must have shape ({MAX_PLAYERS}, {NUM_CARDS})."
            )
        if self.feasible_objective_mask.shape != (MAX_PLAYERS, NUM_OBJECTIVES):
            raise ValueError(
                "feasible_objective_mask must have shape "
                f"({MAX_PLAYERS}, {NUM_OBJECTIVES})."
            )
        if self.available_card_mask.shape != (NUM_CARDS,):
            raise ValueError(
                f"available_card_mask must have shape ({NUM_CARDS},)."
            )

    def objective_distribution_for(self, player_id: str, seat_index: int) -> dict[str, float]:
        if player_id not in self.opponent_player_ids:
            raise ValueError(f"Player {player_id!r} is not an opponent.")
        probs = self.objective_probabilities[seat_index]
        return {
            OBJECTIVE_IDS[idx]: float(probs[idx])
            for idx in range(NUM_OBJECTIVES)
        }


@dataclass(frozen=True)
class CalibrationBinStat:
    """Single reliability-diagram confidence bin."""

    bin_lower: float
    bin_upper: float
    sample_count: int
    mean_confidence: float
    empirical_accuracy: float
    absolute_calibration_gap: float


@dataclass(frozen=True)
class BeliefCalibrationMetrics:
    """Calibration and proper-scoring metrics for an opponent belief estimator."""

    method_id: BeliefMethodId
    num_samples: int
    objective_ece: float
    objective_brier_score: float
    objective_log_loss: float
    objective_top1_accuracy: float
    objective_top3_accuracy: float
    mean_true_objective_probability: float
    card_ece: float
    card_brier_score: float
    card_log_loss: float
    hard_constraint_violation_rate: float
    objective_reliability_bins: tuple[CalibrationBinStat, ...]


@dataclass(frozen=True)
class BeliefComparisonReport:
    """Side-by-side comparison across simple prior, particle filter, and learned model."""

    num_training_observations: int
    num_evaluation_observations: int
    metrics_by_method: Mapping[BeliefMethodId, BeliefCalibrationMetrics]
    best_objective_log_loss_method: BeliefMethodId
    best_objective_brier_method: BeliefMethodId
    best_calibrated_method: BeliefMethodId
    all_respect_hard_constraints: bool


def eligible_objective_ids_for_public_state(
    public_state: PublicGameState,
) -> tuple[str, ...]:
    """Return the 8 + N objective IDs included in the setup deck for the table."""
    active_colors = {player.color for player in public_state.players}
    return tuple(
        objective.objective_id
        for objective in WAR1_BOARD_SPEC.objectives
        if objective.objective_id not in ELIMINATION_OBJECTIVE_COLORS
        or ELIMINATION_OBJECTIVE_COLORS[objective.objective_id] in active_colors
    )


def effective_objective_for_opponent_hypothesis(
    public_state: PublicGameState,
    opponent: PublicPlayerState,
    raw_objective_id: str,
) -> str:
    """Resolve fallback to W1-OBJ-13 if an elimination card targets self or other-eliminated."""
    target_color = ELIMINATION_OBJECTIVE_COLORS.get(raw_objective_id)
    if target_color is None:
        return raw_objective_id
    if target_color == opponent.color:
        return FALLBACK_OBJECTIVE_ID
    target_player = next(
        (p for p in public_state.players if p.color == target_color),
        None,
    )
    if target_player is None:
        return FALLBACK_OBJECTIVE_ID
    if (
        target_player.is_eliminated
        and target_player.eliminated_by_player_id != opponent.player_id
    ):
        return FALLBACK_OBJECTIVE_ID
    return raw_objective_id


def is_objective_already_satisfied_publicly(
    public_state: PublicGameState,
    player: PublicPlayerState,
    raw_objective_id: str,
) -> bool:
    """Check whether raw_objective_id (or its fallback) is already satisfied on board."""
    effective_id = effective_objective_for_opponent_hypothesis(
        public_state,
        player,
        raw_objective_id,
    )
    condition = OBJECTIVE_CONDITIONS[effective_id]
    if isinstance(condition, EliminateColorCondition):
        target_player = next(
            (p for p in public_state.players if p.color == condition.target_color),
            None,
        )
        if target_player is None:
            return False
        return (
            target_player.is_eliminated
            and target_player.eliminated_by_player_id == player.player_id
        )

    if isinstance(condition, TerritoryCountCondition):
        qualified = sum(
            1
            for territory in public_state.territories
            if territory.owner_player_id == player.player_id
            and territory.army_count >= condition.minimum_armies_per_territory
        )
        return qualified >= condition.minimum_territories

    if isinstance(condition, ControlContinentsCondition):
        owned_territories = {
            territory.territory_id
            for territory in public_state.territories
            if territory.owner_player_id == player.player_id
        }
        controlled_continents = {
            continent_id
            for continent_id, terr_ids in _CONTINENT_TERRITORIES.items()
            if terr_ids.issubset(owned_territories)
        }
        if not set(condition.required_continent_ids).issubset(controlled_continents):
            return False
        extra = len(controlled_continents - set(condition.required_continent_ids))
        return extra >= condition.additional_continent_count

    return False


def compute_feasible_objective_mask(
    observation: PlayerObservation,
) -> NDArray[np.bool_]:
    """Return a (MAX_PLAYERS, NUM_OBJECTIVES) mask of rule-compatible objectives."""
    public_state = observation.public_state
    focal_player_id = observation.player_id
    own_raw_objective_id = observation.private_state.objective_id
    eligible_set = set(eligible_objective_ids_for_public_state(public_state))

    mask = np.zeros((MAX_PLAYERS, NUM_OBJECTIVES), dtype=np.bool_)
    for seat_idx, player in enumerate(public_state.players):
        if seat_idx >= MAX_PLAYERS or player.player_id == focal_player_id:
            continue
        if (
            public_state.is_finished
            and public_state.winner_player_id == player.player_id
            and public_state.winner_objective_id is not None
        ):
            win_idx = _OBJECTIVE_INDEX_BY_ID.get(public_state.winner_objective_id)
            if win_idx is not None:
                mask[seat_idx, win_idx] = True
                continue

        for obj_idx, obj_id in enumerate(OBJECTIVE_IDS):
            if obj_id not in eligible_set:
                continue
            if own_raw_objective_id is not None and obj_id == own_raw_objective_id:
                continue
            if (
                not public_state.is_finished
                and public_state.pending_objective_check is None
                and not player.is_eliminated
                and is_objective_already_satisfied_publicly(
                    public_state,
                    player,
                    obj_id,
                )
            ):
                continue
            mask[seat_idx, obj_idx] = True

        if not np.any(mask[seat_idx]):
            for obj_idx, obj_id in enumerate(OBJECTIVE_IDS):
                if obj_id in eligible_set and obj_id != own_raw_objective_id:
                    mask[seat_idx, obj_idx] = True
    return mask


def estimate_public_hand_counts_and_card_availability(
    observation: PlayerObservation,
    *,
    observed_actions: Sequence[Action] = (),
) -> tuple[dict[str, int], NDArray[np.bool_], tuple[str, ...]]:
    """Estimate player hand sizes and available cards from public events and own hand."""
    public_state = observation.public_state
    focal_player_id = observation.player_id
    hand_counts: dict[str, int] = {
        player.player_id: 0 for player in public_state.players
    }

    for event in observation.recent_events:
        actor = event.actor_player_id
        if actor is None or actor not in hand_counts:
            continue
        if event.event_type == "draw_territory_card":
            hand_counts[actor] = min(6, hand_counts[actor] + 1)
        elif event.event_type == "trade_cards":
            cards_traded = event.data.get("cards_traded", 3)
            traded_int = int(cards_traded) if isinstance(cards_traded, int) else 3
            hand_counts[actor] = max(0, hand_counts[actor] - traded_int)
        elif event.event_type == "discard_territory_card":
            cards_discarded = event.data.get("cards_discarded", 1)
            disc_int = int(cards_discarded) if isinstance(cards_discarded, int) else 1
            hand_counts[actor] = max(0, hand_counts[actor] - disc_int)

    for player in public_state.players:
        if player.is_eliminated:
            transferred = hand_counts.get(player.player_id, 0)
            hand_counts[player.player_id] = 0
            eliminator = player.eliminated_by_player_id
            if eliminator is not None and eliminator in hand_counts and transferred > 0:
                hand_counts[eliminator] = min(6, hand_counts[eliminator] + transferred)

    decision_player_id = (
        public_state.decision_player_id or public_state.active_player_id
    )
    for player in public_state.players:
        pid = player.player_id
        if pid == focal_player_id:
            hand_counts[pid] = len(observation.private_state.hand_card_ids)
        elif player.is_eliminated:
            hand_counts[pid] = 0
        elif (
            public_state.phase_id == "card_discard"
            and decision_player_id == pid
        ):
            hand_counts[pid] = 6
        else:
            hand_counts[pid] = min(5, max(0, hand_counts[pid]))

    available_mask = np.ones((NUM_CARDS,), dtype=np.bool_)
    explicitly_unavailable_cards: list[str] = []
    for own_card in observation.private_state.hand_card_ids:
        card_idx = _CARD_INDEX_BY_ID.get(own_card)
        if card_idx is not None:
            available_mask[card_idx] = False
            explicitly_unavailable_cards.append(own_card)

    for pool in public_state.reinforcement_pools:
        if pool.source_card_id is not None:
            card_idx = _CARD_INDEX_BY_ID.get(pool.source_card_id)
            if card_idx is not None and available_mask[card_idx]:
                available_mask[card_idx] = False
                explicitly_unavailable_cards.append(pool.source_card_id)

    for action in observed_actions:
        op = action.operation
        if isinstance(op, DiscardTerritoryCardOperation):
            card_idx = _CARD_INDEX_BY_ID.get(op.card_id)
            if card_idx is not None and available_mask[card_idx]:
                available_mask[card_idx] = False
                explicitly_unavailable_cards.append(op.card_id)
        elif isinstance(op, TradeCardsOperation):
            for traded_card_id in op.card_ids:
                card_idx = _CARD_INDEX_BY_ID.get(traded_card_id)
                if card_idx is not None and available_mask[card_idx]:
                    available_mask[card_idx] = False
                    explicitly_unavailable_cards.append(traded_card_id)

    return hand_counts, available_mask, tuple(explicitly_unavailable_cards)


def build_player_knowledge_state(
    observation: PlayerObservation,
    *,
    observed_actions: Sequence[Action] = (),
) -> PlayerKnowledgeState:
    """Build verified KnownFact entries without promoting probabilistic hypotheses to facts."""
    public_state = observation.public_state
    focal_id = observation.player_id
    known_at = (
        observation.recent_events[-1].occurred_at
        if observation.recent_events
        else _DEFAULT_KNOWN_AT
    )
    last_event_id = (
        observation.recent_events[-1].event_id
        if observation.recent_events
        else None
    )

    facts: list[KnownFact] = []
    if observation.private_state.objective_id is not None:
        facts.append(
            KnownFact(
                fact_id=f"{observation.game_id}:{focal_id}:own_objective",
                subject_id=focal_id,
                predicate="has_objective_card",
                value=observation.private_state.objective_id,
                source_event_id=None,
                known_at=known_at,
            )
        )
    if observation.private_state.objective_fallback_id is not None:
        facts.append(
            KnownFact(
                fact_id=f"{observation.game_id}:{focal_id}:own_objective_fallback",
                subject_id=focal_id,
                predicate="has_objective_fallback",
                value=observation.private_state.objective_fallback_id,
                source_event_id=None,
                known_at=known_at,
            )
        )
    for card_id in observation.private_state.hand_card_ids:
        facts.append(
            KnownFact(
                fact_id=f"{observation.game_id}:{focal_id}:holds_card:{card_id}",
                subject_id=focal_id,
                predicate="holds_territory_card",
                value=card_id,
                source_event_id=None,
                known_at=known_at,
            )
        )
    for pool in public_state.reinforcement_pools:
        if pool.source_card_id is not None:
            facts.append(
                KnownFact(
                    fact_id=(
                        f"{observation.game_id}:pool:{pool.pool_id}:"
                        f"traded_card:{pool.source_card_id}"
                    ),
                    subject_id=pool.player_id,
                    predicate="traded_territory_card",
                    value=pool.source_card_id,
                    source_event_id=last_event_id,
                    known_at=known_at,
                )
            )
    for action in observed_actions:
        if isinstance(action.operation, DiscardTerritoryCardOperation):
            card_id = action.operation.card_id
            facts.append(
                KnownFact(
                    fact_id=f"{observation.game_id}:discarded:{action.action_id}:{card_id}",
                    subject_id=action.actor_player_id,
                    predicate="permanently_discarded_card",
                    value=card_id,
                    source_event_id=last_event_id,
                    known_at=known_at,
                )
            )
    if (
        public_state.is_finished
        and public_state.winner_player_id is not None
        and public_state.winner_objective_id is not None
    ):
        facts.append(
            KnownFact(
                fact_id=f"{observation.game_id}:winner:{public_state.winner_player_id}",
                subject_id=public_state.winner_player_id,
                predicate="revealed_winner_objective",
                value=public_state.winner_objective_id,
                source_event_id=last_event_id,
                known_at=known_at,
            )
        )

    return PlayerKnowledgeState(
        contract_type="player_knowledge_state",
        game_id=observation.game_id,
        player_id=focal_id,
        state_version=observation.state_version,
        known_facts=tuple(facts),
    )


def _normalize_masked_rows(
    scores: NDArray[np.float64],
    feasible_mask: NDArray[np.bool_],
) -> NDArray[np.float64]:
    probs = np.zeros_like(scores, dtype=np.float64)
    for row_idx in range(scores.shape[0]):
        valid = feasible_mask[row_idx]
        if not np.any(valid):
            continue
        row_scores = np.where(valid, np.maximum(scores[row_idx], 1e-8), 0.0)
        total = float(np.sum(row_scores))
        if total <= 0.0:
            probs[row_idx, valid] = 1.0 / float(np.sum(valid))
        else:
            probs[row_idx] = row_scores / total
    return probs


def _compute_card_marginals_from_scores(
    card_scores: NDArray[np.float64],
    available_card_mask: NDArray[np.bool_],
    estimated_hand_counts: Mapping[str, int],
    public_state: PublicGameState,
    focal_player_id: str,
) -> NDArray[np.float64]:
    marginals = np.zeros((MAX_PLAYERS, NUM_CARDS), dtype=np.float64)
    available_indices = np.flatnonzero(available_card_mask)
    if available_indices.size == 0:
        return marginals

    for seat_idx, player in enumerate(public_state.players):
        if (
            seat_idx >= MAX_PLAYERS
            or player.player_id == focal_player_id
            or player.is_eliminated
        ):
            continue
        hand_count = int(estimated_hand_counts.get(player.player_id, 0))
        if hand_count <= 0:
            continue
        weights = np.where(
            available_card_mask,
            np.maximum(card_scores[seat_idx], 1e-6),
            0.0,
        )
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            marginals[seat_idx, available_card_mask] = min(
                1.0,
                float(hand_count) / float(available_indices.size),
            )
        else:
            scaled = weights * (float(hand_count) / weight_sum)
            marginals[seat_idx] = np.clip(scaled, 0.0, 1.0)
            marginals[seat_idx, ~available_card_mask] = 0.0
    return marginals


def _sample_contract_hypotheses(
    *,
    observation: PlayerObservation,
    objective_probabilities: NDArray[np.float64],
    card_probabilities: NDArray[np.float64],
    estimated_hand_counts: Mapping[str, int],
    available_card_mask: NDArray[np.bool_],
    num_hypotheses: int = 8,
    seed: int = 0,
) -> BeliefState:
    """Construct a contract-validated BeliefState with joint opponent hypotheses."""
    rng = np.random.default_rng(seed + int(observation.state_version) * 97)
    public_state = observation.public_state
    focal_id = observation.player_id
    opponent_seats = [
        (seat_idx, player.player_id)
        for seat_idx, player in enumerate(public_state.players)
        if seat_idx < MAX_PLAYERS and player.player_id != focal_id
    ]
    available_cards = [
        CARD_IDS[idx]
        for idx in range(NUM_CARDS)
        if bool(available_card_mask[idx])
    ]

    raw_hypotheses: dict[str, tuple[float, dict[str, JsonValue]]] = {}
    for hyp_idx in range(max(1, num_hypotheses)):
        used_objectives: set[int] = set()
        if observation.private_state.objective_id is not None:
            own_idx = _OBJECTIVE_INDEX_BY_ID.get(observation.private_state.objective_id)
            if own_idx is not None:
                used_objectives.add(own_idx)

        remaining_cards = list(available_cards)
        log_score = 0.0
        objective_map: dict[str, JsonValue] = {}
        hand_map: dict[str, JsonValue] = {}

        for seat_idx, opp_id in opponent_seats:
            row_probs = objective_probabilities[seat_idx].copy()
            for used in used_objectives:
                row_probs[used] = 0.0
            total = float(np.sum(row_probs))
            if total <= 0.0:
                fallback_candidates = [
                    i for i in range(NUM_OBJECTIVES) if i not in used_objectives
                ]
                chosen_obj_idx = int(rng.choice(fallback_candidates))
                chosen_prob = 1.0 / max(1, len(fallback_candidates))
            else:
                row_probs /= total
                if hyp_idx == 0:
                    chosen_obj_idx = int(np.argmax(row_probs))
                else:
                    chosen_obj_idx = int(rng.choice(NUM_OBJECTIVES, p=row_probs))
                chosen_prob = float(max(row_probs[chosen_obj_idx], 1e-6))

            used_objectives.add(chosen_obj_idx)
            objective_map[opp_id] = OBJECTIVE_IDS[chosen_obj_idx]
            log_score += math.log(chosen_prob)

            hand_size = min(
                int(estimated_hand_counts.get(opp_id, 0)),
                len(remaining_cards),
            )
            if hand_size > 0:
                card_weights = np.array(
                    [
                        max(float(card_probabilities[seat_idx, _CARD_INDEX_BY_ID[cid]]), 1e-4)
                        for cid in remaining_cards
                    ],
                    dtype=np.float64,
                )
                card_weights /= float(np.sum(card_weights))
                if hyp_idx == 0:
                    top_indices = np.argsort(-card_weights)[:hand_size]
                    sampled_cards = sorted(remaining_cards[int(i)] for i in top_indices)
                else:
                    chosen_indices = rng.choice(
                        len(remaining_cards),
                        size=hand_size,
                        replace=False,
                        p=card_weights,
                    )
                    sampled_cards = sorted(remaining_cards[int(i)] for i in chosen_indices)
                for cid in sampled_cards:
                    remaining_cards.remove(cid)
                hand_map[opp_id] = list(sampled_cards)
            else:
                hand_map[opp_id] = []

        signature = repr(( sorted(objective_map.items()), sorted(hand_map.items()) ))
        weight = math.exp(max(log_score, -30.0))
        if signature in raw_hypotheses:
            prev_w, prev_assign = raw_hypotheses[signature]
            raw_hypotheses[signature] = (prev_w + weight, prev_assign)
        else:
            raw_hypotheses[signature] = (
                weight,
                {
                    "opponent_objectives": objective_map,
                    "opponent_hands": hand_map,
                },
            )

    sorted_items = sorted(
        raw_hypotheses.values(),
        key=lambda item: item[0],
        reverse=True,
    )
    total_weight = sum(weight for weight, _ in sorted_items) or 1.0
    hypotheses = tuple(
        BeliefHypothesis(
            hypothesis_id=f"hyp-{idx}",
            probability=round(weight / total_weight, 6),
            assignments=assignments,
        )
        for idx, (weight, assignments) in enumerate(sorted_items)
    )
    prob_sum = sum(h.probability for h in hypotheses)
    if prob_sum > 1.0 and hypotheses:
        adjusted = max(0.0, round(hypotheses[-1].probability - (prob_sum - 1.0), 6))
        hypotheses = (
            *hypotheses[:-1],
            BeliefHypothesis(
                hypothesis_id=hypotheses[-1].hypothesis_id,
                probability=adjusted,
                assignments=hypotheses[-1].assignments,
            ),
        )

    event_ids = tuple(event.event_id for event in observation.recent_events[-32:])
    return BeliefState(
        contract_type="belief_state",
        game_id=observation.game_id,
        player_id=focal_id,
        state_version=observation.state_version,
        conditioned_on_event_ids=event_ids,
        hypotheses=hypotheses,
    )


def _behavioral_objective_log_likelihoods(
    observation: PlayerObservation,
    *,
    observed_actions: Sequence[Action] = (),
) -> NDArray[np.float64]:
    """Compute per-seat log-likelihood scores over objectives from history and board."""
    public_state = observation.public_state
    focal_id = observation.player_id
    seat_by_player_id = {
        player.player_id: idx
        for idx, player in enumerate(public_state.players)
        if idx < MAX_PLAYERS
    }
    owner_by_territory = {
        territory.territory_id: territory.owner_player_id
        for territory in public_state.territories
    }
    armies_by_territory = {
        territory.territory_id: territory.army_count
        for territory in public_state.territories
    }
    color_by_player_id = {
        player.player_id: player.color for player in public_state.players
    }

    continent_focus = np.zeros(
        (MAX_PLAYERS, len(WAR1_BOARD_SPEC.map.continents)),
        dtype=np.float64,
    )
    continent_index = {
        c.continent_id: idx
        for idx, c in enumerate(WAR1_BOARD_SPEC.map.continents)
    }
    color_attack_pressure: dict[tuple[int, str], float] = {}
    two_army_spread_signal = np.zeros((MAX_PLAYERS,), dtype=np.float64)
    expansion_signal = np.zeros((MAX_PLAYERS,), dtype=np.float64)

    for territory in public_state.territories:
        owner = territory.owner_player_id
        if owner is None or owner not in seat_by_player_id:
            continue
        seat_idx = seat_by_player_id[owner]
        c_id = _TERRITORY_CONTINENT_BY_ID[territory.territory_id]
        continent_focus[seat_idx, continent_index[c_id]] += 1.0 + 0.15 * min(
            territory.army_count,
            8,
        )
        if territory.army_count >= 2:
            two_army_spread_signal[seat_idx] += 0.25
        expansion_signal[seat_idx] += 0.12

        for neighbor_id in _ADJACENT_TERRITORIES[territory.territory_id]:
            neighbor_owner = owner_by_territory.get(neighbor_id)
            if (
                neighbor_owner is not None
                and neighbor_owner != owner
                and territory.army_count >= 2
            ):
                target_color = color_by_player_id.get(neighbor_owner, "")
                key = (seat_idx, target_color)
                color_attack_pressure[key] = color_attack_pressure.get(key, 0.0) + (
                    0.08 * min(territory.army_count - 1, 5)
                )

    for player in public_state.players:
        if (
            player.is_eliminated
            and player.eliminated_by_player_id is not None
            and player.eliminated_by_player_id in seat_by_player_id
        ):
            eliminator_seat = seat_by_player_id[player.eliminated_by_player_id]
            key = (eliminator_seat, player.color)
            color_attack_pressure[key] = color_attack_pressure.get(key, 0.0) + 2.0

    for event in observation.recent_events:
        actor = event.actor_player_id
        if actor is None or actor not in seat_by_player_id:
            continue
        seat_idx = seat_by_player_id[actor]
        etype = event.event_type
        data = event.data

        if etype in ("attack", "occupy_territory", "move_armies"):
            target_tid = data.get("target_territory_id")
            if isinstance(target_tid, str) and target_tid in _TERRITORY_CONTINENT_BY_ID:
                c_id = _TERRITORY_CONTINENT_BY_ID[target_tid]
                weight = 0.45 if etype == "attack" else 0.30
                continent_focus[seat_idx, continent_index[c_id]] += weight
                if etype in ("attack", "occupy_territory"):
                    expansion_signal[seat_idx] += 0.18
                    target_owner = owner_by_territory.get(target_tid)
                    if target_owner is not None and target_owner != actor:
                        t_color = color_by_player_id.get(target_owner, "")
                        key = (seat_idx, t_color)
                        color_attack_pressure[key] = (
                            color_attack_pressure.get(key, 0.0) + 0.35
                        )
        elif etype == "place_armies":
            placements = data.get("placements")
            if isinstance(placements, list):
                for item in placements:
                    if not isinstance(item, dict):
                        continue
                    tid = item.get("territory_id")
                    if isinstance(tid, str) and tid in _TERRITORY_CONTINENT_BY_ID:
                        c_id = _TERRITORY_CONTINENT_BY_ID[tid]
                        continent_focus[seat_idx, continent_index[c_id]] += 0.25
                        if armies_by_territory.get(tid, 0) == 2:
                            two_army_spread_signal[seat_idx] += 0.15

    for action in observed_actions:
        actor = action.actor_player_id
        if actor not in seat_by_player_id:
            continue
        seat_idx = seat_by_player_id[actor]
        op = action.operation
        if isinstance(op, (AttackOperation, OccupyTerritoryOperation, MoveArmiesOperation)):
            action_c_id = _TERRITORY_CONTINENT_BY_ID.get(op.target_territory_id)
            if action_c_id is not None:
                continent_focus[seat_idx, continent_index[action_c_id]] += 0.35
        elif isinstance(op, ClaimTerritoryOperation):
            action_c_id = _TERRITORY_CONTINENT_BY_ID.get(op.territory_id)
            if action_c_id is not None:
                continent_focus[seat_idx, continent_index[action_c_id]] += 0.40
        elif isinstance(op, PlaceArmiesOperation):
            for placement in op.placements:
                action_c_id = _TERRITORY_CONTINENT_BY_ID.get(placement.territory_id)
                if action_c_id is not None:
                    continent_focus[seat_idx, continent_index[action_c_id]] += 0.25

    log_scores = np.zeros((MAX_PLAYERS, NUM_OBJECTIVES), dtype=np.float64)
    for seat_idx, player in enumerate(public_state.players):
        if seat_idx >= MAX_PLAYERS or player.player_id == focal_id:
            continue
        total_focus = float(np.sum(continent_focus[seat_idx])) + 1e-6
        norm_continent_focus = continent_focus[seat_idx] / total_focus

        for obj_idx, obj_id in enumerate(OBJECTIVE_IDS):
            effective_id = effective_objective_for_opponent_hypothesis(
                public_state,
                player,
                obj_id,
            )
            condition = OBJECTIVE_CONDITIONS[effective_id]
            if isinstance(condition, EliminateColorCondition):
                pressure = color_attack_pressure.get(
                    (seat_idx, condition.target_color),
                    0.0,
                )
                log_scores[seat_idx, obj_idx] = 0.55 * min(pressure, 4.0)
            elif isinstance(condition, ControlContinentsCondition):
                req_share = sum(
                    float(norm_continent_focus[continent_index[cid]])
                    for cid in condition.required_continent_ids
                )
                bonus = 0.15 * float(condition.additional_continent_count)
                log_scores[seat_idx, obj_idx] = 1.6 * req_share + bonus
            elif isinstance(condition, TerritoryCountCondition):
                if condition.minimum_armies_per_territory >= 2:
                    log_scores[seat_idx, obj_idx] = 0.22 * min(
                        two_army_spread_signal[seat_idx],
                        6.0,
                    )
                else:
                    log_scores[seat_idx, obj_idx] = 0.20 * min(
                        expansion_signal[seat_idx],
                        6.0,
                    )
    return log_scores


class SimpleRulePriorBeliefEstimator:
    """Rule-distribution prior conditioned on own private state and public exclusions."""

    method_id: BeliefMethodId = "simple_prior"

    def estimate(
        self,
        observation: PlayerObservation,
        *,
        observed_actions: Sequence[Action] = (),
    ) -> OpponentBeliefEstimate:
        feasible_mask = compute_feasible_objective_mask(observation)
        hand_counts, available_card_mask, _ = (
            estimate_public_hand_counts_and_card_availability(
                observation,
                observed_actions=observed_actions,
            )
        )
        uniform_scores = np.ones((MAX_PLAYERS, NUM_OBJECTIVES), dtype=np.float64)
        objective_probs = _normalize_masked_rows(uniform_scores, feasible_mask)

        uniform_card_scores = np.ones((MAX_PLAYERS, NUM_CARDS), dtype=np.float64)
        card_probs = _compute_card_marginals_from_scores(
            uniform_card_scores,
            available_card_mask,
            hand_counts,
            observation.public_state,
            observation.player_id,
        )
        knowledge_state = build_player_knowledge_state(
            observation,
            observed_actions=observed_actions,
        )
        contract_belief = _sample_contract_hypotheses(
            observation=observation,
            objective_probabilities=objective_probs,
            card_probabilities=card_probs,
            estimated_hand_counts=hand_counts,
            available_card_mask=available_card_mask,
        )
        opponents = tuple(
            p.player_id
            for p in observation.public_state.players
            if p.player_id != observation.player_id
        )
        return OpponentBeliefEstimate(
            method_id="simple_prior",
            game_id=observation.game_id,
            focal_player_id=observation.player_id,
            state_version=int(observation.state_version),
            opponent_player_ids=opponents,
            objective_probabilities=objective_probs,
            card_probabilities=card_probs,
            estimated_hand_counts=hand_counts,
            feasible_objective_mask=feasible_mask,
            available_card_mask=available_card_mask,
            knowledge_state=knowledge_state,
            contract_belief_state=contract_belief,
        )


@dataclass(frozen=True)
class WorldParticle:
    """Single joint hypothesis over opponent objectives and card hands."""

    objective_index_by_seat: Mapping[int, int]
    card_indices_by_seat: Mapping[int, tuple[int, ...]]
    weight: float


class SequentialParticleBeliefFilter:
    """Sequential Monte Carlo particle filter over joint opponent secret assignments."""

    method_id: BeliefMethodId = "particle_filter"

    def __init__(
        self,
        *,
        num_particles: int = 96,
        ess_resample_ratio: float = 0.5,
        seed: int = 42,
    ) -> None:
        if isinstance(num_particles, bool) or num_particles < 8:
            raise ValueError("num_particles must be at least 8.")
        if not 0.0 < ess_resample_ratio <= 1.0:
            raise ValueError("ess_resample_ratio must be in (0, 1].")
        self.num_particles = num_particles
        self.ess_resample_ratio = ess_resample_ratio
        self.seed = seed
        self._particles: list[WorldParticle] = []
        self._last_game_key: tuple[str, str] | None = None

    @property
    def particles(self) -> tuple[WorldParticle, ...]:
        return tuple(self._particles)

    def effective_sample_size(self) -> float:
        if not self._particles:
            return 0.0
        weights = np.array([p.weight for p in self._particles], dtype=np.float64)
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        norm = weights / total
        denom = float(np.sum(norm * norm))
        return 0.0 if denom <= 0.0 else 1.0 / denom

    def estimate(
        self,
        observation: PlayerObservation,
        *,
        observed_actions: Sequence[Action] = (),
    ) -> OpponentBeliefEstimate:
        feasible_mask = compute_feasible_objective_mask(observation)
        hand_counts, available_card_mask, _ = (
            estimate_public_hand_counts_and_card_availability(
                observation,
                observed_actions=observed_actions,
            )
        )
        log_likelihoods = _behavioral_objective_log_likelihoods(
            observation,
            observed_actions=observed_actions,
        )
        rng = np.random.default_rng(
            self.seed + int(observation.state_version) * 131
        )
        public_state = observation.public_state
        focal_id = observation.player_id
        opponent_seats = [
            seat_idx
            for seat_idx, player in enumerate(public_state.players)
            if seat_idx < MAX_PLAYERS and player.player_id != focal_id
        ]
        available_card_indices = np.flatnonzero(available_card_mask).tolist()

        game_key = (observation.game_id, focal_id)
        if (
            self._last_game_key != game_key
            or not self._particles
            or tuple(sorted(self._particles[0].objective_index_by_seat))
            != tuple(opponent_seats)
        ):
            self._particles = self._sample_initial_particles(
                opponent_seats=opponent_seats,
                feasible_mask=feasible_mask,
                hand_counts=hand_counts,
                public_state=public_state,
                available_card_indices=available_card_indices,
                rng=rng,
            )
            self._last_game_key = game_key

        updated_particles: list[WorldParticle] = []
        for particle in self._particles:
            valid = True
            log_w = math.log(max(particle.weight, 1e-12))
            repaired_cards: dict[int, tuple[int, ...]] = {}
            used_cards: set[int] = set()

            for seat_idx in opponent_seats:
                obj_idx = particle.objective_index_by_seat[seat_idx]
                if not feasible_mask[seat_idx, obj_idx]:
                    valid = False
                    break
                log_w += float(log_likelihoods[seat_idx, obj_idx])

                player_id = public_state.players[seat_idx].player_id
                target_hand_size = int(hand_counts.get(player_id, 0))
                existing = [
                    c_idx
                    for c_idx in particle.card_indices_by_seat.get(seat_idx, ())
                    if bool(available_card_mask[c_idx]) and c_idx not in used_cards
                ][:target_hand_size]
                if len(existing) < target_hand_size:
                    pool = [
                        c_idx
                        for c_idx in available_card_indices
                        if c_idx not in used_cards and c_idx not in existing
                    ]
                    need = min(target_hand_size - len(existing), len(pool))
                    if need > 0:
                        sampled = rng.choice(pool, size=need, replace=False).tolist()
                        existing.extend(int(x) for x in sampled)
                used_cards.update(existing)
                repaired_cards[seat_idx] = tuple(sorted(existing))

            if valid:
                updated_particles.append(
                    WorldParticle(
                        objective_index_by_seat=particle.objective_index_by_seat,
                        card_indices_by_seat=repaired_cards,
                        weight=math.exp(max(min(log_w, 30.0), -30.0)),
                    )
                )

        if len(updated_particles) < max(4, self.num_particles // 4):
            fresh = self._sample_initial_particles(
                opponent_seats=opponent_seats,
                feasible_mask=feasible_mask,
                hand_counts=hand_counts,
                public_state=public_state,
                available_card_indices=available_card_indices,
                rng=rng,
                log_likelihoods=log_likelihoods,
            )
            updated_particles.extend(fresh)

        total_w = sum(p.weight for p in updated_particles) or 1.0
        self._particles = [
            WorldParticle(
                objective_index_by_seat=p.objective_index_by_seat,
                card_indices_by_seat=p.card_indices_by_seat,
                weight=p.weight / total_w,
            )
            for p in updated_particles[: self.num_particles]
        ]

        if self.effective_sample_size() < self.ess_resample_ratio * len(self._particles):
            self._particles = self._systematic_resample_and_rejuvenate(
                self._particles,
                opponent_seats=opponent_seats,
                feasible_mask=feasible_mask,
                log_likelihoods=log_likelihoods,
                rng=rng,
            )

        objective_counts = np.zeros((MAX_PLAYERS, NUM_OBJECTIVES), dtype=np.float64)
        card_counts = np.zeros((MAX_PLAYERS, NUM_CARDS), dtype=np.float64)
        for particle in self._particles:
            for seat_idx, obj_idx in particle.objective_index_by_seat.items():
                objective_counts[seat_idx, obj_idx] += particle.weight
            for seat_idx, c_indices in particle.card_indices_by_seat.items():
                for c_idx in c_indices:
                    card_counts[seat_idx, c_idx] += particle.weight

        # Laplace smoothing with feasible rule prior so zero-particle cells remain calibrated
        smoothed_obj = objective_counts + 0.12 * feasible_mask.astype(np.float64)
        objective_probs = _normalize_masked_rows(smoothed_obj, feasible_mask)

        smoothed_card_scores = card_counts + 0.15 * available_card_mask.astype(np.float64)
        card_probs = _compute_card_marginals_from_scores(
            smoothed_card_scores,
            available_card_mask,
            hand_counts,
            public_state,
            focal_id,
        )
        knowledge_state = build_player_knowledge_state(
            observation,
            observed_actions=observed_actions,
        )
        contract_belief = _sample_contract_hypotheses(
            observation=observation,
            objective_probabilities=objective_probs,
            card_probabilities=card_probs,
            estimated_hand_counts=hand_counts,
            available_card_mask=available_card_mask,
            seed=self.seed,
        )
        opponents = tuple(
            p.player_id for p in public_state.players if p.player_id != focal_id
        )
        return OpponentBeliefEstimate(
            method_id="particle_filter",
            game_id=observation.game_id,
            focal_player_id=focal_id,
            state_version=int(observation.state_version),
            opponent_player_ids=opponents,
            objective_probabilities=objective_probs,
            card_probabilities=card_probs,
            estimated_hand_counts=hand_counts,
            feasible_objective_mask=feasible_mask,
            available_card_mask=available_card_mask,
            knowledge_state=knowledge_state,
            contract_belief_state=contract_belief,
        )

    def _sample_initial_particles(
        self,
        *,
        opponent_seats: Sequence[int],
        feasible_mask: NDArray[np.bool_],
        hand_counts: Mapping[str, int],
        public_state: PublicGameState,
        available_card_indices: Sequence[int],
        rng: Generator,
        log_likelihoods: NDArray[np.float64] | None = None,
    ) -> list[WorldParticle]:
        particles: list[WorldParticle] = []
        for _ in range(self.num_particles):
            used_objectives: set[int] = set()
            obj_by_seat: dict[int, int] = {}
            log_w = 0.0
            for seat_idx in opponent_seats:
                candidates = [
                    idx
                    for idx in range(NUM_OBJECTIVES)
                    if bool(feasible_mask[seat_idx, idx]) and idx not in used_objectives
                ]
                if not candidates:
                    candidates = [
                        idx
                        for idx in range(NUM_OBJECTIVES)
                        if bool(feasible_mask[seat_idx, idx])
                    ]
                if log_likelihoods is not None:
                    scores = np.array(
                        [math.exp(float(log_likelihoods[seat_idx, idx])) for idx in candidates],
                        dtype=np.float64,
                    )
                    scores /= float(np.sum(scores))
                    chosen = int(rng.choice(candidates, p=scores))
                else:
                    chosen = int(rng.choice(candidates))
                used_objectives.add(chosen)
                obj_by_seat[seat_idx] = chosen
                if log_likelihoods is not None:
                    log_w += float(log_likelihoods[seat_idx, chosen])

            remaining_cards = list(available_card_indices)
            rng.shuffle(remaining_cards)
            cards_by_seat: dict[int, tuple[int, ...]] = {}
            cursor = 0
            for seat_idx in opponent_seats:
                pid = public_state.players[seat_idx].player_id
                count = max(0, int(hand_counts.get(pid, 0)))
                sliced = remaining_cards[cursor : cursor + count]
                cursor += len(sliced)
                cards_by_seat[seat_idx] = tuple(sorted(int(x) for x in sliced))

            particles.append(
                WorldParticle(
                    objective_index_by_seat=obj_by_seat,
                    card_indices_by_seat=cards_by_seat,
                    weight=math.exp(max(min(log_w, 20.0), -20.0)),
                )
            )
        total = sum(p.weight for p in particles) or 1.0
        return [
            WorldParticle(
                objective_index_by_seat=p.objective_index_by_seat,
                card_indices_by_seat=p.card_indices_by_seat,
                weight=p.weight / total,
            )
            for p in particles
        ]

    def _systematic_resample_and_rejuvenate(
        self,
        particles: Sequence[WorldParticle],
        *,
        opponent_seats: Sequence[int],
        feasible_mask: NDArray[np.bool_],
        log_likelihoods: NDArray[np.float64],
        rng: Generator,
    ) -> list[WorldParticle]:
        n = len(particles)
        weights = np.array([p.weight for p in particles], dtype=np.float64)
        weights /= float(np.sum(weights))
        cdf = np.cumsum(weights)
        u0 = float(rng.uniform(0.0, 1.0 / n))
        resampled: list[WorldParticle] = []
        idx = 0
        for step in range(n):
            u = u0 + step / float(n)
            while u > cdf[idx] and idx < n - 1:
                idx += 1
            chosen = particles[idx]
            obj_map = dict(chosen.objective_index_by_seat)
            if opponent_seats and rng.random() < 0.35:
                seat_to_mutate = int(rng.choice(opponent_seats))
                used_other = {
                    obj_i
                    for s_i, obj_i in obj_map.items()
                    if s_i != seat_to_mutate
                }
                candidates = [
                    obj_i
                    for obj_i in range(NUM_OBJECTIVES)
                    if bool(feasible_mask[seat_to_mutate, obj_i])
                    and obj_i not in used_other
                ]
                if candidates:
                    scores = np.array(
                        [
                            math.exp(float(log_likelihoods[seat_to_mutate, c]))
                            for c in candidates
                        ],
                        dtype=np.float64,
                    )
                    scores /= float(np.sum(scores))
                    obj_map[seat_to_mutate] = int(rng.choice(candidates, p=scores))
            resampled.append(
                WorldParticle(
                    objective_index_by_seat=obj_map,
                    card_indices_by_seat=dict(chosen.card_indices_by_seat),
                    weight=1.0 / float(n),
                )
            )
        return resampled


class LearnedBeliefModelEstimator(nn.Module):
    """Neural belief estimator combining WarArchitectureModel with rule masks and calibration."""

    method_id: BeliefMethodId = "learned_model"

    def __init__(
        self,
        config: ArchitectureConfig = ArchitectureConfig(
            encoder="graph_transformer",
            memory="causal_event_transformer",
            hidden_size=64,
            graph_layers=2,
            attention_heads=4,
            use_belief_head=True,
        ),
        *,
        objective_temperature: float = 1.0,
        card_temperature: float = 1.0,
        behavioral_prior_weight: float = 0.65,
    ) -> None:
        super().__init__()
        if not config.use_belief_head:
            raise ValueError("LearnedBeliefModelEstimator requires use_belief_head=True.")
        if objective_temperature <= 0.0 or card_temperature <= 0.0:
            raise ValueError("Temperatures must be positive.")
        self.config = config
        self.model = WarArchitectureModel(config)
        self.objective_temperature = float(objective_temperature)
        self.card_temperature = float(card_temperature)
        self.behavioral_prior_weight = float(behavioral_prior_weight)
        self._encoders_by_num_players: dict[int, War1ObservationEncoder] = {}

    def _encoder_for(self, num_players: int) -> War1ObservationEncoder:
        encoder = self._encoders_by_num_players.get(num_players)
        if encoder is None:
            encoder = War1ObservationEncoder(num_players)
            self._encoders_by_num_players[num_players] = encoder
        return encoder

    def _to_aec_observation(
        self,
        observation: PlayerObservation,
    ) -> War1AECObservation:
        num_players = len(observation.public_state.players)
        encoder = self._encoder_for(num_players)
        numeric_obs = encoder.encode_player(observation)
        num_legal = max(1, len(observation.legal_actions))
        return {
            "observation": numeric_obs,
            "action_mask": np.ones((num_legal,), dtype=np.int8),
            "action_token_mask": np.ones((64,), dtype=np.int8),
            "engine_observation": observation,
            "legal_action_token_values": np.zeros((1,), dtype=np.int16),
            "legal_action_token_offsets": np.array([0, 1], dtype=np.int64),
        }

    @torch.no_grad()
    def estimate(
        self,
        observation: PlayerObservation,
        *,
        observed_actions: Sequence[Action] = (),
    ) -> OpponentBeliefEstimate:
        self.eval()
        feasible_mask = compute_feasible_objective_mask(observation)
        hand_counts, available_card_mask, _ = (
            estimate_public_hand_counts_and_card_availability(
                observation,
                observed_actions=observed_actions,
            )
        )
        behavioral_log_scores = _behavioral_objective_log_likelihoods(
            observation,
            observed_actions=observed_actions,
        )

        aec_obs = self._to_aec_observation(observation)
        inputs = extract_architecture_inputs(
            aec_obs,
            max_history_events=self.config.max_history_events,
        )
        dummy_tokens = torch.full(
            (1, 2),
            FULL_GAME_INTENT_END_TOKEN,
            dtype=torch.long,
        )
        output = self.model(inputs, dummy_tokens)
        assert output.opponent_objective_logits is not None
        assert output.opponent_card_logits is not None

        raw_obj_logits = (
            output.opponent_objective_logits[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        raw_card_logits = (
            output.opponent_card_logits[0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )

        combined_logits = (
            raw_obj_logits / self.objective_temperature
            + self.behavioral_prior_weight * behavioral_log_scores
        )
        objective_probs = np.zeros((MAX_PLAYERS, NUM_OBJECTIVES), dtype=np.float64)
        for seat_idx in range(MAX_PLAYERS):
            valid = feasible_mask[seat_idx]
            if not np.any(valid):
                continue
            valid_logits = combined_logits[seat_idx, valid]
            shifted = valid_logits - float(np.max(valid_logits))
            exp_vals = np.exp(shifted)
            objective_probs[seat_idx, valid] = exp_vals / float(np.sum(exp_vals))

        card_sigmoid = 1.0 / (
            1.0 + np.exp(-np.clip(raw_card_logits / self.card_temperature, -20.0, 20.0))
        )
        card_probs = _compute_card_marginals_from_scores(
            card_sigmoid,
            available_card_mask,
            hand_counts,
            observation.public_state,
            observation.player_id,
        )
        knowledge_state = build_player_knowledge_state(
            observation,
            observed_actions=observed_actions,
        )
        contract_belief = _sample_contract_hypotheses(
            observation=observation,
            objective_probabilities=objective_probs,
            card_probabilities=card_probs,
            estimated_hand_counts=hand_counts,
            available_card_mask=available_card_mask,
        )
        opponents = tuple(
            p.player_id
            for p in observation.public_state.players
            if p.player_id != observation.player_id
        )
        return OpponentBeliefEstimate(
            method_id="learned_model",
            game_id=observation.game_id,
            focal_player_id=observation.player_id,
            state_version=int(observation.state_version),
            opponent_player_ids=opponents,
            objective_probabilities=objective_probs,
            card_probabilities=card_probs,
            estimated_hand_counts=hand_counts,
            feasible_objective_mask=feasible_mask,
            available_card_mask=available_card_mask,
            knowledge_state=knowledge_state,
            contract_belief_state=contract_belief,
        )

    def fit_supervised(
        self,
        samples: Sequence[tuple[PlayerObservation, PrivilegedTableSecrets]],
        *,
        epochs: int = 12,
        learning_rate: float = 3e-3,
        seed: int = 7,
    ) -> list[float]:
        """Train the neural belief head on training-only secret labels and calibrate temp."""
        if not samples:
            raise ValueError("At least one training sample is required.")
        torch.manual_seed(seed)
        self.train()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

        aec_observations = [self._to_aec_observation(obs) for obs, _ in samples]
        inputs = extract_architecture_inputs(
            aec_observations,
            max_history_events=self.config.max_history_events,
        )
        combined_secrets = PrivilegedTableSecrets(
            opponent_objective_indices=torch.cat(
                [sec.opponent_objective_indices for _, sec in samples],
                dim=0,
            ),
            opponent_card_targets=torch.cat(
                [sec.opponent_card_targets for _, sec in samples],
                dim=0,
            ),
            opponent_seat_mask=torch.cat(
                [sec.opponent_seat_mask for _, sec in samples],
                dim=0,
            ),
        )
        dummy_tokens = torch.full(
            (inputs.batch_size, 2),
            FULL_GAME_INTENT_END_TOKEN,
            dtype=torch.long,
        )

        losses: list[float] = []
        for _ in range(max(1, epochs)):
            optimizer.zero_grad()
            out = self.model(inputs, dummy_tokens)
            assert out.opponent_objective_logits is not None
            assert out.opponent_card_logits is not None
            obj_loss, card_loss = compute_secret_belief_losses(
                opponent_objective_logits=out.opponent_objective_logits,
                opponent_card_logits=out.opponent_card_logits,
                secrets=combined_secrets,
            )
            loss = obj_loss + card_loss
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            losses.append(float(loss.detach().item()))

        self.eval()
        self._calibrate_temperatures(samples)
        return losses

    @torch.no_grad()
    def _calibrate_temperatures(
        self,
        samples: Sequence[tuple[PlayerObservation, PrivilegedTableSecrets]],
    ) -> None:
        candidate_temps = (0.6, 0.8, 1.0, 1.25, 1.6, 2.2)
        best_temp = 1.0
        best_nll = float("inf")
        for temp in candidate_temps:
            self.objective_temperature = temp
            nll_sum = 0.0
            count = 0
            for obs, secrets in samples:
                est = self.estimate(obs)
                seat_mask = secrets.opponent_seat_mask[0].cpu().numpy()
                true_obj = secrets.opponent_objective_indices[0].cpu().numpy()
                for seat_idx in np.flatnonzero(seat_mask):
                    target = int(true_obj[seat_idx])
                    prob = float(
                        np.clip(est.objective_probabilities[seat_idx, target], 1e-6, 1.0)
                    )
                    nll_sum -= math.log(prob)
                    count += 1
            mean_nll = nll_sum / max(1, count)
            if mean_nll < best_nll:
                best_nll = mean_nll
                best_temp = temp
        self.objective_temperature = best_temp


def _compute_ece_and_bins(
    probabilities: NDArray[np.float64],
    outcomes: NDArray[np.float64],
    *,
    num_bins: int = 10,
) -> tuple[float, tuple[CalibrationBinStat, ...]]:
    flat_probs = probabilities.reshape(-1)
    flat_outcomes = outcomes.reshape(-1)
    total = max(1, flat_probs.size)
    bins: list[CalibrationBinStat] = []
    ece = 0.0
    edges = np.linspace(0.0, 1.0, num_bins + 1)

    for b_idx in range(num_bins):
        lo = float(edges[b_idx])
        hi = float(edges[b_idx + 1])
        if b_idx == num_bins - 1:
            in_bin = (flat_probs >= lo) & (flat_probs <= hi)
        else:
            in_bin = (flat_probs >= lo) & (flat_probs < hi)
        count = int(np.sum(in_bin))
        if count == 0:
            bins.append(
                CalibrationBinStat(
                    bin_lower=lo,
                    bin_upper=hi,
                    sample_count=0,
                    mean_confidence=0.0,
                    empirical_accuracy=0.0,
                    absolute_calibration_gap=0.0,
                )
            )
            continue
        mean_conf = float(np.mean(flat_probs[in_bin]))
        emp_acc = float(np.mean(flat_outcomes[in_bin]))
        gap = abs(mean_conf - emp_acc)
        ece += (count / float(total)) * gap
        bins.append(
            CalibrationBinStat(
                bin_lower=lo,
                bin_upper=hi,
                sample_count=count,
                mean_confidence=mean_conf,
                empirical_accuracy=emp_acc,
                absolute_calibration_gap=gap,
            )
        )
    return float(ece), tuple(bins)


def evaluate_belief_calibration(
    estimates_and_secrets: Sequence[tuple[OpponentBeliefEstimate, PrivilegedTableSecrets]],
    *,
    num_bins: int = 10,
) -> BeliefCalibrationMetrics:
    """Compute ECE, Brier score, log-loss, accuracy, and constraint compliance."""
    if not estimates_and_secrets:
        raise ValueError("At least one (estimate, secrets) pair is required.")

    method_id = estimates_and_secrets[0][0].method_id
    obj_prob_rows: list[NDArray[np.float64]] = []
    obj_onehot_rows: list[NDArray[np.float64]] = []
    obj_log_losses: list[float] = []
    obj_top1_hits: list[float] = []
    obj_top3_hits: list[float] = []
    true_obj_probs: list[float] = []

    card_prob_rows: list[NDArray[np.float64]] = []
    card_target_rows: list[NDArray[np.float64]] = []
    card_log_losses: list[float] = []
    violations = 0
    total_checks = 0

    for estimate, secrets in estimates_and_secrets:
        seat_mask = secrets.opponent_seat_mask[0].cpu().numpy().astype(bool)
        true_objs = secrets.opponent_objective_indices[0].cpu().numpy()
        true_cards = secrets.opponent_card_targets[0].cpu().numpy()

        for seat_idx in np.flatnonzero(seat_mask):
            total_checks += 1
            row_probs = estimate.objective_probabilities[seat_idx]
            infeasible_mass = float(
                np.sum(row_probs[~estimate.feasible_objective_mask[seat_idx]])
            )
            unavailable_card_mass = float(
                np.sum(estimate.card_probabilities[seat_idx, ~estimate.available_card_mask])
            )
            if infeasible_mass > 1e-6 or unavailable_card_mass > 1e-6:
                violations += 1

            target_idx = int(true_objs[seat_idx])
            one_hot = np.zeros((NUM_OBJECTIVES,), dtype=np.float64)
            one_hot[target_idx] = 1.0
            obj_prob_rows.append(row_probs)
            obj_onehot_rows.append(one_hot)

            true_p = float(np.clip(row_probs[target_idx], 1e-6, 1.0))
            true_obj_probs.append(true_p)
            obj_log_losses.append(-math.log(true_p))

            ranked = np.argsort(-row_probs)
            obj_top1_hits.append(1.0 if int(ranked[0]) == target_idx else 0.0)
            obj_top3_hits.append(
                1.0 if target_idx in set(int(x) for x in ranked[:3]) else 0.0
            )

            c_probs = np.clip(
                estimate.card_probabilities[seat_idx],
                1e-6,
                1.0 - 1e-6,
            )
            c_targets = true_cards[seat_idx].astype(np.float64)
            card_prob_rows.append(estimate.card_probabilities[seat_idx])
            card_target_rows.append(c_targets)
            bce = -float(
                np.mean(
                    c_targets * np.log(c_probs)
                    + (1.0 - c_targets) * np.log(1.0 - c_probs)
                )
            )
            card_log_losses.append(bce)

    obj_probs_mat = np.vstack(obj_prob_rows)
    obj_onehot_mat = np.vstack(obj_onehot_rows)
    obj_brier = float(np.mean(np.sum((obj_probs_mat - obj_onehot_mat) ** 2, axis=1)))
    obj_ece, obj_bins = _compute_ece_and_bins(
        obj_probs_mat,
        obj_onehot_mat,
        num_bins=num_bins,
    )

    card_probs_mat = np.vstack(card_prob_rows)
    card_targets_mat = np.vstack(card_target_rows)
    card_brier = float(np.mean((card_probs_mat - card_targets_mat) ** 2))
    card_ece, _ = _compute_ece_and_bins(
        card_probs_mat,
        card_targets_mat,
        num_bins=num_bins,
    )

    return BeliefCalibrationMetrics(
        method_id=method_id,
        num_samples=len(obj_prob_rows),
        objective_ece=obj_ece,
        objective_brier_score=obj_brier,
        objective_log_loss=float(np.mean(obj_log_losses)),
        objective_top1_accuracy=float(np.mean(obj_top1_hits)),
        objective_top3_accuracy=float(np.mean(obj_top3_hits)),
        mean_true_objective_probability=float(np.mean(true_obj_probs)),
        card_ece=card_ece,
        card_brier_score=card_brier,
        card_log_loss=float(np.mean(card_log_losses)),
        hard_constraint_violation_rate=float(violations) / float(max(1, total_checks)),
        objective_reliability_bins=obj_bins,
    )


def _select_goal_directed_rollout_action(
    env: War1AECEnv,
    agent_id: str,
    full_state: FullGameState,
    rng: Generator,
) -> int:
    """Select a goal-directed legal action during benchmark data collection."""
    obs = env.observe(agent_id)
    engine_obs = obs["engine_observation"]
    if engine_obs is None or len(engine_obs.legal_actions) <= 1:
        return 0

    player = next(p for p in full_state.players if p.public.player_id == agent_id)
    raw_obj_id = player.private.objective_id or FALLBACK_OBJECTIVE_ID
    eff_obj_id = effective_objective_for_opponent_hypothesis(
        engine_obs.public_state,
        player.public,
        raw_obj_id,
    )
    condition = OBJECTIVE_CONDITIONS[eff_obj_id]
    owner_by_tid = {
        t.territory_id: t.owner_player_id for t in engine_obs.public_state.territories
    }
    color_by_pid = {
        p.player_id: p.color for p in engine_obs.public_state.players
    }

    best_score = -1e9
    best_indices: list[int] = []
    for idx, action in enumerate(engine_obs.legal_actions):
        op = action.operation
        score = 0.0
        target_tid: str | None = None
        if isinstance(op, (AttackOperation, OccupyTerritoryOperation, MoveArmiesOperation)):
            target_tid = op.target_territory_id
        elif isinstance(op, ClaimTerritoryOperation):
            target_tid = op.territory_id
        elif isinstance(op, PlaceArmiesOperation) and op.placements:
            target_tid = op.placements[0].territory_id

        if target_tid is not None:
            if isinstance(condition, ControlContinentsCondition):
                if _TERRITORY_CONTINENT_BY_ID.get(target_tid) in condition.required_continent_ids:
                    score += 3.0
            elif isinstance(condition, EliminateColorCondition):
                t_owner = owner_by_tid.get(target_tid)
                if t_owner is not None and color_by_pid.get(t_owner) == condition.target_color:
                    score += 3.5
            elif isinstance(condition, TerritoryCountCondition):
                score += 1.0

        if score > best_score + 1e-9:
            best_score = score
            best_indices = [idx]
        elif abs(score - best_score) <= 1e-9:
            best_indices.append(idx)

    return int(rng.choice(best_indices))


def collect_belief_benchmark_samples(
    *,
    num_games: int = 6,
    num_players: int = 3,
    steps_per_game: int = 28,
    base_seed: int = 910,
) -> list[tuple[PlayerObservation, PrivilegedTableSecrets]]:
    """Collect player-scoped observations paired with ground-truth secrets for evaluation."""
    if num_games < 1 or steps_per_game < 1:
        raise ValueError("num_games and steps_per_game must be positive.")

    samples: list[tuple[PlayerObservation, PrivilegedTableSecrets]] = []
    rng = np.random.default_rng(base_seed)
    env = War1AECEnv(num_players=num_players)

    for game_idx in range(num_games):
        env.reset(seed=base_seed + game_idx * 17)
        for step_idx in range(steps_per_game):
            agent = env.agent_selection
            if agent is None:
                break
            if env.terminations[agent] or env.truncations[agent]:
                env.step(None)
                continue
            full_state = env._require_state()
            action_idx = _select_goal_directed_rollout_action(
                env,
                agent,
                full_state,
                rng,
            )
            env.step(action_idx)

            if (step_idx + 1) % max(1, steps_per_game // 3) == 0:
                current_state = env._require_state()
                focal_agent = env.possible_agents[step_idx % num_players]
                focal_obs = env.observe(focal_agent)["engine_observation"]
                if focal_obs is None:
                    continue
                # In the benchmark harness, remap raw_objective_id into secrets for exact check
                secrets = _extract_raw_objective_and_card_secrets(
                    current_state,
                    focal_player_id=focal_agent,
                )
                samples.append((focal_obs, secrets))

    return samples


def _extract_raw_objective_and_card_secrets(
    full_state: FullGameState,
    *,
    focal_player_id: str,
) -> PrivilegedTableSecrets:
    """Extract opponent raw drawn objective card and hand cards for belief evaluation."""
    base_secrets = extract_privileged_table_secrets(
        full_state,
        focal_player_id=focal_player_id,
    )
    raw_obj_indices = base_secrets.opponent_objective_indices.clone()
    for seat_idx, player in enumerate(full_state.players):
        if seat_idx >= MAX_PLAYERS or player.public.player_id == focal_player_id:
            continue
        raw_id = player.private.objective_id or FALLBACK_OBJECTIVE_ID
        raw_obj_indices[0, seat_idx] = _OBJECTIVE_INDEX_BY_ID.get(raw_id, 0)
    return PrivilegedTableSecrets(
        opponent_objective_indices=raw_obj_indices,
        opponent_card_targets=base_secrets.opponent_card_targets,
        opponent_seat_mask=base_secrets.opponent_seat_mask,
    )


def compare_belief_estimators(
    *,
    num_train_games: int = 4,
    num_eval_games: int = 4,
    num_players: int = 3,
    steps_per_game: int = 24,
    num_particles: int = 64,
    seed: int = 901,
) -> BeliefComparisonReport:
    """Compare simple prior, particle filter, and learned belief model on held-out games."""
    train_samples = collect_belief_benchmark_samples(
        num_games=num_train_games,
        num_players=num_players,
        steps_per_game=steps_per_game,
        base_seed=seed,
    )
    eval_samples = collect_belief_benchmark_samples(
        num_games=num_eval_games,
        num_players=num_players,
        steps_per_game=steps_per_game,
        base_seed=seed + 1000,
    )

    simple_estimator = SimpleRulePriorBeliefEstimator()
    particle_estimator = SequentialParticleBeliefFilter(
        num_particles=num_particles,
        seed=seed,
    )
    learned_estimator = LearnedBeliefModelEstimator()
    learned_estimator.fit_supervised(train_samples, epochs=10, seed=seed)

    simple_pairs = [(simple_estimator.estimate(obs), sec) for obs, sec in eval_samples]
    particle_pairs = [
        (particle_estimator.estimate(obs), sec) for obs, sec in eval_samples
    ]
    learned_pairs = [
        (learned_estimator.estimate(obs), sec) for obs, sec in eval_samples
    ]

    metrics_by_method: dict[BeliefMethodId, BeliefCalibrationMetrics] = {
        "simple_prior": evaluate_belief_calibration(simple_pairs),
        "particle_filter": evaluate_belief_calibration(particle_pairs),
        "learned_model": evaluate_belief_calibration(learned_pairs),
    }

    best_log_loss_method = min(
        BELIEF_METHODS,
        key=lambda m: metrics_by_method[m].objective_log_loss,
    )
    best_brier_method = min(
        BELIEF_METHODS,
        key=lambda m: metrics_by_method[m].objective_brier_score,
    )
    best_ece_method = min(
        BELIEF_METHODS,
        key=lambda m: metrics_by_method[m].objective_ece,
    )
    all_respect = all(
        metrics_by_method[m].hard_constraint_violation_rate == 0.0
        for m in BELIEF_METHODS
    )
    return BeliefComparisonReport(
        num_training_observations=len(train_samples),
        num_evaluation_observations=len(eval_samples),
        metrics_by_method=metrics_by_method,
        best_objective_log_loss_method=best_log_loss_method,
        best_objective_brier_method=best_brier_method,
        best_calibrated_method=best_ece_method,
        all_respect_hard_constraints=all_respect,
    )


__all__ = [
    "BELIEF_METHODS",
    "BeliefCalibrationMetrics",
    "BeliefComparisonReport",
    "BeliefMethodId",
    "CalibrationBinStat",
    "LearnedBeliefModelEstimator",
    "OpponentBeliefEstimate",
    "SequentialParticleBeliefFilter",
    "SimpleRulePriorBeliefEstimator",
    "WorldParticle",
    "build_player_knowledge_state",
    "collect_belief_benchmark_samples",
    "compare_belief_estimators",
    "compute_feasible_objective_mask",
    "effective_objective_for_opponent_hypothesis",
    "eligible_objective_ids_for_public_state",
    "estimate_public_hand_counts_and_card_availability",
    "evaluate_belief_calibration",
    "is_objective_already_satisfied_publicly",
]
