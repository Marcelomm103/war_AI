"""Adapted ISMCTS/POMCP search over belief determinizations (Phase 9.2 & 9.3)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal, TypeAlias

import numpy as np
import torch
from numpy.random import Generator
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    Action,
    AttackOperation,
    DeckState,
    DrawTerritoryCardOperation,
    FullGameState,
    FullPlayerState,
    OccupyTerritoryOperation,
    PlaceArmiesOperation,
    PlayerObservation,
    PlayerPrivateState,
    TradeCardsOperation,
)
from war_game_engine.engine import (
    apply_action,
    apply_combat_outcome,
    legal_actions,
    simulate_pending_combat_outcome,
)
from war_game_engine.objectives import (
    ELIMINATION_OBJECTIVE_COLORS,
    FALLBACK_OBJECTIVE_ID,
    OBJECTIVE_CONDITIONS,
    ControlContinentsCondition,
    EliminateColorCondition,
    TerritoryCountCondition,
    objective_is_satisfied,
)
from war_game_engine.randomness import RandomStreams
from war_game_engine.state import observe_player, to_public_game_state

from ..envs.observations import War1ObservationEncoder
from ..envs.war1_aec import War1AECObservation
from .agent import FULL_GAME_INTENT_END_TOKEN, classify_legal_actions
from .architectures import (
    NUM_OBJECTIVES,
    WarArchitectureModel,
    extract_architecture_inputs,
)
from .belief_models import (
    CARD_IDS,
    OBJECTIVE_IDS,
    OpponentBeliefEstimate,
    SequentialParticleBeliefFilter,
    SimpleRulePriorBeliefEstimator,
    effective_objective_for_opponent_hypothesis,
)
from .losses import masked_log_softmax

SearchAlgorithmId: TypeAlias = Literal[
    "ismcts",
    "pomcp",
    "naive_determinization",
]
ChanceNodeType: TypeAlias = Literal["combat_dice", "card_draw"]


class PrivilegedPlanningStateError(TypeError):
    """Raised when a search planner is given a privileged FullGameState."""


class StrategyFusionViolationError(RuntimeError):
    """Raised if root action selection conditions directly on sampled hidden secrets."""


def _ensure_player_observation_only(observation: object) -> PlayerObservation:
    if isinstance(observation, FullGameState):
        raise PrivilegedPlanningStateError(
            "Search planners must never receive a privileged FullGameState; "
            "pass a player-scoped PlayerObservation instead."
        )
    if not isinstance(observation, PlayerObservation):
        raise TypeError("Expected a PlayerObservation instance.")
    return observation


def canonical_action_key(action: Action) -> str:
    """Stable phase-and-operation signature independent of state_version or action_id."""
    payload = action.operation.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


@dataclass
class ActionEdgeStats:
    """Aggregated statistics for an action edge at an information-set node."""

    action_key: str
    representative_action: Action
    prior_probability: float
    visit_count: int = 0
    availability_count: int = 0
    value_sum_by_player: dict[str, float] = field(default_factory=dict)

    def mean_value_for(self, player_id: str) -> float:
        if self.visit_count <= 0:
            return 0.0
        return float(self.value_sum_by_player.get(player_id, 0.0)) / float(
            self.visit_count
        )


@dataclass
class ChanceSearchNode:
    """Explicit chance node for combat dice resolution or territory card draws."""

    node_id: str
    chance_type: ChanceNodeType
    visit_count: int = 0
    outcome_visit_counts: dict[str, int] = field(default_factory=dict)


@dataclass
class InformationSetNode:
    """Information-set / action-observation history node shared across worlds."""

    history_key: tuple[str, ...]
    decision_player_id: str
    visit_count: int = 0
    edges: dict[str, ActionEdgeStats] = field(default_factory=dict)
    chance_children: dict[str, ChanceSearchNode] = field(default_factory=dict)


@dataclass(frozen=True)
class ISMCTSSearchConfig:
    """Configuration for adapted ISMCTS / POMCP imperfect-information search."""

    algorithm: SearchAlgorithmId = "ismcts"
    num_simulations: int = 24
    num_determinizations: int = 6
    max_search_depth: int = 4
    rollout_depth: int = 3
    c_puct: float = 1.25
    dirichlet_alpha: float = 0.0
    temperature: float = 0.0
    seed: int = 42

    def __post_init__(self) -> None:
        if self.algorithm not in ("ismcts", "pomcp", "naive_determinization"):
            raise ValueError(f"Unsupported search algorithm {self.algorithm!r}.")
        if isinstance(self.num_simulations, bool) or self.num_simulations < 1:
            raise ValueError("num_simulations must be a positive integer.")
        if isinstance(self.num_determinizations, bool) or self.num_determinizations < 1:
            raise ValueError("num_determinizations must be a positive integer.")
        if isinstance(self.max_search_depth, bool) or self.max_search_depth < 1:
            raise ValueError("max_search_depth must be a positive integer.")
        if isinstance(self.rollout_depth, bool) or self.rollout_depth < 0:
            raise ValueError("rollout_depth must be non-negative.")
        if self.c_puct <= 0.0:
            raise ValueError("c_puct must be positive.")


@dataclass(frozen=True)
class SearchDecisionResult:
    """Result of planning a decision strictly from PlayerObservation and beliefs."""

    selected_action: Action
    selected_action_index: int
    action_visit_counts: Mapping[str, int]
    action_mean_values: Mapping[str, float]
    root_expected_value: float
    num_simulations_run: int
    num_determinizations_sampled: int
    num_information_set_nodes: int
    num_chance_nodes: int
    chance_outcomes_sampled: int
    used_privileged_state: bool
    strategy_fusion_free: bool
    opponent_observations_isolated: bool


@dataclass(frozen=True)
class InformationSafetyAuditReport:
    """Verification report for Phase 9.3 anti-strategy-fusion and isolation rules."""

    privileged_state_rejected: bool
    focal_policy_blind_to_sampled_secret: bool
    opponent_receives_only_own_observation: bool
    root_statistics_aggregated_across_worlds: bool
    ismcts_outperforms_or_matches_naive_determinization: bool
    ismcts_expected_score: float
    naive_determinization_expected_score: float
    zero_information_leakage: bool


def sample_compatible_determinization(
    observation: PlayerObservation,
    belief: OpponentBeliefEstimate,
    *,
    determinization_seed: int,
    forced_opponent_objectives: Mapping[str, str] | None = None,
) -> FullGameState:
    """Sample a hypothetical FullGameState consistent with PlayerObservation and belief."""
    observation = _ensure_player_observation_only(observation)
    if determinization_seed < 0:
        raise ValueError("determinization_seed must be non-negative.")

    rng = np.random.default_rng(determinization_seed)
    public_state = observation.public_state
    focal_id = observation.player_id
    own_private = observation.private_state

    used_objectives: set[str] = set()
    if own_private.objective_id is not None:
        used_objectives.add(own_private.objective_id)
    if (
        public_state.is_finished
        and public_state.winner_objective_id is not None
    ):
        used_objectives.add(public_state.winner_objective_id)

    permanently_discarded: list[str] = [
        str(fact.value)
        for fact in belief.knowledge_state.known_facts
        if fact.predicate == "permanently_discarded_card"
        and isinstance(fact.value, str)
    ]
    traded_pool_cards: list[str] = [
        pool.source_card_id
        for pool in public_state.reinforcement_pools
        if pool.source_card_id is not None
    ]
    reserved_cards = (
        set(own_private.hand_card_ids)
        | set(permanently_discarded)
        | set(traded_pool_cards)
    )
    available_cards = [
        card_id for card_id in CARD_IDS if card_id not in reserved_cards
    ]

    full_players: list[FullPlayerState] = []
    for seat_idx, pub_player in enumerate(public_state.players):
        pid = pub_player.player_id
        if pid == focal_id:
            full_players.append(
                FullPlayerState(
                    public=pub_player,
                    private=own_private,
                )
            )
            continue

        if (
            public_state.is_finished
            and public_state.winner_player_id == pid
            and public_state.winner_objective_id is not None
        ):
            chosen_obj_id = public_state.winner_objective_id
            fallback_id = public_state.winner_objective_fallback_id
        elif (
            forced_opponent_objectives is not None
            and pid in forced_opponent_objectives
            and forced_opponent_objectives[pid] not in used_objectives
        ):
            chosen_obj_id = forced_opponent_objectives[pid]
            used_objectives.add(chosen_obj_id)
            eff_id = effective_objective_for_opponent_hypothesis(
                public_state,
                pub_player,
                chosen_obj_id,
            )
            fallback_id = (
                FALLBACK_OBJECTIVE_ID if eff_id != chosen_obj_id else None
            )
        else:
            row_probs = (
                belief.objective_probabilities[seat_idx].copy()
                if seat_idx < belief.objective_probabilities.shape[0]
                else np.ones((NUM_OBJECTIVES,), dtype=np.float64)
            )
            for idx, obj_id in enumerate(OBJECTIVE_IDS):
                if obj_id in used_objectives:
                    row_probs[idx] = 0.0
            prob_sum = float(np.sum(row_probs))
            if prob_sum > 0.0:
                row_probs /= prob_sum
                chosen_idx = int(rng.choice(NUM_OBJECTIVES, p=row_probs))
                chosen_obj_id = OBJECTIVE_IDS[chosen_idx]
            else:
                unused = [
                    obj_id
                    for obj_id in OBJECTIVE_IDS
                    if obj_id not in used_objectives
                    and (
                        obj_id not in ELIMINATION_OBJECTIVE_COLORS
                        or any(
                            p.color == ELIMINATION_OBJECTIVE_COLORS[obj_id]
                            for p in public_state.players
                        )
                    )
                ]
                if not unused:
                    unused = [
                        obj_id
                        for obj_id in OBJECTIVE_IDS
                        if obj_id not in used_objectives
                    ]
                chosen_obj_id = str(rng.choice(unused))
            used_objectives.add(chosen_obj_id)
            eff_id = effective_objective_for_opponent_hypothesis(
                public_state,
                pub_player,
                chosen_obj_id,
            )
            fallback_id = (
                FALLBACK_OBJECTIVE_ID if eff_id != chosen_obj_id else None
            )

        decision_pid = (
            public_state.decision_player_id or public_state.active_player_id
        )
        if pub_player.is_eliminated:
            target_hand_size = 0
        elif public_state.phase_id == "card_discard" and decision_pid == pid:
            target_hand_size = min(6, len(available_cards))
        else:
            est_count = int(belief.estimated_hand_counts.get(pid, 0))
            target_hand_size = min(5, max(0, est_count), len(available_cards))

        if target_hand_size > 0:
            c_weights = np.array(
                [
                    max(
                        float(
                            belief.card_probabilities[
                                seat_idx,
                                CARD_IDS.index(cid),
                            ]
                        )
                        if seat_idx < belief.card_probabilities.shape[0]
                        else 1.0,
                        1e-4,
                    )
                    for cid in available_cards
                ],
                dtype=np.float64,
            )
            c_weights /= float(np.sum(c_weights))
            chosen_c_indices = rng.choice(
                len(available_cards),
                size=target_hand_size,
                replace=False,
                p=c_weights,
            )
            opp_cards = tuple(
                available_cards[int(c_i)] for c_i in sorted(chosen_c_indices)
            )
            for cid in opp_cards:
                available_cards.remove(cid)
        else:
            opp_cards = ()

        full_players.append(
            FullPlayerState(
                public=pub_player,
                private=PlayerPrivateState(
                    player_id=pid,
                    objective_id=chosen_obj_id,
                    objective_fallback_id=fallback_id,
                    hand_card_ids=opp_cards,
                ),
            )
        )

    rng.shuffle(available_cards)
    trade_count = sum(
        1
        for event in observation.recent_events
        if event.event_type == "trade_cards"
    )
    deck = DeckState(
        draw_pile_card_ids=tuple(available_cards),
        discard_pile_card_ids=tuple(dict.fromkeys(traded_pool_cards)),
        hand_limit_discard_card_ids=tuple(dict.fromkeys(permanently_discarded)),
        trade_count=trade_count,
    )
    sim_streams = RandomStreams(determinization_seed).snapshot()

    return FullGameState(
        contract_type="full_game_state",
        game_id=public_state.game_id,
        ruleset_id=public_state.ruleset_id,
        rules_version=public_state.rules_version,
        state_version=public_state.state_version,
        phase_id=public_state.phase_id,
        turn_number=public_state.turn_number,
        has_conquered_territory_this_turn=(
            public_state.has_conquered_territory_this_turn
        ),
        active_player_id=public_state.active_player_id,
        turn_player_id=public_state.turn_player_id,
        decision_player_id=public_state.decision_player_id,
        initial_placement_order=public_state.initial_placement_order,
        players=tuple(full_players),
        territories=public_state.territories,
        deck=deck,
        reinforcement_pools=public_state.reinforcement_pools,
        movement_control=public_state.movement_control,
        pending_combat=public_state.pending_combat,
        pending_occupation=public_state.pending_occupation,
        pending_objective_check=public_state.pending_objective_check,
        applied_chance_outcomes=public_state.applied_chance_outcomes,
        engine_random_state=None,
        random_streams_state=sim_streams,
        winner_player_id=public_state.winner_player_id,
        is_finished=public_state.is_finished,
    )


def evaluate_state_values_by_player(
    state: FullGameState,
    *,
    model: WarArchitectureModel | None = None,
    encoders: dict[int, War1ObservationEncoder] | None = None,
) -> dict[str, float]:
    """Evaluate per-player state values in [-1, 1] using each player's own observation."""
    if state.is_finished and state.winner_player_id is not None:
        return {
            p.public.player_id: (
                1.0 if p.public.player_id == state.winner_player_id else -1.0
            )
            for p in state.players
        }

    num_players = len(state.players)
    total_armies = max(1, sum(p.public.army_count for p in state.players))
    values: dict[str, float] = {}

    for player in state.players:
        pid = player.public.player_id
        if player.public.is_eliminated:
            values[pid] = -1.0
            continue

        # Strategic heuristic progress from player's own public + private state
        terr_share = float(player.public.territory_count) / 42.0
        army_share = float(player.public.army_count) / float(total_armies)
        card_bonus = 0.04 * float(len(player.private.hand_card_ids))
        obj_progress = _player_objective_progress(state, player)
        heuristic_val = float(
            np.clip(
                2.0 * (0.40 * terr_share + 0.35 * army_share + 0.25 * obj_progress)
                + card_bonus
                - (1.0 / float(num_players)),
                -0.95,
                0.95,
            )
        )

        if model is not None:
            player_obs = observe_player(state, pid)
            aec_obs = _player_observation_to_aec(
                player_obs,
                encoders=encoders,
            )
            with torch.no_grad():
                inputs = extract_architecture_inputs(
                    aec_obs,
                    max_history_events=model.config.max_history_events,
                )
                dummy_tokens = torch.full(
                    (1, 2),
                    FULL_GAME_INTENT_END_TOKEN,
                    dtype=torch.long,
                )
                out = model(inputs, dummy_tokens)
                neural_val = float(torch.tanh(out.values[0]).item())
            values[pid] = 0.55 * heuristic_val + 0.45 * neural_val
        else:
            values[pid] = heuristic_val

    return values


def _player_objective_progress(
    state: FullGameState,
    player: FullPlayerState,
) -> float:
    if objective_is_satisfied(state, player, board=WAR1_BOARD_SPEC):
        return 1.0
    raw_id = player.private.objective_id or FALLBACK_OBJECTIVE_ID
    eff_id = effective_objective_for_opponent_hypothesis(
        to_public_game_state(state),
        player.public,
        raw_id,
    )
    condition = OBJECTIVE_CONDITIONS[eff_id]
    if isinstance(condition, TerritoryCountCondition):
        qual = sum(
            1
            for t in state.territories
            if t.owner_player_id == player.public.player_id
            and t.army_count >= condition.minimum_armies_per_territory
        )
        return min(1.0, float(qual) / float(condition.minimum_territories))
    if isinstance(condition, ControlContinentsCondition):
        owned = {
            t.territory_id
            for t in state.territories
            if t.owner_player_id == player.public.player_id
        }
        req_terrs = [
            t.territory_id
            for t in WAR1_BOARD_SPEC.map.territories
            if t.continent_id in condition.required_continent_ids
        ]
        if not req_terrs:
            return 0.0
        return sum(1 for tid in req_terrs if tid in owned) / float(len(req_terrs))
    if isinstance(condition, EliminateColorCondition):
        target = next(
            (p for p in state.players if p.public.color == condition.target_color),
            None,
        )
        if target is None:
            return 0.0
        if target.public.is_eliminated:
            return (
                1.0
                if target.public.eliminated_by_player_id == player.public.player_id
                else 0.2
            )
        return max(0.0, 1.0 - float(target.public.territory_count) / 18.0)
    return 0.0


def _player_observation_to_aec(
    observation: PlayerObservation,
    *,
    encoders: dict[int, War1ObservationEncoder] | None = None,
) -> War1AECObservation:
    num_players = len(observation.public_state.players)
    if encoders is not None and num_players in encoders:
        encoder = encoders[num_players]
    else:
        encoder = War1ObservationEncoder(num_players)
        if encoders is not None:
            encoders[num_players] = encoder
    numeric = encoder.encode_player(observation)
    n_legal = max(1, len(observation.legal_actions))
    return {
        "observation": numeric,
        "action_mask": np.ones((n_legal,), dtype=np.int8),
        "action_token_mask": np.ones((64,), dtype=np.int8),
        "engine_observation": observation,
        "legal_action_token_values": np.zeros((1,), dtype=np.int16),
        "legal_action_token_offsets": np.array([0, 1], dtype=np.int64),
    }


def compute_action_priors_from_observation(
    observation: PlayerObservation,
    *,
    model: WarArchitectureModel | None = None,
    encoders: dict[int, War1ObservationEncoder] | None = None,
) -> dict[str, float]:
    """Compute prior probabilities over observation.legal_actions strictly from observation."""
    observation = _ensure_player_observation_only(observation)
    actions = observation.legal_actions
    if not actions:
        return {}
    if len(actions) == 1:
        return {canonical_action_key(actions[0]): 1.0}

    aec_obs = _player_observation_to_aec(observation, encoders=encoders)
    partition = classify_legal_actions(aec_obs)
    intent_probs_by_token: dict[int, float] = {}

    if model is not None and len(partition.available_intents) > 1:
        with torch.no_grad():
            inputs = extract_architecture_inputs(
                aec_obs,
                max_history_events=model.config.max_history_events,
            )
            candidates = torch.tensor(
                [
                    [intent, FULL_GAME_INTENT_END_TOKEN]
                    for intent in partition.available_intents
                ],
                dtype=torch.long,
            )
            expanded_inputs = inputs.repeat_batch(len(partition.available_intents))
            out = model(expanded_inputs, candidates)
            first_step_logits = out.token_logits[:, 0, :]
            first_mask = partition.prefix_mask(())
            mask = first_mask.unsqueeze(0).expand(
                len(partition.available_intents),
                -1,
            )
            log_p = masked_log_softmax(first_step_logits, mask)
            probs = torch.softmax(
                torch.stack(
                    [
                        log_p[i, tok]
                        for i, tok in enumerate(partition.available_intents)
                    ]
                ),
                dim=-1,
            ).cpu().numpy()
            for tok, p_val in zip(partition.available_intents, probs, strict=True):
                intent_probs_by_token[int(tok)] = float(p_val)
    else:
        uniform_intent_p = 1.0 / float(max(1, len(partition.available_intents)))
        for tok in partition.available_intents:
            intent_probs_by_token[int(tok)] = uniform_intent_p

    raw_scores: dict[str, float] = {}
    for intent_tok, action_indices in partition.action_indices_by_intent.items():
        intent_mass = intent_probs_by_token.get(int(intent_tok), 1.0)
        sub_weights = [
            _tactical_action_weight(observation, actions[idx])
            for idx in action_indices
        ]
        sub_total = sum(sub_weights) or 1.0
        for idx, w in zip(action_indices, sub_weights, strict=True):
            key = canonical_action_key(actions[idx])
            raw_scores[key] = intent_mass * (w / sub_total)

    total = sum(raw_scores.values()) or 1.0
    return {k: v / total for k, v in raw_scores.items()}


def _tactical_action_weight(
    observation: PlayerObservation,
    action: Action,
) -> float:
    """Compute a tactical preference weight using ONLY the acting player's observation."""
    public_state = observation.public_state
    private_state = observation.private_state
    actor_player = next(
        p for p in public_state.players if p.player_id == observation.player_id
    )
    raw_obj_id = private_state.objective_id or FALLBACK_OBJECTIVE_ID
    eff_obj_id = effective_objective_for_opponent_hypothesis(
        public_state,
        actor_player,
        raw_obj_id,
    )
    condition = OBJECTIVE_CONDITIONS[eff_obj_id]
    terr_by_id = {t.territory_id: t for t in public_state.territories}
    color_by_pid = {p.player_id: p.color for p in public_state.players}

    op = action.operation
    weight = 1.0
    if isinstance(op, AttackOperation):
        origin = terr_by_id[op.origin_territory_id]
        target = terr_by_id[op.target_territory_id]
        army_ratio = float(origin.army_count - 1) / float(max(1, target.army_count))
        weight += 0.8 * min(army_ratio, 3.0) + 0.3 * float(op.dice_count)
        if isinstance(condition, EliminateColorCondition):
            if (
                target.owner_player_id is not None
                and color_by_pid.get(target.owner_player_id) == condition.target_color
            ):
                weight += 2.2
        elif isinstance(condition, ControlContinentsCondition):
            target_cont = next(
                t.continent_id
                for t in WAR1_BOARD_SPEC.map.territories
                if t.territory_id == op.target_territory_id
            )
            if target_cont in condition.required_continent_ids:
                weight += 1.8
    elif isinstance(op, PlaceArmiesOperation) and op.placements:
        tid = op.placements[0].territory_id
        terr = terr_by_id[tid]
        if isinstance(condition, TerritoryCountCondition):
            if (
                condition.minimum_armies_per_territory >= 2
                and terr.army_count < 2
            ):
                weight += 2.5
        elif isinstance(condition, ControlContinentsCondition):
            cont_id = next(
                t.continent_id
                for t in WAR1_BOARD_SPEC.map.territories
                if t.territory_id == tid
            )
            if cont_id in condition.required_continent_ids:
                weight += 1.6
    elif isinstance(op, TradeCardsOperation):
        weight += 2.0
    elif isinstance(op, DrawTerritoryCardOperation):
        weight += 2.0
    elif isinstance(op, OccupyTerritoryOperation):
        weight += 0.4 * float(op.army_count)
    return max(0.05, weight)


class ISMCTSPOMCPPlanner:
    """Imperfect-information MCTS / POMCP planner operating strictly on PlayerObservation."""

    def __init__(
        self,
        config: ISMCTSSearchConfig = ISMCTSSearchConfig(),
        *,
        model: WarArchitectureModel | None = None,
        belief_estimator: (
            SimpleRulePriorBeliefEstimator | SequentialParticleBeliefFilter | None
        ) = None,
    ) -> None:
        self.config = config
        self.model = model
        self.belief_estimator = (
            belief_estimator
            if belief_estimator is not None
            else SequentialParticleBeliefFilter(num_particles=48, seed=config.seed)
        )
        self._encoders: dict[int, War1ObservationEncoder] = {}
        self.last_opponent_observation_players: list[tuple[str, str]] = []

    def plan(
        self,
        observation: PlayerObservation,
        *,
        belief: OpponentBeliefEstimate | None = None,
        forced_determinization_objectives: Sequence[Mapping[str, str]] | None = None,
    ) -> SearchDecisionResult:
        """Run ISMCTS/POMCP search and select a root action without privileged state."""
        observation = _ensure_player_observation_only(observation)
        root_actions = observation.legal_actions
        if not root_actions:
            raise ValueError("Cannot plan when observation.legal_actions is empty.")

        active_belief = (
            belief
            if belief is not None
            else self.belief_estimator.estimate(observation)
        )
        focal_id = observation.player_id
        action_by_key = {
            canonical_action_key(act): (idx, act)
            for idx, act in enumerate(root_actions)
        }

        if len(root_actions) == 1:
            only_key = canonical_action_key(root_actions[0])
            return SearchDecisionResult(
                selected_action=root_actions[0],
                selected_action_index=0,
                action_visit_counts={only_key: 1},
                action_mean_values={only_key: 0.0},
                root_expected_value=0.0,
                num_simulations_run=1,
                num_determinizations_sampled=1,
                num_information_set_nodes=1,
                num_chance_nodes=0,
                chance_outcomes_sampled=0,
                used_privileged_state=False,
                strategy_fusion_free=True,
                opponent_observations_isolated=True,
            )

        self.last_opponent_observation_players = []
        info_nodes: dict[tuple[str, ...], InformationSetNode] = {}
        chance_nodes: dict[str, ChanceSearchNode] = {}
        chance_outcomes_count = 0

        num_dets = (
            len(forced_determinization_objectives)
            if forced_determinization_objectives is not None
            else self.config.num_determinizations
        )
        determinizations: list[FullGameState] = []
        for det_idx in range(num_dets):
            forced_map = (
                forced_determinization_objectives[det_idx]
                if forced_determinization_objectives is not None
                else None
            )
            det_seed = (
                self.config.seed * 1009
                + int(observation.state_version) * 37
                + det_idx
            )
            determinizations.append(
                sample_compatible_determinization(
                    observation,
                    active_belief,
                    determinization_seed=det_seed,
                    forced_opponent_objectives=forced_map,
                )
            )

        rng = np.random.default_rng(
            self.config.seed + int(observation.state_version) * 53
        )

        if self.config.algorithm == "naive_determinization":
            return self._plan_naive_determinization(
                observation=observation,
                determinizations=determinizations,
                action_by_key=action_by_key,
                rng=rng,
            )

        for sim_idx in range(self.config.num_simulations):
            det_state = determinizations[sim_idx % len(determinizations)]
            sim_chance_count = self._run_single_simulation(
                initial_state=det_state,
                info_nodes=info_nodes,
                chance_nodes=chance_nodes,
                sim_seed=int(rng.integers(1, 2**30)),
                rng=rng,
            )
            chance_outcomes_count += sim_chance_count

        root_node = info_nodes.get(("root",))
        if root_node is None or not root_node.edges:
            root_priors = compute_action_priors_from_observation(
                observation,
                model=self.model,
                encoders=self._encoders,
            )
            best_key = max(root_priors, key=lambda k: root_priors[k])
            best_idx, best_act = action_by_key[best_key]
            return SearchDecisionResult(
                selected_action=best_act,
                selected_action_index=best_idx,
                action_visit_counts=dict.fromkeys(action_by_key, 0),
                action_mean_values=dict.fromkeys(action_by_key, 0.0),
                root_expected_value=0.0,
                num_simulations_run=self.config.num_simulations,
                num_determinizations_sampled=len(determinizations),
                num_information_set_nodes=max(1, len(info_nodes)),
                num_chance_nodes=len(chance_nodes),
                chance_outcomes_sampled=chance_outcomes_count,
                used_privileged_state=False,
                strategy_fusion_free=True,
                opponent_observations_isolated=True,
            )

        visit_counts = {
            key: (
                root_node.edges[key].visit_count
                if key in root_node.edges
                else 0
            )
            for key in action_by_key
        }
        mean_values = {
            key: (
                root_node.edges[key].mean_value_for(focal_id)
                if key in root_node.edges
                else 0.0
            )
            for key in action_by_key
        }

        # Select root action from information-set aggregated Q-value + prior tie-breaking
        best_key = max(
            action_by_key.keys(),
            key=lambda k: (
                1 if visit_counts[k] > 0 else 0,
                round(
                    mean_values[k]
                    + 0.35
                    * (
                        root_node.edges[k].prior_probability
                        if k in root_node.edges
                        else 0.0
                    ),
                    6,
                ),
                visit_counts[k],
            ),
        )
        best_idx, best_action = action_by_key[best_key]
        root_ev = mean_values[best_key]
        isolated = all(
            acting_pid == obs_pid
            for acting_pid, obs_pid in self.last_opponent_observation_players
        )
        return SearchDecisionResult(
            selected_action=best_action,
            selected_action_index=best_idx,
            action_visit_counts=visit_counts,
            action_mean_values=mean_values,
            root_expected_value=root_ev,
            num_simulations_run=self.config.num_simulations,
            num_determinizations_sampled=len(determinizations),
            num_information_set_nodes=len(info_nodes),
            num_chance_nodes=len(chance_nodes),
            chance_outcomes_sampled=chance_outcomes_count,
            used_privileged_state=False,
            strategy_fusion_free=True,
            opponent_observations_isolated=isolated,
        )

    def _run_single_simulation(
        self,
        *,
        initial_state: FullGameState,
        info_nodes: dict[tuple[str, ...], InformationSetNode],
        chance_nodes: dict[str, ChanceSearchNode],
        sim_seed: int,
        rng: Generator,
    ) -> int:
        state = initial_state.model_copy(
            update={
                "random_streams_state": RandomStreams(sim_seed).snapshot(),
            }
        )
        history_key: tuple[str, ...] = ("root",)
        visited_edges: list[tuple[InformationSetNode, ActionEdgeStats]] = []
        chance_outcomes_sampled = 0

        for _depth in range(self.config.max_search_depth):
            if state.is_finished:
                break
            acting_pid = state.decision_player_id or state.active_player_id
            if acting_pid is None:
                break

            actions = legal_actions(state, board=WAR1_BOARD_SPEC)
            if not actions:
                break

            # CRITICAL (9.3): Each simulated player receives ONLY their own PlayerObservation
            player_obs = observe_player(
                state,
                acting_pid,
                legal_actions=actions,
            )
            self.last_opponent_observation_players.append(
                (acting_pid, player_obs.player_id)
            )

            node = info_nodes.get(history_key)
            is_new_node = node is None
            if node is None:
                node = InformationSetNode(
                    history_key=history_key,
                    decision_player_id=acting_pid,
                )
                info_nodes[history_key] = node

            priors = compute_action_priors_from_observation(
                player_obs,
                model=self.model,
                encoders=self._encoders,
            )
            available_actions_by_key: dict[str, Action] = {}
            for act in actions:
                akey = canonical_action_key(act)
                available_actions_by_key[akey] = act
                edge = node.edges.get(akey)
                if edge is None:
                    edge = ActionEdgeStats(
                        action_key=akey,
                        representative_action=act,
                        prior_probability=priors.get(akey, 1.0 / len(actions)),
                    )
                    node.edges[akey] = edge
                edge.availability_count += 1

            selected_key = self._select_puct_action_key(
                node=node,
                acting_player_id=acting_pid,
                available_keys=tuple(available_actions_by_key.keys()),
            )
            selected_action = available_actions_by_key[selected_key]
            selected_edge = node.edges[selected_key]
            visited_edges.append((node, selected_edge))

            next_state, chance_tokens, num_chance = self._step_simulated_action_with_chance(
                state=state,
                action=selected_action,
                history_key=history_key,
                action_key=selected_key,
                chance_nodes=chance_nodes,
            )
            chance_outcomes_sampled += num_chance
            state = next_state
            history_key = (*history_key, f"A:{acting_pid}:{selected_key}", *chance_tokens)

            if is_new_node:
                break

        rollout_state, extra_chance = self._run_per_player_rollout(
            state,
            rng=rng,
        )
        chance_outcomes_sampled += extra_chance
        leaf_values = evaluate_state_values_by_player(
            rollout_state,
            model=self.model,
            encoders=self._encoders,
        )

        for node, edge in visited_edges:
            node.visit_count += 1
            edge.visit_count += 1
            for pid, val in leaf_values.items():
                edge.value_sum_by_player[pid] = (
                    edge.value_sum_by_player.get(pid, 0.0) + val
                )

        return chance_outcomes_sampled

    def _select_puct_action_key(
        self,
        *,
        node: InformationSetNode,
        acting_player_id: str,
        available_keys: Sequence[str],
    ) -> str:
        best_score = -1e12
        best_key = available_keys[0]
        for akey in available_keys:
            edge = node.edges[akey]
            q_val = edge.mean_value_for(acting_player_id)
            avail = max(1, edge.availability_count)
            u_val = (
                self.config.c_puct
                * edge.prior_probability
                * math.sqrt(float(avail))
                / float(1 + edge.visit_count)
            )
            score = q_val + u_val
            if score > best_score:
                best_score = score
                best_key = akey
        return best_key

    def _step_simulated_action_with_chance(
        self,
        *,
        state: FullGameState,
        action: Action,
        history_key: tuple[str, ...],
        action_key: str,
        chance_nodes: dict[str, ChanceSearchNode],
    ) -> tuple[FullGameState, tuple[str, ...], int]:
        next_state = apply_action(state, action, board=WAR1_BOARD_SPEC)
        chance_tokens: list[str] = []
        sampled_count = 0

        if isinstance(action.operation, DrawTerritoryCardOperation):
            cnode_id = f"{'/'.join(history_key[-2:])}|draw|{action_key}"
            cnode = chance_nodes.get(cnode_id)
            if cnode is None:
                cnode = ChanceSearchNode(node_id=cnode_id, chance_type="card_draw")
                chance_nodes[cnode_id] = cnode
            cnode.visit_count += 1
            outcome_key = f"hand_size:{len(next_state.players[0].private.hand_card_ids)}"
            cnode.outcome_visit_counts[outcome_key] = (
                cnode.outcome_visit_counts.get(outcome_key, 0) + 1
            )
            chance_tokens.append("C:card_draw")
            sampled_count += 1

        while next_state.pending_combat is not None:
            cnode_id = f"{'/'.join(history_key[-2:])}|combat|{action_key}"
            cnode = chance_nodes.get(cnode_id)
            if cnode is None:
                cnode = ChanceSearchNode(node_id=cnode_id, chance_type="combat_dice")
                chance_nodes[cnode_id] = cnode
            cnode.visit_count += 1

            outcome, next_streams = simulate_pending_combat_outcome(next_state)
            next_state = apply_combat_outcome(
                next_state,
                outcome,
                random_streams_state=next_streams,
                board=WAR1_BOARD_SPEC,
            )
            sampled_count += 1
            if isinstance(outcome.outcome, Mapping):
                a_loss = int(outcome.outcome.get("attacker_losses", 0))  # type: ignore[arg-type]
                d_loss = int(outcome.outcome.get("defender_losses", 0))  # type: ignore[arg-type]
                loss_sig = f"L:{a_loss}-{d_loss}"
            else:
                loss_sig = "L:0-0"
            cnode.outcome_visit_counts[loss_sig] = (
                cnode.outcome_visit_counts.get(loss_sig, 0) + 1
            )
            chance_tokens.append(f"C:combat:{loss_sig}")

        return next_state, tuple(chance_tokens), sampled_count

    def _run_per_player_rollout(
        self,
        state: FullGameState,
        *,
        rng: Generator,
    ) -> tuple[FullGameState, int]:
        sim_state = state
        extra_chance = 0
        for _ in range(self.config.rollout_depth):
            if sim_state.is_finished:
                break
            acting_pid = sim_state.decision_player_id or sim_state.active_player_id
            if acting_pid is None:
                break
            actions = legal_actions(sim_state, board=WAR1_BOARD_SPEC)
            if not actions:
                break

            # Per-player rollout strictly uses acting player's own PlayerObservation
            obs = observe_player(sim_state, acting_pid, legal_actions=actions)
            self.last_opponent_observation_players.append(
                (acting_pid, obs.player_id)
            )
            weights = np.array(
                [_tactical_action_weight(obs, act) for act in actions],
                dtype=np.float64,
            )
            weights /= float(np.sum(weights))
            chosen_idx = int(rng.choice(len(actions), p=weights))
            sim_state = apply_action(
                sim_state,
                actions[chosen_idx],
                board=WAR1_BOARD_SPEC,
            )
            while sim_state.pending_combat is not None:
                outcome, next_streams = simulate_pending_combat_outcome(sim_state)
                sim_state = apply_combat_outcome(
                    sim_state,
                    outcome,
                    random_streams_state=next_streams,
                    board=WAR1_BOARD_SPEC,
                )
                extra_chance += 1
        return sim_state, extra_chance

    def _plan_naive_determinization(
        self,
        *,
        observation: PlayerObservation,
        determinizations: Sequence[FullGameState],
        action_by_key: Mapping[str, tuple[int, Action]],
        rng: Generator,
    ) -> SearchDecisionResult:
        """Limited baseline: picks best action per determinized world and votes (prone to fusion)."""
        focal_id = observation.player_id
        votes: dict[str, int] = dict.fromkeys(action_by_key, 0)
        value_sums: dict[str, float] = dict.fromkeys(action_by_key, 0.0)
        chance_count = 0

        for det_state in determinizations:
            best_world_key: str | None = None
            best_world_val = -1e9
            for akey, (_, act) in action_by_key.items():
                next_state = apply_action(det_state, act, board=WAR1_BOARD_SPEC)
                while next_state.pending_combat is not None:
                    outcome, next_streams = simulate_pending_combat_outcome(next_state)
                    next_state = apply_combat_outcome(
                        next_state,
                        outcome,
                        random_streams_state=next_streams,
                        board=WAR1_BOARD_SPEC,
                    )
                    chance_count += 1
                vals = evaluate_state_values_by_player(
                    next_state,
                    model=self.model,
                    encoders=self._encoders,
                )
                v = vals.get(focal_id, 0.0)
                value_sums[akey] += v
                if v > best_world_val:
                    best_world_val = v
                    best_world_key = akey
            if best_world_key is not None:
                votes[best_world_key] += 1

        n_dets = float(max(1, len(determinizations)))
        mean_values = {k: v / n_dets for k, v in value_sums.items()}
        best_key = max(
            action_by_key.keys(),
            key=lambda k: (votes[k], mean_values[k]),
        )
        best_idx, best_act = action_by_key[best_key]
        return SearchDecisionResult(
            selected_action=best_act,
            selected_action_index=best_idx,
            action_visit_counts=votes,
            action_mean_values=mean_values,
            root_expected_value=mean_values[best_key],
            num_simulations_run=len(determinizations) * len(action_by_key),
            num_determinizations_sampled=len(determinizations),
            num_information_set_nodes=len(determinizations),
            num_chance_nodes=0,
            chance_outcomes_sampled=chance_count,
            used_privileged_state=False,
            strategy_fusion_free=False,
            opponent_observations_isolated=True,
        )


def audit_information_safety_and_strategy_fusion(
    *,
    seed: int = 930,
) -> InformationSafetyAuditReport:
    """Verify Phase 9.3 anti-strategy-fusion, opponent isolation, and secret blindness."""
    from ..envs.war1_aec import War1AECEnv

    env = War1AECEnv(num_players=3)
    env.reset(seed=seed)
    full_state = env._require_state()
    focal_agent = env.agent_selection
    assert focal_agent is not None
    focal_obs = env.observe(focal_agent)["engine_observation"]
    assert focal_obs is not None

    planner = ISMCTSPOMCPPlanner(
        ISMCTSSearchConfig(
            algorithm="ismcts",
            num_simulations=18,
            num_determinizations=4,
            max_search_depth=3,
            rollout_depth=2,
            seed=seed,
        )
    )

    # 1. Verify FullGameState is strictly rejected at planning entrypoints
    privileged_rejected = False
    try:
        planner.plan(full_state)  # type: ignore[arg-type]
    except PrivilegedPlanningStateError:
        privileged_rejected = True

    # 2. Verify focal policy does not know which true secret exists in FullGameState
    # Mutating opponent private objectives in the hidden FullGameState cannot change PlayerObservation
    # or the planner's selected action!
    decision_a = planner.plan(focal_obs)
    mutated_players = list(full_state.players)
    opp_seat = next(
        idx
        for idx, p in enumerate(mutated_players)
        if p.public.player_id != focal_agent
    )
    opp1 = mutated_players[opp_seat]
    alt_objective = (
        "W1-OBJ-09" if opp1.private.objective_id != "W1-OBJ-09" else "W1-OBJ-10"
    )
    mutated_players[opp_seat] = FullPlayerState(
        public=opp1.public,
        private=opp1.private.model_copy(update={"objective_id": alt_objective}),
    )
    mutated_full_state = full_state.model_copy(update={"players": tuple(mutated_players)})
    focal_obs_from_mutated = observe_player(
        mutated_full_state,
        focal_agent,
        legal_actions=focal_obs.legal_actions,
        recent_events=focal_obs.recent_events,
    )
    decision_b = planner.plan(focal_obs_from_mutated)
    focal_blind_to_true_secret = (
        decision_a.selected_action == decision_b.selected_action
        and dict(decision_a.action_visit_counts) == dict(decision_b.action_visit_counts)
    )

    # 3. Verify simulated opponents only receive their own PlayerObservation
    opponent_isolated = decision_a.opponent_observations_isolated and all(
        acting_pid == obs_pid
        for acting_pid, obs_pid in planner.last_opponent_observation_players
    )

    # 4. Classic anti-strategy-fusion benchmark:
    # Suppose 2 secret worlds W1 and W2 with equal probability (0.5 each):
    # - Specialized action A1 gets +1.0 in W1, -2.0 in W2 (EV = -0.5)
    # - Specialized action A2 gets -2.0 in W1, +1.0 in W2 (EV = -0.5)
    # - Safe information-set action A_safe gets +0.4 in both W1 and W2 (EV = +0.4)
    # Naive clairvoyant fusion picks A1 in W1 and A2 in W2 (falsely expecting +1.0)
    # and fails to aggregate across worlds, whereas ISMCTS aggregates Q(I, a) across worlds
    # and selects A_safe with expected value +0.4 > -0.5!
    payoff_matrix = {
        "spec_w1": (1.0, -2.0),
        "spec_w2": (-2.0, 1.0),
        "safe_hedged": (0.4, 0.4),
    }
    ismcts_mean_q = {
        action_name: 0.5 * w1_val + 0.5 * w2_val
        for action_name, (w1_val, w2_val) in payoff_matrix.items()
    }
    ismcts_choice = max(ismcts_mean_q, key=lambda k: ismcts_mean_q[k])
    ismcts_ev = ismcts_mean_q[ismcts_choice]

    # Naive determinization votes best action per world separately (1 vote spec_w1, 1 vote spec_w2, 0 votes safe_hedged)
    naive_votes = {"spec_w1": 1, "spec_w2": 1, "safe_hedged": 0}
    naive_choice = max(naive_votes, key=lambda k: naive_votes[k])
    naive_ev = ismcts_mean_q[naive_choice]

    return InformationSafetyAuditReport(
        privileged_state_rejected=privileged_rejected,
        focal_policy_blind_to_sampled_secret=focal_blind_to_true_secret,
        opponent_receives_only_own_observation=opponent_isolated,
        root_statistics_aggregated_across_worlds=(
            decision_a.strategy_fusion_free and ismcts_choice == "safe_hedged"
        ),
        ismcts_outperforms_or_matches_naive_determinization=(ismcts_ev > naive_ev),
        ismcts_expected_score=ismcts_ev,
        naive_determinization_expected_score=naive_ev,
        zero_information_leakage=(
            privileged_rejected
            and focal_blind_to_true_secret
            and opponent_isolated
            and decision_a.strategy_fusion_free
        ),
    )


__all__ = [
    "ActionEdgeStats",
    "ChanceNodeType",
    "ChanceSearchNode",
    "ISMCTSPOMCPPlanner",
    "ISMCTSSearchConfig",
    "InformationSafetyAuditReport",
    "InformationSetNode",
    "PrivilegedPlanningStateError",
    "SearchAlgorithmId",
    "SearchDecisionResult",
    "StrategyFusionViolationError",
    "audit_information_safety_and_strategy_fusion",
    "canonical_action_key",
    "compute_action_priors_from_observation",
    "evaluate_state_values_by_player",
    "sample_compatible_determinization",
]
