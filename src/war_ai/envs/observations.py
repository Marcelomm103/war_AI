"""Lossless structured observations for the WAR 1 classic profile."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any, TypedDict

import numpy as np
from gymnasium import spaces
from gymnasium.spaces import Space
from numpy.typing import NDArray
from war_game_engine.board_spec import WAR1_BOARD_SPEC
from war_game_engine.contracts import (
    GameEvent,
    PlayerObservation,
    PlayerPrivateState,
    PublicGameState,
)
from war_game_engine.objectives import PlayerColor

WAR1_RL_OBSERVATION_ENCODING_VERSION = "1.0.0"
MAX_PLAYERS = 6
PLAYER_COLOR_IDS: tuple[PlayerColor, ...] = (
    "white",
    "red",
    "black",
    "blue",
    "yellow",
    "green",
)
PHASE_IDS = (
    "territory_claim",
    "initial_placement",
    "reinforcement",
    "reinforcement_placement",
    "attack",
    "occupation",
    "objective_check",
    "movement",
    "card_purchase",
    "card_discard",
    "turn_end",
    "finished",
)
REINFORCEMENT_ORIGINS = (
    "territory_count",
    "continent_bonus",
    "card_trade",
    "territory_card",
)
PUBLIC_EVENT_TYPES = (
    "claim_territory",
    "place_armies",
    "trade_cards",
    "draw_territory_card",
    "discard_territory_card",
    "attack",
    "occupy_territory",
    "move_armies",
    "end_phase",
    "end_turn",
    "combat_outcome",
)


class War1NumericObservation(TypedDict):
    adjacency: NDArray[np.int8]
    territory_continent_indices: NDArray[np.int8]
    continent_bonuses: NDArray[np.int8]
    continent_owner_indices: NDArray[np.int8]
    territory_owner_indices: NDArray[np.int8]
    territory_owner_mask: NDArray[np.int8]
    territory_troop_log2: NDArray[np.float64]
    territory_troop_bytes: NDArray[np.uint8]
    territory_troop_offsets: NDArray[np.int64]
    phase_one_hot: NDArray[np.int8]
    player_mask: NDArray[np.int8]
    player_color_indices: NDArray[np.int8]
    player_territory_counts: NDArray[np.int8]
    player_troop_log2: NDArray[np.float64]
    player_troop_bytes: NDArray[np.uint8]
    player_troop_offsets: NDArray[np.int64]
    player_eliminated: NDArray[np.int8]
    player_eliminated_by_indices: NDArray[np.int8]
    initial_placement_order_indices: NDArray[np.int8]
    active_player_index: NDArray[np.int8]
    turn_player_index: NDArray[np.int8]
    decision_player_index: NDArray[np.int8]
    winner_player_index: NDArray[np.int8]
    winner_objective_index: NDArray[np.int8]
    game_flags: NDArray[np.int8]
    turn_number_log2: NDArray[np.float64]
    turn_number_bytes: NDArray[np.uint8]
    turn_number_offsets: NDArray[np.int64]
    state_version_log2: NDArray[np.float64]
    state_version_bytes: NDArray[np.uint8]
    state_version_offsets: NDArray[np.int64]
    reinforcement_pool_player_indices: NDArray[np.int8]
    reinforcement_pool_origin_indices: NDArray[np.int8]
    reinforcement_pool_log2: NDArray[np.float64]
    reinforcement_pool_bytes: NDArray[np.uint8]
    reinforcement_pool_offsets: NDArray[np.int64]
    reinforcement_pool_restricted: NDArray[np.int8]
    reinforcement_pool_allowed_masks: NDArray[np.int8]
    reinforcement_pool_source_card_indices: NDArray[np.int8]
    movement_origin_indices: NDArray[np.int8]
    movement_target_indices: NDArray[np.int8]
    movement_troop_log2: NDArray[np.float64]
    movement_troop_bytes: NDArray[np.uint8]
    movement_troop_offsets: NDArray[np.int64]
    pending_occupation_actor_index: NDArray[np.int8]
    pending_occupation_defender_index: NDArray[np.int8]
    pending_occupation_origin_index: NDArray[np.int8]
    pending_occupation_target_index: NDArray[np.int8]
    pending_occupation_max_troop_log2: NDArray[np.float64]
    pending_occupation_max_troop_bytes: NDArray[np.uint8]
    pending_occupation_max_troop_offsets: NDArray[np.int64]
    own_objective_index: NDArray[np.int8]
    own_fallback_objective_index: NDArray[np.int8]
    own_hand_card_indices: NDArray[np.int8]
    public_history_event_type_indices: NDArray[np.int8]
    public_history_actor_indices: NDArray[np.int8]
    public_history_bytes: NDArray[np.uint8]
    public_history_offsets: NDArray[np.int64]


class _VariableArraySpace(Space[np.ndarray[Any, Any]]):
    def __init__(
        self,
        dtype: np.dtype[Any],
        *,
        trailing_shape: tuple[int, ...] = (),
        minimum: int | float | None = None,
        maximum: int | float | None = None,
    ) -> None:
        self._array_dtype = np.dtype(dtype)
        self._trailing_shape = trailing_shape
        self._minimum = minimum
        self._maximum = maximum
        super().__init__(shape=None, dtype=self._array_dtype)

    def sample(
        self,
        mask: Any | None = None,
        probability: Any | None = None,
    ) -> np.ndarray[Any, Any]:
        if mask is not None or probability is not None:
            raise ValueError("Variable observation arrays do not accept masks.")
        return np.zeros((0, *self._trailing_shape), dtype=self._array_dtype)

    def contains(self, value: object) -> bool:
        if (
            not isinstance(value, np.ndarray)
            or value.dtype != self._array_dtype
            or value.ndim != len(self._trailing_shape) + 1
            or tuple(value.shape[1:]) != self._trailing_shape
        ):
            return False
        if self._minimum is not None and np.any(value < self._minimum):
            return False
        if self._maximum is not None and np.any(value > self._maximum):
            return False
        return True


def encode_integer_records(
    values: Iterable[int],
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Encode arbitrary non-negative integers as concatenated big-endian bytes."""
    data: list[int] = []
    offsets = [0]
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("Integer observations must be non-negative integers.")
        byte_count = max(1, (value.bit_length() + 7) // 8)
        data.extend(value.to_bytes(byte_count, byteorder="big", signed=False))
        offsets.append(len(data))
    return np.asarray(data, dtype=np.uint8), np.asarray(offsets, dtype=np.int64)


def decode_integer_records(
    data: NDArray[np.uint8],
    offsets: NDArray[np.int64],
) -> tuple[int, ...]:
    if data.ndim != 1 or offsets.ndim != 1 or offsets.size == 0:
        raise ValueError("Integer record arrays must be one-dimensional.")
    if (
        offsets[0] != 0
        or offsets[-1] != data.size
        or np.any(offsets[1:] < offsets[:-1])
    ):
        raise ValueError("Integer record offsets are inconsistent with the data.")

    decoded: list[int] = []
    for start, end in zip(offsets[:-1], offsets[1:], strict=True):
        if end <= start:
            raise ValueError("Integer records must contain at least one byte.")
        record = data[int(start) : int(end)].tobytes()
        if len(record) > 1 and record[0] == 0:
            raise ValueError("Integer records must use their shortest encoding.")
        decoded.append(int.from_bytes(record, byteorder="big", signed=False))
    return tuple(decoded)


class War1ObservationEncoder:
    def __init__(self, num_players: int) -> None:
        if isinstance(num_players, bool) or not 3 <= num_players <= MAX_PLAYERS:
            raise ValueError(f"WAR 1 classic requires 3–{MAX_PLAYERS} players.")
        self.num_players = num_players
        self.territory_ids = tuple(
            territory.territory_id for territory in WAR1_BOARD_SPEC.map.territories
        )
        self.continent_ids = tuple(
            continent.continent_id for continent in WAR1_BOARD_SPEC.map.continents
        )
        self.objective_ids = tuple(
            objective.objective_id for objective in WAR1_BOARD_SPEC.objectives
        )
        self.card_ids = tuple(WAR1_BOARD_SPEC.territory_deck.card_ids)
        self._territory_indices = {
            territory_id: index for index, territory_id in enumerate(self.territory_ids)
        }
        self._continent_indices = {
            continent_id: index for index, continent_id in enumerate(self.continent_ids)
        }
        self._objective_indices = {
            objective_id: index for index, objective_id in enumerate(self.objective_ids)
        }
        self._card_indices = {
            card_id: index for index, card_id in enumerate(self.card_ids)
        }
        self._phase_indices = {phase_id: index for index, phase_id in enumerate(PHASE_IDS)}
        self._origin_indices = {
            origin: index for index, origin in enumerate(REINFORCEMENT_ORIGINS)
        }
        self._event_type_indices = {
            event_type: index for index, event_type in enumerate(PUBLIC_EVENT_TYPES)
        }
        self._adjacency = self._build_adjacency()
        self._continent_indices_by_territory = np.asarray(
            [
                self._continent_indices[territory.continent_id]
                for territory in WAR1_BOARD_SPEC.map.territories
            ],
            dtype=np.int8,
        )
        self._continent_bonuses = np.asarray(
            [
                continent.reinforcement_bonus
                for continent in WAR1_BOARD_SPEC.map.continents
            ],
            dtype=np.int8,
        )
        self.observation_space = self._build_space()

    def encode_player(
        self,
        observation: PlayerObservation,
    ) -> War1NumericObservation:
        return self._encode(
            observation.public_state,
            player_id=observation.player_id,
            private_state=observation.private_state,
            recent_events=observation.recent_events,
        )

    def encode_public(
        self,
        public_state: PublicGameState,
        *,
        player_id: str | None,
        recent_events: Sequence[GameEvent] = (),
    ) -> War1NumericObservation:
        return self._encode(
            public_state,
            player_id=player_id,
            private_state=None,
            recent_events=recent_events,
        )

    def encode_empty(self) -> War1NumericObservation:
        return self._encode(None, player_id=None, private_state=None)

    def _encode(
        self,
        public_state: PublicGameState | None,
        *,
        player_id: str | None,
        private_state: PlayerPrivateState | None,
        recent_events: Sequence[GameEvent] = (),
    ) -> War1NumericObservation:
        players = () if public_state is None else public_state.players
        if len(players) > MAX_PLAYERS or (
            public_state is not None and len(players) != self.num_players
        ):
            raise ValueError("Player table size does not match the encoder.")
        player_indices = {
            player.player_id: index for index, player in enumerate(players)
        }
        if player_id is not None and player_id not in player_indices:
            raise ValueError(f"Unknown observation player {player_id!r}.")

        territories_by_id = (
            {}
            if public_state is None
            else {
                territory.territory_id: territory
                for territory in public_state.territories
            }
        )
        if public_state is not None and set(territories_by_id) != set(
            self.territory_ids
        ):
            raise ValueError("Observation state does not contain the certified map.")
        territory_owner_indices = np.zeros(len(self.territory_ids), dtype=np.int8)
        territory_troops: list[int] = []
        for index, territory_id in enumerate(self.territory_ids):
            territory = territories_by_id.get(territory_id)
            if territory is None:
                territory_troops.append(0)
                continue
            territory_troops.append(territory.army_count)
            if territory.owner_player_id is not None:
                owner_index = player_indices.get(territory.owner_player_id)
                if owner_index is None:
                    raise ValueError(
                        f"Unknown territory owner {territory.owner_player_id!r}."
                    )
                territory_owner_indices[index] = owner_index + 1
        territory_owner_mask = np.zeros(
            (len(self.territory_ids), MAX_PLAYERS + 1),
            dtype=np.int8,
        )
        territory_owner_mask[
            np.arange(len(self.territory_ids)),
            territory_owner_indices,
        ] = 1
        continent_owner_indices = np.full(
            len(self.continent_ids),
            -1,
            dtype=np.int8,
        )
        if public_state is not None:
            owner_ids_by_continent: dict[str, set[str | None]] = {
                continent_id: set() for continent_id in self.continent_ids
            }
            for territory_definition in WAR1_BOARD_SPEC.map.territories:
                territory = territories_by_id[territory_definition.territory_id]
                owner_ids_by_continent[territory_definition.continent_id].add(
                    territory.owner_player_id
                )
            for continent_id, owner_ids in owner_ids_by_continent.items():
                if len(owner_ids) == 1 and None not in owner_ids:
                    owner_id = next(iter(owner_ids))
                    if owner_id is not None:
                        continent_owner_indices[
                            self._continent_indices[continent_id]
                        ] = self._player_index(player_indices, owner_id)
        territory_troop_bytes, territory_troop_offsets = encode_integer_records(
            territory_troops
        )

        player_mask = np.zeros(MAX_PLAYERS, dtype=np.int8)
        player_mask[: len(players)] = 1
        player_color_indices = np.full(MAX_PLAYERS, -1, dtype=np.int8)
        player_territory_counts = np.zeros(MAX_PLAYERS, dtype=np.int8)
        player_troops: list[int] = []
        player_eliminated = np.zeros(MAX_PLAYERS, dtype=np.int8)
        player_eliminated_by_indices = np.full(MAX_PLAYERS, -1, dtype=np.int8)
        for index, player in enumerate(players):
            try:
                player_color_indices[index] = PLAYER_COLOR_IDS.index(player.color)
            except ValueError as error:
                raise ValueError(f"Unknown player color {player.color!r}.") from error
            player_territory_counts[index] = player.territory_count
            player_troops.append(player.army_count)
            player_eliminated[index] = int(player.is_eliminated)
            if player.eliminated_by_player_id is not None:
                player_eliminated_by_indices[index] = self._player_index(
                    player_indices,
                    player.eliminated_by_player_id,
                )
        player_troops.extend([0] * (MAX_PLAYERS - len(players)))
        player_troop_bytes, player_troop_offsets = encode_integer_records(
            player_troops
        )

        phase = "initial_placement" if public_state is None else public_state.phase_id
        phase_one_hot = np.zeros(len(PHASE_IDS), dtype=np.int8)
        phase_one_hot[self._phase_index(phase)] = 1
        active_player_index = self._slot_array(
            None if public_state is None else public_state.active_player_id,
            player_indices,
        )
        turn_player_index = self._slot_array(
            None if public_state is None else public_state.turn_player_id,
            player_indices,
        )
        decision_player_index = self._slot_array(
            None if public_state is None else public_state.decision_player_id,
            player_indices,
        )
        winner_player_index = self._slot_array(
            None if public_state is None else public_state.winner_player_id,
            player_indices,
        )
        placement_order = np.full(MAX_PLAYERS, -1, dtype=np.int8)
        if public_state is not None:
            placement_order[: len(public_state.initial_placement_order)] = [
                self._player_index(player_indices, item)
                for item in public_state.initial_placement_order
            ]

        turn_number = 0 if public_state is None else public_state.turn_number
        state_version = 0 if public_state is None else public_state.state_version
        turn_number_bytes, turn_number_offsets = encode_integer_records(
            (turn_number,)
        )
        state_version_bytes, state_version_offsets = encode_integer_records(
            (state_version,)
        )

        pools = () if public_state is None else public_state.reinforcement_pools
        pool_player_indices = np.asarray(
            [self._player_index(player_indices, pool.player_id) for pool in pools],
            dtype=np.int8,
        )
        pool_origin_indices = np.asarray(
            [self._origin_index(pool.origin) for pool in pools],
            dtype=np.int8,
        )
        pool_log2 = np.asarray(
            [_log2_integer(pool.remaining_armies) for pool in pools],
            dtype=np.float64,
        )
        pool_bytes, pool_offsets = encode_integer_records(
            pool.remaining_armies for pool in pools
        )
        pool_restricted = np.zeros(len(pools), dtype=np.int8)
        pool_allowed_masks = np.zeros(
            (len(pools), len(self.territory_ids)),
            dtype=np.int8,
        )
        pool_source_card_indices = np.full(len(pools), -1, dtype=np.int8)
        for index, pool in enumerate(pools):
            if pool.allowed_territory_ids is None:
                pool_allowed_masks[index, :] = 1
            else:
                pool_restricted[index] = 1
                for territory_id in pool.allowed_territory_ids:
                    pool_allowed_masks[
                        index,
                        self._territory_index(territory_id),
                    ] = 1
            if pool.source_card_id is not None:
                pool_source_card_indices[index] = self._card_index(
                    pool.source_card_id
                )

        movement_transfers = (
            ()
            if public_state is None
            else public_state.movement_control.moved_transfers
        )
        movement_origin_indices = np.asarray(
            [
                self._territory_index(transfer.origin_territory_id)
                for transfer in movement_transfers
            ],
            dtype=np.int8,
        )
        movement_target_indices = np.asarray(
            [
                self._territory_index(transfer.target_territory_id)
                for transfer in movement_transfers
            ],
            dtype=np.int8,
        )
        movement_troop_log2 = np.asarray(
            [_log2_integer(transfer.army_count) for transfer in movement_transfers],
            dtype=np.float64,
        )
        movement_troop_bytes, movement_troop_offsets = encode_integer_records(
            transfer.army_count for transfer in movement_transfers
        )

        pending_occupation = (
            None if public_state is None else public_state.pending_occupation
        )
        pending_occupation_actor_index = self._slot_array(
            (
                None
                if pending_occupation is None
                else pending_occupation.attacker_player_id
            ),
            player_indices,
        )
        pending_occupation_defender_index = self._slot_array(
            (
                None
                if pending_occupation is None
                else pending_occupation.defender_player_id
            ),
            player_indices,
        )
        pending_occupation_origin_index = self._territory_slot_array(
            None
            if pending_occupation is None
            else pending_occupation.origin_territory_id
        )
        pending_occupation_target_index = self._territory_slot_array(
            None
            if pending_occupation is None
            else pending_occupation.target_territory_id
        )
        pending_occupation_max_count = (
            0 if pending_occupation is None else pending_occupation.maximum_army_count
        )
        pending_occupation_max_troop_bytes, pending_occupation_max_troop_offsets = (
            encode_integer_records((pending_occupation_max_count,))
        )

        own_objective_index = -1
        own_fallback_objective_index = -1
        own_hand_card_indices: NDArray[np.int8] = np.zeros(0, dtype=np.int8)
        if private_state is not None:
            if private_state.player_id != player_id:
                raise ValueError("Private state must belong to the focal player.")
            own_objective_index = self._optional_index(
                self._objective_indices,
                private_state.objective_id,
                "objective",
            )
            own_fallback_objective_index = self._optional_index(
                self._objective_indices,
                private_state.objective_fallback_id,
                "fallback objective",
            )
            own_hand_card_indices = np.asarray(
                [self._card_index(card_id) for card_id in private_state.hand_card_ids],
                dtype=np.int8,
            )

        public_events = tuple(recent_events)
        history_event_type_indices = np.asarray(
            [
                self._event_type_indices.get(event.event_type, -1)
                for event in public_events
            ],
            dtype=np.int8,
        )
        history_actor_indices = np.asarray(
            [
                -1
                if event.actor_player_id is None
                else self._player_index(player_indices, event.actor_player_id)
                for event in public_events
            ],
            dtype=np.int8,
        )
        history_bytes, history_offsets = _pack_byte_records(
            json.dumps(
                event.model_dump(mode="json"),
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
            for event in public_events
        )

        return {
            "adjacency": self._adjacency.copy(),
            "territory_continent_indices": (
                self._continent_indices_by_territory.copy()
            ),
            "continent_bonuses": self._continent_bonuses.copy(),
            "continent_owner_indices": continent_owner_indices,
            "territory_owner_indices": territory_owner_indices,
            "territory_owner_mask": territory_owner_mask,
            "territory_troop_log2": np.asarray(
                [_log2_integer(value) for value in territory_troops],
                dtype=np.float64,
            ),
            "territory_troop_bytes": territory_troop_bytes,
            "territory_troop_offsets": territory_troop_offsets,
            "phase_one_hot": phase_one_hot,
            "player_mask": player_mask,
            "player_color_indices": player_color_indices,
            "player_territory_counts": player_territory_counts,
            "player_troop_log2": np.asarray(
                [_log2_integer(value) for value in player_troops],
                dtype=np.float64,
            ),
            "player_troop_bytes": player_troop_bytes,
            "player_troop_offsets": player_troop_offsets,
            "player_eliminated": player_eliminated,
            "player_eliminated_by_indices": player_eliminated_by_indices,
            "initial_placement_order_indices": placement_order,
            "active_player_index": active_player_index,
            "turn_player_index": turn_player_index,
            "decision_player_index": decision_player_index,
            "winner_player_index": winner_player_index,
            "winner_objective_index": np.asarray(
                [
                    self._optional_index(
                        self._objective_indices,
                        (
                            None
                            if public_state is None
                            else public_state.winner_objective_id
                        ),
                        "winner objective",
                    )
                ],
                dtype=np.int8,
            ),
            "game_flags": np.asarray(
                [
                    int(public_state is not None and public_state.is_finished),
                    int(
                        public_state is not None
                        and public_state.has_conquered_territory_this_turn
                    ),
                ],
                dtype=np.int8,
            ),
            "turn_number_log2": np.asarray(
                [_log2_integer(turn_number)],
                dtype=np.float64,
            ),
            "turn_number_bytes": turn_number_bytes,
            "turn_number_offsets": turn_number_offsets,
            "state_version_log2": np.asarray(
                [_log2_integer(state_version)],
                dtype=np.float64,
            ),
            "state_version_bytes": state_version_bytes,
            "state_version_offsets": state_version_offsets,
            "reinforcement_pool_player_indices": pool_player_indices,
            "reinforcement_pool_origin_indices": pool_origin_indices,
            "reinforcement_pool_log2": pool_log2,
            "reinforcement_pool_bytes": pool_bytes,
            "reinforcement_pool_offsets": pool_offsets,
            "reinforcement_pool_restricted": pool_restricted,
            "reinforcement_pool_allowed_masks": pool_allowed_masks,
            "reinforcement_pool_source_card_indices": pool_source_card_indices,
            "movement_origin_indices": movement_origin_indices,
            "movement_target_indices": movement_target_indices,
            "movement_troop_log2": movement_troop_log2,
            "movement_troop_bytes": movement_troop_bytes,
            "movement_troop_offsets": movement_troop_offsets,
            "pending_occupation_actor_index": pending_occupation_actor_index,
            "pending_occupation_defender_index": pending_occupation_defender_index,
            "pending_occupation_origin_index": pending_occupation_origin_index,
            "pending_occupation_target_index": pending_occupation_target_index,
            "pending_occupation_max_troop_log2": np.asarray(
                [_log2_integer(pending_occupation_max_count)],
                dtype=np.float64,
            ),
            "pending_occupation_max_troop_bytes": (
                pending_occupation_max_troop_bytes
            ),
            "pending_occupation_max_troop_offsets": (
                pending_occupation_max_troop_offsets
            ),
            "own_objective_index": np.asarray(
                [own_objective_index],
                dtype=np.int8,
            ),
            "own_fallback_objective_index": np.asarray(
                [own_fallback_objective_index],
                dtype=np.int8,
            ),
            "own_hand_card_indices": own_hand_card_indices,
            "public_history_event_type_indices": history_event_type_indices,
            "public_history_actor_indices": history_actor_indices,
            "public_history_bytes": history_bytes,
            "public_history_offsets": history_offsets,
        }

    def _build_adjacency(self) -> NDArray[np.int8]:
        graph = np.zeros(
            (len(self.territory_ids), len(self.territory_ids)),
            dtype=np.int8,
        )
        for edge in WAR1_BOARD_SPEC.map.adjacencies:
            first = self._territory_index(edge.first_territory_id)
            second = self._territory_index(edge.second_territory_id)
            graph[first, second] = 1
            graph[second, first] = 1
        return graph

    def _build_space(self) -> spaces.Dict:
        int64_max = np.iinfo(np.int64).max
        float64_max = float(np.finfo(np.float64).max)
        territory_count = len(self.territory_ids)
        continent_count = len(self.continent_ids)
        return spaces.Dict(
            {
                "adjacency": spaces.MultiBinary(
                    (territory_count, territory_count)
                ),
                "territory_continent_indices": spaces.Box(
                    0,
                    continent_count - 1,
                    shape=(territory_count,),
                    dtype=np.int8,
                ),
                "continent_bonuses": spaces.Box(
                    0,
                    max(int(value) for value in self._continent_bonuses),
                    shape=(continent_count,),
                    dtype=np.int8,
                ),
                "continent_owner_indices": spaces.Box(
                    -1,
                    MAX_PLAYERS - 1,
                    shape=(continent_count,),
                    dtype=np.int8,
                ),
                "territory_owner_indices": spaces.Box(
                    0,
                    MAX_PLAYERS,
                    shape=(territory_count,),
                    dtype=np.int8,
                ),
                "territory_owner_mask": spaces.MultiBinary(
                    (territory_count, MAX_PLAYERS + 1)
                ),
                "territory_troop_log2": spaces.Box(
                    0.0,
                    float64_max,
                    shape=(territory_count,),
                    dtype=np.float64,
                ),
                "territory_troop_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "territory_troop_offsets": spaces.Box(
                    0,
                    int64_max,
                    shape=(territory_count + 1,),
                    dtype=np.int64,
                ),
                "phase_one_hot": spaces.MultiBinary(len(PHASE_IDS)),
                "player_mask": spaces.MultiBinary(MAX_PLAYERS),
                "player_color_indices": spaces.Box(
                    -1,
                    len(PLAYER_COLOR_IDS) - 1,
                    shape=(MAX_PLAYERS,),
                    dtype=np.int8,
                ),
                "player_territory_counts": spaces.Box(
                    0,
                    territory_count,
                    shape=(MAX_PLAYERS,),
                    dtype=np.int8,
                ),
                "player_troop_log2": spaces.Box(
                    0.0,
                    float64_max,
                    shape=(MAX_PLAYERS,),
                    dtype=np.float64,
                ),
                "player_troop_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "player_troop_offsets": spaces.Box(
                    0,
                    int64_max,
                    shape=(MAX_PLAYERS + 1,),
                    dtype=np.int64,
                ),
                "player_eliminated": spaces.MultiBinary(MAX_PLAYERS),
                "player_eliminated_by_indices": spaces.Box(
                    -1,
                    MAX_PLAYERS - 1,
                    shape=(MAX_PLAYERS,),
                    dtype=np.int8,
                ),
                "initial_placement_order_indices": spaces.Box(
                    -1,
                    MAX_PLAYERS - 1,
                    shape=(MAX_PLAYERS,),
                    dtype=np.int8,
                ),
                "active_player_index": spaces.Box(-1, 5, shape=(1,), dtype=np.int8),
                "turn_player_index": spaces.Box(-1, 5, shape=(1,), dtype=np.int8),
                "decision_player_index": spaces.Box(-1, 5, shape=(1,), dtype=np.int8),
                "winner_player_index": spaces.Box(-1, 5, shape=(1,), dtype=np.int8),
                "winner_objective_index": spaces.Box(
                    -1,
                    len(self.objective_ids) - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "game_flags": spaces.MultiBinary(2),
                "turn_number_log2": spaces.Box(
                    0.0,
                    float64_max,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "turn_number_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "turn_number_offsets": spaces.Box(
                    0,
                    int64_max,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "state_version_log2": spaces.Box(
                    0.0,
                    float64_max,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "state_version_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "state_version_offsets": spaces.Box(
                    0,
                    int64_max,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "reinforcement_pool_player_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=0,
                    maximum=MAX_PLAYERS - 1,
                ),
                "reinforcement_pool_origin_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=0,
                    maximum=len(REINFORCEMENT_ORIGINS) - 1,
                ),
                "reinforcement_pool_log2": _VariableArraySpace(
                    np.dtype(np.float64),
                    minimum=0.0,
                    maximum=float64_max,
                ),
                "reinforcement_pool_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "reinforcement_pool_offsets": _VariableArraySpace(
                    np.dtype(np.int64),
                    minimum=0,
                ),
                "reinforcement_pool_restricted": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=0,
                    maximum=1,
                ),
                "reinforcement_pool_allowed_masks": _VariableArraySpace(
                    np.dtype(np.int8),
                    trailing_shape=(territory_count,),
                    minimum=0,
                    maximum=1,
                ),
                "reinforcement_pool_source_card_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=-1,
                    maximum=len(self.card_ids) - 1,
                ),
                "movement_origin_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=0,
                    maximum=territory_count - 1,
                ),
                "movement_target_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=0,
                    maximum=territory_count - 1,
                ),
                "movement_troop_log2": _VariableArraySpace(
                    np.dtype(np.float64),
                    minimum=0.0,
                    maximum=float64_max,
                ),
                "movement_troop_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "movement_troop_offsets": _VariableArraySpace(
                    np.dtype(np.int64),
                    minimum=0,
                ),
                "pending_occupation_actor_index": spaces.Box(
                    -1,
                    MAX_PLAYERS - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "pending_occupation_defender_index": spaces.Box(
                    -1,
                    MAX_PLAYERS - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "pending_occupation_origin_index": spaces.Box(
                    -1,
                    territory_count - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "pending_occupation_target_index": spaces.Box(
                    -1,
                    territory_count - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "pending_occupation_max_troop_log2": spaces.Box(
                    0.0,
                    float64_max,
                    shape=(1,),
                    dtype=np.float64,
                ),
                "pending_occupation_max_troop_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "pending_occupation_max_troop_offsets": spaces.Box(
                    0,
                    int64_max,
                    shape=(2,),
                    dtype=np.int64,
                ),
                "own_objective_index": spaces.Box(
                    -1,
                    len(self.objective_ids) - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "own_fallback_objective_index": spaces.Box(
                    -1,
                    len(self.objective_ids) - 1,
                    shape=(1,),
                    dtype=np.int8,
                ),
                "own_hand_card_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=0,
                    maximum=len(self.card_ids) - 1,
                ),
                "public_history_event_type_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=-1,
                    maximum=len(PUBLIC_EVENT_TYPES) - 1,
                ),
                "public_history_actor_indices": _VariableArraySpace(
                    np.dtype(np.int8),
                    minimum=-1,
                    maximum=MAX_PLAYERS - 1,
                ),
                "public_history_bytes": _VariableArraySpace(
                    np.dtype(np.uint8),
                    minimum=0,
                    maximum=255,
                ),
                "public_history_offsets": _VariableArraySpace(
                    np.dtype(np.int64),
                    minimum=0,
                ),
            }
        )

    @staticmethod
    def _player_index(player_indices: dict[str, int], player_id: str) -> int:
        try:
            return player_indices[player_id]
        except KeyError as error:
            raise ValueError(f"Unknown player ID {player_id!r}.") from error

    @staticmethod
    def _slot_array(
        player_id: str | None,
        player_indices: dict[str, int],
    ) -> NDArray[np.int8]:
        return np.asarray(
            [
                -1
                if player_id is None
                else War1ObservationEncoder._player_index(
                    player_indices,
                    player_id,
                )
            ],
            dtype=np.int8,
        )

    def _phase_index(self, phase_id: str) -> int:
        try:
            return self._phase_indices[phase_id]
        except KeyError as error:
            raise ValueError(f"Unknown game phase {phase_id!r}.") from error

    def _origin_index(self, origin: str) -> int:
        try:
            return self._origin_indices[origin]
        except KeyError as error:
            raise ValueError(f"Unknown reinforcement origin {origin!r}.") from error

    def _territory_index(self, territory_id: str) -> int:
        try:
            return self._territory_indices[territory_id]
        except KeyError as error:
            raise ValueError(f"Unknown territory ID {territory_id!r}.") from error

    def _card_index(self, card_id: str) -> int:
        try:
            return self._card_indices[card_id]
        except KeyError as error:
            raise ValueError(f"Unknown card ID {card_id!r}.") from error

    def _territory_slot_array(
        self,
        territory_id: str | None,
    ) -> NDArray[np.int8]:
        return np.asarray(
            [
                -1
                if territory_id is None
                else self._territory_index(territory_id)
            ],
            dtype=np.int8,
        )

    @staticmethod
    def _optional_index(
        index_by_value: dict[str, int],
        value: str | None,
        label: str,
    ) -> int:
        if value is None:
            return -1
        try:
            return index_by_value[value]
        except KeyError as error:
            raise ValueError(f"Unknown {label} ID {value!r}.") from error

def _log2_integer(value: int) -> float:
    if value < 0:
        raise ValueError("Logarithmic observation features require non-negative values.")
    if value == 0:
        return 0.0
    shift = max(0, value.bit_length() - 53)
    leading_bits = value >> shift
    return math.log2(float(leading_bits)) + shift


def _pack_byte_records(
    records: Iterable[bytes],
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    data = bytearray()
    offsets = [0]
    for record in records:
        data.extend(record)
        offsets.append(len(data))
    return (
        np.frombuffer(bytes(data), dtype=np.uint8).copy(),
        np.asarray(offsets, dtype=np.int64),
    )


__all__ = [
    "MAX_PLAYERS",
    "PHASE_IDS",
    "PLAYER_COLOR_IDS",
    "PUBLIC_EVENT_TYPES",
    "REINFORCEMENT_ORIGINS",
    "WAR1_RL_OBSERVATION_ENCODING_VERSION",
    "War1NumericObservation",
    "War1ObservationEncoder",
    "decode_integer_records",
    "encode_integer_records",
]
