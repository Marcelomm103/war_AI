"""Lossless structured action tokens and conditional legal-prefix masks."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from typing import TypeAlias

import numpy as np
from gymnasium.spaces import Space
from numpy.typing import NDArray
from war_game_engine.contracts import Action

from ..action_tokens import (
    ACTION_BODY_END_TOKEN,
    ACTION_CONFIRM_TOKEN,
    ACTION_TOKEN_ENCODING_VERSION,
    ACTION_TOKEN_VOCAB_SIZE,
    ACTION_TYPE_IDS,
    ACTION_TYPE_TOKEN_OFFSET,
)

ActionTokenSequence: TypeAlias = tuple[int, ...]


class ActionTokenValuesSpace(Space[NDArray[np.int16]]):
    def __init__(self) -> None:
        super().__init__(shape=None, dtype=np.int16)

    def sample(
        self,
        mask: NDArray[np.int8] | None = None,
        probability: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int16]:
        if mask is not None or probability is not None:
            raise ValueError("Action token values do not accept sampling masks.")
        return np.zeros(0, dtype=np.int16)

    def contains(self, value: object) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(np.int16)
            and value.ndim == 1
            and bool(np.all(value >= 0))
            and bool(np.all(value < ACTION_TOKEN_VOCAB_SIZE))
        )


class ActionTokenOffsetsSpace(Space[NDArray[np.int64]]):
    def __init__(self) -> None:
        super().__init__(shape=None, dtype=np.int64)

    def sample(
        self,
        mask: NDArray[np.int8] | None = None,
        probability: NDArray[np.float64] | None = None,
    ) -> NDArray[np.int64]:
        if mask is not None or probability is not None:
            raise ValueError("Action token offsets do not accept sampling masks.")
        return np.zeros(1, dtype=np.int64)

    def contains(self, value: object) -> bool:
        return (
            isinstance(value, np.ndarray)
            and value.dtype == np.dtype(np.int64)
            and value.ndim == 1
            and value.size > 0
            and value[0] == 0
            and bool(np.all(value[1:] >= value[:-1]))
        )


class StructuredActionCodec:
    """Encode engine operations as unbounded JSON-byte tokens with confirmation."""

    def encode_action(self, action: Action) -> ActionTokenSequence:
        action_type = action.operation.action_type
        try:
            action_type_token = (
                ACTION_TYPE_TOKEN_OFFSET + ACTION_TYPE_IDS.index(action_type)
            )
        except ValueError as error:
            raise ValueError(f"Unknown action type {action_type!r}.") from error

        payload = json.dumps(
            action.operation.model_dump(
                mode="json",
                exclude={"action_type"},
            ),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=False,
        ).encode("utf-8")
        action_id = action.action_id.encode("utf-8")
        if not action_id:
            raise ValueError("Action IDs must not be empty.")
        return (
            action_type_token,
            *payload,
            ACTION_BODY_END_TOKEN,
            *action_id,
            ACTION_CONFIRM_TOKEN,
        )

    def encode_table(
        self,
        legal_actions: Sequence[Action],
    ) -> tuple[NDArray[np.int16], NDArray[np.int64]]:
        values: list[int] = []
        offsets = [0]
        for action in legal_actions:
            values.extend(self.encode_action(action))
            offsets.append(len(values))
        return np.asarray(values, dtype=np.int16), np.asarray(offsets, dtype=np.int64)

    def decode_action(
        self,
        tokens: Iterable[int],
        legal_actions: Sequence[Action],
    ) -> Action:
        normalized = self._validated_tokens(tokens)
        if not normalized or normalized[-1] != ACTION_CONFIRM_TOKEN:
            raise ValueError("A structured action requires its confirmation token.")
        if normalized.count(ACTION_BODY_END_TOKEN) != 1:
            raise ValueError("A structured action requires exactly one body separator.")
        separator_index = normalized.index(ACTION_BODY_END_TOKEN)
        if separator_index <= 0 or separator_index + 2 >= len(normalized):
            raise ValueError("Structured action body and action ID must be complete.")
        if not ACTION_TYPE_TOKEN_OFFSET <= normalized[0] < ACTION_TOKEN_VOCAB_SIZE:
            raise ValueError("Structured action must begin with a known action type.")
        if any(
            token > 255
            for token in normalized[1:separator_index]
            + normalized[separator_index + 1 : -1]
        ):
            raise ValueError("Only byte tokens are valid inside action fields.")

        matches = [
            action
            for action in legal_actions
            if self.encode_action(action) == normalized
        ]
        if len(matches) != 1:
            raise ValueError(
                "Structured action does not identify exactly one currently legal action."
            )
        return matches[0]

    def decode_table(
        self,
        values: NDArray[np.int16],
        offsets: NDArray[np.int64],
        legal_actions: Sequence[Action],
    ) -> tuple[Action, ...]:
        if values.ndim != 1 or offsets.ndim != 1:
            raise ValueError("Encoded action values and offsets must be vectors.")
        if (
            offsets.size != len(legal_actions) + 1
            or offsets.size == 0
            or offsets[0] != 0
            or offsets[-1] != values.size
            or np.any(offsets[1:] < offsets[:-1])
        ):
            raise ValueError("Encoded action offsets are inconsistent.")
        decoded: list[Action] = []
        for action, start, end in zip(
            legal_actions,
            offsets[:-1],
            offsets[1:],
            strict=True,
        ):
            encoded = tuple(int(token) for token in values[int(start) : int(end)])
            if encoded != self.encode_action(action):
                raise ValueError(
                    "Encoded action table does not round-trip its legal actions."
                )
            decoded.append(action)
        return tuple(decoded)

    def valid_next_token_mask(
        self,
        prefix: Iterable[int],
        legal_actions: Sequence[Action],
    ) -> NDArray[np.int8]:
        normalized_prefix = self._validated_tokens(prefix)
        mask = np.zeros(ACTION_TOKEN_VOCAB_SIZE, dtype=np.int8)
        if not legal_actions:
            if normalized_prefix:
                raise ValueError("An empty legal-action set has no valid prefix.")
            return mask

        candidates: list[ActionTokenSequence] = []
        for action in legal_actions:
            encoded = self.encode_action(action)
            if encoded[: len(normalized_prefix)] == normalized_prefix:
                candidates.append(encoded)
        if not candidates:
            raise ValueError("Prefix is not a prefix of any legal action.")
        for candidate in candidates:
            if len(candidate) > len(normalized_prefix):
                mask[candidate[len(normalized_prefix)]] = 1
        return mask

    @staticmethod
    def _validated_tokens(tokens: Iterable[int]) -> ActionTokenSequence:
        normalized: list[int] = []
        for token in tokens:
            if isinstance(token, (bool, np.bool_)) or not isinstance(
                token, (int, np.integer)
            ):
                raise ValueError("Action tokens must be integer values.")
            value = int(token)
            if not 0 <= value < ACTION_TOKEN_VOCAB_SIZE:
                raise ValueError(f"Action token {value} is outside the vocabulary.")
            normalized.append(value)
        return tuple(normalized)


__all__ = [
    "ACTION_BODY_END_TOKEN",
    "ACTION_CONFIRM_TOKEN",
    "ACTION_TOKEN_VOCAB_SIZE",
    "ACTION_TOKEN_ENCODING_VERSION",
    "ACTION_TYPE_IDS",
    "ACTION_TYPE_TOKEN_OFFSET",
    "ActionTokenOffsetsSpace",
    "ActionTokenSequence",
    "ActionTokenValuesSpace",
    "StructuredActionCodec",
]
