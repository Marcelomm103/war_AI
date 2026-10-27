"""Stable token IDs shared by the action codec and RL policies."""

ACTION_TOKEN_ENCODING_VERSION = "1.0.0"
ACTION_TYPE_IDS = (
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
)
ACTION_BODY_END_TOKEN = 256
ACTION_CONFIRM_TOKEN = 257
ACTION_TYPE_TOKEN_OFFSET = 258
ACTION_TOKEN_VOCAB_SIZE = ACTION_TYPE_TOKEN_OFFSET + len(ACTION_TYPE_IDS)

__all__ = [
    "ACTION_BODY_END_TOKEN",
    "ACTION_CONFIRM_TOKEN",
    "ACTION_TOKEN_ENCODING_VERSION",
    "ACTION_TOKEN_VOCAB_SIZE",
    "ACTION_TYPE_IDS",
    "ACTION_TYPE_TOKEN_OFFSET",
]
