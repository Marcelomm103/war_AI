from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("war_game_engine")

from war_game_engine.contracts import (
    Action,
    ActionOperation,
    ArmyPlacement,
    AttackOperation,
    ClaimTerritoryOperation,
    DiscardTerritoryCardOperation,
    DrawTerritoryCardOperation,
    EndPhaseOperation,
    EndTurnOperation,
    MoveArmiesOperation,
    OccupyTerritoryOperation,
    PlaceArmiesOperation,
    TradeCardsOperation,
)
from war_game_engine.engine import legal_actions
from war_game_engine.setup import SetupPlayer, prepare_game
from war_game_engine.versioning import ACTION_SCHEMA_VERSION

from war_ai.envs.actions import (
    ACTION_BODY_END_TOKEN,
    ACTION_CONFIRM_TOKEN,
    ACTION_TOKEN_VOCAB_SIZE,
    ACTION_TYPE_IDS,
    ACTION_TYPE_TOKEN_OFFSET,
    StructuredActionCodec,
)

PLAYERS = (
    SetupPlayer(player_id="player_0", color="white"),
    SetupPlayer(player_id="player_1", color="red"),
    SetupPlayer(player_id="player_2", color="black"),
)


def test_codec_round_trips_every_current_legal_action() -> None:
    state = prepare_game("action-codec", PLAYERS, seed=23).state
    legal = legal_actions(state)
    codec = StructuredActionCodec()

    values, offsets = codec.encode_table(legal)

    assert values.dtype == np.int16
    assert offsets.shape == (len(legal) + 1,)
    assert codec.decode_table(values, offsets, legal) == legal
    for action in legal:
        tokens = codec.encode_action(action)
        assert tokens[0] == (
            ACTION_TYPE_TOKEN_OFFSET
            + ACTION_TYPE_IDS.index(action.operation.action_type)
        )
        assert tokens[-1] == ACTION_CONFIRM_TOKEN
        assert codec.decode_action(tokens, legal) == action


def test_codec_round_trips_each_semantic_operation_type() -> None:
    operations: tuple[ActionOperation, ...] = (
        ClaimTerritoryOperation(action_type="claim_territory", territory_id="T01"),
        PlaceArmiesOperation(
            action_type="place_armies",
            placements=(
                ArmyPlacement(
                    territory_id="T01",
                    army_count=1,
                    source_pool_id="player_0:initial",
                ),
            ),
            placement_phase="initial",
        ),
        TradeCardsOperation(
            action_type="trade_cards",
            card_ids=("T01", "T02", "T03"),
        ),
        DrawTerritoryCardOperation(action_type="draw_territory_card"),
        DiscardTerritoryCardOperation(
            action_type="discard_territory_card",
            card_id="JOKER-01",
        ),
        AttackOperation(
            action_type="attack",
            origin_territory_id="T01",
            target_territory_id="T02",
            dice_count=3,
        ),
        OccupyTerritoryOperation(
            action_type="occupy_territory",
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=7,
        ),
        MoveArmiesOperation(
            action_type="move_armies",
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=9,
        ),
        EndPhaseOperation(action_type="end_phase", phase_id="attack"),
        EndTurnOperation(action_type="end_turn"),
    )
    actions = tuple(
        Action(
            contract_type="action",
            action_id=f"legal-operation-{index}",
            actor_player_id="player_0",
            action_schema_version=ACTION_SCHEMA_VERSION,
            expected_state_version=3,
            operation=operation,
        )
        for index, operation in enumerate(operations)
    )
    codec = StructuredActionCodec()

    values, offsets = codec.encode_table(actions)

    assert codec.decode_table(values, offsets, actions) == actions
    assert np.count_nonzero(codec.valid_next_token_mask((), actions)) == len(
        ACTION_TYPE_IDS
    )


def test_conditional_prefix_masks_wait_for_explicit_confirmation() -> None:
    state = prepare_game("action-prefix", PLAYERS, seed=31).state
    legal = legal_actions(state)
    codec = StructuredActionCodec()
    tokens = codec.encode_action(legal[0])

    initial_mask = codec.valid_next_token_mask((), legal)
    before_separator = codec.valid_next_token_mask(
        tokens[: tokens.index(ACTION_BODY_END_TOKEN)],
        legal,
    )
    before_confirmation = codec.valid_next_token_mask(tokens[:-1], legal)
    completed_mask = codec.valid_next_token_mask(tokens, legal)

    assert initial_mask.shape == (ACTION_TOKEN_VOCAB_SIZE,)
    assert initial_mask[tokens[0]] == 1
    assert before_separator[ACTION_BODY_END_TOKEN] == 1
    assert before_confirmation[ACTION_CONFIRM_TOKEN] == 1
    assert not np.any(completed_mask)
    with pytest.raises(ValueError, match="confirmation"):
        codec.decode_action(tokens[:-1], legal)
    with pytest.raises(ValueError, match="not a prefix"):
        codec.valid_next_token_mask((ACTION_TOKEN_VOCAB_SIZE - 1,), legal)


def test_variable_integer_parameters_have_no_arbitrary_token_cap() -> None:
    huge_count = (1 << 4096) + 987654321
    action = Action(
        contract_type="action",
        action_id="legal-huge-movement",
        actor_player_id="player_0",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=17,
        operation=MoveArmiesOperation(
            action_type="move_armies",
            origin_territory_id="T01",
            target_territory_id="T02",
            army_count=huge_count,
        ),
    )
    codec = StructuredActionCodec()

    tokens = codec.encode_action(action)

    assert len(tokens) > 1000
    assert codec.decode_action(tokens, (action,)) == action


def test_decoder_rejects_an_action_not_in_the_current_legal_set() -> None:
    state = prepare_game("action-stale", PLAYERS, seed=7).state
    legal = legal_actions(state)
    codec = StructuredActionCodec()
    tokens = codec.encode_action(legal[0])

    with pytest.raises(ValueError, match="currently legal"):
        codec.decode_action(tokens, ())
