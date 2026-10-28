"""Tests for Phase 7.2 modular neural architectures."""

from __future__ import annotations

from dataclasses import replace

import torch

from war_ai.envs.observations import MAX_PLAYERS
from war_ai.envs.war1_aec import War1AECEnv, War1AECObservation
from war_ai.rl.agent import FULL_GAME_INTENT_END_TOKEN, FULL_GAME_INTENT_VOCAB_SIZE
from war_ai.rl.architectures import (
    NUM_CARDS,
    NUM_OBJECTIVES,
    ArchitectureConfig,
    CausalEventTransformer,
    WarArchitectureModel,
    build_phase7_architecture_catalog,
    evaluate_architecture_catalog,
    extract_architecture_inputs,
)


def _sample_observations(count: int = 3) -> list[War1AECObservation]:
    env = War1AECEnv()
    env.reset(seed=2026)
    collected: list[War1AECObservation] = []
    for _ in range(count):
        agent_id = env.agent_selection
        assert agent_id is not None
        obs = env.observe(agent_id)
        collected.append(obs)
        env.step(0)
    return collected


def test_architecture_catalog_evaluates_all_seven_variants() -> None:
    observations = _sample_observations(3)
    summaries = evaluate_architecture_catalog(
        observations,
        hidden_size=32,
        graph_layers=3,
        max_history_events=64,
    )

    catalog = build_phase7_architecture_catalog(hidden_size=32)
    assert [summary.variant_id for summary in summaries] == list(catalog.keys())
    for summary in summaries:
        assert summary.parameter_count > 0
        assert summary.legal_action_selection_valid
        assert summary.policy_logits_shape == (3, 2, FULL_GAME_INTENT_VOCAB_SIZE)
        assert summary.value_shape == (3,)
        if summary.config.use_belief_head:
            assert summary.belief_objective_shape == (
                3,
                MAX_PLAYERS,
                NUM_OBJECTIVES,
            )
            assert summary.belief_card_shape == (3, MAX_PLAYERS, NUM_CARDS)
        else:
            assert summary.belief_objective_shape is None
            assert summary.belief_card_shape is None


def test_causal_event_transformer_never_leaks_future_events() -> None:
    torch.manual_seed(42)
    transformer = CausalEventTransformer(
        hidden_size=32,
        num_heads=4,
        max_history_events=16,
    )
    transformer.eval()

    event_types_a = torch.tensor([[1, 2, 3, 4, 5, 6]], dtype=torch.long)
    event_actors_a = torch.tensor([[1, 2, 3, 1, 2, 3]], dtype=torch.long)
    mask = torch.ones((1, 6), dtype=torch.bool)

    event_types_b = event_types_a.clone()
    event_actors_b = event_actors_a.clone()
    event_types_b[0, 3:] = torch.tensor([9, 10, 11], dtype=torch.long)
    event_actors_b[0, 3:] = torch.tensor([3, 1, 2], dtype=torch.long)

    with torch.no_grad():
        seq_a = transformer.encode_sequence(event_types_a, event_actors_a, mask)
        seq_b = transformer.encode_sequence(event_types_b, event_actors_b, mask)

    assert torch.allclose(seq_a[:, :3, :], seq_b[:, :3, :], atol=1e-6)
    assert not torch.allclose(seq_a[:, 3:, :], seq_b[:, 3:, :], atol=1e-6)


def test_phase_conditioned_head_and_objective_card_encoder_respond_to_inputs() -> None:
    observations = _sample_observations(1)
    inputs = extract_architecture_inputs(observations)
    tokens = torch.tensor([[0, FULL_GAME_INTENT_END_TOKEN]], dtype=torch.long)

    torch.manual_seed(7)
    model = WarArchitectureModel(
        ArchitectureConfig(
            encoder="graph_transformer",
            memory="none",
            hidden_size=32,
            graph_layers=3,
            phase_conditioned_heads=True,
            use_objective_and_card_encoder=True,
            use_belief_head=True,
        )
    )
    model.eval()

    altered_phase_inputs = replace(
        inputs,
        phase_indices=(inputs.phase_indices + 3) % 10,
    )
    altered_objective_inputs = replace(
        inputs,
        objective_indices=(inputs.objective_indices + 5) % NUM_OBJECTIVES,
    )

    with torch.no_grad():
        base_out = model(inputs, tokens)
        phase_out = model(altered_phase_inputs, tokens)
        obj_out = model(altered_objective_inputs, tokens)

    assert not torch.allclose(base_out.token_logits, phase_out.token_logits)
    assert not torch.allclose(base_out.values, obj_out.values)


def test_gru_and_lstm_architectures_maintain_recurrent_state_and_gradients() -> None:
    observations = _sample_observations(2)
    inputs = extract_architecture_inputs(observations)
    tokens = torch.tensor(
        [[0, FULL_GAME_INTENT_END_TOKEN], [1, FULL_GAME_INTENT_END_TOKEN]],
        dtype=torch.long,
    )

    for memory_mode in ("gru", "lstm"):
        model = WarArchitectureModel(
            ArchitectureConfig(
                encoder="message_passing_gnn",
                memory=memory_mode,
                hidden_size=32,
                graph_layers=3,
                use_belief_head=True,
            )
        )
        out1 = model(inputs, tokens)
        assert out1.recurrent_state is not None
        out2 = model(inputs, tokens, recurrent_state=out1.recurrent_state)
        assert not torch.allclose(out1.values, out2.values)

        assert out2.opponent_objective_logits is not None
        assert out2.opponent_card_logits is not None
        total_loss = (
            out2.token_logits.sum()
            + out2.values.sum()
            + out2.opponent_objective_logits.sum()
            + out2.opponent_card_logits.sum()
        )
        total_loss.backward()
        grad_norms = [
            p.grad.norm().item()
            for p in model.parameters()
            if p.grad is not None
        ]
        assert grad_norms and any(norm > 0.0 for norm in grad_norms)
