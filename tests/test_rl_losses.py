from __future__ import annotations

import pytest

pytest.importorskip("torch")

import torch

from war_ai.rl.losses import (
    joint_action_entropy,
    joint_action_kl_divergence,
    joint_action_log_prob,
    masked_categorical_entropy,
    masked_log_softmax,
)
from war_ai.rl.memory import PerPlayerRecurrentMemory


def _token_batch() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = torch.tensor(
        [
            [
                [0.0, 2.0, 50.0],
                [99.0, 1.0, 3.0],
                [4.0, 5.0, 6.0],
            ]
        ],
        dtype=torch.float64,
        requires_grad=True,
    )
    legal_mask = torch.tensor(
        [
            [
                [True, True, False],
                [False, True, True],
                [False, False, False],
            ]
        ]
    )
    token_mask = torch.tensor([[True, True, False]])
    actions = torch.tensor([[1, 2, 0]], dtype=torch.long)
    return logits, legal_mask, token_mask, actions


def test_joint_probability_and_entropy_follow_each_token_mask() -> None:
    logits, legal_mask, token_mask, actions = _token_batch()

    log_probability = joint_action_log_prob(
        logits,
        actions,
        legal_mask,
        token_mask,
    )
    entropy_by_token = masked_categorical_entropy(
        logits,
        legal_mask,
        active_mask=token_mask,
    )
    joint_entropy = joint_action_entropy(logits, legal_mask, token_mask)
    expected = torch.log_softmax(logits[0, 0, :2], dim=0)[1]
    expected += torch.log_softmax(logits[0, 1, 1:], dim=0)[1]

    assert torch.allclose(log_probability[0], expected)
    assert entropy_by_token[0, 2] == 0.0
    assert torch.allclose(joint_entropy, entropy_by_token.sum(dim=-1))

    log_probability.sum().backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, 0, 2] == 0.0
    assert logits.grad[0, 1, 0] == 0.0
    assert torch.all(logits.grad[0, 2] == 0.0)


def test_masked_kl_uses_the_same_legal_support() -> None:
    old_logits, legal_mask, token_mask, _ = _token_batch()
    same_policy_kl = joint_action_kl_divergence(
        old_logits,
        old_logits.clone(),
        legal_mask,
        token_mask,
    )
    changed_logits = old_logits.detach().clone()
    changed_logits[0, 0, 0] += 1.0
    changed_policy_kl = joint_action_kl_divergence(
        old_logits.detach(),
        changed_logits,
        legal_mask,
        token_mask,
    )

    assert torch.allclose(same_policy_kl, torch.zeros_like(same_policy_kl))
    assert torch.isfinite(changed_policy_kl).all()
    assert torch.all(changed_policy_kl >= 0.0)


def test_invalid_or_masked_action_tokens_are_rejected() -> None:
    logits = torch.zeros((1, 3), dtype=torch.float32)
    with pytest.raises(ValueError, match="legal token"):
        masked_log_softmax(logits, torch.zeros_like(logits, dtype=torch.bool))
    with pytest.raises(ValueError, match="zero and one"):
        masked_log_softmax(
            logits,
            torch.tensor([[0, 2, 0]], dtype=torch.int8),
        )

    with pytest.raises(ValueError, match="masked as illegal"):
        joint_action_log_prob(
            logits=torch.zeros((1, 1, 3)),
            actions=torch.tensor([[2]], dtype=torch.long),
            legal_mask=torch.tensor([[[True, True, False]]]),
            token_mask=torch.tensor([[True]]),
        )


def test_inactive_padded_tokens_with_empty_masks_contribute_zero() -> None:
    logits = torch.tensor(
        [[[float("nan"), float("inf")], [0.0, 1.0]]],
        requires_grad=True,
    )
    legal_mask = torch.tensor([[[False, False], [True, False]]])
    token_mask = torch.tensor([[False, True]])
    actions = torch.tensor([[999, 0]], dtype=torch.long)

    log_probabilities = masked_log_softmax(
        logits,
        legal_mask,
        active_mask=token_mask,
    )
    joint_log_probability = joint_action_log_prob(
        logits,
        actions,
        legal_mask,
        token_mask,
    )

    assert torch.all(log_probabilities[0, 0] == 0.0)
    assert torch.isfinite(log_probabilities[0, 0]).all()
    assert torch.isneginf(log_probabilities[0, 1, 1])
    assert torch.allclose(joint_log_probability, torch.tensor([0.0]))
    joint_log_probability.sum().backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.all(logits.grad[0, 0] == 0.0)


def test_token_vocabulary_boundaries_have_finite_log_probabilities() -> None:
    vocab_size = 268
    logits = torch.zeros((1, 1, vocab_size), requires_grad=True)
    legal_mask = torch.zeros_like(logits, dtype=torch.bool)
    legal_mask[0, 0, 0] = True
    legal_mask[0, 0, vocab_size - 1] = True
    token_mask = torch.ones((1, 1), dtype=torch.bool)

    lower = joint_action_log_prob(
        logits,
        torch.tensor([[0]], dtype=torch.long),
        legal_mask,
        token_mask,
    )
    upper = joint_action_log_prob(
        logits,
        torch.tensor([[vocab_size - 1]], dtype=torch.long),
        legal_mask,
        token_mask,
    )

    assert torch.allclose(lower, torch.tensor([-torch.log(torch.tensor(2.0))]))
    assert torch.allclose(upper, lower)


def test_rollout_prefix_masks_match_joint_action_learning_masks() -> None:
    pytest.importorskip("pettingzoo")
    pytest.importorskip("war_game_engine")
    from war_ai.envs import War1AECEnv

    env = War1AECEnv()
    env.reset(seed=37)
    agent = env.agent_selection
    assert agent is not None
    observation = env.observe(agent)
    player_observation = observation["engine_observation"]
    assert player_observation is not None
    action = player_observation.legal_actions[0]
    tokens = env.action_codec.encode_action(action)
    masks = []
    for index, token in enumerate(tokens):
        prefix_mask = env.action_prefix_mask(agent, tokens[:index])
        assert prefix_mask[token] == 1
        masks.append(torch.from_numpy(prefix_mask.astype(bool)))
    legal_mask = torch.stack(masks).unsqueeze(0)
    action_tokens = torch.tensor([tokens], dtype=torch.long)
    token_mask = torch.ones_like(action_tokens, dtype=torch.bool)
    logits = torch.zeros(
        (1, len(tokens), legal_mask.shape[-1]),
        requires_grad=True,
    )

    log_probability = joint_action_log_prob(
        logits,
        action_tokens,
        legal_mask,
        token_mask,
    )
    log_probability.sum().backward()  # type: ignore[no-untyped-call]

    assert torch.isfinite(log_probability).all()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_recurrent_memory_isolated_by_game_and_player_with_lstm_state() -> None:
    memory = PerPlayerRecurrentMemory.zeros(
        batch_size=2,
        num_players=3,
        num_layers=2,
        hidden_size=2,
        with_cell_state=True,
    )
    next_hidden = torch.arange(8, dtype=torch.float32).reshape(2, 2, 2)
    next_hidden.requires_grad_()
    next_cell = (next_hidden.detach() + 10.0).requires_grad_()
    player_indices = torch.tensor([0, 2], dtype=torch.long)

    updated = memory.update(
        player_indices,
        (next_hidden, next_cell),
    )
    selected_hidden, selected_cell = updated.select(player_indices)
    hidden_sum = selected_hidden.sum()
    cell_sum = selected_cell.sum()
    (hidden_sum + cell_sum).backward()

    assert torch.equal(selected_hidden, next_hidden)
    assert torch.equal(selected_cell, next_cell)
    assert next_hidden.grad is not None
    assert torch.all(next_hidden.grad == 1.0)
    assert next_cell.grad is not None
    assert torch.all(next_cell.grad == 1.0)

    inactive = updated.select(
        torch.tensor([-1, -1], dtype=torch.long),
        active_mask=torch.tensor([False, False]),
    )
    assert isinstance(inactive, tuple)
    assert torch.all(inactive[0] == 0.0)
    assert torch.all(inactive[1] == 0.0)

    reset = updated.reset_players(
        player_indices,
        reset_mask=torch.tensor([True, False]),
    )
    reset_selected, _ = reset.select(
        player_indices,
        active_mask=torch.tensor([True, False]),
    )
    assert torch.all(reset_selected[:, 0] == 0.0)


def test_recurrent_memory_rejects_an_invalid_active_seat() -> None:
    memory = PerPlayerRecurrentMemory.zeros(
        batch_size=1,
        num_players=3,
        num_layers=1,
        hidden_size=4,
    )

    with pytest.raises(IndexError, match="outside the table"):
        memory.select(torch.tensor([3], dtype=torch.long))
