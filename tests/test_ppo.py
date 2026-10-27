from __future__ import annotations

import math
from typing import Literal

import pytest

pytest.importorskip("torch")

import torch

from war_ai.rl.losses import joint_action_log_prob
from war_ai.rl.ppo import (
    PPOActorCritic,
    compute_gae,
    compute_masked_ppo_loss,
    normalize_advantages,
)


def test_masked_ppo_clips_ratios_and_reports_entropy() -> None:
    probability = torch.tensor([0.7, 0.3])
    logits = probability.log().reshape(1, 1, 2).repeat(2, 1, 1)
    logits.requires_grad_()
    legal_masks = torch.ones_like(logits, dtype=torch.bool)

    loss = compute_masked_ppo_loss(
        new_logits=logits,
        actions=torch.zeros((2, 1), dtype=torch.long),
        legal_masks=legal_masks,
        token_mask=torch.ones((2, 1), dtype=torch.bool),
        old_log_probabilities=torch.full((2,), math.log(0.5)),
        advantages=torch.tensor([1.0, -1.0]),
        value_predictions=torch.zeros(2),
        returns=torch.zeros(2),
        clip_range=0.2,
        value_coefficient=0.0,
        entropy_coefficient=0.0,
        normalize_advs=False,
    )

    expected_entropy = -(0.7 * math.log(0.7) + 0.3 * math.log(0.3))
    assert torch.allclose(
        loss.importance_ratios,
        torch.tensor([1.4, 1.4]),
        atol=1e-6,
    )
    assert torch.allclose(loss.policy_loss, torch.tensor(0.1), atol=1e-6)
    assert torch.allclose(loss.clip_fraction, torch.tensor(1.0))
    assert torch.allclose(loss.entropy, torch.tensor(expected_entropy), atol=1e-6)
    loss.total_loss.backward()  # type: ignore[no-untyped-call]
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_advantage_normalization_ignores_padding_and_handles_single_sample() -> None:
    advantages = torch.tensor([[1.0, 3.0, 100.0]])
    active = torch.tensor([[True, True, False]])

    normalized = normalize_advantages(advantages, sample_mask=active)
    single = normalize_advantages(
        torch.tensor([4.0]),
        sample_mask=torch.tensor([True]),
    )

    assert torch.allclose(normalized, torch.tensor([[-1.0, 1.0, 0.0]]))
    assert torch.equal(single, torch.zeros_like(single))

    with pytest.raises(ValueError, match="At least one advantage sample"):
        normalize_advantages(
            advantages,
            sample_mask=torch.zeros_like(active),
        )


def test_gae_stops_at_termination_and_bootstraps_truncation() -> None:
    rewards = torch.tensor([1.0, 100.0, 2.0, 0.0])
    values = torch.zeros_like(rewards)
    next_values = torch.tensor([500.0, 0.0, 10.0, 20.0])
    terminated = torch.tensor([True, False, False, False])
    truncated = torch.tensor([False, False, True, False])

    advantages, returns = compute_gae(
        rewards,
        values,
        next_values,
        terminated,
        truncated,
        gamma=1.0,
        gae_lambda=1.0,
    )

    assert torch.equal(advantages, torch.tensor([1.0, 112.0, 12.0, 20.0]))
    assert torch.equal(returns, advantages)


def test_masked_policy_rollout_log_probability_matches_teacher_forcing() -> None:
    model = PPOActorCritic(
        observation_size=3,
        token_vocab_size=4,
        hidden_size=8,
        end_token_id=3,
    )
    observation = torch.tensor([0.2, -0.5, 1.0])

    def prefix_mask(prefix: tuple[int, ...]) -> torch.Tensor:
        if not prefix:
            return torch.tensor([True, True, False, False])
        if len(prefix) == 1 and prefix[0] in (0, 1):
            return torch.tensor([False, False, False, True])
        raise AssertionError(f"Unexpected action prefix: {prefix!r}")

    sample = model.sample_action(
        observation,
        prefix_mask,
        deterministic=True,
    )
    action_tokens = torch.tensor([sample.action_tokens], dtype=torch.long)
    output = model(observation.unsqueeze(0), action_tokens)
    reevaluated_log_probability = joint_action_log_prob(
        output.token_logits,
        action_tokens,
        sample.legal_masks,
        sample.token_mask,
    )[0]

    assert sample.action_tokens[-1] == 3
    assert sample.legal_masks.shape == (1, len(sample.action_tokens), 4)
    assert torch.allclose(
        sample.old_log_probability,
        reevaluated_log_probability,
    )
    assert sample.entropy > 0.0
    assert torch.isfinite(sample.value)


def test_actor_critic_supports_feedforward_gru_and_lstm_with_local_values() -> None:
    recurrent_modes: tuple[Literal["none", "gru", "lstm"], ...] = (
        "none",
        "gru",
        "lstm",
    )
    observations = torch.randn(3, 2, 4)
    action_tokens = torch.tensor(
        [
            [[0, 3], [1, 3]],
            [[2, 3], [0, 3]],
            [[1, 3], [2, 3]],
        ],
        dtype=torch.long,
    )

    for recurrent in recurrent_modes:
        model = PPOActorCritic(
            observation_size=4,
            token_vocab_size=4,
            hidden_size=8,
            recurrent=recurrent,
            recurrent_layers=2,
            end_token_id=3,
        )
        output = model(observations, action_tokens)
        different_actions = model(
            observations,
            torch.flip(action_tokens, dims=(-1,)),
        )

        assert output.token_logits.shape == (3, 2, 2, 4)
        assert output.values.shape == (3, 2)
        assert torch.allclose(output.values, different_actions.values)
        if recurrent == "none":
            assert output.recurrent_state is None
        elif recurrent == "gru":
            assert isinstance(output.recurrent_state, torch.Tensor)
            assert output.recurrent_state.shape == (2, 2, 8)
        else:
            assert isinstance(output.recurrent_state, tuple)
            assert output.recurrent_state[0].shape == (2, 2, 8)
            assert output.recurrent_state[1].shape == (2, 2, 8)


@pytest.mark.parametrize("recurrent", ["gru", "lstm"])
def test_recurrent_actor_critic_resets_episode_start_state(
    recurrent: Literal["gru", "lstm"],
) -> None:
    model = PPOActorCritic(
        observation_size=3,
        token_vocab_size=4,
        hidden_size=8,
        recurrent=recurrent,
        end_token_id=3,
    )
    observations = torch.randn(2, 1, 3)
    action_tokens = torch.tensor([[[0]], [[3]]], dtype=torch.long)
    initial_hidden = torch.ones((1, 1, 8))
    recurrent_state: torch.Tensor | tuple[torch.Tensor, torch.Tensor]
    if recurrent == "gru":
        recurrent_state = initial_hidden
    else:
        recurrent_state = (initial_hidden, torch.full_like(initial_hidden, 2.0))

    reset_output = model(
        observations,
        action_tokens,
        recurrent_state=recurrent_state,
        episode_start_mask=torch.tensor([[False], [True]]),
    )
    fresh_output = model(
        observations[1],
        action_tokens[1],
    )

    assert torch.allclose(reset_output.values[1], fresh_output.values[0])


def test_ppo_loss_ignores_masked_padding_samples() -> None:
    logits = torch.tensor(
        [
            [[0.0, 0.0]],
            [[float("nan"), float("nan")]],
        ],
        requires_grad=True,
    )
    loss = compute_masked_ppo_loss(
        new_logits=logits,
        actions=torch.tensor([[0], [999]], dtype=torch.long),
        legal_masks=torch.tensor(
            [
                [[True, True]],
                [[False, False]],
            ]
        ),
        token_mask=torch.tensor([[True], [False]]),
        old_log_probabilities=torch.tensor([math.log(0.5), float("nan")]),
        advantages=torch.tensor([1.0, float("nan")]),
        value_predictions=torch.tensor([0.0, float("nan")]),
        returns=torch.tensor([1.0, float("nan")]),
        sample_mask=torch.tensor([True, False]),
        normalize_advs=False,
    )

    assert torch.isfinite(loss.total_loss)
    assert torch.equal(loss.importance_ratios, torch.tensor([1.0, 1.0]))
