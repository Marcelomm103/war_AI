from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("war_game_engine")

import torch

from war_ai.envs.actions import ACTION_CONFIRM_TOKEN, ACTION_TOKEN_VOCAB_SIZE
from war_ai.rl.ppo import PPOActorCritic
from war_ai.rl.small_tasks import (
    build_small_tasks,
    run_small_task_suite,
    train_small_task,
)
from war_ai.rl.telemetry import TrainingRunJournal


def test_small_task_suite_enumerates_exact_legal_action_sets() -> None:
    tasks = build_small_tasks()

    assert tuple(task.task_id for task in tasks) == (
        "legal_placement",
        "attack_or_stop",
        "occupation",
        "movement",
        "trades",
        "objective_completion",
    )
    for task in tasks:
        assert len(task.actions) >= 2
        assert task.exact_action_labels
        for action in task.actions:
            assert task.prefix_mask(())[action.tokens[0]]
            assert not task.prefix_mask(action.tokens).any()


def test_small_task_policy_samples_only_enumerated_legal_actions() -> None:
    for task in build_small_tasks():
        model = PPOActorCritic(
            observation_size=len(task.observation),
            token_vocab_size=ACTION_TOKEN_VOCAB_SIZE,
            hidden_size=8,
            end_token_id=ACTION_CONFIRM_TOKEN,
        )
        sample = model.sample_action(
            torch.tensor(task.observation, dtype=torch.float32),
            task.prefix_mask,
            deterministic=False,
            generator=torch.Generator().manual_seed(7),
        )

        assert any(
            action.tokens == sample.action_tokens
            for action in task.actions
        )
        assert sample.action_tokens[-1] == ACTION_CONFIRM_TOKEN


def test_six_ppo_microtasks_learn_against_exact_solutions() -> None:
    results = run_small_task_suite(
        seed=2026,
        updates=100,
        batch_size=32,
    )

    assert len(results) == 6
    for result in results:
        assert result.learned_expected_return >= result.initial_expected_return
        assert result.learned_expected_return >= result.exact_return - 0.02
        assert result.optimal_action_probability >= 0.98
        assert result.most_likely_action in result.exact_action_labels


def test_microtask_rejects_a_prefix_outside_its_legal_engine_actions() -> None:
    placement_task = build_small_tasks()[0]

    with pytest.raises(ValueError):
        placement_task.prefix_mask((ACTION_TOKEN_VOCAB_SIZE - 1,))


def test_microtask_ppo_updates_can_be_written_to_the_training_journal(
    tmp_path: Path,
) -> None:
    journal = TrainingRunJournal(
        tmp_path / "microtask-training.jsonl",
        run_id="microtask-training",
    )

    train_small_task(
        build_small_tasks()[0],
        seed=9,
        updates=2,
        batch_size=4,
        journal=journal,
    )

    assert journal.update_count == 2
    assert journal.environment_steps == 8
    assert journal.episode_count == 0
