from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("torch")

import torch

from war_ai.reproducibility import ArtifactStore
from war_ai.rl.ppo import PPOLoss
from war_ai.rl.telemetry import (
    EpisodeAccumulator,
    TrainingEpisodeRecord,
    TrainingRunJournal,
    apply_ppo_update,
)


def test_episode_records_separate_official_wins_from_shaped_truncations(
    tmp_path: Path,
) -> None:
    journal_path = tmp_path / "run.jsonl"
    journal = TrainingRunJournal(journal_path, run_id="ppo-run")
    accumulator = EpisodeAccumulator(
        run_id="ppo-run",
        episode_index=0,
        seed=17,
        reset_info={"objective_id": "W1-OBJ-07", "num_players": 3},
    )
    assert (
        accumulator.add_transition(
            0.25,
            terminated=False,
            truncated=False,
            info={
                "official_reward": 0.0,
                "shaping_reward": 0.25,
                "semantic_duration": 2,
            },
        )
        is None
    )
    resumed_accumulator = EpisodeAccumulator.from_state_dict(
        accumulator.state_dict()
    )
    truncated_record = resumed_accumulator.add_transition(
        0.75,
        terminated=False,
        truncated=True,
        info={
            "official_reward": 0.0,
            "shaping_reward": 0.75,
            "semantic_duration": 3,
        },
    )
    assert truncated_record is not None
    assert truncated_record.official_reward == 0.0
    assert truncated_record.shaped_reward == 1.0
    assert truncated_record.semantic_decisions == 5
    journal.record_episode(truncated_record)

    journal.record_episode(
        TrainingEpisodeRecord(
            run_id="ppo-run",
            episode_index=1,
            seed=18,
            objective_id="W1-OBJ-07",
            table_size=3,
            official_reward=1.0,
            shaped_reward=0.1,
            decision_count=4,
            semantic_decisions=7,
            terminated=True,
            truncated=False,
            elapsed_seconds=2.0,
        )
    )
    journal.record_episode(
        TrainingEpisodeRecord(
            run_id="ppo-run",
            episode_index=2,
            seed=19,
            objective_id="W1-OBJ-13",
            table_size=4,
            official_reward=0.0,
            shaped_reward=-0.2,
            decision_count=2,
            semantic_decisions=4,
            terminated=True,
            truncated=False,
            elapsed_seconds=1.0,
        )
    )

    curves = journal.curves_by_objective_and_table
    first = next(
        curve
        for curve in curves
        if curve.objective_id == "W1-OBJ-07" and curve.table_size == 3
    )
    assert first.episodes == 2
    assert first.completed_episodes == 1
    assert first.official_wins == 1
    assert first.official_win_rate == 1.0
    assert first.shaped_reward_total == pytest.approx(1.1)
    assert first.truncations == 1
    assert first.truncation_rate == 0.5

    restored_journal = TrainingRunJournal(journal_path, run_id="ppo-run")
    assert restored_journal.episode_count == 3
    assert restored_journal.curves_by_objective_and_table == curves
    assert len(journal_path.read_text(encoding="utf-8").splitlines()) == 3
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    metrics_artifact = restored_journal.store_artifact(artifact_store)
    assert artifact_store.verify(metrics_artifact).read_bytes() == (
        journal_path.read_bytes()
    )


def test_ppo_update_records_losses_gradients_throughput_and_resources(
    tmp_path: Path,
) -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    prediction = model(torch.tensor([[1.0, -2.0]]))
    loss_value = prediction.square().mean()
    loss = PPOLoss(
        total_loss=loss_value,
        policy_loss=loss_value.detach(),
        value_loss=loss_value.detach(),
        entropy=torch.tensor(0.4),
        approximate_kl=torch.tensor(0.03),
        clip_fraction=torch.tensor(0.1),
        importance_ratios=torch.ones(1),
    )

    gradient_norm = apply_ppo_update(model, optimizer, loss)
    journal = TrainingRunJournal(tmp_path / "updates.jsonl", run_id="update-run")
    record = journal.record_ppo_update(
        update_index=1,
        environment_steps=32,
        loss=loss,
        gradient_norm=gradient_norm,
        learning_rate=0.01,
        update_seconds=0.25,
    )

    assert record.approximate_kl == pytest.approx(0.03)
    assert record.entropy == pytest.approx(0.4)
    assert record.gradient_norm > 0.0
    assert record.environment_steps_per_second > 0.0
    assert record.updates_per_second > 0.0
    assert record.cpu_max_rss_bytes is None or record.cpu_max_rss_bytes > 0
    stored = json.loads(
        (tmp_path / "updates.jsonl").read_text(encoding="utf-8")
    )
    assert stored["event_type"] == "ppo_update"
    assert stored["policy_loss"] == pytest.approx(float(loss_value.detach()))


def test_episode_accumulator_rejects_mixed_reward_accounting() -> None:
    accumulator = EpisodeAccumulator(
        run_id="bad-run",
        episode_index=0,
        seed=1,
        reset_info={"objective_id": "W1-OBJ-07", "num_players": 3},
    )

    with pytest.raises(ValueError, match="must equal official plus shaping"):
        accumulator.add_transition(
            1.0,
            terminated=True,
            truncated=False,
            info={
                "official_reward": 0.0,
                "shaping_reward": 0.25,
                "semantic_duration": 1,
            },
        )
