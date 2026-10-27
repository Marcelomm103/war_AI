from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

import torch

from war_ai.baselines import create_baseline
from war_ai.envs.gymnasium_env import War1GymEnv
from war_ai.envs.war1_aec import War1AECEnv
from war_ai.reproducibility import ArtifactStore
from war_ai.rl.checkpoint import (
    PPOResumeState,
    restore_ppo_checkpoint,
    save_ppo_checkpoint,
)
from war_ai.rl.ppo import PPOActorCritic, PPOLoss
from war_ai.rl.small_tasks import build_intermediate_scenario_state
from war_ai.rl.telemetry import apply_ppo_update


def _intermediate_gym_env() -> War1GymEnv:
    aec_env = War1AECEnv(
        initial_state_factory=lambda game_id, seed: build_intermediate_scenario_state(
            "attack",
            game_id=game_id,
            seed=seed,
        )
    )
    return War1GymEnv(
        opponent_policies={
            "player_1": create_baseline("random_legal"),
            "player_2": create_baseline("conservative_borders"),
        },
        aec_env=aec_env,
        max_semantic_decisions=32,
    )


def test_ppo_checkpoint_restores_model_optimizer_rng_and_live_environment(
    tmp_path: Path,
) -> None:
    random.seed(31)
    np.random.seed(37)
    torch.manual_seed(41)
    model = PPOActorCritic(
        observation_size=3,
        token_vocab_size=4,
        hidden_size=8,
        end_token_id=3,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    prediction = model(
        torch.tensor([[0.1, 0.2, 0.3]]),
        torch.tensor([[0, 3]], dtype=torch.long),
    ).values
    loss_value = prediction.square().mean()
    loss = PPOLoss(
        total_loss=loss_value,
        policy_loss=loss_value.detach(),
        value_loss=loss_value.detach(),
        entropy=torch.tensor(0.1),
        approximate_kl=torch.tensor(0.0),
        clip_fraction=torch.tensor(0.0),
        importance_ratios=torch.ones(1),
    )
    apply_ppo_update(model, optimizer, loss)
    expected_model = {
        name: value.detach().clone()
        for name, value in model.state_dict().items()
    }
    expected_optimizer = copy.deepcopy(optimizer.state_dict())

    env = _intermediate_gym_env()
    observation, reset_info = env.reset(
        seed=43,
        options={"game_id": "checkpoint-resume-game"},
    )
    assert observation["engine_observation"] is not None
    assert reset_info["objective_id"]
    assert reset_info["num_players"] == 3

    numpy_generator = np.random.default_rng(47)
    python_generator = random.Random(53)
    resume_state = PPOResumeState(
        run_id="checkpoint-run",
        update_index=1,
        episode_count=0,
        environment_steps=0,
        environment_state={"gym_env": env.state_dict()},
        rollout_state={"cursor": 0, "buffer": ()},
        recurrent_state={"memory": ()},
        curriculum_state={"stage_id": "valid-intermediate-states"},
        journal_state={"episode_index": 0},
    )
    artifact_store = ArtifactStore(tmp_path / "artifacts")
    references = save_ppo_checkpoint(
        artifact_store,
        model,
        optimizer,
        resume_state,
        numpy_generators={"training": numpy_generator},
        python_generators={"training": python_generator},
    )

    expected_random_values = (
        random.random(),
        float(np.random.random()),
        float(numpy_generator.random()),
        python_generator.random(),
        torch.rand(3),
    )
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.add_(2.0)
    random.seed(59)
    np.random.seed(61)
    torch.manual_seed(67)
    numpy_generator.random()
    python_generator.random()

    restored_state = restore_ppo_checkpoint(
        artifact_store,
        references,
        model,
        optimizer,
        numpy_generators={"training": numpy_generator},
        python_generators={"training": python_generator},
    )

    assert restored_state.run_id == resume_state.run_id
    assert restored_state.update_index == resume_state.update_index
    for name, expected in expected_model.items():
        assert torch.equal(model.state_dict()[name], expected)
    restored_optimizer = optimizer.state_dict()
    for parameter_id, expected_values in expected_optimizer["state"].items():
        actual_values = restored_optimizer["state"][parameter_id]
        for key, expected in expected_values.items():
            if isinstance(expected, torch.Tensor):
                assert torch.equal(actual_values[key], expected)
            else:
                assert actual_values[key] == expected

    assert random.random() == expected_random_values[0]
    assert float(np.random.random()) == expected_random_values[1]
    assert float(numpy_generator.random()) == expected_random_values[2]
    assert python_generator.random() == expected_random_values[3]
    assert torch.equal(torch.rand(3), expected_random_values[4])

    restored_env = _intermediate_gym_env()
    gym_state = restored_state.environment_state["gym_env"]
    assert isinstance(gym_state, dict)
    restored_env.load_state_dict(gym_state)
    original_step = env.step(0)
    restored_step = restored_env.step(0)
    assert original_step[1:] == restored_step[1:]
    assert env._aec_env._require_state() == restored_env._aec_env._require_state()
