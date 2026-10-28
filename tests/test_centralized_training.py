"""Tests for Phase 7.4 centralized training and production artifact isolation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from war_ai.envs.observations import MAX_PLAYERS
from war_ai.envs.war1_aec import War1AECEnv
from war_ai.rl.architectures import NUM_CARDS, NUM_OBJECTIVES, ArchitectureConfig
from war_ai.rl.centralized_training import (
    CentralizedTrainingHarness,
    PrivilegedTableSecrets,
    compare_local_vs_privileged_critic,
    export_production_policy_artifact,
    extract_privileged_table_secrets,
    verify_counterfactual_non_interference,
    verify_production_artifact_isolation,
)


def _setup_env_and_secrets() -> tuple[War1AECEnv, str, PrivilegedTableSecrets]:
    env = War1AECEnv()
    env.reset(seed=2026)
    assert env._game_state is not None
    active_player = env.agent_selection
    assert active_player is not None
    secrets = extract_privileged_table_secrets(
        env._game_state,
        focal_player_id=active_player,
    )
    return env, active_player, secrets


def test_extract_privileged_secrets_masks_out_focal_player() -> None:
    env, active_player, secrets = _setup_env_and_secrets()
    assert env._game_state is not None
    focal_idx = next(
        idx
        for idx, p in enumerate(env._game_state.players)
        if p.public.player_id == active_player
    )
    for seat in range(3):
        if seat == focal_idx:
            assert not bool(secrets.opponent_seat_mask[0, seat].item())
        else:
            assert bool(secrets.opponent_seat_mask[0, seat].item())


def test_counterfactual_non_interference_guarantees_actor_independence() -> None:
    env, active_player, secrets_a = _setup_env_and_secrets()
    observation = env.observe(active_player)

    alt_objectives = (secrets_a.opponent_objective_indices + 7) % NUM_OBJECTIVES
    alt_cards = torch.ones_like(secrets_a.opponent_card_targets) - (
        secrets_a.opponent_card_targets
    )
    secrets_b = PrivilegedTableSecrets(
        opponent_objective_indices=alt_objectives,
        opponent_card_targets=alt_cards,
        opponent_seat_mask=secrets_a.opponent_seat_mask,
    )

    torch.manual_seed(42)
    harness = CentralizedTrainingHarness(
        ArchitectureConfig(hidden_size=32, use_belief_head=True),
        critic_mode="privileged",
    )
    harness.eval()

    report = verify_counterfactual_non_interference(
        harness,
        observation,
        secrets_a=secrets_a,
        secrets_b=secrets_b,
    )
    assert report.actor_logits_identical
    assert report.local_values_identical
    assert report.selected_actions_identical
    assert report.max_actor_logit_abs_diff == 0.0
    assert report.max_local_value_abs_diff == 0.0
    assert report.privileged_critic_value_abs_diff > 0.0


def test_production_artifact_strips_privileged_critic_and_executes_in_arena() -> None:
    env, active_player, _ = _setup_env_and_secrets()
    observation = env.observe(active_player)

    torch.manual_seed(99)
    harness = CentralizedTrainingHarness(
        ArchitectureConfig(hidden_size=32, use_belief_head=True),
        critic_mode="privileged",
    )
    artifact = export_production_policy_artifact(
        harness,
        policy_id="ctde_prod_v1",
    )

    assert artifact.checkpoint_sha256.startswith("sha256:")
    assert any(
        key.startswith("privileged_critic.")
        for key in artifact.excluded_training_parameter_keys
    )
    policy_fn = artifact.policy_spec.factory(2026)
    chosen_action = policy_fn(observation, np.random.default_rng(2026))
    assert 0 <= chosen_action < int(observation["action_mask"].shape[0])

    with pytest.raises(ValueError, match="privileged training parameters"):
        verify_production_artifact_isolation(
            {"privileged_critic.value_network.0.weight": torch.zeros((1,))}
        )


def test_compare_local_vs_privileged_critic_runs_both_modes() -> None:
    env, active_player, secrets = _setup_env_and_secrets()
    observation = env.observe(active_player)
    losses = compare_local_vs_privileged_critic(
        observation,
        secrets,
        updates=6,
        seed=2026,
    )
    assert set(losses.keys()) == {"local", "privileged"}
    assert losses["local"] >= 0.0
    assert losses["privileged"] >= 0.0
    assert MAX_PLAYERS == 6
    assert NUM_CARDS == 44
