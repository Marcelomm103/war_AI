"""Tests for adapted ISMCTS/POMCP search and anti-strategy-fusion (Phase 9.2 & 9.3)."""

from __future__ import annotations

import pytest
from war_game_engine.state import observe_player, to_public_game_state

from war_ai.envs.war1_aec import War1AECEnv
from war_ai.rl.architectures import ArchitectureConfig, WarArchitectureModel
from war_ai.rl.belief_models import SimpleRulePriorBeliefEstimator
from war_ai.rl.imperfect_info_search import (
    ISMCTSPOMCPPlanner,
    ISMCTSSearchConfig,
    PrivilegedPlanningStateError,
    audit_information_safety_and_strategy_fusion,
    sample_compatible_determinization,
)


def test_sample_compatible_determinization_preserves_public_and_own_private_state() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=920)
    obs = env.observe("player_0")["engine_observation"]
    assert obs is not None

    belief = SimpleRulePriorBeliefEstimator().estimate(obs)
    det_state = sample_compatible_determinization(
        obs,
        belief,
        determinization_seed=1234,
    )

    assert to_public_game_state(det_state) == obs.public_state
    focal_in_det = observe_player(det_state, "player_0")
    assert focal_in_det.private_state == obs.private_state
    # Opponents have distinct objectives from focal player
    assigned_objectives = [
        p.private.objective_id for p in det_state.players if p.private.objective_id
    ]
    assert len(assigned_objectives) == len(set(assigned_objectives))


def test_planner_rejects_privileged_full_game_state() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=921)
    full_state = env._require_state()
    planner = ISMCTSPOMCPPlanner()

    with pytest.raises(PrivilegedPlanningStateError):
        planner.plan(full_state)  # type: ignore[arg-type]


def test_ismcts_and_pomcp_plan_with_neural_policy_value_and_chance_nodes() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=922)
    # Advance until an attack phase where combat chance nodes are exercised
    for _ in range(50):
        agent = env.agent_selection
        assert agent is not None
        obs = env.observe(agent)["engine_observation"]
        assert obs is not None
        if obs.public_state.phase_id == "attack" and len(obs.legal_actions) > 1:
            break
        env.step(0)

    agent = env.agent_selection
    assert agent is not None
    obs = env.observe(agent)["engine_observation"]
    assert obs is not None

    model = WarArchitectureModel(
        ArchitectureConfig(
            encoder="mlp",
            memory="none",
            hidden_size=32,
            use_belief_head=True,
        )
    )
    planner = ISMCTSPOMCPPlanner(
        ISMCTSSearchConfig(
            algorithm="ismcts",
            num_simulations=20,
            num_determinizations=4,
            max_search_depth=3,
            rollout_depth=2,
            seed=922,
        ),
        model=model,
    )
    decision = planner.plan(obs)
    assert decision.selected_action in obs.legal_actions
    assert decision.selected_action == obs.legal_actions[decision.selected_action_index]
    assert not decision.used_privileged_state
    assert decision.strategy_fusion_free
    assert decision.opponent_observations_isolated
    assert decision.num_simulations_run == 20
    assert decision.num_determinizations_sampled == 4
    assert decision.num_information_set_nodes >= 1
    assert decision.num_chance_nodes >= 1
    assert decision.chance_outcomes_sampled >= 1


def test_information_safety_and_anti_strategy_fusion_audit() -> None:
    audit = audit_information_safety_and_strategy_fusion(seed=930)
    assert audit.privileged_state_rejected
    assert audit.focal_policy_blind_to_sampled_secret
    assert audit.opponent_receives_only_own_observation
    assert audit.root_statistics_aggregated_across_worlds
    assert audit.ismcts_outperforms_or_matches_naive_determinization
    assert audit.ismcts_expected_score > audit.naive_determinization_expected_score
    assert audit.zero_information_leakage


def test_pomcp_and_naive_determinization_modes_on_live_observation() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=933)
    active_agent = env.agent_selection
    assert active_agent is not None
    obs = env.observe(active_agent)["engine_observation"]
    assert obs is not None

    pomcp_planner = ISMCTSPOMCPPlanner(
        ISMCTSSearchConfig(
            algorithm="pomcp",
            num_simulations=12,
            num_determinizations=3,
            max_search_depth=2,
            rollout_depth=1,
            seed=933,
        )
    )
    naive_planner = ISMCTSPOMCPPlanner(
        ISMCTSSearchConfig(
            algorithm="naive_determinization",
            num_simulations=12,
            num_determinizations=3,
            max_search_depth=2,
            rollout_depth=1,
            seed=933,
        )
    )
    pomcp_res = pomcp_planner.plan(obs)
    naive_res = naive_planner.plan(obs)

    assert pomcp_res.selected_action in obs.legal_actions
    assert pomcp_res.strategy_fusion_free
    assert naive_res.selected_action in obs.legal_actions
    assert not naive_res.strategy_fusion_free

