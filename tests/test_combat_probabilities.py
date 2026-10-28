"""Tests for exact combat probability distributions and DP campaigns (Phase 9.4)."""

from __future__ import annotations

from fractions import Fraction

import pytest
from war_game_engine.contracts import AttackOperation

from war_ai.envs.war1_aec import War1AECEnv
from war_ai.rl.combat_probabilities import (
    CERTIFIED_COMBAT_CATALOG,
    CampaignMacroSpec,
    UnsafeMacroSubstitutionError,
    compare_aggregated_dp_vs_round_by_round,
    evaluate_attack_action_with_exact_combat_distribution,
    evaluate_territory_campaign_dp,
    get_certified_combat_branches,
    validate_campaign_macro_equivalence,
)
from war_ai.rl.imperfect_info_search import evaluate_state_values_by_player


def test_certified_combat_catalog_probabilities_sum_to_one_exactly() -> None:
    assert len(CERTIFIED_COMBAT_CATALOG) == 9
    for a_dice in (1, 2, 3):
        for d_dice in (1, 2, 3):
            branches = get_certified_combat_branches(a_dice, d_dice)
            exact_sum = sum(
                (br.exact_probability for br in branches),
                start=Fraction(0, 1),
            )
            assert exact_sum == Fraction(1, 1)
            for br in branches:
                assert br.attacker_losses + br.defender_losses == min(a_dice, d_dice)


def test_unsafe_combat_macros_ignoring_stop_occupation_or_victory_are_rejected() -> None:
    with pytest.raises(UnsafeMacroSubstitutionError):
        validate_campaign_macro_equivalence(
            CampaignMacroSpec(allow_optional_stop=False)
        )
    with pytest.raises(UnsafeMacroSubstitutionError):
        validate_campaign_macro_equivalence(
            CampaignMacroSpec(model_mandatory_occupation=False)
        )
    with pytest.raises(UnsafeMacroSubstitutionError):
        validate_campaign_macro_equivalence(
            CampaignMacroSpec(check_intermediate_victory=False)
        )
    with pytest.raises(UnsafeMacroSubstitutionError):
        evaluate_territory_campaign_dp(
            4,
            2,
            macro_spec=CampaignMacroSpec(model_mandatory_occupation=False),
        )


def test_dp_campaign_matches_round_by_round_tree_and_exposes_naive_macro_bias() -> None:
    comp = compare_aggregated_dp_vs_round_by_round(
        initial_attacker_armies=5,
        initial_defender_armies=3,
        requires_two_armies_per_territory=True,
        two_army_conquest_triggers_victory=True,
    )
    assert comp.semantically_equivalent
    assert comp.max_abs_value_difference < 1e-12
    assert abs(comp.dp_conquest_probability - comp.round_by_round_conquest_probability) < 1e-12
    assert comp.dp_optimal_choice == comp.round_by_round_optimal_choice
    assert comp.dp_states_evaluated < comp.round_by_round_tree_leaves
    assert comp.naive_unsafe_macro_value_bias > 0.05


def test_exact_combat_distribution_evaluates_live_attack_action() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=940)
    for _ in range(100):
        agent = env.agent_selection
        assert agent is not None
        obs = env.observe(agent)["engine_observation"]
        assert obs is not None
        if obs.public_state.phase_id == "attack" and any(
            isinstance(act.operation, AttackOperation) for act in obs.legal_actions
        ):
            break
        env.step(0)

    state = env._require_state()
    agent = env.agent_selection
    assert agent is not None
    obs = env.observe(agent)["engine_observation"]
    assert obs is not None
    attack_action = next(
        act for act in obs.legal_actions if isinstance(act.operation, AttackOperation)
    )

    ev = evaluate_attack_action_with_exact_combat_distribution(
        state,
        attack_action,
        lambda s: evaluate_state_values_by_player(s)[agent],
    )
    assert -1.0 <= ev <= 1.0
