"""Tests for Phase 8.1 league population management."""

from __future__ import annotations

from dataclasses import replace

import pytest

from war_ai.arena import ArenaMatchConfig, run_arena
from war_ai.rl.league_population import (
    POPULATION_ROLES,
    LeaguePopulation,
    LeaguePopulationMember,
    build_default_league_population,
    build_specialist_or_exploiter_policy_spec,
)


def test_default_league_population_covers_all_six_roles_and_isolates_lineages() -> None:
    population = build_default_league_population(seed=2026)
    report = population.validate_completeness()

    assert report.is_complete is True
    assert report.missing_roles == ()
    assert report.independent_lineages_disjoint is True
    for role in POPULATION_ROLES:
        assert report.counts_by_role[role] >= 2
        assert len(population.members_by_role(role)) == report.counts_by_role[role]

    trainable = population.trainable_opponent_pool()
    held_out = population.held_out_generalization_pool()
    assert len(trainable) + len(held_out) == report.total_members
    assert all(not m.held_out_from_training for m in trainable)
    assert all(m.held_out_from_training for m in held_out)
    assert {m.lineage_id for m in trainable}.isdisjoint(
        {m.lineage_id for m in held_out}
    )


def test_promote_main_agent_and_upsert_architecture_seed_leader() -> None:
    population = build_default_league_population(seed=2026)
    initial_hist_count = len(population.historical_snapshots)

    spec_gen3, fp_gen3 = build_specialist_or_exploiter_policy_spec(
        policy_id="league_main_ppo_recurrent_gen3",
        primary_style="exact_probability_planner",
        secondary_style="objective_oriented",
        primary_weight=0.9,
        tactical_tier_bias=0.95,
        seed_salt=999,
    )
    promoted_main = LeaguePopulationMember(
        member_id="main_ppo_recurrent_gen3",
        role="main_agent",
        policy_spec=spec_gen3,
        architecture_id="gru_recurrent",
        algorithm_branch="ppo_recurrent",
        seed=2026,
        generation=3,
        lineage_id="lineage_main_ppo_recurrent",
        style_tag="recurrent_tactical_main_gen3",
        validation_score=0.999,
        parameter_fingerprint=fp_gen3,
    )
    archived = population.promote_main_agent("main_ppo_recurrent_gen2", promoted_main)
    assert archived.role == "historical_snapshot"
    assert len(population.historical_snapshots) == initial_hist_count + 1
    assert population.get_member("main_ppo_recurrent_gen3").role == "main_agent"

    # Upsert leader for (mlp_gru, 2036): lower score rejected, higher score accepted
    incumbent_leader = population.architecture_seed_leaders[("mlp_gru", 2036)]
    low_spec, low_fp = build_specialist_or_exploiter_policy_spec(
        policy_id="leader_gru_low",
        primary_style="expansionist",
        secondary_style="objective_oriented",
        primary_weight=0.5,
        tactical_tier_bias=0.2,
        seed_salt=701,
    )
    low_candidate = LeaguePopulationMember(
        member_id="leader_gru_seed2026_low",
        role="architecture_seed_leader",
        policy_spec=low_spec,
        architecture_id="mlp_gru",
        algorithm_branch="ppo_recurrent",
        seed=2036,
        generation=3,
        lineage_id="lineage_leader_gru",
        style_tag="weak_candidate",
        validation_score=0.10,
        parameter_fingerprint=low_fp,
    )
    assert population.upsert_architecture_seed_leader(low_candidate) is False
    assert (
        population.architecture_seed_leaders[("mlp_gru", 2036)].member_id
        == incumbent_leader.member_id
    )

    high_spec, high_fp = build_specialist_or_exploiter_policy_spec(
        policy_id="leader_gru_high",
        primary_style="exact_probability_planner",
        secondary_style="continent_control",
        primary_weight=0.95,
        tactical_tier_bias=0.90,
        seed_salt=702,
    )
    high_candidate = LeaguePopulationMember(
        member_id="leader_gru_seed2026_high",
        role="architecture_seed_leader",
        policy_spec=high_spec,
        architecture_id="mlp_gru",
        algorithm_branch="ppo_recurrent",
        seed=2036,
        generation=4,
        lineage_id="lineage_leader_gru",
        style_tag="strong_candidate",
        validation_score=1.0,
        parameter_fingerprint=high_fp,
    )
    assert population.upsert_architecture_seed_leader(high_candidate) is True
    assert (
        population.architecture_seed_leaders[("mlp_gru", 2036)].member_id
        == "leader_gru_seed2026_high"
    )
    assert population.get_member(incumbent_leader.member_id).role == "historical_snapshot"


def test_population_validation_guards_and_arena_compatibility() -> None:
    population = LeaguePopulation()
    spec, fp = build_specialist_or_exploiter_policy_spec(
        policy_id="spec_valid_main",
        primary_style="objective_oriented",
        secondary_style="continent_control",
        primary_weight=0.8,
        tactical_tier_bias=0.6,
        seed_salt=1,
    )
    main_member = LeaguePopulationMember(
        member_id="main_1",
        role="main_agent",
        policy_spec=spec,
        architecture_id="gru_recurrent",
        algorithm_branch="ppo_recurrent",
        seed=1,
        generation=1,
        lineage_id="lineage_a",
        style_tag="main_style",
        validation_score=0.9,
        parameter_fingerprint=fp,
    )
    population.register_member(main_member)

    # Duplicate checkpoint digest rejected
    with pytest.raises(ValueError, match="Duplicate checkpoint_sha256"):
        population.register_member(
            replace(
                main_member,
                member_id="main_2",
                policy_spec=replace(spec, policy_id="spec_dup"),
            )
        )

    # Independent generalization model sharing training lineage rejected
    held_spec, held_fp = build_specialist_or_exploiter_policy_spec(
        policy_id="spec_heldout_leak",
        primary_style="exact_probability_planner",
        secondary_style="objective_oriented",
        primary_weight=0.8,
        tactical_tier_bias=0.7,
        seed_salt=2,
    )
    with pytest.raises(ValueError, match="cannot share lineage"):
        population.register_member(
            LeaguePopulationMember(
                member_id="heldout_bad",
                role="independent_generalization_model",
                policy_spec=held_spec,
                architecture_id="gru_recurrent",
                algorithm_branch="ppo_recurrent",
                seed=2,
                generation=1,
                lineage_id="lineage_a",
                style_tag="heldout_style",
                held_out_from_training=True,
                validation_score=0.8,
                parameter_fingerprint=held_fp,
            )
        )

    # Full-game arena execution with policies from the default population
    full_pop = build_default_league_population(seed=2026)
    registry = full_pop.policy_registry()
    config = ArenaMatchConfig(
        match_id="pop-smoke-3p",
        seed=2026,
        num_players=3,
        policy_ids_by_player={
            "player_0": "league_main_ppo_recurrent_gen2",
            "player_1": "league_spec_continent_control",
            "player_2": "league_exploiter_anti_main_ppo",
        },
        max_semantic_decisions=5000,
    )
    arena_report = run_arena((config,), registry)
    assert arena_report.matches[0].status == "completed"
