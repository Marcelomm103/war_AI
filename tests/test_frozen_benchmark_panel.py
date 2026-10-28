"""Tests for Phase 8.6 frozen benchmark panel formation and diversity/strength gates."""

from __future__ import annotations

from dataclasses import replace

import pytest

from war_ai.rl.frozen_benchmark_panel import (
    MINIMUM_BENCHMARK_APPROACHES,
    MINIMUM_BENCHMARK_PANEL_POLICIES,
    NearDuplicateCheckpointError,
    WeakOrRandomPanelPolicyError,
    assemble_frozen_benchmark_panel,
    validate_and_create_frozen_panel,
)
from war_ai.rl.league_population import (
    build_default_league_population,
    build_specialist_or_exploiter_policy_spec,
)


def test_assemble_frozen_benchmark_panel_satisfies_all_phase8_criteria() -> None:
    population = build_default_league_population(seed=2026)
    panel = assemble_frozen_benchmark_panel(population)
    report = panel.validation_report

    assert report.passed is True
    assert report.distinct_policies_count >= MINIMUM_BENCHMARK_PANEL_POLICIES
    assert len(report.distinct_successful_approaches) >= MINIMUM_BENCHMARK_APPROACHES
    assert report.distinct_lineages_count == report.distinct_policies_count
    assert report.includes_leaders is True
    assert report.includes_independent_opponents is True
    assert report.includes_strong_exploiters is True
    assert report.includes_mission_specialists is True
    assert report.zero_random_or_weak_fillers is True
    assert report.zero_near_duplicate_checkpoints is True

    candidate = population.get_member("main_ppo_recurrent_gen2").policy_spec
    homog_configs = panel.build_homogeneous_scenario_configs(
        candidate,
        adversary_policy_id=panel.policy_ids[0],
        num_players=4,
        seeds=(2026, 2027, 2028, 2029),
    )
    assert len(homog_configs) == 4
    for cfg in homog_configs:
        opp_pids = [
            pid
            for pid in cfg.policy_ids_by_player.values()
            if pid != candidate.policy_id
        ]
        assert len(opp_pids) == 3
        assert set(opp_pids) == {panel.policy_ids[0]}

    mixed_configs = panel.build_mixed_scenario_configs(
        candidate,
        num_players=4,
        seeds=(2026, 2027),
    )
    assert len(mixed_configs) == 2
    for cfg in mixed_configs:
        opp_pids = [
            pid
            for pid in cfg.policy_ids_by_player.values()
            if pid != candidate.policy_id
        ]
        assert len(opp_pids) == 3
        assert len(set(opp_pids)) == 3


def test_frozen_panel_rejects_random_weak_and_near_duplicate_checkpoints() -> None:
    population = build_default_league_population(seed=2026)
    panel = assemble_frozen_benchmark_panel(population)
    base_members = list(panel.members[:8])

    # 1. Reject random_legal / random filler policy
    random_member = replace(
        base_members[0],
        member_id="random_legal_filler",
        lineage_id="lineage_random_filler",
    )
    with pytest.raises(WeakOrRandomPanelPolicyError, match="forbidden random/weak"):
        validate_and_create_frozen_panel(
            "bad_random_panel",
            [random_member, *base_members[1:]],
        )

    # 2. Reject generation 0 checkpoint and low validation score
    gen0_member = replace(
        base_members[0],
        member_id="early_gen0_snapshot",
        generation=0,
        lineage_id="lineage_gen0",
    )
    with pytest.raises(WeakOrRandomPanelPolicyError, match="min_generation"):
        validate_and_create_frozen_panel(
            "bad_gen0_panel",
            [gen0_member, *base_members[1:]],
        )

    low_score_member = replace(
        base_members[0],
        member_id="unproven_candidate",
        validation_score=0.50,
        lineage_id="lineage_low_score",
    )
    with pytest.raises(WeakOrRandomPanelPolicyError, match="validation_score"):
        validate_and_create_frozen_panel(
            "bad_low_score_panel",
            [low_score_member, *base_members[1:]],
        )

    # 3. Reject near-duplicate checkpoints (identical or near-identical parameter fingerprint)
    dup_spec, _ = build_specialist_or_exploiter_policy_spec(
        policy_id="near_dup_policy",
        primary_style="continent_control",
        secondary_style="exact_probability_planner",
        primary_weight=0.85,
        tactical_tier_bias=0.65,
        seed_salt=777,
    )
    near_dup_member = replace(
        base_members[0],
        member_id="near_duplicate_checkpoint_v2",
        policy_spec=dup_spec,
        lineage_id="lineage_distinct_for_dup_test",
        parameter_fingerprint=tuple(
            val + 1e-6 for val in base_members[1].parameter_fingerprint
        ),
    )
    with pytest.raises(NearDuplicateCheckpointError, match="near-duplicates"):
        validate_and_create_frozen_panel(
            "bad_near_dup_panel",
            [near_dup_member, *base_members[1:]],
        )
