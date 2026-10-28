"""Tests for Phase 7.7 rigorous comparison, 20-seed finalist gate, ablations, negative results, and scaling."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from war_ai.rl.experimental_axes import ComputeBudget, ExperimentalAxisSpec
from war_ai.rl.rigorous_comparison import (
    DEFAULT_FINALIST_SEEDS,
    FINALIST_MIN_SEEDS,
    NegativeResultRecord,
    assess_finalist_selection,
    collect_sample_compute_time_curve,
    evaluate_finalist_branch,
    run_single_axis_ablations,
    run_unbounded_scaling_study,
)
from war_ai.rl.seed_evaluation import SeedComparison


def test_finalist_selection_enforces_at_least_20_seeds() -> None:
    assert FINALIST_MIN_SEEDS == 20
    assert len(DEFAULT_FINALIST_SEEDS) == 20

    with pytest.raises(ValueError, match="at least 20 seeds"):
        assess_finalist_selection((), minimum_seed_count=10)

    ten_seed_comparisons = tuple(
        SeedComparison(
            seed=2026 + idx,
            num_players=3,
            candidate_policy_id="cand_v1",
            baseline_policy_id="random_legal",
            candidate_wins=3,
            candidate_appearances=3,
            baseline_wins=0,
            baseline_appearances=6,
            candidate_mission_ids=("W1-OBJ-01", "W1-OBJ-02", "W1-OBJ-03"),
        )
        for idx in range(10)
    )
    decision_10 = assess_finalist_selection(ten_seed_comparisons)
    assert decision_10.promote is False
    assert any(
        "requires at least 20 paired seeds" in reason
        for reason in decision_10.reasons
    )

    twenty_seed_comparisons = tuple(
        SeedComparison(
            seed=2026 + idx,
            num_players=3,
            candidate_policy_id="cand_v1",
            baseline_policy_id="random_legal",
            candidate_wins=3,
            candidate_appearances=3,
            baseline_wins=0,
            baseline_appearances=6,
            candidate_mission_ids=("W1-OBJ-01", "W1-OBJ-02", "W1-OBJ-03"),
        )
        for idx in range(20)
    )
    decision_20 = assess_finalist_selection(twenty_seed_comparisons)
    assert decision_20.promote is True
    assert decision_20.seed_count == 20


def test_sample_compute_time_curve_and_unbounded_scaling_study() -> None:
    budget = ComputeBudget(
        semantic_decisions=48,
        learner_updates=6,
        inference_budget_ms=250,
    )
    curve, _ = collect_sample_compute_time_curve(
        "graph_actor_critic",
        compute_budget=budget,
        hidden_size=32,
        seed=2026,
    )
    assert len(curve.points) == 6
    assert curve.points[-1].environment_samples == 48
    assert curve.points[-1].compute_units > curve.points[0].compute_units
    assert curve.points[-1].optimal_intent_probability > (
        curve.points[0].optimal_intent_probability
    )

    scaling = run_unbounded_scaling_study(
        "graph_actor_critic",
        hidden_sizes=(16, 32, 64),
        updates_per_scale=(2, 4, 6),
        seed=2026,
    )
    assert len(scaling.points) == 3
    assert (
        scaling.points[0].parameter_count
        < scaling.points[1].parameter_count
        < scaling.points[2].parameter_count
    )
    assert (
        scaling.points[0].compute_units
        < scaling.points[1].compute_units
        < scaling.points[2].compute_units
    )


def test_single_axis_ablations_and_negative_result_registry() -> None:
    ref_spec = ExperimentalAxisSpec(
        spec_id="ref_graph_search",
        algorithm="policy_value_search",
        encoder="graph_transformer",
        memory="causal_event_transformer",
        action_representation="phase_conditioned_tokens",
        curriculum="three_stage_progressive",
        shaping="potential_bounded",
        league="historical_self_play",
        search="one_step_expectimax",
        compute_budget=ComputeBudget(
            semantic_decisions=32,
            learner_updates=6,
            inference_budget_ms=250,
        ),
    )
    ablations, registry = run_single_axis_ablations(ref_spec, seed=2026)
    assert len(ablations) == 4
    assert len(registry.records) >= 1

    with pytest.raises(ValueError, match="already registered"):
        registry.register(
            NegativeResultRecord(
                record_id=registry.records[0].record_id,
                candidate_id="dup",
                reference_id="ref",
                axis="encoder",
                metric_name="optimal_intent_probability",
                candidate_metric=0.1,
                reference_metric=0.9,
                metric_delta=-0.8,
                rejection_reason="Duplicate test",
            )
        )


def test_evaluate_finalist_branch_rejects_fewer_than_20_seeds() -> None:
    with pytest.raises(ValueError, match="at least 20 seeds"):
        evaluate_finalist_branch(
            "ppo_feedforward",
            finalist_seeds=tuple(range(2026, 2036)),
        )


def test_persisted_phase_7_7_benchmark_artifact_is_complete_and_promoted() -> None:
    benchmark_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "phase-7.7-comparison-2026-10-28.json"
    )
    assert benchmark_path.is_file()
    payload = json.loads(benchmark_path.read_text(encoding="utf-8"))

    assert len(payload["catalog"]) == 6
    assert len(payload["selected_strong_branches"]) == 2
    assert len(payload["ablations"]) == 4
    assert len(payload["negative_results"]) >= 4
    assert len(payload["scaling_study"]["points"]) >= 3
    assert len(payload["finalist_reports"]) == 2
    for finalist in payload["finalist_reports"]:
        assert len(finalist["finalist_comparisons"]) == 20
        assert finalist["decision"]["seed_count"] == 20
        assert finalist["decision"]["promote"] is True
        assert finalist["decision"]["sign_test_p_value"] < 0.05

