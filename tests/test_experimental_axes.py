"""Tests for Phase 7.1 experimental axes separation and compute-budget parity."""

from __future__ import annotations

from dataclasses import replace

import pytest

from war_ai.rl.experimental_axes import (
    ALLOWED_AXIS_VALUES,
    EXPERIMENTAL_AXES,
    ComputeBudget,
    ExperimentalAxisSpec,
    SingleAxisComparisonPlan,
    build_orthogonal_axis_sweep,
    validate_single_axis_comparison,
)


def test_experimental_axes_cover_all_eight_roadmap_dimensions() -> None:
    assert EXPERIMENTAL_AXES == (
        "algorithm",
        "encoder",
        "memory",
        "action_representation",
        "curriculum",
        "shaping",
        "league",
        "search",
    )
    spec = ExperimentalAxisSpec(spec_id="reference_v1")
    assert tuple(spec.axis_values().keys()) == EXPERIMENTAL_AXES


def test_single_axis_comparison_accepts_isolated_axis_change() -> None:
    reference = ExperimentalAxisSpec(spec_id="ref_mlp")
    candidate = reference.with_axis(
        "encoder",
        "graph_transformer",
        spec_id="cand_graph_transformer",
    )

    changed = validate_single_axis_comparison(
        reference,
        candidate,
        expected_axis="encoder",
    )
    assert changed == "encoder"

    plan = SingleAxisComparisonPlan(
        axis="encoder",
        reference=reference,
        candidate=candidate,
    )
    assert plan.reference_value == "mlp"
    assert plan.candidate_value == "graph_transformer"


def test_single_axis_comparison_rejects_confounded_multi_axis_changes() -> None:
    reference = ExperimentalAxisSpec(spec_id="ref_mlp")
    confounded = replace(
        reference,
        spec_id="confounded_transformer_and_search",
        encoder="graph_transformer",
        search="one_step_expectimax",
    )

    with pytest.raises(ValueError, match="Confounded comparison"):
        validate_single_axis_comparison(reference, confounded)


def test_single_axis_comparison_rejects_mismatched_compute_budget() -> None:
    reference = ExperimentalAxisSpec(
        spec_id="ref_mlp",
        compute_budget=ComputeBudget(
            semantic_decisions=2_000,
            learner_updates=30,
            inference_budget_ms=250,
        ),
    )
    extra_compute_candidate = replace(
        reference,
        spec_id="cand_transformer_more_compute",
        encoder="graph_transformer",
        compute_budget=ComputeBudget(
            semantic_decisions=8_000,
            learner_updates=120,
            inference_budget_ms=250,
        ),
    )

    with pytest.raises(ValueError, match="identical compute_budget"):
        validate_single_axis_comparison(reference, extra_compute_candidate)


def test_orthogonal_axis_sweep_covers_every_axis_independently() -> None:
    reference = ExperimentalAxisSpec(spec_id="baseline_spec")
    plans = build_orthogonal_axis_sweep(reference)

    covered_axes = {plan.axis for plan in plans}
    assert covered_axes == set(EXPERIMENTAL_AXES)
    expected_count = sum(len(values) - 1 for values in ALLOWED_AXIS_VALUES.values())
    assert len(plans) == expected_count
    for plan in plans:
        assert plan.reference.compute_budget == plan.candidate.compute_budget
        assert plan.reference.differing_axes(plan.candidate) == (plan.axis,)
