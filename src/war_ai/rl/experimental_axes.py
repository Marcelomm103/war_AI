"""Orthogonal experimental axes and compute-parity guards for Phase 7 studies."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias, cast

ExperimentalAxisName: TypeAlias = Literal[
    "algorithm",
    "encoder",
    "memory",
    "action_representation",
    "curriculum",
    "shaping",
    "league",
    "search",
]

AlgorithmAxis: TypeAlias = Literal[
    "ppo_feedforward",
    "ppo_recurrent",
    "graph_actor_critic",
    "impala_vtrace",
    "recurrent_off_policy_r2d2",
    "policy_value_search",
]
EncoderAxis: TypeAlias = Literal[
    "mlp",
    "message_passing_gnn",
    "graph_transformer",
]
MemoryAxis: TypeAlias = Literal[
    "none",
    "gru",
    "lstm",
    "causal_event_transformer",
]
ActionRepresentationAxis: TypeAlias = Literal[
    "tactical_intent_tokens",
    "structured_action_tokens",
    "phase_conditioned_tokens",
]
CurriculumAxis: TypeAlias = Literal[
    "none",
    "three_stage_progressive",
    "full_game_only",
]
ShapingAxis: TypeAlias = Literal[
    "none",
    "potential_bounded",
]
LeagueAxis: TypeAlias = Literal[
    "fixed_baselines",
    "historical_self_play",
    "mixed_league",
]
SearchAxis: TypeAlias = Literal[
    "none",
    "one_step_expectimax",
    "rollout_value_search",
]

EXPERIMENTAL_AXES: tuple[ExperimentalAxisName, ...] = (
    "algorithm",
    "encoder",
    "memory",
    "action_representation",
    "curriculum",
    "shaping",
    "league",
    "search",
)

ALLOWED_AXIS_VALUES: dict[ExperimentalAxisName, tuple[str, ...]] = {
    "algorithm": (
        "ppo_feedforward",
        "ppo_recurrent",
        "graph_actor_critic",
        "impala_vtrace",
        "recurrent_off_policy_r2d2",
        "policy_value_search",
    ),
    "encoder": (
        "mlp",
        "message_passing_gnn",
        "graph_transformer",
    ),
    "memory": (
        "none",
        "gru",
        "lstm",
        "causal_event_transformer",
    ),
    "action_representation": (
        "tactical_intent_tokens",
        "structured_action_tokens",
        "phase_conditioned_tokens",
    ),
    "curriculum": (
        "none",
        "three_stage_progressive",
        "full_game_only",
    ),
    "shaping": (
        "none",
        "potential_bounded",
    ),
    "league": (
        "fixed_baselines",
        "historical_self_play",
        "mixed_league",
    ),
    "search": (
        "none",
        "one_step_expectimax",
        "rollout_value_search",
    ),
}


@dataclass(frozen=True)
class ComputeBudget:
    """Controlled compute budget shared across independent axis comparisons."""

    semantic_decisions: int = 2_000
    learner_updates: int = 30
    inference_budget_ms: int = 250

    def __post_init__(self) -> None:
        for label, value in (
            ("semantic_decisions", self.semantic_decisions),
            ("learner_updates", self.learner_updates),
            ("inference_budget_ms", self.inference_budget_ms),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{label} must be a positive integer.")


@dataclass(frozen=True)
class ExperimentalAxisSpec:
    """Complete assignment of all eight experimental axes plus compute budget."""

    spec_id: str
    algorithm: AlgorithmAxis = "ppo_feedforward"
    encoder: EncoderAxis = "mlp"
    memory: MemoryAxis = "none"
    action_representation: ActionRepresentationAxis = "tactical_intent_tokens"
    curriculum: CurriculumAxis = "three_stage_progressive"
    shaping: ShapingAxis = "potential_bounded"
    league: LeagueAxis = "historical_self_play"
    search: SearchAxis = "none"
    compute_budget: ComputeBudget = ComputeBudget()

    def __post_init__(self) -> None:
        if not isinstance(self.spec_id, str) or not self.spec_id.strip():
            raise ValueError("spec_id must be a non-empty string.")
        if not isinstance(self.compute_budget, ComputeBudget):
            raise TypeError("compute_budget must be a ComputeBudget instance.")
        for axis in EXPERIMENTAL_AXES:
            value = getattr(self, axis)
            allowed = ALLOWED_AXIS_VALUES[axis]
            if value not in allowed:
                raise ValueError(
                    f"Invalid value {value!r} for axis {axis!r}; "
                    f"expected one of {allowed}."
                )

    def axis_values(self) -> dict[ExperimentalAxisName, str]:
        """Return the eight experimental axis assignments in canonical order."""
        return {axis: str(getattr(self, axis)) for axis in EXPERIMENTAL_AXES}

    def differing_axes(
        self,
        other: ExperimentalAxisSpec,
    ) -> tuple[ExperimentalAxisName, ...]:
        """List experimental axes whose values differ from ``other``."""
        return tuple(
            axis
            for axis in EXPERIMENTAL_AXES
            if getattr(self, axis) != getattr(other, axis)
        )

    def with_axis(
        self,
        axis: ExperimentalAxisName,
        value: str,
        *,
        spec_id: str | None = None,
    ) -> ExperimentalAxisSpec:
        """Return a copy with only ``axis`` changed under the same compute budget."""
        if axis not in ALLOWED_AXIS_VALUES:
            raise ValueError(f"Unknown experimental axis {axis!r}.")
        if value not in ALLOWED_AXIS_VALUES[axis]:
            raise ValueError(
                f"Invalid value {value!r} for axis {axis!r}; "
                f"expected one of {ALLOWED_AXIS_VALUES[axis]}."
            )
        resolved_id = spec_id or f"{self.spec_id}__{axis}_{value}"
        updated = self.axis_values()
        updated[axis] = value
        return ExperimentalAxisSpec(
            spec_id=resolved_id,
            algorithm=cast(AlgorithmAxis, updated["algorithm"]),
            encoder=cast(EncoderAxis, updated["encoder"]),
            memory=cast(MemoryAxis, updated["memory"]),
            action_representation=cast(
                ActionRepresentationAxis, updated["action_representation"]
            ),
            curriculum=cast(CurriculumAxis, updated["curriculum"]),
            shaping=cast(ShapingAxis, updated["shaping"]),
            league=cast(LeagueAxis, updated["league"]),
            search=cast(SearchAxis, updated["search"]),
            compute_budget=self.compute_budget,
        )


@dataclass(frozen=True)
class SingleAxisComparisonPlan:
    """Validated comparison isolating a single experimental axis."""

    axis: ExperimentalAxisName
    reference: ExperimentalAxisSpec
    candidate: ExperimentalAxisSpec

    def __post_init__(self) -> None:
        differing_axis = validate_single_axis_comparison(
            self.reference,
            self.candidate,
            expected_axis=self.axis,
        )
        if differing_axis != self.axis:
            raise ValueError(
                f"Comparison expected axis {self.axis!r} but changed {differing_axis!r}."
            )

    @property
    def reference_value(self) -> str:
        return str(getattr(self.reference, self.axis))

    @property
    def candidate_value(self) -> str:
        return str(getattr(self.candidate, self.axis))


def validate_single_axis_comparison(
    reference: ExperimentalAxisSpec,
    candidate: ExperimentalAxisSpec,
    *,
    expected_axis: ExperimentalAxisName | None = None,
) -> ExperimentalAxisName:
    """Verify that two specs isolate one axis under identical compute budgets."""
    if reference.spec_id == candidate.spec_id:
        raise ValueError("Reference and candidate specs must have distinct IDs.")
    if reference.compute_budget != candidate.compute_budget:
        raise ValueError(
            "Single-axis comparisons require identical compute_budget so gains "
            "from extra compute are not misattributed to an architectural or "
            "algorithmic axis."
        )
    differing = reference.differing_axes(candidate)
    if not differing:
        raise ValueError(
            "Reference and candidate specs are identical across all experimental axes."
        )
    if len(differing) > 1:
        raise ValueError(
            f"Confounded comparison changes multiple axes simultaneously: {differing}. "
            "Vary one axis at a time."
        )
    changed_axis = differing[0]
    if expected_axis is not None and changed_axis != expected_axis:
        raise ValueError(
            f"Expected comparison on axis {expected_axis!r}, "
            f"but {changed_axis!r} changed."
        )
    return changed_axis


def build_orthogonal_axis_sweep(
    reference: ExperimentalAxisSpec,
    alternatives_by_axis: Mapping[ExperimentalAxisName, Sequence[str]] | None = None,
) -> tuple[SingleAxisComparisonPlan, ...]:
    """Generate single-axis comparisons across all eight experimental axes."""
    source = (
        ALLOWED_AXIS_VALUES
        if alternatives_by_axis is None
        else alternatives_by_axis
    )
    unknown = set(source) - set(EXPERIMENTAL_AXES)
    if unknown:
        raise ValueError(f"Unknown experimental axes in sweep: {sorted(unknown)}.")

    plans: list[SingleAxisComparisonPlan] = []
    for axis in EXPERIMENTAL_AXES:
        if axis not in source:
            continue
        reference_value = getattr(reference, axis)
        seen_values: set[str] = set()
        for candidate_value in source[axis]:
            if candidate_value in seen_values:
                raise ValueError(
                    f"Duplicate candidate value {candidate_value!r} on axis {axis!r}."
                )
            seen_values.add(candidate_value)
            if candidate_value == reference_value:
                continue
            candidate_spec = reference.with_axis(
                axis,
                candidate_value,
                spec_id=f"{reference.spec_id}__{axis}__{candidate_value}",
            )
            plans.append(
                SingleAxisComparisonPlan(
                    axis=axis,
                    reference=reference,
                    candidate=candidate_spec,
                )
            )
    if not plans:
        raise ValueError("Orthogonal sweep produced no single-axis comparisons.")
    return tuple(plans)


__all__ = [
    "ALLOWED_AXIS_VALUES",
    "ActionRepresentationAxis",
    "AlgorithmAxis",
    "ComputeBudget",
    "CurriculumAxis",
    "EXPERIMENTAL_AXES",
    "EncoderAxis",
    "ExperimentalAxisName",
    "ExperimentalAxisSpec",
    "LeagueAxis",
    "MemoryAxis",
    "SearchAxis",
    "ShapingAxis",
    "SingleAxisComparisonPlan",
    "build_orthogonal_axis_sweep",
    "validate_single_axis_comparison",
]
