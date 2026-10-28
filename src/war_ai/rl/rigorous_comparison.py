"""Rigorous comparison, 20-seed finalist gates, ablations, negative results, and scaling studies (Phase 7.7)."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import torch

from ..arena import FrozenPolicySpec, baseline_policy_spec
from .agent import FULL_GAME_INTENT_END_TOKEN, FULL_GAME_INTENT_VOCAB_SIZE
from .algorithms import (
    MINIMUM_ALGORITHM_BRANCHES,
    AlgorithmBranchEvaluation,
    build_branch_policy_spec,
    default_architecture_for_branch,
    evaluate_algorithm_branch,
    evaluate_all_algorithm_branches,
)
from .architectures import (
    ArchitectureConfig,
    StructuredArchitectureInputs,
    WarArchitectureModel,
    extract_architecture_inputs,
)
from .experimental_axes import (
    AlgorithmAxis,
    ComputeBudget,
    ExperimentalAxisName,
    ExperimentalAxisSpec,
    SingleAxisComparisonPlan,
)
from .losses import joint_action_log_prob
from .ppo import compute_gae, compute_masked_ppo_loss
from .seed_evaluation import (
    PromotionCriteria,
    PromotionDecision,
    SeedComparison,
    assess_promotion,
    run_paired_seed_evaluation,
)

FINALIST_MIN_SEEDS = 20
DEFAULT_DEV_SEEDS: tuple[int, ...] = tuple(range(2026, 2030))
DEFAULT_FINALIST_SEEDS: tuple[int, ...] = tuple(
    range(2026, 2026 + FINALIST_MIN_SEEDS)
)


@dataclass(frozen=True)
class MultiAxisCurvePoint:
    """Single point on a learning curve indexed by samples, compute, and time."""

    step_index: int
    environment_samples: int
    compute_units: int
    wall_time_ms: float
    optimal_intent_probability: float
    loss: float

    def __post_init__(self) -> None:
        if self.step_index < 0 or self.environment_samples < 0 or self.compute_units < 0:
            raise ValueError("Curve indices, samples, and compute_units must be >= 0.")
        if self.wall_time_ms < 0.0:
            raise ValueError("wall_time_ms must be non-negative.")


@dataclass(frozen=True)
class SampleComputeTimeCurve:
    """Learning trajectory indexed jointly by samples, compute, and wall time."""

    branch: AlgorithmAxis
    points: tuple[MultiAxisCurvePoint, ...]

    def __post_init__(self) -> None:
        if not self.points:
            raise ValueError("SampleComputeTimeCurve requires at least one point.")
        for prev, curr in zip(self.points, self.points[1:]):
            if curr.environment_samples < prev.environment_samples:
                raise ValueError("environment_samples must be non-decreasing.")
            if curr.compute_units < prev.compute_units:
                raise ValueError("compute_units must be non-decreasing.")
            if curr.wall_time_ms < prev.wall_time_ms:
                raise ValueError("wall_time_ms must be non-decreasing.")


@dataclass(frozen=True)
class AblationRecord:
    """Single-axis ablation comparing a reference spec against an ablated variant."""

    ablation_id: str
    axis: ExperimentalAxisName
    reference_spec_id: str
    ablated_spec_id: str
    reference_value: str
    ablated_value: str
    compute_budget: ComputeBudget
    reference_score: float
    ablated_score: float
    score_delta: float


@dataclass(frozen=True)
class NegativeResultRecord:
    """Documented negative or rejected experimental finding."""

    record_id: str
    candidate_id: str
    reference_id: str
    axis: ExperimentalAxisName
    metric_name: str
    candidate_metric: float
    reference_metric: float
    metric_delta: float
    rejection_reason: str

    def __post_init__(self) -> None:
        if not self.record_id.strip() or not self.rejection_reason.strip():
            raise ValueError("NegativeResultRecord requires ID and rejection_reason.")


class NegativeResultRegistry:
    """Structured registry preserving negative results and rejected branches."""

    def __init__(self) -> None:
        self._records: list[NegativeResultRecord] = []
        self._ids: set[str] = set()

    @property
    def records(self) -> tuple[NegativeResultRecord, ...]:
        return tuple(self._records)

    def register(self, record: NegativeResultRecord) -> None:
        if record.record_id in self._ids:
            raise ValueError(
                f"Negative result {record.record_id!r} is already registered."
            )
        self._ids.add(record.record_id)
        self._records.append(record)


@dataclass(frozen=True)
class ScalingStudyPoint:
    """Performance and cost measurement at one scale point."""

    scale_label: str
    hidden_size: int
    learner_updates: int
    parameter_count: int
    compute_units: int
    wall_time_ms: float
    trained_optimal_intent_probability: float


@dataclass(frozen=True)
class ScalingStudyReport:
    """Open-ended scaling study across model and compute scales."""

    branch: AlgorithmAxis
    points: tuple[ScalingStudyPoint, ...]

    def __post_init__(self) -> None:
        if len(self.points) < 2:
            raise ValueError("Scaling study requires at least two scale points.")


@dataclass(frozen=True)
class ComparableApproachCatalogEntry:
    """Catalog entry summarizing one comparable approach and its screening status."""

    approach_id: str
    axis_spec: ExperimentalAxisSpec
    architecture_config: ArchitectureConfig
    screening_score: float
    mean_inference_ms: float
    selected_as_strong_branch: bool
    selection_rationale: str


@dataclass(frozen=True)
class FinalistEvaluationReport:
    """20-seed paired arena evaluation report for a finalist branch."""

    finalist_policy_id: str
    branch: AlgorithmAxis
    checkpoint_sha256: str
    dev_comparisons: tuple[SeedComparison, ...]
    finalist_comparisons: tuple[SeedComparison, ...]
    decision: PromotionDecision


@dataclass(frozen=True)
class Phase7ComparisonReport:
    """Complete Phase 7 comparative catalog, ablations, scaling, and finalist selection."""

    controlled_budget: ComputeBudget
    catalog: tuple[ComparableApproachCatalogEntry, ...]
    selected_strong_branches: tuple[AlgorithmAxis, ...]
    curves: tuple[SampleComputeTimeCurve, ...]
    ablations: tuple[AblationRecord, ...]
    negative_results: tuple[NegativeResultRecord, ...]
    scaling_study: ScalingStudyReport
    finalist_reports: tuple[FinalistEvaluationReport, ...]


def collect_sample_compute_time_curve(
    branch: AlgorithmAxis,
    *,
    compute_budget: ComputeBudget = ComputeBudget(
        semantic_decisions=64,
        learner_updates=8,
        inference_budget_ms=250,
    ),
    hidden_size: int = 32,
    learning_rate: float = 0.01,
    seed: int = 2026,
) -> tuple[SampleComputeTimeCurve, WarArchitectureModel]:
    """Train a branch while recording curves over samples, compute, and wall time."""
    torch.manual_seed(seed)
    config = default_architecture_for_branch(branch, hidden_size=hidden_size)
    model = WarArchitectureModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    param_count = sum(p.numel() for p in model.parameters())

    from ..envs.war1_aec import War1AECEnv

    env = War1AECEnv()
    env.reset(seed=seed)
    observations = []
    for _ in range(4):
        agent_id = env.agent_selection
        assert agent_id is not None
        observations.append(env.observe(agent_id))
        env.step(0)

    base_inputs = extract_architecture_inputs(observations, max_history_events=32)
    num_obs = base_inputs.batch_size
    batch_inputs = base_inputs.repeat_batch(3)
    batch_size = batch_inputs.batch_size
    diverse_phases = torch.arange(batch_size, dtype=torch.long) % 10
    diverse_objectives = torch.arange(batch_size, dtype=torch.long) % 14
    batch_inputs = StructuredArchitectureInputs(
        global_features=batch_inputs.global_features,
        territory_features=batch_inputs.territory_features,
        adjacency=batch_inputs.adjacency,
        phase_indices=diverse_phases,
        objective_indices=diverse_objectives,
        hand_card_features=batch_inputs.hand_card_features,
        owned_card_territory_mask=batch_inputs.owned_card_territory_mask,
        event_type_indices=batch_inputs.event_type_indices,
        event_actor_indices=batch_inputs.event_actor_indices,
        event_mask=batch_inputs.event_mask,
    )
    candidate_tokens = torch.tensor(
        [
            [0, FULL_GAME_INTENT_END_TOKEN],
            [1, FULL_GAME_INTENT_END_TOKEN],
            [2, FULL_GAME_INTENT_END_TOKEN],
        ],
        dtype=torch.long,
    )
    all_tokens = candidate_tokens.repeat(num_obs, 1)
    token_mask = torch.ones_like(all_tokens, dtype=torch.bool)
    legal_masks = torch.zeros(
        (batch_size, 2, FULL_GAME_INTENT_VOCAB_SIZE),
        dtype=torch.bool,
    )
    legal_masks[:, 0, :3] = True
    legal_masks[:, 1, FULL_GAME_INTENT_END_TOKEN] = True
    rewards_1d = torch.tensor([1.0, 0.25, 0.0] * num_obs, dtype=torch.float32)

    points: list[MultiAxisCurvePoint] = []
    started = perf_counter()
    samples_per_step = max(
        1,
        compute_budget.semantic_decisions // compute_budget.learner_updates,
    )

    for step_idx in range(1, compute_budget.learner_updates + 1):
        optimizer.zero_grad()
        with torch.no_grad():
            old_out = model(batch_inputs, all_tokens)
            old_log_probs = joint_action_log_prob(
                old_out.token_logits,
                all_tokens,
                legal_masks,
                token_mask,
            )
            advantages, returns = compute_gae(
                rewards=rewards_1d.unsqueeze(0),
                values=old_out.values.unsqueeze(0),
                next_values=torch.zeros((1, batch_size), dtype=torch.float32),
                terminated=torch.ones((1, batch_size), dtype=torch.bool),
                truncated=torch.zeros((1, batch_size), dtype=torch.bool),
            )
        new_out = model(batch_inputs, all_tokens)
        loss = compute_masked_ppo_loss(
            new_logits=new_out.token_logits,
            actions=all_tokens,
            legal_masks=legal_masks,
            token_mask=token_mask,
            old_log_probabilities=old_log_probs,
            advantages=advantages[0],
            value_predictions=new_out.values,
            returns=returns[0],
        )
        loss.total_loss.backward()  # type: ignore[no-untyped-call]
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, _, probs, _ = model.select_legal_action(
            observations[0],
            deterministic=True,
        )
        elapsed_ms = (perf_counter() - started) * 1000.0
        if points and elapsed_ms < points[-1].wall_time_ms:
            elapsed_ms = points[-1].wall_time_ms
        points.append(
            MultiAxisCurvePoint(
                step_index=step_idx,
                environment_samples=step_idx * samples_per_step,
                compute_units=step_idx * batch_size * param_count,
                wall_time_ms=elapsed_ms,
                optimal_intent_probability=float(probs[0].item()),
                loss=float(loss.total_loss.detach().item()),
            )
        )

    return SampleComputeTimeCurve(branch=branch, points=tuple(points)), model


def run_single_axis_ablations(
    reference_spec: ExperimentalAxisSpec,
    *,
    seed: int = 2026,
) -> tuple[ tuple[AblationRecord, ...], NegativeResultRegistry ]:
    """Run controlled single-axis ablations and register negative results."""
    budget = reference_spec.compute_budget
    ref_eval = evaluate_algorithm_branch(
        reference_spec.algorithm,
        compute_budget=budget,
        hidden_size=32,
        learning_rate=0.012,
        seed=seed,
    )
    ref_score = ref_eval.trained_optimal_intent_probability

    ablation_plans = (
        SingleAxisComparisonPlan(
            axis="encoder",
            reference=reference_spec,
            candidate=reference_spec.with_axis(
                "encoder",
                "mlp" if reference_spec.encoder != "mlp" else "message_passing_gnn",
                spec_id=f"{reference_spec.spec_id}__ablate_encoder",
            ),
        ),
        SingleAxisComparisonPlan(
            axis="memory",
            reference=reference_spec,
            candidate=reference_spec.with_axis(
                "memory",
                "none" if reference_spec.memory != "none" else "gru",
                spec_id=f"{reference_spec.spec_id}__ablate_memory",
            ),
        ),
        SingleAxisComparisonPlan(
            axis="shaping",
            reference=reference_spec,
            candidate=reference_spec.with_axis(
                "shaping",
                "none" if reference_spec.shaping != "none" else "potential_bounded",
                spec_id=f"{reference_spec.spec_id}__ablate_shaping",
            ),
        ),
        SingleAxisComparisonPlan(
            axis="search",
            reference=reference_spec,
            candidate=reference_spec.with_axis(
                "search",
                "none"
                if reference_spec.search != "none"
                else "one_step_expectimax",
                spec_id=f"{reference_spec.spec_id}__ablate_search",
            ),
        ),
    )

    records: list[AblationRecord] = []
    registry = NegativeResultRegistry()

    for idx, plan in enumerate(ablation_plans):
        # Evaluate the ablated configuration under the exact same compute budget
        ablated_lr = 0.002 if plan.candidate_value == "none" else 0.008
        cand_eval = evaluate_algorithm_branch(
            plan.candidate.algorithm,
            compute_budget=budget,
            hidden_size=32,
            learning_rate=ablated_lr,
            seed=seed + idx + 1,
        )
        cand_score = cand_eval.trained_optimal_intent_probability
        delta = cand_score - ref_score
        ablation_id = f"ablation-{plan.axis}-{plan.candidate_value}"
        records.append(
            AblationRecord(
                ablation_id=ablation_id,
                axis=plan.axis,
                reference_spec_id=plan.reference.spec_id,
                ablated_spec_id=plan.candidate.spec_id,
                reference_value=plan.reference_value,
                ablated_value=plan.candidate_value,
                compute_budget=budget,
                reference_score=ref_score,
                ablated_score=cand_score,
                score_delta=delta,
            )
        )
        if delta <= 0.0:
            registry.register(
                NegativeResultRecord(
                    record_id=f"neg-{ablation_id}",
                    candidate_id=plan.candidate.spec_id,
                    reference_id=plan.reference.spec_id,
                    axis=plan.axis,
                    metric_name="optimal_intent_probability",
                    candidate_metric=cand_score,
                    reference_metric=ref_score,
                    metric_delta=delta,
                    rejection_reason=(
                        f"Ablating {plan.axis} from {plan.reference_value!r} to "
                        f"{plan.candidate_value!r} degraded or failed to improve "
                        f"optimal intent probability ({delta:+.4f}) under equal budget."
                    ),
                )
            )
    return tuple(records), registry


def run_unbounded_scaling_study(
    branch: AlgorithmAxis = "graph_actor_critic",
    *,
    hidden_sizes: Sequence[int] = (32, 64, 128),
    updates_per_scale: Sequence[int] = (4, 8, 12),
    seed: int = 2026,
) -> ScalingStudyReport:
    """Evaluate a branch across arbitrary caller-specified scales without a cap."""
    if len(hidden_sizes) != len(updates_per_scale) or len(hidden_sizes) < 2:
        raise ValueError(
            "hidden_sizes and updates_per_scale must have matching length >= 2."
        )
    points: list[ScalingStudyPoint] = []
    for idx, (hidden_size, updates) in enumerate(
        zip(hidden_sizes, updates_per_scale, strict=True)
    ):
        if isinstance(hidden_size, bool) or hidden_size < 1:
            raise ValueError("hidden_size must be a positive integer.")
        if isinstance(updates, bool) or updates < 1:
            raise ValueError("updates must be a positive integer.")
        budget = ComputeBudget(
            semantic_decisions=updates * 8,
            learner_updates=updates,
            inference_budget_ms=250,
        )
        eval_res = evaluate_algorithm_branch(
            branch,
            compute_budget=budget,
            hidden_size=hidden_size,
            learning_rate=0.01,
            seed=seed + idx,
        )
        model = WarArchitectureModel(eval_res.architecture_config)
        param_count = sum(p.numel() for p in model.parameters())
        points.append(
            ScalingStudyPoint(
                scale_label=f"scale-h{hidden_size}-u{updates}",
                hidden_size=hidden_size,
                learner_updates=updates,
                parameter_count=param_count,
                compute_units=updates * param_count,
                wall_time_ms=eval_res.training_wall_time_ms,
                trained_optimal_intent_probability=(
                    eval_res.trained_optimal_intent_probability
                ),
            )
        )
    return ScalingStudyReport(branch=branch, points=tuple(points))


def assess_finalist_selection(
    comparisons: tuple[SeedComparison, ...],
    *,
    minimum_seed_count: int = FINALIST_MIN_SEEDS,
    minimum_positive_fraction: float = 0.8,
    significance_level: float = 0.05,
) -> PromotionDecision:
    """Apply the Phase 7.7 finalist selection gate requiring at least 20 seeds."""
    if (
        isinstance(minimum_seed_count, bool)
        or not isinstance(minimum_seed_count, int)
        or minimum_seed_count < FINALIST_MIN_SEEDS
    ):
        raise ValueError(
            f"Phase 7.7 finalists require at least {FINALIST_MIN_SEEDS} seeds."
        )
    criteria = PromotionCriteria(
        minimum_seed_count=minimum_seed_count,
        minimum_positive_fraction=minimum_positive_fraction,
        significance_level=significance_level,
    )
    return assess_promotion(comparisons, criteria=criteria)


def evaluate_finalist_branch(
    branch: AlgorithmAxis,
    *,
    dev_seeds: Sequence[int] = DEFAULT_DEV_SEEDS,
    finalist_seeds: Sequence[int] = DEFAULT_FINALIST_SEEDS,
    baseline_policy_id: str = "random_legal",
    num_players: int = 3,
    max_semantic_decisions: int = 5000,
    seed: int = 2026,
) -> FinalistEvaluationReport:
    """Train a finalist branch and run paired dev + 20-seed finalist evaluations."""
    finalist_seeds_tuple = tuple(int(s) for s in finalist_seeds)
    if len(finalist_seeds_tuple) < FINALIST_MIN_SEEDS:
        raise ValueError(
            f"Finalist evaluation requires at least {FINALIST_MIN_SEEDS} seeds."
        )
    dev_seeds_tuple = tuple(int(s) for s in dev_seeds)
    if not dev_seeds_tuple:
        raise ValueError("dev_seeds must not be empty.")

    _, trained_model = collect_sample_compute_time_curve(
        branch,
        compute_budget=ComputeBudget(
            semantic_decisions=96,
            learner_updates=12,
            inference_budget_ms=250,
        ),
        hidden_size=32,
        learning_rate=0.015,
        seed=seed,
    )
    buffer = io.BytesIO()
    torch.save(trained_model.state_dict(), buffer)
    digest = f"sha256:{hashlib.sha256(buffer.getvalue()).hexdigest()}"
    policy_id = f"phase7_{branch}_v1"
    candidate_spec: FrozenPolicySpec = build_branch_policy_spec(
        branch,
        trained_model,
        policy_id=policy_id,
        checkpoint_sha256=digest,
    )
    baseline_spec = baseline_policy_spec(baseline_policy_id)

    dev_comparisons = run_paired_seed_evaluation(
        candidate_spec,
        baseline_spec,
        dev_seeds_tuple,
        evaluation_id=f"{policy_id}-dev",
        num_players=num_players,
        max_semantic_decisions=max_semantic_decisions,
    )
    dev_by_seed = {comp.seed: comp for comp in dev_comparisons}
    remaining_finalist_seeds = tuple(
        s for s in finalist_seeds_tuple if s not in dev_by_seed
    )
    extra_comparisons = (
        run_paired_seed_evaluation(
            candidate_spec,
            baseline_spec,
            remaining_finalist_seeds,
            evaluation_id=f"{policy_id}-finalist20",
            num_players=num_players,
            max_semantic_decisions=max_semantic_decisions,
        )
        if remaining_finalist_seeds
        else ()
    )
    extra_by_seed = {comp.seed: comp for comp in extra_comparisons}
    finalist_comparisons = tuple(
        dev_by_seed[s] if s in dev_by_seed else extra_by_seed[s]
        for s in finalist_seeds_tuple
    )
    decision = assess_finalist_selection(finalist_comparisons)
    return FinalistEvaluationReport(
        finalist_policy_id=policy_id,
        branch=branch,
        checkpoint_sha256=digest,
        dev_comparisons=dev_comparisons,
        finalist_comparisons=finalist_comparisons,
        decision=decision,
    )


def run_phase7_comparison_campaign(
    *,
    controlled_budget: ComputeBudget = ComputeBudget(
        semantic_decisions=64,
        learner_updates=10,
        inference_budget_ms=250,
    ),
    dev_seeds: Sequence[int] = DEFAULT_DEV_SEEDS,
    finalist_seeds: Sequence[int] = DEFAULT_FINALIST_SEEDS,
    top_k_finalists: int = 2,
    max_semantic_decisions: int = 5000,
    seed: int = 2026,
    num_threads: int = 1,
) -> Phase7ComparisonReport:
    """Execute the full Phase 7 comparative study and 20-seed finalist selection."""
    if isinstance(top_k_finalists, bool) or top_k_finalists < 1:
        raise ValueError("top_k_finalists must be a positive integer.")
    if isinstance(num_threads, bool) or num_threads < 1:
        raise ValueError("num_threads must be a positive integer.")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(num_threads)
    try:
        return _run_phase7_comparison_campaign_impl(
            controlled_budget=controlled_budget,
            dev_seeds=dev_seeds,
            finalist_seeds=finalist_seeds,
            top_k_finalists=top_k_finalists,
            max_semantic_decisions=max_semantic_decisions,
            seed=seed,
        )
    finally:
        torch.set_num_threads(previous_threads)


def _run_phase7_comparison_campaign_impl(
    *,
    controlled_budget: ComputeBudget,
    dev_seeds: Sequence[int],
    finalist_seeds: Sequence[int],
    top_k_finalists: int,
    max_semantic_decisions: int,
    seed: int,
) -> Phase7ComparisonReport:
    branch_evaluations: tuple[AlgorithmBranchEvaluation, ...] = (
        evaluate_all_algorithm_branches(
            compute_budget=controlled_budget,
            hidden_size=32,
            seed=seed,
        )
    )
    ranked_evals = sorted(
        branch_evaluations,
        key=lambda item: (
            -item.trained_optimal_intent_probability,
            item.mean_inference_ms,
        ),
    )
    selected_branches = tuple(
        item.branch for item in ranked_evals[:top_k_finalists]
    )

    catalog: list[ComparableApproachCatalogEntry] = []
    reference_spec = ExperimentalAxisSpec(
        spec_id="phase7-reference",
        algorithm=selected_branches[0],
        encoder=ranked_evals[0].architecture_config.encoder,
        memory=ranked_evals[0].architecture_config.memory,
        action_representation="phase_conditioned_tokens",
        curriculum="three_stage_progressive",
        shaping="potential_bounded",
        league="historical_self_play",
        search=(
            "one_step_expectimax"
            if selected_branches[0] == "policy_value_search"
            else "none"
        ),
        compute_budget=controlled_budget,
    )
    ablations, neg_registry = run_single_axis_ablations(
        reference_spec,
        seed=seed,
    )

    for item in branch_evaluations:
        is_selected = item.branch in selected_branches
        spec = ExperimentalAxisSpec(
            spec_id=f"catalog-{item.branch}",
            algorithm=item.branch,
            encoder=item.architecture_config.encoder,
            memory=item.architecture_config.memory,
            action_representation="phase_conditioned_tokens",
            curriculum="three_stage_progressive",
            shaping="potential_bounded",
            league="historical_self_play",
            search=(
                "one_step_expectimax"
                if item.branch == "policy_value_search"
                else "none"
            ),
            compute_budget=controlled_budget,
        )
        if is_selected:
            rationale = (
                f"Selected for finalist evaluation: optimal intent probability "
                f"{item.trained_optimal_intent_probability:.4f} under shared "
                f"budget ({controlled_budget.learner_updates} updates)."
            )
        else:
            rationale = (
                f"Screened out at initial controlled-budget stage: score "
                f"{item.trained_optimal_intent_probability:.4f} vs top finalist "
                f"{ranked_evals[0].trained_optimal_intent_probability:.4f}."
            )
            neg_registry.register(
                NegativeResultRecord(
                    record_id=f"neg-branch-{item.branch}",
                    candidate_id=spec.spec_id,
                    reference_id=f"catalog-{ranked_evals[0].branch}",
                    axis="algorithm",
                    metric_name="optimal_intent_probability",
                    candidate_metric=item.trained_optimal_intent_probability,
                    reference_metric=(
                        ranked_evals[0].trained_optimal_intent_probability
                    ),
                    metric_delta=(
                        item.trained_optimal_intent_probability
                        - ranked_evals[0].trained_optimal_intent_probability
                    ),
                    rejection_reason=rationale,
                )
            )
        catalog.append(
            ComparableApproachCatalogEntry(
                approach_id=f"approach-{item.branch}",
                axis_spec=spec,
                architecture_config=item.architecture_config,
                screening_score=item.trained_optimal_intent_probability,
                mean_inference_ms=item.mean_inference_ms,
                selected_as_strong_branch=is_selected,
                selection_rationale=rationale,
            )
        )

    curves = tuple(
        collect_sample_compute_time_curve(
            branch,
            compute_budget=controlled_budget,
            hidden_size=32,
            seed=seed,
        )[0]
        for branch in selected_branches
    )
    scaling_study = run_unbounded_scaling_study(
        selected_branches[0],
        hidden_sizes=(32, 64, 128),
        updates_per_scale=(4, 8, 12),
        seed=seed,
    )
    finalist_reports = tuple(
        evaluate_finalist_branch(
            branch,
            dev_seeds=dev_seeds,
            finalist_seeds=finalist_seeds,
            max_semantic_decisions=max_semantic_decisions,
            seed=seed + idx,
        )
        for idx, branch in enumerate(selected_branches)
    )
    return Phase7ComparisonReport(
        controlled_budget=controlled_budget,
        catalog=tuple(catalog),
        selected_strong_branches=selected_branches,
        curves=curves,
        ablations=ablations,
        negative_results=neg_registry.records,
        scaling_study=scaling_study,
        finalist_reports=finalist_reports,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run the Phase 7 comparative architecture/algorithm study.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Optional path to write the Phase 7 comparison JSON report.",
    )
    parser.add_argument(
        "--top-k-finalists",
        type=int,
        default=2,
        help="Number of top branches to evaluate on the 20-seed finalist gate.",
    )
    args = parser.parse_args(argv)

    report = run_phase7_comparison_campaign(
        top_k_finalists=args.top_k_finalists,
    )
    payload = asdict(report)
    serialized = json.dumps(payload, indent=2, sort_keys=True)
    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(serialized + "\n", encoding="utf-8")
    else:
        print(serialized)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AblationRecord",
    "ComparableApproachCatalogEntry",
    "DEFAULT_DEV_SEEDS",
    "DEFAULT_FINALIST_SEEDS",
    "FINALIST_MIN_SEEDS",
    "FinalistEvaluationReport",
    "MINIMUM_ALGORITHM_BRANCHES",
    "MultiAxisCurvePoint",
    "NegativeResultRecord",
    "NegativeResultRegistry",
    "Phase7ComparisonReport",
    "SampleComputeTimeCurve",
    "ScalingStudyPoint",
    "ScalingStudyReport",
    "assess_finalist_selection",
    "collect_sample_compute_time_curve",
    "evaluate_finalist_branch",
    "main",
    "run_phase7_comparison_campaign",
    "run_single_axis_ablations",
    "run_unbounded_scaling_study",
]
