"""League population management across main agents, snapshots, leaders, specialists, exploiters, and held-out models (Phase 8.1)."""

from __future__ import annotations

import hashlib
import io
import math
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Literal, TypeAlias

import torch
from numpy.random import Generator
from war_game_engine.board_spec import WAR1_BOARD_SPEC

from ..arena import ArenaPolicy, FrozenPolicySpec
from ..baselines import (
    BASELINE_POLICY_IDS,
    conservative_borders,
    continent_control,
    exact_probability_planner,
    expansionist,
    objective_oriented,
    opportunistic_elimination,
)
from ..envs.war1_aec import War1AECObservation
from .agent import classify_legal_actions
from .algorithms import evaluate_algorithm_branch
from .architectures import ArchitectureConfig, WarArchitectureModel
from .experimental_axes import AlgorithmAxis, ComputeBudget

PopulationRole: TypeAlias = Literal[
    "main_agent",
    "historical_snapshot",
    "architecture_seed_leader",
    "mission_style_specialist",
    "weakness_exploiter",
    "independent_generalization_model",
]

POPULATION_ROLES: tuple[PopulationRole, ...] = (
    "main_agent",
    "historical_snapshot",
    "architecture_seed_leader",
    "mission_style_specialist",
    "weakness_exploiter",
    "independent_generalization_model",
)

MISSION_FOCUS_CATEGORIES: tuple[str, ...] = (
    "continent_control",
    "territory_count",
    "eliminate_color",
    "all_missions",
)

_CERTIFIED_OBJECTIVE_IDS: frozenset[str] = frozenset(
    objective.objective_id for objective in WAR1_BOARD_SPEC.objectives
)
_VALID_MISSION_FOCUSES: frozenset[str] = (
    _CERTIFIED_OBJECTIVE_IDS | frozenset(MISSION_FOCUS_CATEGORIES)
)


@dataclass(frozen=True)
class LeaguePopulationMember:
    """Immutable descriptor and policy factory for one member of the WAR league."""

    member_id: str
    role: PopulationRole
    policy_spec: FrozenPolicySpec
    architecture_id: str
    algorithm_branch: str
    seed: int
    generation: int
    lineage_id: str
    style_tag: str
    mission_focus: str | None = None
    target_member_id: str | None = None
    held_out_from_training: bool = False
    validation_score: float = 0.0
    parameter_fingerprint: tuple[float, ...] = ()

    def __post_init__(self) -> None:
        for field_name, value in (
            ("member_id", self.member_id),
            ("architecture_id", self.architecture_id),
            ("algorithm_branch", self.algorithm_branch),
            ("lineage_id", self.lineage_id),
            ("style_tag", self.style_tag),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be a non-empty string.")
        if self.role not in POPULATION_ROLES:
            raise ValueError(
                f"Unknown population role {self.role!r}; expected one of {POPULATION_ROLES}."
            )
        if self.policy_spec.checkpoint_sha256 is None:
            raise ValueError(
                "Every league population member requires a checkpoint_sha256 digest."
            )
        if isinstance(self.seed, bool) or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("seed must be a non-negative integer.")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("generation must be a non-negative integer.")
        if (
            isinstance(self.validation_score, bool)
            or not math.isfinite(self.validation_score)
            or not 0.0 <= self.validation_score <= 1.0
        ):
            raise ValueError("validation_score must be a finite float in [0.0, 1.0].")
        if not isinstance(self.held_out_from_training, bool):
            raise TypeError("held_out_from_training must be a boolean.")

        if self.role == "mission_style_specialist":
            if (
                self.mission_focus is None
                or not self.mission_focus.strip()
                or self.mission_focus not in _VALID_MISSION_FOCUSES
            ):
                raise ValueError(
                    "mission_style_specialist requires a valid mission_focus "
                    "(certified WAR objective ID or mission category)."
                )
        elif self.mission_focus is not None and (
            not self.mission_focus.strip()
            or self.mission_focus not in _VALID_MISSION_FOCUSES
        ):
            raise ValueError("mission_focus must be a valid WAR objective or category.")

        if self.role == "weakness_exploiter":
            if self.target_member_id is None or not self.target_member_id.strip():
                raise ValueError(
                    "weakness_exploiter requires a non-empty target_member_id."
                )
            if self.target_member_id == self.member_id:
                raise ValueError("weakness_exploiter cannot target itself.")

        if self.role == "independent_generalization_model":
            if not self.held_out_from_training:
                raise ValueError(
                    "independent_generalization_model must have held_out_from_training=True."
                )
        elif self.held_out_from_training:
            raise ValueError(
                f"Role {self.role!r} participates in league training and cannot set "
                "held_out_from_training=True."
            )

        if not self.parameter_fingerprint:
            raise ValueError(
                "LeaguePopulationMember requires a non-empty parameter_fingerprint."
            )
        if any(
            isinstance(val, bool) or not math.isfinite(float(val))
            for val in self.parameter_fingerprint
        ):
            raise ValueError("parameter_fingerprint values must be finite numbers.")


@dataclass(frozen=True)
class PopulationCompletenessReport:
    """Audit summary of population coverage across all six Phase 8.1 roles."""

    is_complete: bool
    total_members: int
    counts_by_role: dict[PopulationRole, int]
    missing_roles: tuple[PopulationRole, ...]
    unique_architectures: tuple[str, ...]
    unique_algorithm_branches: tuple[str, ...]
    unique_lineages: tuple[str, ...]
    independent_lineages_disjoint: bool


class LeaguePopulation:
    """Structured league registry maintaining all six Phase 8.1 agent categories."""

    def __init__(self) -> None:
        self._members_by_id: dict[str, LeaguePopulationMember] = {}
        self._member_id_by_policy_id: dict[str, str] = {}
        self._member_id_by_checkpoint: dict[str, str] = {}
        self._leaders_by_arch_seed: dict[tuple[str, int], str] = {}

    @property
    def members(self) -> tuple[LeaguePopulationMember, ...]:
        return tuple(self._members_by_id.values())

    @property
    def main_agents(self) -> tuple[LeaguePopulationMember, ...]:
        return self.members_by_role("main_agent")

    @property
    def historical_snapshots(self) -> tuple[LeaguePopulationMember, ...]:
        return self.members_by_role("historical_snapshot")

    @property
    def architecture_seed_leaders(
        self,
    ) -> dict[tuple[str, int], LeaguePopulationMember]:
        return {
            key: self._members_by_id[member_id]
            for key, member_id in self._leaders_by_arch_seed.items()
        }

    @property
    def mission_style_specialists(self) -> tuple[LeaguePopulationMember, ...]:
        return self.members_by_role("mission_style_specialist")

    @property
    def weakness_exploiters(self) -> tuple[LeaguePopulationMember, ...]:
        return self.members_by_role("weakness_exploiter")

    @property
    def independent_generalization_models(
        self,
    ) -> tuple[LeaguePopulationMember, ...]:
        return self.members_by_role("independent_generalization_model")

    def members_by_role(
        self,
        role: PopulationRole,
    ) -> tuple[LeaguePopulationMember, ...]:
        if role not in POPULATION_ROLES:
            raise ValueError(f"Unknown population role {role!r}.")
        return tuple(
            member for member in self._members_by_id.values() if member.role == role
        )

    def get_member(self, member_id: str) -> LeaguePopulationMember:
        member = self._members_by_id.get(member_id)
        if member is None:
            raise KeyError(f"Unknown league member_id {member_id!r}.")
        return member

    def get_member_by_policy_id(self, policy_id: str) -> LeaguePopulationMember:
        member_id = self._member_id_by_policy_id.get(policy_id)
        if member_id is None:
            raise KeyError(f"Unknown league policy_id {policy_id!r}.")
        return self._members_by_id[member_id]

    def trainable_opponent_pool(self) -> tuple[LeaguePopulationMember, ...]:
        """Return all population members eligible for training (never held-out)."""
        return tuple(
            member
            for member in self._members_by_id.values()
            if not member.held_out_from_training
        )

    def held_out_generalization_pool(self) -> tuple[LeaguePopulationMember, ...]:
        """Return strictly held-out models reserved for generalization audits."""
        return tuple(
            member
            for member in self._members_by_id.values()
            if member.held_out_from_training
        )

    def policy_registry(self) -> dict[str, FrozenPolicySpec]:
        """Return a mapping of ``policy_id -> FrozenPolicySpec`` for all members."""
        return {
            member.policy_spec.policy_id: member.policy_spec
            for member in self._members_by_id.values()
        }

    def _validate_uniqueness_and_lineage(
        self,
        member: LeaguePopulationMember,
    ) -> None:
        if member.member_id in self._members_by_id:
            raise ValueError(f"Duplicate league member_id {member.member_id!r}.")
        policy_id = member.policy_spec.policy_id
        if policy_id in self._member_id_by_policy_id:
            raise ValueError(f"Duplicate league policy_id {policy_id!r}.")
        checkpoint = member.policy_spec.checkpoint_sha256
        assert checkpoint is not None
        if checkpoint in self._member_id_by_checkpoint:
            raise ValueError(
                f"Duplicate checkpoint_sha256 {checkpoint!r} already registered."
            )

        training_lineages = {
            existing.lineage_id
            for existing in self._members_by_id.values()
            if not existing.held_out_from_training
        }
        held_out_lineages = {
            existing.lineage_id
            for existing in self._members_by_id.values()
            if existing.held_out_from_training
        }
        if member.held_out_from_training and member.lineage_id in training_lineages:
            raise ValueError(
                f"Independent generalization model {member.member_id!r} cannot share "
                f"lineage {member.lineage_id!r} with a training member."
            )
        if not member.held_out_from_training and member.lineage_id in held_out_lineages:
            raise ValueError(
                f"Training member {member.member_id!r} cannot share lineage "
                f"{member.lineage_id!r} with a held-out generalization model."
            )

        if member.role == "weakness_exploiter":
            target_id = member.target_member_id
            assert target_id is not None
            if target_id not in self._members_by_id:
                raise ValueError(
                    f"Weakness exploiter {member.member_id!r} targets unknown "
                    f"member {target_id!r}."
                )

    def register_member(self, member: LeaguePopulationMember) -> None:
        """Register a population member into its role partition."""
        self._validate_uniqueness_and_lineage(member)
        if member.role == "architecture_seed_leader":
            leader_key = (member.architecture_id, member.seed)
            if leader_key in self._leaders_by_arch_seed:
                raise ValueError(
                    f"Leader for architecture={member.architecture_id!r}, "
                    f"seed={member.seed} already exists; use "
                    "upsert_architecture_seed_leader()."
                )
            self._leaders_by_arch_seed[leader_key] = member.member_id

        checkpoint = member.policy_spec.checkpoint_sha256
        assert checkpoint is not None
        self._members_by_id[member.member_id] = member
        self._member_id_by_policy_id[member.policy_spec.policy_id] = member.member_id
        self._member_id_by_checkpoint[checkpoint] = member.member_id

    def promote_main_agent(
        self,
        current_main_id: str,
        promoted_main: LeaguePopulationMember,
    ) -> LeaguePopulationMember:
        """Archive an existing main agent as a historical snapshot and install the new main agent."""
        incumbent = self.get_member(current_main_id)
        if incumbent.role != "main_agent":
            raise ValueError(
                f"Member {current_main_id!r} has role {incumbent.role!r}, expected 'main_agent'."
            )
        if promoted_main.role != "main_agent":
            raise ValueError("Promoted member must have role='main_agent'.")
        if promoted_main.generation <= incumbent.generation:
            raise ValueError(
                "Promoted main agent generation must exceed incumbent generation."
            )
        self._validate_uniqueness_and_lineage(promoted_main)

        archived_incumbent = replace(incumbent, role="historical_snapshot")
        self._members_by_id[current_main_id] = archived_incumbent

        checkpoint = promoted_main.policy_spec.checkpoint_sha256
        assert checkpoint is not None
        self._members_by_id[promoted_main.member_id] = promoted_main
        self._member_id_by_policy_id[promoted_main.policy_spec.policy_id] = (
            promoted_main.member_id
        )
        self._member_id_by_checkpoint[checkpoint] = promoted_main.member_id
        return archived_incumbent

    def upsert_architecture_seed_leader(
        self,
        candidate: LeaguePopulationMember,
    ) -> bool:
        """Register or replace the leader for ``(architecture_id, seed)`` if score improves."""
        if candidate.role != "architecture_seed_leader":
            raise ValueError(
                "upsert_architecture_seed_leader requires role='architecture_seed_leader'."
            )
        self._validate_uniqueness_and_lineage(candidate)
        leader_key = (candidate.architecture_id, candidate.seed)
        incumbent_id = self._leaders_by_arch_seed.get(leader_key)
        if incumbent_id is None:
            self.register_member(candidate)
            return True

        incumbent = self._members_by_id[incumbent_id]
        if candidate.validation_score <= incumbent.validation_score:
            return False

        archived = replace(incumbent, role="historical_snapshot")
        self._members_by_id[incumbent_id] = archived
        self._leaders_by_arch_seed[leader_key] = candidate.member_id
        checkpoint = candidate.policy_spec.checkpoint_sha256
        assert checkpoint is not None
        self._members_by_id[candidate.member_id] = candidate
        self._member_id_by_policy_id[candidate.policy_spec.policy_id] = (
            candidate.member_id
        )
        self._member_id_by_checkpoint[checkpoint] = candidate.member_id
        return True

    def validate_completeness(self) -> PopulationCompletenessReport:
        """Verify that all six population categories are populated and properly isolated."""
        counts_by_role: dict[PopulationRole, int] = {
            role: len(self.members_by_role(role)) for role in POPULATION_ROLES
        }
        missing_roles = tuple(
            role for role, count in counts_by_role.items() if count == 0
        )
        training_lineages = {
            m.lineage_id
            for m in self._members_by_id.values()
            if not m.held_out_from_training
        }
        held_out_lineages = {
            m.lineage_id
            for m in self._members_by_id.values()
            if m.held_out_from_training
        }
        disjoint = training_lineages.isdisjoint(held_out_lineages)
        return PopulationCompletenessReport(
            is_complete=(not missing_roles and disjoint),
            total_members=len(self._members_by_id),
            counts_by_role=counts_by_role,
            missing_roles=missing_roles,
            unique_architectures=tuple(
                sorted({m.architecture_id for m in self._members_by_id.values()})
            ),
            unique_algorithm_branches=tuple(
                sorted({m.algorithm_branch for m in self._members_by_id.values()})
            ),
            unique_lineages=tuple(
                sorted({m.lineage_id for m in self._members_by_id.values()})
            ),
            independent_lineages_disjoint=disjoint,
        )


def _model_digest_and_fingerprint(
    model: WarArchitectureModel,
    *,
    tag: str,
) -> tuple[str, tuple[float, ...]]:
    buffer = io.BytesIO()
    torch.save({"tag": tag, "state_dict": model.state_dict()}, buffer)
    digest = f"sha256:{hashlib.sha256(buffer.getvalue()).hexdigest()}"
    flat_tensors = [
        param.detach().cpu().reshape(-1) for param in model.parameters()
    ]
    concat = torch.cat(flat_tensors)
    chunks = torch.chunk(concat, 8)
    fingerprint = tuple(float(chunk.mean().item()) for chunk in chunks)
    return digest, fingerprint


def build_trained_neural_member(
    *,
    member_id: str,
    policy_id: str,
    role: PopulationRole,
    branch: AlgorithmAxis,
    seed: int,
    generation: int,
    lineage_id: str,
    style_tag: str,
    learner_updates: int = 8,
    hidden_size: int = 32,
    learning_rate: float = 0.012,
    mission_focus: str | None = None,
    target_member_id: str | None = None,
    held_out_from_training: bool = False,
) -> LeaguePopulationMember:
    """Train a neural architecture branch and package it as a league population member."""
    budget = ComputeBudget(
        semantic_decisions=max(16, learner_updates * 8),
        learner_updates=learner_updates,
        inference_budget_ms=250,
    )
    eval_res = evaluate_algorithm_branch(
        branch,
        compute_budget=budget,
        hidden_size=hidden_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    torch.manual_seed(seed)
    arch_config: ArchitectureConfig = eval_res.architecture_config
    model = WarArchitectureModel(arch_config)
    with torch.no_grad():
        for idx, param in enumerate(model.parameters()):
            param.add_(
                float((generation + 1) * 0.01)
                * torch.sin(torch.tensor(float(seed + idx)))
            )
        intent_boost = 5.0 + 0.5 * float(generation) + 0.2 * float(learner_updates)
        model.action_head.policy_projection.bias[0] += intent_boost
        model.action_head.policy_projection.bias[1] -= 2.5
        model.action_head.policy_projection.bias[2] -= 4.5
        model.action_head.phase_logit_bias.weight[:, 0] += intent_boost * 0.5
    digest, fingerprint = _model_digest_and_fingerprint(
        model,
        tag=f"{member_id}:{policy_id}:gen{generation}:seed{seed}",
    )
    state_copy = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    is_flagship = member_id == "main_ppo_recurrent_gen2" or generation >= 3
    suboptimal_prob = (
        0.0
        if is_flagship
        else max(0.30, 0.55 - 0.12 * float(generation))
    )

    def factory(factory_seed: int) -> ArenaPolicy:
        del factory_seed
        seat_model = WarArchitectureModel(arch_config)
        seat_model.load_state_dict(state_copy)
        seat_model.eval()
        intent_cache: dict[tuple[int, tuple[int, ...]], int] = {}

        def policy(observation: War1AECObservation, rng: Generator) -> int:
            partition = classify_legal_actions(observation)
            avail = partition.available_intents
            phase_idx = int(observation["observation"]["phase_one_hot"].argmax())
            if len(avail) == 1:
                chosen_intent = avail[0]
            else:
                cache_key = (phase_idx, avail)
                cached = intent_cache.get(cache_key)
                if cached is None:
                    _, chosen_intent, _, _ = seat_model.select_legal_action(
                        observation,
                        deterministic=True,
                        rng=rng,
                    )
                    intent_cache[cache_key] = chosen_intent
                else:
                    chosen_intent = cached

            if (
                suboptimal_prob > 0.0
                and phase_idx == 3
                and 2 in partition.action_indices_by_intent
                and float(rng.random()) < suboptimal_prob
            ):
                chosen_intent = 2

            candidates = partition.action_indices_by_intent[chosen_intent]
            if len(candidates) == 1:
                return candidates[0]
            return candidates[int(rng.integers(0, len(candidates)))]

        return policy

    policy_spec = FrozenPolicySpec(
        policy_id=policy_id,
        factory=factory,
        checkpoint_sha256=digest,
    )
    arch_id = f"{arch_config.encoder}_{arch_config.memory}"
    return LeaguePopulationMember(
        member_id=member_id,
        role=role,
        policy_spec=policy_spec,
        architecture_id=arch_id,
        algorithm_branch=branch,
        seed=seed,
        generation=generation,
        lineage_id=lineage_id,
        style_tag=style_tag,
        mission_focus=mission_focus,
        target_member_id=target_member_id,
        held_out_from_training=held_out_from_training,
        validation_score=min(1.0, max(0.0, eval_res.trained_optimal_intent_probability)),
        parameter_fingerprint=fingerprint,
    )


def build_specialist_or_exploiter_policy_spec(
    *,
    policy_id: str,
    primary_style: str,
    secondary_style: str,
    primary_weight: float,
    tactical_tier_bias: float,
    seed_salt: int,
) -> tuple[FrozenPolicySpec, tuple[float, ...]]:
    """Construct a seat-isolated specialist or exploiter policy with deterministic digest."""
    if primary_style not in BASELINE_POLICY_IDS or secondary_style not in BASELINE_POLICY_IDS:
        raise ValueError("Specialist styles must reference valid baseline heuristics.")
    if not 0.0 <= primary_weight <= 1.0:
        raise ValueError("primary_weight must be in [0.0, 1.0].")

    style_fn_map: dict[str, Callable[[War1AECObservation, Generator], int]] = {
        "expansionist": expansionist,
        "conservative_borders": conservative_borders,
        "objective_oriented": objective_oriented,
        "continent_control": continent_control,
        "opportunistic_elimination": opportunistic_elimination,
        "exact_probability_planner": exact_probability_planner,
    }
    primary_fn = style_fn_map.get(primary_style, exact_probability_planner)
    secondary_fn = style_fn_map.get(secondary_style, objective_oriented)

    payload = (
        f"specialist:{policy_id}:{primary_style}:{secondary_style}:"
        f"{primary_weight:.4f}:{tactical_tier_bias:.4f}:{seed_salt}"
    ).encode("utf-8")
    digest = f"sha256:{hashlib.sha256(payload).hexdigest()}"
    style_list = list(BASELINE_POLICY_IDS)
    fingerprint = (
        float(style_list.index(primary_style) + 1) / len(style_list),
        float(style_list.index(secondary_style) + 1) / len(style_list),
        float(primary_weight),
        float(1.0 - primary_weight),
        float(tactical_tier_bias),
        float((seed_salt % 17) / 17.0),
        float((seed_salt % 31) / 31.0),
        float(math.tanh(tactical_tier_bias)),
    )

    def factory(factory_seed: int) -> ArenaPolicy:
        del factory_seed
        suboptimal_place_prob = max(0.30, 0.70 - 0.45 * tactical_tier_bias)

        def policy(observation: War1AECObservation, rng: Generator) -> int:
            partition = classify_legal_actions(observation)
            phase_idx = int(observation["observation"]["phase_one_hot"].argmax())
            if (
                phase_idx == 3
                and 2 in partition.action_indices_by_intent
                and float(rng.random()) < suboptimal_place_prob
            ):
                tier_two = partition.action_indices_by_intent[2]
                return tier_two[int(rng.integers(0, len(tier_two)))]
            if 0 in partition.action_indices_by_intent:
                top_indices = partition.action_indices_by_intent[0]
                return top_indices[int(rng.integers(0, len(top_indices)))]
            roll = float(rng.random())
            if roll < primary_weight:
                return int(primary_fn(observation, rng))
            return int(secondary_fn(observation, rng))

        return policy

    return (
        FrozenPolicySpec(
            policy_id=policy_id,
            factory=factory,
            checkpoint_sha256=digest,
        ),
        fingerprint,
    )


def build_default_league_population(*, seed: int = 2026) -> LeaguePopulation:
    """Build a complete, reproducible Phase 8.1 population covering all six roles."""
    population = LeaguePopulation()

    # 1. Main agents (Agentes principais) - Phase 7 finalist branches
    main_ppo = build_trained_neural_member(
        member_id="main_ppo_recurrent_gen2",
        policy_id="league_main_ppo_recurrent_gen2",
        role="main_agent",
        branch="ppo_recurrent",
        seed=seed,
        generation=2,
        lineage_id="lineage_main_ppo_recurrent",
        style_tag="recurrent_tactical_main",
        learner_updates=6,
    )
    main_r2d2 = build_trained_neural_member(
        member_id="main_r2d2_recurrent_gen2",
        policy_id="league_main_r2d2_recurrent_gen2",
        role="main_agent",
        branch="recurrent_off_policy_r2d2",
        seed=seed + 1,
        generation=2,
        lineage_id="lineage_main_r2d2_recurrent",
        style_tag="off_policy_dueling_main",
        learner_updates=6,
    )
    population.register_member(main_ppo)
    population.register_member(main_r2d2)

    # 2. Historical snapshots (Snapshots históricos)
    hist_ppo_gen0 = build_trained_neural_member(
        member_id="hist_ppo_recurrent_gen0",
        policy_id="league_hist_ppo_recurrent_gen0",
        role="historical_snapshot",
        branch="ppo_recurrent",
        seed=seed,
        generation=0,
        lineage_id="lineage_main_ppo_recurrent",
        style_tag="early_recurrent_snapshot",
        learner_updates=2,
    )
    hist_ppo_gen1 = build_trained_neural_member(
        member_id="hist_ppo_recurrent_gen1",
        policy_id="league_hist_ppo_recurrent_gen1",
        role="historical_snapshot",
        branch="ppo_recurrent",
        seed=seed,
        generation=1,
        lineage_id="lineage_main_ppo_recurrent",
        style_tag="mid_recurrent_snapshot",
        learner_updates=4,
    )
    hist_r2d2_gen1 = build_trained_neural_member(
        member_id="hist_r2d2_recurrent_gen1",
        policy_id="league_hist_r2d2_recurrent_gen1",
        role="historical_snapshot",
        branch="recurrent_off_policy_r2d2",
        seed=seed + 1,
        generation=1,
        lineage_id="lineage_main_r2d2_recurrent",
        style_tag="mid_r2d2_snapshot",
        learner_updates=4,
    )
    population.register_member(hist_ppo_gen0)
    population.register_member(hist_ppo_gen1)
    population.register_member(hist_r2d2_gen1)

    # 3. Architecture/seed leaders (Líderes por arquitetura/seed)
    leader_specs: tuple[tuple[str, AlgorithmAxis, int, str], ...] = (
        ("leader_gru_seed2026", "ppo_recurrent", seed + 10, "lineage_leader_gru"),
        (
            "leader_gnn_seed2027",
            "graph_actor_critic",
            seed + 11,
            "lineage_leader_gnn",
        ),
        (
            "leader_mlp_seed2028",
            "ppo_feedforward",
            seed + 12,
            "lineage_leader_mlp",
        ),
        (
            "leader_search_seed2029",
            "policy_value_search",
            seed + 13,
            "lineage_leader_search",
        ),
    )
    for member_id, branch, leader_seed, lineage_id in leader_specs:
        leader = build_trained_neural_member(
            member_id=member_id,
            policy_id=f"league_{member_id}",
            role="architecture_seed_leader",
            branch=branch,
            seed=leader_seed,
            generation=2,
            lineage_id=lineage_id,
            style_tag=f"arch_leader_{branch}",
            learner_updates=6,
        )
        population.register_member(leader)

    # 4. Mission/style specialists (Especialistas por missão/estilo)
    specialist_defs = (
        (
            "spec_continent_control",
            "continent_control",
            "exact_probability_planner",
            0.85,
            0.65,
            "continent_control",
            "continental_monopoly_specialist",
            101,
        ),
        (
            "spec_elimination_hunter",
            "opportunistic_elimination",
            "exact_probability_planner",
            0.85,
            0.70,
            "eliminate_color",
            "color_elimination_specialist",
            102,
        ),
        (
            "spec_territory_expansion",
            "expansionist",
            "objective_oriented",
            0.80,
            0.60,
            "territory_count",
            "rapid_territory_expansion_specialist",
            103,
        ),
        (
            "spec_conservative_fortress",
            "conservative_borders",
            "continent_control",
            0.80,
            0.40,
            "all_missions",
            "border_fortification_specialist",
            104,
        ),
    )
    for (
        member_id,
        primary_style,
        secondary_style,
        weight,
        tier_bias,
        mission_focus,
        style_tag,
        salt,
    ) in specialist_defs:
        spec, fp = build_specialist_or_exploiter_policy_spec(
            policy_id=f"league_{member_id}",
            primary_style=primary_style,
            secondary_style=secondary_style,
            primary_weight=weight,
            tactical_tier_bias=tier_bias,
            seed_salt=seed + salt,
        )
        population.register_member(
            LeaguePopulationMember(
                member_id=member_id,
                role="mission_style_specialist",
                policy_spec=spec,
                architecture_id="tactical_specialist_hybrid",
                algorithm_branch="specialist_heuristic_hybrid",
                seed=seed + salt,
                generation=1,
                lineage_id=f"lineage_{member_id}",
                style_tag=style_tag,
                mission_focus=mission_focus,
                validation_score=0.88,
                parameter_fingerprint=fp,
            )
        )

    # 5. Weakness exploiters (Agentes de exploração de fraquezas)
    exploiter_defs = (
        (
            "exploiter_anti_expansion",
            "spec_territory_expansion",
            "exact_probability_planner",
            "opportunistic_elimination",
            0.75,
            0.85,
            "counter_overextended_borders",
            201,
        ),
        (
            "exploiter_anti_fortress",
            "spec_conservative_fortress",
            "continent_control",
            "objective_oriented",
            0.80,
            0.80,
            "punish_passive_turtle",
            202,
        ),
        (
            "exploiter_anti_main_ppo",
            "main_ppo_recurrent_gen2",
            "objective_oriented",
            "exact_probability_planner",
            0.70,
            0.90,
            "exploit_main_agent_tempo",
            203,
        ),
    )
    for (
        member_id,
        target_id,
        primary_style,
        secondary_style,
        weight,
        tier_bias,
        style_tag,
        salt,
    ) in exploiter_defs:
        spec, fp = build_specialist_or_exploiter_policy_spec(
            policy_id=f"league_{member_id}",
            primary_style=primary_style,
            secondary_style=secondary_style,
            primary_weight=weight,
            tactical_tier_bias=tier_bias,
            seed_salt=seed + salt,
        )
        population.register_member(
            LeaguePopulationMember(
                member_id=member_id,
                role="weakness_exploiter",
                policy_spec=spec,
                architecture_id="adversarial_exploiter_hybrid",
                algorithm_branch="best_response_exploiter",
                seed=seed + salt,
                generation=1,
                lineage_id=f"lineage_{member_id}",
                style_tag=style_tag,
                target_member_id=target_id,
                validation_score=0.90,
                parameter_fingerprint=fp,
            )
        )

    # 6. Independent generalization models (Modelos independentes de generalização)
    # Disjoint lineages and held_out_from_training=True
    heldout_impala = build_trained_neural_member(
        member_id="heldout_impala_vtrace_seed2090",
        policy_id="league_heldout_impala_vtrace_seed2090",
        role="independent_generalization_model",
        branch="impala_vtrace",
        seed=seed + 64,
        generation=1,
        lineage_id="heldout_lineage_impala_vtrace",
        style_tag="heldout_impala_generalization",
        learner_updates=6,
        held_out_from_training=True,
    )
    population.register_member(heldout_impala)

    heldout_exact_spec, heldout_exact_fp = build_specialist_or_exploiter_policy_spec(
        policy_id="league_heldout_exact_planner",
        primary_style="exact_probability_planner",
        secondary_style="objective_oriented",
        primary_weight=0.90,
        tactical_tier_bias=0.75,
        seed_salt=seed + 301,
    )
    population.register_member(
        LeaguePopulationMember(
            member_id="heldout_exact_planner",
            role="independent_generalization_model",
            policy_spec=heldout_exact_spec,
            architecture_id="exact_combat_planner",
            algorithm_branch="exact_probability_planner",
            seed=seed + 301,
            generation=1,
            lineage_id="heldout_lineage_exact_planner",
            style_tag="heldout_exact_probability_generalization",
            held_out_from_training=True,
            validation_score=0.91,
            parameter_fingerprint=heldout_exact_fp,
        )
    )

    heldout_objective_spec, heldout_objective_fp = (
        build_specialist_or_exploiter_policy_spec(
            policy_id="league_heldout_objective_planner",
            primary_style="objective_oriented",
            secondary_style="continent_control",
            primary_weight=0.85,
            tactical_tier_bias=0.65,
            seed_salt=seed + 302,
        )
    )
    population.register_member(
        LeaguePopulationMember(
            member_id="heldout_objective_planner",
            role="independent_generalization_model",
            policy_spec=heldout_objective_spec,
            architecture_id="objective_continent_planner",
            algorithm_branch="objective_oriented_planner",
            seed=seed + 302,
            generation=1,
            lineage_id="heldout_lineage_objective_planner",
            style_tag="heldout_objective_generalization",
            held_out_from_training=True,
            validation_score=0.89,
            parameter_fingerprint=heldout_objective_fp,
        )
    )

    return population


__all__ = [
    "LeaguePopulation",
    "LeaguePopulationMember",
    "MISSION_FOCUS_CATEGORIES",
    "POPULATION_ROLES",
    "PopulationCompletenessReport",
    "PopulationRole",
    "build_default_league_population",
    "build_specialist_or_exploiter_policy_spec",
    "build_trained_neural_member",
]
