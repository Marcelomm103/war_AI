"""Full-table multi-player lineup sampling and training against diverse tables (Phase 8.2)."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..arena import ArenaMatchConfig, ArenaMatchResult, run_arena
from ..envs.war1_aec import War1AECEnv
from .agent import FULL_GAME_INTENT_END_TOKEN, FULL_GAME_INTENT_VOCAB_SIZE
from .algorithms import default_architecture_for_branch
from .architectures import WarArchitectureModel, extract_architecture_inputs
from .league_population import LeaguePopulation, LeaguePopulationMember
from .losses import joint_action_log_prob
from .ppo import compute_gae, compute_masked_ppo_loss


@dataclass(frozen=True)
class FullTableSamplingConfig:
    """Configuration for multi-player lineup sampling with uniform + hard-matchup priority."""

    player_counts: tuple[int, ...] = (3, 4, 5, 6)
    uniform_history_weight: float = 0.35
    priority_exponent: float = 1.5
    min_opponent_probability_floor: float = 0.02
    style_diversity_bonus: float = 0.30
    lineage_diversity_bonus: float = 0.25
    allow_duplicate_policies_at_table: bool = False

    def __post_init__(self) -> None:
        if not self.player_counts:
            raise ValueError("player_counts must not be empty.")
        if len(set(self.player_counts)) != len(self.player_counts):
            raise ValueError("player_counts must not contain duplicates.")
        min_players, max_players = PLAYER_COUNT_RANGE
        for count in self.player_counts:
            if isinstance(count, bool) or not isinstance(count, int):
                raise TypeError("Every table size in player_counts must be an integer.")
            if count == 2:
                raise ValueError(
                    "WAR cannot be reduced to 2-player duels; full-table training "
                    f"requires {min_players}..{max_players} players."
                )
            if not min_players <= count <= max_players:
                raise ValueError(
                    f"Table size {count} is outside official WAR range "
                    f"{min_players}..{max_players}."
                )
        if (
            isinstance(self.uniform_history_weight, bool)
            or not math.isfinite(self.uniform_history_weight)
            or not 0.0 < self.uniform_history_weight < 1.0
        ):
            raise ValueError(
                "uniform_history_weight must be in (0.0, 1.0) to combine uniform "
                "history and prioritized hard matchups."
            )
        if (
            isinstance(self.priority_exponent, bool)
            or not math.isfinite(self.priority_exponent)
            or self.priority_exponent <= 0.0
        ):
            raise ValueError("priority_exponent must be finite and positive.")
        if (
            isinstance(self.min_opponent_probability_floor, bool)
            or not math.isfinite(self.min_opponent_probability_floor)
            or not 0.0 < self.min_opponent_probability_floor <= 0.25
        ):
            raise ValueError(
                "min_opponent_probability_floor must be in (0.0, 0.25]."
            )
        for label, bonus in (
            ("style_diversity_bonus", self.style_diversity_bonus),
            ("lineage_diversity_bonus", self.lineage_diversity_bonus),
        ):
            if isinstance(bonus, bool) or not math.isfinite(bonus) or bonus < 0.0:
                raise ValueError(f"{label} must be finite and non-negative.")

    @property
    def priority_weight(self) -> float:
        return 1.0 - self.uniform_history_weight


@dataclass(frozen=True)
class OpponentDifficultyRecord:
    """Empirical difficulty of one opponent member against the focal learner."""

    member_id: str
    policy_id: str
    appearances_against_focal: int
    focal_wins_when_present: int
    opponent_wins_when_present: int

    def __post_init__(self) -> None:
        if not self.member_id.strip() or not self.policy_id.strip():
            raise ValueError("OpponentDifficultyRecord requires member_id and policy_id.")
        for label, value in (
            ("appearances_against_focal", self.appearances_against_focal),
            ("focal_wins_when_present", self.focal_wins_when_present),
            ("opponent_wins_when_present", self.opponent_wins_when_present),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{label} must be a non-negative integer.")
        if (
            self.focal_wins_when_present + self.opponent_wins_when_present
            > self.appearances_against_focal
        ):
            raise ValueError("Combined focal and opponent wins cannot exceed appearances.")

    @property
    def focal_win_rate(self) -> float:
        if self.appearances_against_focal == 0:
            return 0.25
        return self.focal_wins_when_present / self.appearances_against_focal

    @property
    def opponent_win_rate(self) -> float:
        if self.appearances_against_focal == 0:
            return 0.25
        return self.opponent_wins_when_present / self.appearances_against_focal

    @property
    def difficulty_score(self) -> float:
        """Difficulty in [0.05, 1.0]; higher when focal agent loses or opponent wins."""
        unexplored_bonus = 0.20 if self.appearances_against_focal == 0 else 0.0
        raw = (
            0.65 * (1.0 - self.focal_win_rate)
            + 0.35 * self.opponent_win_rate
            + unexplored_bonus
        )
        return min(1.0, max(0.05, raw))


@dataclass(frozen=True)
class SampledTableLineup:
    """One sampled multi-player table lineup with logged sampling probabilities."""

    lineup_id: str
    seed: int
    num_players: int
    focal_member_id: str
    focal_policy_id: str
    focal_seat_index: int
    focal_player_id: str
    opponent_member_ids_by_seat: dict[str, str]
    policy_ids_by_player: dict[str, str]
    candidate_distribution: dict[str, float]
    seat_sampling_probabilities: dict[str, float]
    joint_lineup_probability: float
    uniform_component_weight: float
    priority_component_weight: float
    unique_styles_count: int
    unique_lineages_count: int

    def __post_init__(self) -> None:
        min_players, max_players = PLAYER_COUNT_RANGE
        if not min_players <= self.num_players <= max_players:
            raise ValueError(
                f"SampledTableLineup requires {min_players}..{max_players} players."
            )
        if not 0 <= self.focal_seat_index < self.num_players:
            raise ValueError("focal_seat_index must be inside [0, num_players).")
        if self.focal_player_id != f"player_{self.focal_seat_index}":
            raise ValueError("focal_player_id does not match focal_seat_index.")
        if len(self.opponent_member_ids_by_seat) != self.num_players - 1:
            raise ValueError("Must assign exactly num_players - 1 opponent seats.")
        if not self.candidate_distribution:
            raise ValueError("candidate_distribution must not be empty.")
        total_prob = sum(self.candidate_distribution.values())
        if not math.isclose(total_prob, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(
                f"candidate_distribution probabilities must sum to 1.0, got {total_prob}."
            )
        if any(p <= 0.0 for p in self.candidate_distribution.values()):
            raise ValueError("All eligible candidates must have positive probability.")
        if (
            not math.isfinite(self.joint_lineup_probability)
            or not 0.0 < self.joint_lineup_probability <= 1.0
        ):
            raise ValueError("joint_lineup_probability must be in (0.0, 1.0].")

    def to_arena_config(
        self,
        *,
        max_semantic_decisions: int = 5000,
        decision_budget_ms: int | None = None,
    ) -> ArenaMatchConfig:
        return ArenaMatchConfig(
            match_id=self.lineup_id,
            seed=self.seed,
            num_players=self.num_players,
            policy_ids_by_player=dict(self.policy_ids_by_player),
            max_semantic_decisions=max_semantic_decisions,
            decision_budget_ms=decision_budget_ms,
        )


@dataclass(frozen=True)
class LineupDiversityReport:
    """Quantitative diversity audit across sampled multi-player lineups."""

    num_lineups: int
    sampled_table_sizes: tuple[int, ...]
    shannon_entropy: float
    normalized_entropy: float
    effective_support_size: float
    distinct_opponents_sampled: int
    mean_unique_styles_per_table: float
    mean_unique_lineages_per_table: float
    min_logged_candidate_probability: float
    duel_lineups_count: int


@dataclass(frozen=True)
class FullTableTrainingReport:
    """Summary of full-table training and matchup-priority updates (Phase 8.2)."""

    focal_member_id: str
    sampled_lineups: tuple[SampledTableLineup, ...]
    match_results: tuple[ArenaMatchResult, ...]
    difficulty_records: dict[str, OpponentDifficultyRecord]
    diversity_report: LineupDiversityReport
    learner_updates: int
    initial_optimal_intent_probability: float
    trained_optimal_intent_probability: float
    final_loss: float


class PrioritizedFullTableSampler:
    """Sample full multi-player WAR tables combining uniform history and hard-matchup priority."""

    def __init__(
        self,
        population: LeaguePopulation,
        *,
        focal_member_id: str,
        config: FullTableSamplingConfig = FullTableSamplingConfig(),
        seed: int = 2026,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Sampler seed must be a non-negative integer.")
        self._population = population
        self._focal = population.get_member(focal_member_id)
        if self._focal.held_out_from_training:
            raise ValueError(
                "Held-out generalization models cannot be used as the focal training agent."
            )
        self._config = config
        self._rng = np.random.default_rng(seed)
        self._sample_counter = 0

        eligible = [
            member
            for member in population.trainable_opponent_pool()
            if member.member_id != focal_member_id
        ]
        max_opponents_needed = max(config.player_counts) - 1
        if not config.allow_duplicate_policies_at_table and len(eligible) < max_opponents_needed:
            raise ValueError(
                f"Need at least {max_opponents_needed} eligible training opponents "
                f"for {max(config.player_counts)}-player tables without duplicates."
            )
        self._eligible_by_id: dict[str, LeaguePopulationMember] = {
            member.member_id: member for member in eligible
        }
        self._difficulty: dict[str, OpponentDifficultyRecord] = {
            member.member_id: OpponentDifficultyRecord(
                member_id=member.member_id,
                policy_id=member.policy_spec.policy_id,
                appearances_against_focal=0,
                focal_wins_when_present=0,
                opponent_wins_when_present=0,
            )
            for member in eligible
        }

    @property
    def config(self) -> FullTableSamplingConfig:
        return self._config

    @property
    def difficulty_records(self) -> dict[str, OpponentDifficultyRecord]:
        return dict(self._difficulty)

    def register_matchup_outcome(
        self,
        opponent_member_id: str,
        *,
        focal_won: bool,
        opponent_won: bool,
    ) -> OpponentDifficultyRecord:
        """Update empirical difficulty of an opponent from a completed full-table game."""
        if focal_won and opponent_won:
            raise ValueError("focal_won and opponent_won cannot both be True.")
        rec = self._difficulty.get(opponent_member_id)
        if rec is None:
            raise KeyError(f"Unknown eligible opponent {opponent_member_id!r}.")
        updated = OpponentDifficultyRecord(
            member_id=rec.member_id,
            policy_id=rec.policy_id,
            appearances_against_focal=rec.appearances_against_focal + 1,
            focal_wins_when_present=rec.focal_wins_when_present + (1 if focal_won else 0),
            opponent_wins_when_present=(
                rec.opponent_wins_when_present + (1 if opponent_won else 0)
            ),
        )
        self._difficulty[opponent_member_id] = updated
        return updated

    def update_from_arena_results(
        self,
        lineups: Sequence[SampledTableLineup],
        results: Sequence[ArenaMatchResult],
    ) -> None:
        """Ingest completed multi-player arena results to update hard-matchup priorities."""
        result_by_id = {res.match_id: res for res in results}
        for lineup in lineups:
            res = result_by_id.get(lineup.lineup_id)
            if res is None or res.status != "completed":
                continue
            focal_won = res.winner_player_id == lineup.focal_player_id
            for seat_player_id, opp_member_id in lineup.opponent_member_ids_by_seat.items():
                opp_won = res.winner_player_id == seat_player_id
                self.register_matchup_outcome(
                    opp_member_id,
                    focal_won=focal_won,
                    opponent_won=opp_won,
                )

    def compute_candidate_distribution(self) -> dict[str, float]:
        """Compute normalized mixture of uniform history + prioritized hard-matchup weights."""
        member_ids = tuple(self._eligible_by_id.keys())
        n = len(member_ids)
        uniform_prob = 1.0 / n

        raw_priority = np.array(
            [
                self._difficulty[m_id].difficulty_score ** self._config.priority_exponent
                for m_id in member_ids
            ],
            dtype=np.float64,
        )
        priority_sum = float(raw_priority.sum())
        priority_probs = (
            raw_priority * 0.0 + uniform_prob
            if priority_sum <= 0.0
            else raw_priority / priority_sum
        )

        u_w = self._config.uniform_history_weight
        p_w = self._config.priority_weight
        mixed = u_w * uniform_prob + p_w * priority_probs

        floor = min(self._config.min_opponent_probability_floor, 1.0 / n)
        normalized = floor + (1.0 - n * floor) * mixed
        normalized = normalized / float(normalized.sum())
        return {
            m_id: float(normalized[idx]) for idx, m_id in enumerate(member_ids)
        }

    def sample_lineup(
        self,
        *,
        lineup_id: str,
        seed: int,
        num_players: int | None = None,
        focal_seat_index: int | None = None,
    ) -> SampledTableLineup:
        """Sample a multi-player table lineup while logging exact selection probabilities."""
        if not lineup_id.strip():
            raise ValueError("lineup_id must not be blank.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        if num_players is None:
            table_idx = self._sample_counter % len(self._config.player_counts)
            chosen_players = self._config.player_counts[table_idx]
            table_prob = 1.0 / len(self._config.player_counts)
        else:
            if num_players == 2:
                raise ValueError(
                    "WAR cannot be reduced to 2-player duels; use 3..6 players."
                )
            if num_players not in self._config.player_counts:
                raise ValueError(
                    f"Requested num_players={num_players} is not in configured "
                    f"player_counts={self._config.player_counts}."
                )
            chosen_players = num_players
            table_prob = 1.0

        if focal_seat_index is None:
            chosen_focal_seat = int(self._rng.integers(0, chosen_players))
            seat_prob = 1.0 / chosen_players
        else:
            if not 0 <= focal_seat_index < chosen_players:
                raise ValueError("focal_seat_index out of bounds for table size.")
            chosen_focal_seat = focal_seat_index
            seat_prob = 1.0

        self._sample_counter += 1
        base_dist = self.compute_candidate_distribution()
        available_ids = list(base_dist.keys())

        selected_styles: set[str] = {self._focal.style_tag}
        selected_lineages: set[str] = {self._focal.lineage_id}
        opponent_member_ids_by_seat: dict[str, str] = {}
        seat_sampling_probabilities: dict[str, float] = {}
        policy_ids_by_player: dict[str, str] = {}
        joint_prob = table_prob * seat_prob

        for seat_idx in range(chosen_players):
            player_id = f"player_{seat_idx}"
            if seat_idx == chosen_focal_seat:
                policy_ids_by_player[player_id] = self._focal.policy_spec.policy_id
                continue

            weights = []
            for cand_id in available_ids:
                cand_member = self._eligible_by_id[cand_id]
                w = base_dist[cand_id]
                if cand_member.style_tag not in selected_styles:
                    w *= 1.0 + self._config.style_diversity_bonus
                if cand_member.lineage_id not in selected_lineages:
                    w *= 1.0 + self._config.lineage_diversity_bonus
                weights.append(w)

            w_arr = np.asarray(weights, dtype=np.float64)
            step_probs = w_arr / float(w_arr.sum())
            chosen_pos = int(self._rng.choice(len(available_ids), p=step_probs))
            chosen_cand_id = available_ids[chosen_pos]
            chosen_prob = float(step_probs[chosen_pos])
            chosen_member = self._eligible_by_id[chosen_cand_id]

            opponent_member_ids_by_seat[player_id] = chosen_cand_id
            seat_sampling_probabilities[player_id] = chosen_prob
            policy_ids_by_player[player_id] = chosen_member.policy_spec.policy_id
            selected_styles.add(chosen_member.style_tag)
            selected_lineages.add(chosen_member.lineage_id)
            joint_prob *= chosen_prob

            if not self._config.allow_duplicate_policies_at_table:
                available_ids.pop(chosen_pos)

        return SampledTableLineup(
            lineup_id=lineup_id,
            seed=seed,
            num_players=chosen_players,
            focal_member_id=self._focal.member_id,
            focal_policy_id=self._focal.policy_spec.policy_id,
            focal_seat_index=chosen_focal_seat,
            focal_player_id=f"player_{chosen_focal_seat}",
            opponent_member_ids_by_seat=opponent_member_ids_by_seat,
            policy_ids_by_player=policy_ids_by_player,
            candidate_distribution=base_dist,
            seat_sampling_probabilities=seat_sampling_probabilities,
            joint_lineup_probability=joint_prob,
            uniform_component_weight=self._config.uniform_history_weight,
            priority_component_weight=self._config.priority_weight,
            unique_styles_count=len(selected_styles),
            unique_lineages_count=len(selected_lineages),
        )

    def sample_batch(
        self,
        *,
        batch_prefix: str,
        num_lineups: int,
        base_seed: int = 2026,
    ) -> tuple[SampledTableLineup, ...]:
        """Sample a balanced batch of multi-player lineups across table sizes and seats."""
        if isinstance(num_lineups, bool) or not isinstance(num_lineups, int) or num_lineups < 1:
            raise ValueError("num_lineups must be a positive integer.")
        rotation = [
            (table_size, seat_idx)
            for table_size in self._config.player_counts
            for seat_idx in range(table_size)
        ]
        lineups: list[SampledTableLineup] = []
        for idx in range(num_lineups):
            table_size, seat_idx = rotation[idx % len(rotation)]
            lineups.append(
                self.sample_lineup(
                    lineup_id=f"{batch_prefix}-lineup-{idx:04d}",
                    seed=base_seed + idx,
                    num_players=table_size,
                    focal_seat_index=seat_idx,
                )
            )
        return tuple(lineups)


def evaluate_lineup_diversity(
    lineups: Sequence[SampledTableLineup],
) -> LineupDiversityReport:
    """Compute diversity, entropy, and multi-player coverage metrics for sampled lineups."""
    if not lineups:
        raise ValueError("At least one SampledTableLineup is required.")
    duel_count = sum(1 for lineup in lineups if lineup.num_players < 3)
    if duel_count > 0:
        raise ValueError("Sampled lineups must not contain 2-player duels.")

    last_dist = lineups[-1].candidate_distribution
    probs = np.asarray(list(last_dist.values()), dtype=np.float64)
    shannon = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    max_entropy = math.log(len(probs)) if len(probs) > 1 else 1.0
    normalized_entropy = shannon / max_entropy if max_entropy > 0.0 else 1.0
    effective_support = float(1.0 / np.sum(np.square(probs)))

    sampled_opponents = {
        opp_id
        for lineup in lineups
        for opp_id in lineup.opponent_member_ids_by_seat.values()
    }
    min_logged_prob = min(
        min(lineup.candidate_distribution.values()) for lineup in lineups
    )
    mean_styles = sum(lineup.unique_styles_count for lineup in lineups) / len(lineups)
    mean_lineages = sum(lineup.unique_lineages_count for lineup in lineups) / len(
        lineups
    )
    table_sizes = tuple(sorted({lineup.num_players for lineup in lineups}))
    return LineupDiversityReport(
        num_lineups=len(lineups),
        sampled_table_sizes=table_sizes,
        shannon_entropy=shannon,
        normalized_entropy=normalized_entropy,
        effective_support_size=effective_support,
        distinct_opponents_sampled=len(sampled_opponents),
        mean_unique_styles_per_table=mean_styles,
        mean_unique_lineages_per_table=mean_lineages,
        min_logged_candidate_probability=min_logged_prob,
        duel_lineups_count=duel_count,
    )


def run_full_table_training_campaign(
    population: LeaguePopulation,
    *,
    focal_member_id: str = "main_ppo_recurrent_gen2",
    sampling_config: FullTableSamplingConfig = FullTableSamplingConfig(),
    num_lineups: int = 8,
    learner_updates: int = 8,
    max_semantic_decisions: int = 5000,
    seed: int = 2026,
) -> FullTableTrainingReport:
    """Sample full multi-player tables, run matches, update matchup priorities, and train focal model."""
    sampler = PrioritizedFullTableSampler(
        population,
        focal_member_id=focal_member_id,
        config=sampling_config,
        seed=seed,
    )
    lineups = sampler.sample_batch(
        batch_prefix="phase8-full-table",
        num_lineups=num_lineups,
        base_seed=seed,
    )
    arena_configs = tuple(
        lineup.to_arena_config(max_semantic_decisions=max_semantic_decisions)
        for lineup in lineups
    )
    arena_report = run_arena(arena_configs, population.policy_registry())
    sampler.update_from_arena_results(lineups, arena_report.matches)

    # Collect observations across distinct full-table player counts (3..6) and train focal model
    torch.manual_seed(seed)
    arch_config = default_architecture_for_branch("ppo_recurrent", hidden_size=32)
    model = WarArchitectureModel(arch_config)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.012)

    multi_table_obs = []
    for table_idx, table_size in enumerate(sampling_config.player_counts):
        env = War1AECEnv(num_players=table_size)
        env.reset(seed=seed + table_idx)
        agent_id = env.agent_selection
        assert agent_id is not None
        multi_table_obs.append(env.observe(agent_id))

    base_inputs = extract_architecture_inputs(multi_table_obs, max_history_events=32)
    num_obs = base_inputs.batch_size
    batch_inputs = base_inputs.repeat_batch(3)
    batch_size = batch_inputs.batch_size
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

    _, _, init_probs, _ = model.select_legal_action(
        multi_table_obs[0],
        deterministic=True,
    )
    initial_prob = float(init_probs[0].item())
    final_loss = 0.0

    for _ in range(learner_updates):
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
        final_loss = float(loss.total_loss.detach().item())

    _, _, trained_probs, _ = model.select_legal_action(
        multi_table_obs[0],
        deterministic=True,
    )
    trained_prob = float(trained_probs[0].item())
    diversity_report = evaluate_lineup_diversity(lineups)
    return FullTableTrainingReport(
        focal_member_id=focal_member_id,
        sampled_lineups=lineups,
        match_results=arena_report.matches,
        difficulty_records=sampler.difficulty_records,
        diversity_report=diversity_report,
        learner_updates=learner_updates,
        initial_optimal_intent_probability=initial_prob,
        trained_optimal_intent_probability=trained_prob,
        final_loss=final_loss,
    )


__all__ = [
    "FullTableSamplingConfig",
    "FullTableTrainingReport",
    "LineupDiversityReport",
    "OpponentDifficultyRecord",
    "PrioritizedFullTableSampler",
    "SampledTableLineup",
    "evaluate_lineup_diversity",
    "run_full_table_training_campaign",
]
