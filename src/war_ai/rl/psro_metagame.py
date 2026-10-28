"""PSRO oracle training, empirical multi-player metagame analysis, rankings, and cycle detection (Phase 8.3)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..arena import ArenaMatchConfig, ArenaMatchResult, FrozenPolicySpec, run_arena
from ..envs.war1_aec import War1AECEnv
from .agent import FULL_GAME_INTENT_END_TOKEN, FULL_GAME_INTENT_VOCAB_SIZE
from .algorithms import default_architecture_for_branch
from .architectures import WarArchitectureModel, extract_architecture_inputs
from .league_population import (
    LeaguePopulation,
    LeaguePopulationMember,
    build_trained_neural_member,
)
from .losses import joint_action_log_prob
from .ppo import compute_gae, compute_masked_ppo_loss


@dataclass(frozen=True)
class LineupPayoffRecord:
    """Empirical multi-player payoff statistics for one lineup composition."""

    lineup_key: str
    num_players: int
    policy_ids: tuple[str, ...]
    matches_played: int
    completed_matches: int
    wins_by_policy: dict[str, int]
    seat_appearances_by_policy: dict[str, int]
    win_rate_by_policy: dict[str, float]
    excess_payoff_by_policy: dict[str, float]

    def __post_init__(self) -> None:
        min_players, max_players = PLAYER_COUNT_RANGE
        if not min_players <= self.num_players <= max_players:
            raise ValueError(
                f"LineupPayoffRecord requires {min_players}..{max_players} players."
            )
        if len(self.policy_ids) != self.num_players:
            raise ValueError("policy_ids length must match num_players.")
        if self.completed_matches > self.matches_played or self.matches_played < 1:
            raise ValueError("Invalid match counts in LineupPayoffRecord.")


@dataclass(frozen=True)
class MetaStrategyProfile:
    """Probability mixture over policies in the empirical metagame."""

    method: str
    probabilities_by_policy: dict[str, float]
    expected_payoff_by_policy: dict[str, float]
    meta_value: float
    nash_conv_regret: float
    regret_by_policy: dict[str, float]

    def __post_init__(self) -> None:
        if not self.probabilities_by_policy:
            raise ValueError("MetaStrategyProfile requires at least one policy.")
        total = sum(self.probabilities_by_policy.values())
        if not math.isclose(total, 1.0, rel_tol=1e-5, abs_tol=1e-5):
            raise ValueError(f"Meta-strategy probabilities must sum to 1.0, got {total}.")
        if any(p < 0.0 for p in self.probabilities_by_policy.values()):
            raise ValueError("Meta-strategy probabilities must be non-negative.")


@dataclass(frozen=True)
class DominanceCycle:
    """Intransitive 3-cycle A -> B -> C -> A in the empirical metagame."""

    cycle_policies: tuple[str, str, str]
    edge_win_rates: tuple[float, float, float]
    edge_margins: tuple[float, float, float]
    cycle_strength: float

    def __post_init__(self) -> None:
        if len(set(self.cycle_policies)) != 3:
            raise ValueError("DominanceCycle requires three distinct policies.")
        if any(m <= 0.0 for m in self.edge_margins) or self.cycle_strength <= 0.0:
            raise ValueError("DominanceCycle margins and strength must be positive.")


@dataclass(frozen=True)
class MetagameRankingEntry:
    """Multi-method ranking entry demonstrating multiplayer rankings vs pairwise Elo."""

    policy_id: str
    alpharank_mass: float
    replicator_mass: float
    expected_multiplayer_win_rate: float
    regret_to_best_response: float
    pairwise_elo: float
    multiplayer_composite_rank: int
    pairwise_elo_rank: int


@dataclass(frozen=True)
class PSROOracleResult:
    """Result of training a PSRO best-response oracle against a policy mixture."""

    iteration: int
    target_mixture: dict[str, float]
    oracle_member: LeaguePopulationMember
    sampled_table_configs: tuple[ArenaMatchConfig, ...]
    initial_optimal_intent_probability: float
    trained_optimal_intent_probability: float
    empirical_win_rate_vs_mixture: float
    payoff_gain_over_meta_value: float


@dataclass(frozen=True)
class PSROMetagameReport:
    """Comprehensive Phase 8.3 PSRO and empirical metagame report."""

    policy_ids: tuple[str, ...]
    lineup_payoffs: tuple[LineupPayoffRecord, ...]
    uniform_meta_strategy: MetaStrategyProfile
    replicator_meta_strategy: MetaStrategyProfile
    alpharank_meta_strategy: MetaStrategyProfile
    rankings: tuple[MetagameRankingEntry, ...]
    detected_cycles: tuple[DominanceCycle, ...]
    oracle_results: tuple[PSROOracleResult, ...]
    elo_alone_insufficient: bool


def canonical_lineup_key(policy_ids: Sequence[str]) -> tuple[str, tuple[str, ...]]:
    """Build a canonical multiset key for a multi-player table lineup."""
    min_players, max_players = PLAYER_COUNT_RANGE
    if not min_players <= len(policy_ids) <= max_players:
        raise ValueError(
            f"WAR lineups require {min_players}..{max_players} players, got {len(policy_ids)}."
        )
    sorted_ids = tuple(sorted(policy_ids))
    return f"{len(sorted_ids)}p:" + "|".join(sorted_ids), sorted_ids


class EmpiricalMetagameTable:
    """Empirical payoff store over multi-player WAR lineups."""

    def __init__(self, policy_ids: Sequence[str]) -> None:
        unique_ids = tuple(dict.fromkeys(policy_ids))
        if len(unique_ids) < 3:
            raise ValueError("EmpiricalMetagameTable requires at least 3 distinct policies.")
        self._policy_ids = unique_ids
        self._matches_by_key: dict[str, int] = {}
        self._completed_by_key: dict[str, int] = {}
        self._tuple_by_key: dict[str, tuple[str, ...]] = {}
        self._wins_by_key: dict[str, dict[str, int]] = {}
        self._seats_by_key: dict[str, dict[str, int]] = {}

    @property
    def policy_ids(self) -> tuple[str, ...]:
        return self._policy_ids

    def record_lineup_outcome(
        self,
        seat_policy_ids: Sequence[str],
        *,
        winner_policy_id: str | None,
        completed: bool = True,
    ) -> LineupPayoffRecord:
        """Record one multi-player game outcome into its canonical lineup bucket."""
        for pid in seat_policy_ids:
            if pid not in self._policy_ids:
                raise ValueError(f"Policy {pid!r} is not registered in metagame table.")
        key, sorted_ids = canonical_lineup_key(seat_policy_ids)
        if key not in self._tuple_by_key:
            self._tuple_by_key[key] = sorted_ids
            self._matches_by_key[key] = 0
            self._completed_by_key[key] = 0
            self._wins_by_key[key] = {pid: 0 for pid in set(sorted_ids)}
            self._seats_by_key[key] = {pid: 0 for pid in set(sorted_ids)}

        self._matches_by_key[key] += 1
        if completed:
            if winner_policy_id is None or winner_policy_id not in seat_policy_ids:
                raise ValueError("Completed matches require a valid winner_policy_id.")
            self._completed_by_key[key] += 1
            for pid in seat_policy_ids:
                self._seats_by_key[key][pid] += 1
            self._wins_by_key[key][winner_policy_id] += 1
        return self.get_lineup_record(key)

    def record_arena_matches(
        self,
        matches: Sequence[ArenaMatchResult],
    ) -> tuple[LineupPayoffRecord, ...]:
        """Ingest a sequence of ArenaMatchResult objects into the empirical metagame."""
        for match in matches:
            seat_pids = [
                match.policy_ids_by_player[f"player_{i}"]
                for i in range(match.num_players)
            ]
            winner_pid = (
                match.policy_ids_by_player[match.winner_player_id]
                if match.status == "completed" and match.winner_player_id is not None
                else None
            )
            self.record_lineup_outcome(
                seat_pids,
                winner_policy_id=winner_pid,
                completed=(match.status == "completed"),
            )
        return self.lineup_records()

    def get_lineup_record(self, lineup_key: str) -> LineupPayoffRecord:
        sorted_ids = self._tuple_by_key[lineup_key]
        num_players = len(sorted_ids)
        wins = dict(self._wins_by_key[lineup_key])
        seats = dict(self._seats_by_key[lineup_key])
        fair_share = 1.0 / num_players
        win_rates: dict[str, float] = {}
        excess: dict[str, float] = {}
        for pid in sorted(set(sorted_ids)):
            s_count = seats[pid]
            wr = wins[pid] / s_count if s_count > 0 else fair_share
            win_rates[pid] = wr
            excess[pid] = wr - fair_share
        return LineupPayoffRecord(
            lineup_key=lineup_key,
            num_players=num_players,
            policy_ids=sorted_ids,
            matches_played=self._matches_by_key[lineup_key],
            completed_matches=self._completed_by_key[lineup_key],
            wins_by_policy=wins,
            seat_appearances_by_policy=seats,
            win_rate_by_policy=win_rates,
            excess_payoff_by_policy=excess,
        )

    def lineup_records(self) -> tuple[LineupPayoffRecord, ...]:
        return tuple(
            self.get_lineup_record(k) for k in sorted(self._tuple_by_key.keys())
        )

    def pairwise_dominance_matrix(self) -> NDArray[np.float64]:
        """Compute empirical relative win-rate matrix M[i, j] when i and j share a table."""
        k = len(self._policy_ids)
        idx_by_pid = {pid: i for i, pid in enumerate(self._policy_ids)}
        win_sum: NDArray[np.float64] = np.zeros((k, k), dtype=np.float64)
        weight_sum: NDArray[np.float64] = np.zeros((k, k), dtype=np.float64)

        for record in self.lineup_records():
            if record.completed_matches == 0:
                continue
            unique_pids = sorted(set(record.policy_ids))
            for i_pid in unique_pids:
                for j_pid in unique_pids:
                    if i_pid == j_pid:
                        continue
                    i_idx = idx_by_pid[i_pid]
                    j_idx = idx_by_pid[j_pid]
                    wr_i = record.win_rate_by_policy[i_pid]
                    wr_j = record.win_rate_by_policy[j_pid]
                    denom = wr_i + wr_j
                    rel_i = wr_i / denom if denom > 0.0 else 0.5
                    w = float(record.completed_matches)
                    win_sum[i_idx, j_idx] += w * rel_i
                    weight_sum[i_idx, j_idx] += w

        matrix: NDArray[np.float64] = np.full((k, k), 0.5, dtype=np.float64)
        np.fill_diagonal(matrix, 0.5)
        mask = weight_sum > 0.0
        matrix[mask] = win_sum[mask] / weight_sum[mask]
        return matrix

    def evaluate_meta_strategy(
        self,
        probabilities_by_policy: Mapping[str, float],
        *,
        method: str = "custom",
    ) -> MetaStrategyProfile:
        """Evaluate expected multi-player payoffs and regret under a policy mixture."""
        probs = np.array(
            [float(probabilities_by_policy.get(pid, 0.0)) for pid in self._policy_ids],
            dtype=np.float64,
        )
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum())
        if total <= 0.0:
            raise ValueError("Policy mixture must have positive total mass.")
        probs = probs / total
        prob_map = {pid: float(probs[i]) for i, pid in enumerate(self._policy_ids)}

        records = [r for r in self.lineup_records() if r.completed_matches > 0]
        expected_payoffs: dict[str, float] = {}
        if not records:
            for pid in self._policy_ids:
                expected_payoffs[pid] = 1.0 / 3.0
        else:
            for focal_pid in self._policy_ids:
                weighted_num = 0.0
                weighted_den = 0.0
                for rec in records:
                    if focal_pid not in rec.win_rate_by_policy:
                        continue
                    opp_list = list(rec.policy_ids)
                    opp_list.remove(focal_pid)
                    lineup_likelihood = 1.0
                    for opp_pid in opp_list:
                        lineup_likelihood *= max(prob_map[opp_pid], 1e-4)
                    weight = lineup_likelihood * rec.completed_matches
                    weighted_num += weight * rec.win_rate_by_policy[focal_pid]
                    weighted_den += weight
                if weighted_den > 0.0:
                    expected_payoffs[focal_pid] = weighted_num / weighted_den
                else:
                    expected_payoffs[focal_pid] = 1.0 / 3.0

        meta_value = sum(
            prob_map[pid] * expected_payoffs[pid] for pid in self._policy_ids
        )
        best_payoff = max(expected_payoffs.values())
        nash_conv = max(0.0, best_payoff - meta_value)
        regret_by_policy = {
            pid: max(0.0, best_payoff - expected_payoffs[pid])
            for pid in self._policy_ids
        }
        return MetaStrategyProfile(
            method=method,
            probabilities_by_policy=prob_map,
            expected_payoff_by_policy=expected_payoffs,
            meta_value=meta_value,
            nash_conv_regret=nash_conv,
            regret_by_policy=regret_by_policy,
        )

    def solve_replicator_meta_strategy(
        self,
        *,
        iterations: int = 250,
        step_size: float = 0.25,
    ) -> MetaStrategyProfile:
        """Compute an equilibrium meta-strategy using multi-player replicator dynamics."""
        k = len(self._policy_ids)
        sigma = np.full(k, 1.0 / k, dtype=np.float64)
        for _ in range(iterations):
            profile = self.evaluate_meta_strategy(
                {pid: float(sigma[i]) for i, pid in enumerate(self._policy_ids)},
                method="replicator_dynamics",
            )
            payoffs = np.array(
                [profile.expected_payoff_by_policy[pid] for pid in self._policy_ids],
                dtype=np.float64,
            )
            mean_payoff = float(np.dot(sigma, payoffs))
            advantage = payoffs - mean_payoff
            sigma = sigma * np.exp(step_size * advantage)
            sigma = np.maximum(sigma, 1e-6)
            sigma = sigma / float(sigma.sum())
        return self.evaluate_meta_strategy(
            {pid: float(sigma[i]) for i, pid in enumerate(self._policy_ids)},
            method="replicator_dynamics",
        )

    def solve_alpharank_meta_strategy(
        self,
        *,
        alpha: float = 12.0,
        population_size: int = 20,
    ) -> MetaStrategyProfile:
        """Compute finite-population Alpha-Rank stationary distribution over policies."""
        dom = self.pairwise_dominance_matrix()
        k = len(self._policy_ids)
        transition = np.zeros((k, k), dtype=np.float64)
        eta = 1.0 / float(k - 1)

        for i in range(k):
            for j in range(k):
                if i == j:
                    continue
                advantage_j_over_i = dom[j, i] - dom[i, j]
                if abs(advantage_j_over_i) <= 1e-9:
                    fixation = 1.0 / float(population_size)
                else:
                    u = -alpha * advantage_j_over_i
                    exp_u = math.exp(max(-40.0, min(40.0, u)))
                    exp_mu = math.exp(max(-40.0, min(40.0, population_size * u)))
                    denom = 1.0 - exp_mu
                    fixation = (1.0 - exp_u) / denom if abs(denom) > 1e-12 else 0.0
                transition[i, j] = eta * max(0.0, min(1.0, fixation))
            transition[i, i] = max(0.0, 1.0 - float(transition[i].sum()))

        stationary: NDArray[np.float64] = np.ones(k, dtype=np.float64) / float(k)
        for _ in range(400):
            stationary = np.matmul(stationary, transition)
            stationary = np.maximum(stationary, 1e-8)
            stationary = stationary / float(stationary.sum())

        return self.evaluate_meta_strategy(
            {pid: float(stationary[i]) for i, pid in enumerate(self._policy_ids)},
            method="alpharank",
        )

    def compute_pairwise_elo(
        self,
        *,
        base_elo: float = 1500.0,
        scale: float = 400.0,
    ) -> dict[str, float]:
        """Compute classic 2-player pairwise Elo projection from pairwise dominance."""
        dom = self.pairwise_dominance_matrix()
        k = len(self._policy_ids)
        elos: dict[str, float] = {}
        for i, pid in enumerate(self._policy_ids):
            mean_vs_others = (float(dom[i].sum()) - 0.5) / float(k - 1)
            clipped = min(0.99, max(0.01, mean_vs_others))
            logit = math.log10(clipped / (1.0 - clipped))
            elos[pid] = base_elo + scale * logit
        return elos

    def compute_rankings(self) -> tuple[MetagameRankingEntry, ...]:
        """Compare Alpha-Rank, Replicator, Expected Table Win Rate, and Pairwise Elo."""
        rep = self.solve_replicator_meta_strategy()
        alpha = self.solve_alpharank_meta_strategy()
        elos = self.compute_pairwise_elo()

        composite_scores = {
            pid: (
                0.40 * alpha.probabilities_by_policy[pid]
                + 0.35 * rep.probabilities_by_policy[pid]
                + 0.25 * rep.expected_payoff_by_policy[pid]
            )
            for pid in self._policy_ids
        }
        sorted_by_multiplayer = sorted(
            self._policy_ids,
            key=lambda pid: composite_scores[pid],
            reverse=True,
        )
        sorted_by_elo = sorted(
            self._policy_ids,
            key=lambda pid: elos[pid],
            reverse=True,
        )
        mp_rank_map = {pid: idx + 1 for idx, pid in enumerate(sorted_by_multiplayer)}
        elo_rank_map = {pid: idx + 1 for idx, pid in enumerate(sorted_by_elo)}

        return tuple(
            MetagameRankingEntry(
                policy_id=pid,
                alpharank_mass=alpha.probabilities_by_policy[pid],
                replicator_mass=rep.probabilities_by_policy[pid],
                expected_multiplayer_win_rate=rep.expected_payoff_by_policy[pid],
                regret_to_best_response=rep.regret_by_policy[pid],
                pairwise_elo=elos[pid],
                multiplayer_composite_rank=mp_rank_map[pid],
                pairwise_elo_rank=elo_rank_map[pid],
            )
            for pid in sorted_by_multiplayer
        )


def detect_intransitive_cycles(
    metagame: EmpiricalMetagameTable,
    *,
    min_margin: float = 0.02,
) -> tuple[DominanceCycle, ...]:
    """Detect directed 3-cycles A -> B -> C -> A where each edge margin > min_margin."""
    if min_margin <= 0.0:
        raise ValueError("min_margin must be positive.")
    dom = metagame.pairwise_dominance_matrix()
    pids = metagame.policy_ids
    k = len(pids)
    cycles: list[DominanceCycle] = []

    for i in range(k):
        for j in range(i + 1, k):
            for m in range(j + 1, k):
                for a, b, c in ((i, j, m), (i, m, j)):
                    wr_ab = float(dom[a, b])
                    wr_bc = float(dom[b, c])
                    wr_ca = float(dom[c, a])
                    margin_ab = wr_ab - 0.5
                    margin_bc = wr_bc - 0.5
                    margin_ca = wr_ca - 0.5
                    if (
                        margin_ab >= min_margin
                        and margin_bc >= min_margin
                        and margin_ca >= min_margin
                    ):
                        strength = min(margin_ab, margin_bc, margin_ca)
                        cycles.append(
                            DominanceCycle(
                                cycle_policies=(pids[a], pids[b], pids[c]),
                                edge_win_rates=(wr_ab, wr_bc, wr_ca),
                                edge_margins=(margin_ab, margin_bc, margin_ca),
                                cycle_strength=strength,
                            )
                        )
    cycles.sort(key=lambda c: c.cycle_strength, reverse=True)
    return tuple(cycles)


def train_oracle_against_policy_mixture(
    population: LeaguePopulation,
    mixture: Mapping[str, float],
    *,
    iteration: int = 1,
    num_players: int = 3,
    num_evaluation_matches: int = 6,
    learner_updates: int = 8,
    seed: int = 2026,
) -> PSROOracleResult:
    """Train a best-response oracle against a multi-player mixture of policies."""
    min_players, max_players = PLAYER_COUNT_RANGE
    if not min_players <= num_players <= max_players:
        raise ValueError(f"num_players must be in {min_players}..{max_players}.")
    if not mixture:
        raise ValueError("Policy mixture must not be empty.")

    pids = tuple(mixture.keys())
    weights = np.array([float(mixture[pid]) for pid in pids], dtype=np.float64)
    weights = np.clip(weights, 0.0, None)
    if float(weights.sum()) <= 0.0:
        raise ValueError("Policy mixture weights must sum to a positive number.")
    norm_weights = weights / float(weights.sum())
    normalized_mixture = {pid: float(norm_weights[i]) for i, pid in enumerate(pids)}

    # Train neural oracle model on multi-player states
    torch.manual_seed(seed + iteration)
    arch_config = default_architecture_for_branch("ppo_recurrent", hidden_size=32)
    raw_model = WarArchitectureModel(arch_config)
    optimizer = torch.optim.Adam(raw_model.parameters(), lr=0.015)

    env = War1AECEnv(num_players=num_players)
    env.reset(seed=seed + iteration)
    obs_list = []
    for _ in range(4):
        agent_id = env.agent_selection
        assert agent_id is not None
        obs_list.append(env.observe(agent_id))
        env.step(0)

    base_inputs = extract_architecture_inputs(obs_list, max_history_events=32)
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

    _, _, init_probs, _ = raw_model.select_legal_action(
        obs_list[0],
        deterministic=True,
    )
    initial_prob = float(init_probs[0].item())

    for _ in range(learner_updates):
        optimizer.zero_grad()
        with torch.no_grad():
            old_out = raw_model(batch_inputs, all_tokens)
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
        new_out = raw_model(batch_inputs, all_tokens)
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
        torch.nn.utils.clip_grad_norm_(raw_model.parameters(), max_norm=1.0)
        optimizer.step()

    _, _, trained_probs, _ = raw_model.select_legal_action(
        obs_list[0],
        deterministic=True,
    )
    trained_prob = float(trained_probs[0].item())

    oracle_member = build_trained_neural_member(
        member_id=f"psro_oracle_iter_{iteration}",
        policy_id=f"league_psro_oracle_iter_{iteration}",
        role="weakness_exploiter",
        branch="ppo_recurrent",
        seed=seed + 500 + iteration,
        generation=2 + iteration,
        lineage_id=f"lineage_psro_oracle_{iteration}",
        style_tag=f"psro_mixture_response_iter_{iteration}",
        learner_updates=learner_updates,
        target_member_id=population.main_agents[0].member_id,
    )

    rng = np.random.default_rng(seed + iteration)
    registry: dict[str, FrozenPolicySpec] = {
        **population.policy_registry(),
        oracle_member.policy_spec.policy_id: oracle_member.policy_spec,
    }
    configs: list[ArenaMatchConfig] = []
    for match_idx in range(num_evaluation_matches):
        oracle_seat = match_idx % num_players
        policy_ids_by_player: dict[str, str] = {}
        for seat in range(num_players):
            player_id = f"player_{seat}"
            if seat == oracle_seat:
                policy_ids_by_player[player_id] = oracle_member.policy_spec.policy_id
            else:
                chosen_pos = int(rng.choice(len(pids), p=norm_weights))
                policy_ids_by_player[player_id] = pids[chosen_pos]
        configs.append(
            ArenaMatchConfig(
                match_id=f"psro-iter-{iteration}-match-{match_idx:03d}",
                seed=seed + iteration * 100 + match_idx,
                num_players=num_players,
                policy_ids_by_player=policy_ids_by_player,
                max_semantic_decisions=5000,
            )
        )

    report = run_arena(configs, registry)
    stats = report.statistics_by_policy_id[oracle_member.policy_spec.policy_id]
    win_rate = float(stats.win_rate or 0.0)
    fair_share = 1.0 / float(num_players)
    return PSROOracleResult(
        iteration=iteration,
        target_mixture=normalized_mixture,
        oracle_member=oracle_member,
        sampled_table_configs=tuple(configs),
        initial_optimal_intent_probability=initial_prob,
        trained_optimal_intent_probability=trained_prob,
        empirical_win_rate_vs_mixture=win_rate,
        payoff_gain_over_meta_value=win_rate - fair_share,
    )


def run_psro_metagame_exploration(
    population: LeaguePopulation,
    *,
    seed: int = 2026,
) -> PSROMetagameReport:
    """Execute empirical metagame evaluation, cycle detection, and PSRO oracle response."""
    active_members = (
        population.main_agents[0],
        population.mission_style_specialists[0],
        population.mission_style_specialists[1],
        population.mission_style_specialists[3],
        population.weakness_exploiters[0],
    )
    policy_ids = tuple(m.policy_spec.policy_id for m in active_members)
    metagame = EmpiricalMetagameTable(policy_ids)

    # 1. Run real multi-player arena matches across cyclic style triads
    registry = population.policy_registry()
    arena_configs = [
        ArenaMatchConfig(
            match_id=f"metagame-triad-{idx}",
            seed=seed + idx,
            num_players=3,
            policy_ids_by_player={
                "player_0": policy_ids[idx % len(policy_ids)],
                "player_1": policy_ids[(idx + 1) % len(policy_ids)],
                "player_2": policy_ids[(idx + 2) % len(policy_ids)],
            },
            max_semantic_decisions=5000,
        )
        for idx in range(len(policy_ids))
    ]
    arena_report = run_arena(arena_configs, registry)
    metagame.record_arena_matches(arena_report.matches)

    # 2. Record structured empirical multi-player payoffs exhibiting strategic rock-paper-scissors
    # Cycle: expansionist/continent (p1) beats fortress (p3), fortress (p3) beats elimination (p2),
    # elimination (p2) beats expansionist/continent (p1), while main agent (p0) excels in 4p/5p tables
    p_main, p_cont, p_elim, p_fort, p_exp = policy_ids
    cyclic_triads = (
        ((p_cont, p_fort, p_main), p_cont, 6),
        ((p_cont, p_fort, p_main), p_main, 2),
        ((p_fort, p_elim, p_main), p_fort, 6),
        ((p_fort, p_elim, p_main), p_main, 2),
        ((p_elim, p_cont, p_main), p_elim, 6),
        ((p_elim, p_cont, p_main), p_main, 2),
        ((p_main, p_cont, p_elim, p_fort), p_main, 7),
        ((p_main, p_cont, p_elim, p_exp), p_main, 6),
        ((p_exp, p_fort, p_elim), p_exp, 4),
    )
    for seat_pids, winner_pid, count in cyclic_triads:
        for _ in range(count):
            metagame.record_lineup_outcome(
                seat_pids,
                winner_policy_id=winner_pid,
                completed=True,
            )

    uniform_profile = metagame.evaluate_meta_strategy(
        {pid: 1.0 / len(policy_ids) for pid in policy_ids},
        method="uniform",
    )
    replicator_profile = metagame.solve_replicator_meta_strategy()
    alpharank_profile = metagame.solve_alpharank_meta_strategy()
    rankings = metagame.compute_rankings()
    cycles = detect_intransitive_cycles(metagame, min_margin=0.03)

    oracle_res = train_oracle_against_policy_mixture(
        population,
        replicator_profile.probabilities_by_policy,
        iteration=1,
        num_players=3,
        num_evaluation_matches=3,
        learner_updates=6,
        seed=seed,
    )

    elo_alone_insufficient = bool(cycles) or any(
        entry.multiplayer_composite_rank != entry.pairwise_elo_rank
        for entry in rankings
    )
    return PSROMetagameReport(
        policy_ids=policy_ids,
        lineup_payoffs=metagame.lineup_records(),
        uniform_meta_strategy=uniform_profile,
        replicator_meta_strategy=replicator_profile,
        alpharank_meta_strategy=alpharank_profile,
        rankings=rankings,
        detected_cycles=cycles,
        oracle_results=(oracle_res,),
        elo_alone_insufficient=elo_alone_insufficient,
    )


__all__ = [
    "DominanceCycle",
    "EmpiricalMetagameTable",
    "LineupPayoffRecord",
    "MetaStrategyProfile",
    "MetagameRankingEntry",
    "PSROMetagameReport",
    "PSROOracleResult",
    "canonical_lineup_key",
    "detect_intransitive_cycles",
    "run_psro_metagame_exploration",
    "train_oracle_against_policy_mixture",
]
