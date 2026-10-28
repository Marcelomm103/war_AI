"""Tests for opponent belief estimation and calibration (Phase 9.1)."""

from __future__ import annotations

import numpy as np
from war_game_engine.contracts import (
    Action,
    ContractEnvelope,
    DiscardTerritoryCardOperation,
    parse_contract_json,
)
from war_game_engine.versioning import ACTION_SCHEMA_VERSION

from war_ai.envs.war1_aec import War1AECEnv
from war_ai.rl.belief_models import (
    BELIEF_METHODS,
    CARD_IDS,
    OBJECTIVE_IDS,
    LearnedBeliefModelEstimator,
    SequentialParticleBeliefFilter,
    SimpleRulePriorBeliefEstimator,
    build_player_knowledge_state,
    collect_belief_benchmark_samples,
    compare_belief_estimators,
    compute_feasible_objective_mask,
    eligible_objective_ids_for_public_state,
    estimate_public_hand_counts_and_card_availability,
    evaluate_belief_calibration,
)


def test_rule_prior_excludes_own_objective_and_absent_color_objectives() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=91)
    obs = env.observe("player_0")["engine_observation"]
    assert obs is not None

    eligible = eligible_objective_ids_for_public_state(obs.public_state)
    assert len(eligible) == 8 + 3
    # 3 players use white, red, black; yellow/green/blue elimination objectives are absent
    assert "W1-OBJ-02" not in eligible  # blue
    assert "W1-OBJ-05" not in eligible  # yellow
    assert "W1-OBJ-06" not in eligible  # green

    feasible_mask = compute_feasible_objective_mask(obs)
    own_obj_id = obs.private_state.objective_id
    assert own_obj_id is not None
    own_obj_idx = OBJECTIVE_IDS.index(own_obj_id)

    # Focal seat has no opponent belief row
    assert not np.any(feasible_mask[0])
    # Opponents cannot hold focal player's physical objective card or absent-color cards
    for opp_seat in (1, 2):
        assert not feasible_mask[opp_seat, own_obj_idx]
        for idx, obj_id in enumerate(OBJECTIVE_IDS):
            if obj_id not in eligible:
                assert not feasible_mask[opp_seat, idx]


def test_knowledge_state_keeps_facts_separate_from_belief_hypotheses() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=92)
    obs = env.observe("player_0")["engine_observation"]
    assert obs is not None

    discard_action = Action(
        contract_type="action",
        action_id="obs-discard-1",
        actor_player_id="player_1",
        action_schema_version=ACTION_SCHEMA_VERSION,
        expected_state_version=obs.state_version,
        operation=DiscardTerritoryCardOperation(
            action_type="discard_territory_card",
            card_id=CARD_IDS[0],
        ),
    )
    knowledge = build_player_knowledge_state(
        obs,
        observed_actions=(discard_action,),
    )
    belief = SimpleRulePriorBeliefEstimator().estimate(
        obs,
        observed_actions=(discard_action,),
    )

    # Known facts only record verified facts (own objective, own cards, permanent discard)
    predicates = {fact.predicate for fact in knowledge.known_facts}
    assert "has_objective_card" in predicates
    assert "permanently_discarded_card" in predicates
    for fact in knowledge.known_facts:
        if fact.predicate == "has_objective_card":
            assert fact.subject_id == "player_0"

    # Contract round-trip validation for both PlayerKnowledgeState and BeliefState
    knowledge_env = ContractEnvelope(
        schema_version="1.0",
        api_version="1.0.0",
        payload=knowledge,
    )
    belief_env = ContractEnvelope(
        schema_version="1.0",
        api_version="1.0.0",
        payload=belief.contract_belief_state,
    )
    assert parse_contract_json(knowledge_env.model_dump_json()).payload == knowledge
    assert (
        parse_contract_json(belief_env.model_dump_json()).payload
        == belief.contract_belief_state
    )

    # Permanently discarded card has 0 probability for all opponents
    assert not belief.available_card_mask[0]
    assert np.all(belief.card_probabilities[:, 0] == 0.0)


def test_particle_filter_updates_from_public_history_and_maintains_ess() -> None:
    samples = collect_belief_benchmark_samples(
        num_games=2,
        num_players=3,
        steps_per_game=24,
        base_seed=105,
    )
    assert len(samples) >= 4
    pf = SequentialParticleBeliefFilter(num_particles=48, seed=105)
    for obs, _ in samples:
        estimate = pf.estimate(obs)
        assert estimate.method_id == "particle_filter"
        for seat_idx, player in enumerate(obs.public_state.players):
            if player.player_id == obs.player_id:
                assert float(np.sum(estimate.objective_probabilities[seat_idx])) == 0.0
                continue
            row_sum = float(np.sum(estimate.objective_probabilities[seat_idx]))
            assert abs(row_sum - 1.0) < 1e-6
            assert np.all(
                estimate.objective_probabilities[
                    seat_idx, ~estimate.feasible_objective_mask[seat_idx]
                ]
                == 0.0
            )
    assert pf.effective_sample_size() > 1.0


def test_compare_simple_prior_particles_and_learned_model_calibration() -> None:
    report = compare_belief_estimators(
        num_train_games=3,
        num_eval_games=3,
        num_players=3,
        steps_per_game=21,
        num_particles=48,
        seed=301,
    )
    assert report.num_training_observations >= 6
    assert report.num_evaluation_observations >= 6
    assert report.all_respect_hard_constraints
    assert set(report.metrics_by_method) == set(BELIEF_METHODS)

    simple_metrics = report.metrics_by_method["simple_prior"]
    particle_metrics = report.metrics_by_method["particle_filter"]
    learned_metrics = report.metrics_by_method["learned_model"]

    for metrics in (simple_metrics, particle_metrics, learned_metrics):
        assert 0.0 <= metrics.objective_ece <= 1.0
        assert 0.0 <= metrics.objective_brier_score <= 2.0
        assert metrics.objective_log_loss > 0.0
        assert 0.0 <= metrics.card_ece <= 1.0
        assert metrics.hard_constraint_violation_rate == 0.0
        assert len(metrics.objective_reliability_bins) == 10

    # Particle filter or learned model incorporates behavioral likelihoods and improves log-loss
    assert (
        min(particle_metrics.objective_log_loss, learned_metrics.objective_log_loss)
        <= simple_metrics.objective_log_loss
    )


def test_learned_belief_model_supervised_training_reduces_loss() -> None:
    samples = collect_belief_benchmark_samples(
        num_games=2,
        num_players=3,
        steps_per_game=18,
        base_seed=411,
    )
    estimator = LearnedBeliefModelEstimator()
    losses = estimator.fit_supervised(samples, epochs=8, seed=411)
    assert len(losses) == 8
    assert losses[-1] < losses[0]

    pairs = [(estimator.estimate(obs), sec) for obs, sec in samples]
    cal = evaluate_belief_calibration(pairs)
    assert cal.hard_constraint_violation_rate == 0.0
    assert cal.mean_true_objective_probability > 0.05


def test_public_hand_count_and_card_availability_accounting() -> None:
    env = War1AECEnv(num_players=3)
    env.reset(seed=512)
    obs = env.observe("player_0")["engine_observation"]
    assert obs is not None

    counts, mask, unavailable = estimate_public_hand_counts_and_card_availability(obs)
    assert counts["player_0"] == len(obs.private_state.hand_card_ids)
    for card_id in obs.private_state.hand_card_ids:
        assert card_id in unavailable
        assert not mask[CARD_IDS.index(card_id)]
