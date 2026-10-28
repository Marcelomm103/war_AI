from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

pytest.importorskip("torch")
pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from war_game_engine.contracts import OfficialVictoryOutcome

from war_ai.arena import (
    ArenaMatchResult,
    ArenaReport,
    ArenaSeatResult,
    FrozenPolicySpec,
    baseline_policy_spec,
)
from war_ai.reproducibility import ReproducibilityStore
from war_ai.rl.agent import (
    DEFAULT_PROMOTION_SEEDS,
    export_trained_ppo_policy_spec,
    train_full_game_ppo_agent,
)
from war_ai.rl.seed_evaluation import (
    PromotionCriteria,
    SeedComparison,
    assess_promotion,
    build_paired_seed_configs,
    run_paired_seed_evaluation,
)


def _comparison(seed: int, *, gain: bool) -> SeedComparison:
    return SeedComparison(
        seed=seed,
        num_players=3,
        candidate_policy_id="candidate",
        baseline_policy_id="baseline",
        candidate_wins=1 if gain else 0,
        candidate_appearances=1,
        baseline_wins=0 if gain else 1,
        baseline_appearances=1,
        candidate_mission_ids=("W1-OBJ-07",),
    )


def test_promotion_requires_ten_seeds_reproducibility_and_significance() -> None:
    criteria = PromotionCriteria(
        minimum_seed_count=10,
        minimum_positive_fraction=0.8,
        significance_level=0.05,
    )
    nine_seed_decision = assess_promotion(
        tuple(_comparison(seed, gain=True) for seed in range(9)),
        criteria=criteria,
    )
    eight_of_ten_decision = assess_promotion(
        tuple(
            _comparison(seed, gain=seed < 8)
            for seed in range(10)
        ),
        criteria=criteria,
    )
    nine_of_ten_decision = assess_promotion(
        tuple(
            _comparison(seed, gain=seed < 9)
            for seed in range(10)
        ),
        criteria=criteria,
    )

    assert not nine_seed_decision.promote
    assert "requires at least 10 paired seeds" in nine_seed_decision.reasons
    assert not eight_of_ten_decision.promote
    assert eight_of_ten_decision.sign_test_p_value == pytest.approx(0.0546875)
    assert nine_of_ten_decision.promote
    assert nine_of_ten_decision.positive_fraction == 0.9
    assert nine_of_ten_decision.sign_test_p_value == pytest.approx(11 / 1024)
    assert nine_of_ten_decision.candidate_policy_id == "candidate"
    assert nine_of_ten_decision.baseline_policy_id == "baseline"
    assert nine_of_ten_decision.evaluated_seeds == tuple(range(10))


def test_paired_seed_configs_rotate_candidate_through_every_seat() -> None:
    configs = build_paired_seed_configs(
        evaluation_id="candidate-vs-baseline",
        candidate_policy_id="candidate",
        baseline_policy_id="baseline",
        seed=71,
        num_players=4,
    )

    assert len(configs) == 4
    assert {config.seed for config in configs} == {71}
    assert len({config.match_id for config in configs}) == 4
    assert {
        next(
            player_id
            for player_id, policy_id in config.policy_ids_by_player.items()
            if policy_id == "candidate"
        )
        for config in configs
    } == {f"player_{index}" for index in range(4)}


def test_seed_runner_summarizes_only_complete_paired_arena_results(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import war_ai.rl.seed_evaluation as seed_evaluation

    candidate = FrozenPolicySpec(
        policy_id="candidate",
        factory=lambda _seed: lambda _observation, _rng: 0,
        checkpoint_sha256=f"sha256:{'a' * 64}",
    )
    baseline = FrozenPolicySpec(
        policy_id="baseline",
        factory=lambda _seed: lambda _observation, _rng: 0,
    )

    def fake_run_arena(configs: tuple[Any, ...], _registry: Any) -> Any:
        results: list[ArenaMatchResult] = []
        for config in configs:
            candidate_seat = next(
                seat
                for seat, policy_id in config.policy_ids_by_player.items()
                if policy_id == "candidate"
            )
            seat_results = tuple(
                ArenaSeatResult.model_construct(
                    player_id=seat,
                    policy_id=config.policy_ids_by_player[seat],
                    checkpoint_sha256=None,
                    mission_id=f"W1-OBJ-0{index + 1}",
                    is_eliminated=False,
                    is_winner=seat == candidate_seat,
                )
                for index, seat in enumerate(config.policy_ids_by_player)
            )
            outcome = OfficialVictoryOutcome(
                outcome_type="official_victory",
                winner_player_id=candidate_seat,
            )
            results.append(
                ArenaMatchResult.model_construct(
                    match_id=config.match_id,
                    game_id=f"arena:{config.match_id}",
                    status="completed",
                    seed=config.seed,
                    num_players=config.num_players,
                    policy_ids_by_player=config.policy_ids_by_player,
                    outcome=outcome,
                    seat_results=seat_results,
                )
            )
        return ArenaReport.model_construct(matches=tuple(results))

    monkeypatch.setattr(seed_evaluation, "run_arena", fake_run_arena)
    comparisons = run_paired_seed_evaluation(
        candidate,
        baseline,
        (41, 42),
        evaluation_id="paired-test",
        num_players=3,
    )

    assert len(comparisons) == 2
    assert all(comparison.candidate_win_rate == 1.0 for comparison in comparisons)
    assert all(comparison.baseline_win_rate == 0.0 for comparison in comparisons)


def test_promotion_rejects_duplicate_seeds_and_too_small_seed_floor() -> None:
    comparison = _comparison(1, gain=True)
    with pytest.raises(ValueError, match="unique seeds"):
        assess_promotion((comparison, comparison))
    with pytest.raises(ValueError, match="at least ten"):
        PromotionCriteria(minimum_seed_count=9)


def test_full_game_ppo_checkpoint_exports_and_completes_paired_arena_matches(
    tmp_path: Path,
) -> None:
    store = ReproducibilityStore(tmp_path / "store")
    trained = train_full_game_ppo_agent(
        store,
        run_id="test-full-game-ppo",
        seed=2026,
        training_episodes=2,
        num_threads=1,
    )
    assert trained.episodes_completed == 2
    assert trained.official_wins == 2
    assert (
        trained.trained_optimal_intent_probability
        > trained.initial_optimal_intent_probability
    )
    assert trained.trained_optimal_intent_probability >= 0.99

    exported = export_trained_ppo_policy_spec(
        store,
        "test-full-game-ppo",
        policy_id="ppo_full_game_v1",
    )
    assert exported.checkpoint_sha256 == (
        trained.manifest.weights_artifact.sha256
    )

    comparisons = run_paired_seed_evaluation(
        exported,
        baseline_policy_spec("random_legal"),
        (2026,),
        evaluation_id="exported-ppo-smoke",
        num_players=3,
        max_semantic_decisions=2500,
    )
    assert len(comparisons) == 1
    assert comparisons[0].candidate_wins == 3
    assert comparisons[0].baseline_wins == 0
    assert comparisons[0].paired_gain == 1.0


def test_recorded_phase6_ten_seed_campaign_passes_promotion_gate() -> None:
    campaign_path = (
        Path(__file__).resolve().parents[1]
        / "benchmarks"
        / "phase-6.6-promotion-2026-10-27.json"
    )
    payload = json.loads(campaign_path.read_text(encoding="utf-8"))
    comparisons = tuple(
        SeedComparison(
            seed=item["seed"],
            num_players=item["num_players"],
            candidate_policy_id=item["candidate_policy_id"],
            baseline_policy_id=item["baseline_policy_id"],
            candidate_wins=item["candidate_wins"],
            candidate_appearances=item["candidate_appearances"],
            baseline_wins=item["baseline_wins"],
            baseline_appearances=item["baseline_appearances"],
            candidate_mission_ids=tuple(item["candidate_mission_ids"]),
        )
        for item in payload["comparisons"]
    )
    decision = assess_promotion(comparisons)

    assert payload["checkpoint_sha256"].startswith("sha256:")
    assert payload["manifest_sha256"].startswith("sha256:")
    assert payload["training_episodes"] == 4
    assert payload["training_official_wins"] == 4
    assert payload["trained_optimal_intent_probability"] >= 0.999
    assert decision.promote
    assert decision.seed_count == 10
    assert decision.positive_seed_count == 10
    assert decision.positive_fraction == 1.0
    assert decision.mean_paired_gain == pytest.approx(0.9)
    assert decision.sign_test_p_value == pytest.approx(0.0009765625)
    assert decision.evaluated_seeds == DEFAULT_PROMOTION_SEEDS
