from __future__ import annotations

from collections.abc import Callable

import numpy as np
import pytest

pytest.importorskip("gymnasium")
pytest.importorskip("pettingzoo")
pytest.importorskip("war_game_engine")

from war_game_engine.contracts import FullGameState

from war_ai.arena import FrozenPolicySpec
from war_ai.baselines import BASELINE_POLICY_IDS
from war_ai.envs import War1AECEnv, War1AECObservation
from war_ai.rl.self_play import (
    HistoricalPolicySnapshot,
    HistoricalSelfPlayLeague,
)


def _policy_factory(
    _seed: int,
) -> Callable[[War1AECObservation, np.random.Generator], int]:
    def policy(
        _observation: War1AECObservation,
        _rng: np.random.Generator,
    ) -> int:
        return 0

    return policy


def _spec(policy_id: str, digest: str | None = None) -> FrozenPolicySpec:
    return FrozenPolicySpec(
        policy_id=policy_id,
        factory=_policy_factory,
        checkpoint_sha256=None if digest is None else f"sha256:{digest * 64}",
    )


def _league() -> HistoricalSelfPlayLeague:
    league = HistoricalSelfPlayLeague(
        _spec("live-ppo"),
        current_generation=10,
        snapshot_capacity=8,
        seed=71,
    )
    assert league.add_snapshot(
        HistoricalPolicySnapshot(
            snapshot_id="ppo-generation-2",
            generation=2,
            style_id="ppo-mlp",
            policy_spec=_spec("ppo-snapshot-2", "a"),
        )
    )
    assert league.add_snapshot(
        HistoricalPolicySnapshot(
            snapshot_id="ppo-generation-7",
            generation=7,
            style_id="ppo-gru",
            policy_spec=_spec("ppo-snapshot-7", "b"),
        )
    )
    return league


def test_campaign_rotates_seats_colors_missions_and_table_sizes() -> None:
    campaign = _league().build_campaign(
        "historical-rotation",
        num_matches=sum((3, 4, 5, 6)),
        seed=2026,
        player_counts=(3, 4, 5, 6),
    )
    colors_by_size: dict[int, set[str]] = {}
    missions: set[str] = set()
    historical_ids: set[str] = set()
    opponent_styles: set[str] = set()

    for match in campaign.matches:
        colors_by_size.setdefault(match.config.num_players, set()).add(
            match.live_player_color
        )
        historical_ids.update(match.historical_snapshot_ids)
        opponent_styles.update(
            style for _, style in match.opponent_styles_by_player
        )
        assert match.config.policy_ids_by_player[match.live_player_id] == (
            "live-ppo"
        )
        assert all(
            policy_id != "live-ppo"
            for player_id, policy_id in match.config.policy_ids_by_player.items()
            if player_id != match.live_player_id
        )
        assert match.config.seed >= 2026

        env = War1AECEnv(num_players=match.config.num_players)
        env.reset(
            seed=match.config.seed,
            options={"game_id": f"arena:{match.config.match_id}"},
        )
        state = env._require_state()
        live_player = next(
            player
            for player in state.players
            if player.public.player_id == match.live_player_id
        )
        assert live_player.private.objective_id is not None
        missions.add(live_player.private.objective_id)

    assert set(colors_by_size) == {3, 4, 5, 6}
    for player_count, colors in colors_by_size.items():
        assert len(colors) == player_count
    assert len(missions) > 1
    assert historical_ids == {"ppo-generation-2", "ppo-generation-7"}
    assert set(BASELINE_POLICY_IDS).intersection(opponent_styles)
    assert campaign.policy_registry["live-ppo"] == campaign.policy_specs[0]


def test_historical_league_requires_immutable_older_snapshots() -> None:
    league = HistoricalSelfPlayLeague(
        _spec("live-ppo"),
        current_generation=3,
        snapshot_capacity=2,
    )

    with pytest.raises(RuntimeError, match="historical checkpoint"):
        league.build_campaign("missing-history", num_matches=18, seed=4)
    with pytest.raises(ValueError, match="must predate"):
        league.add_snapshot(
            HistoricalPolicySnapshot(
                snapshot_id="future",
                generation=3,
                style_id="future-style",
                policy_spec=_spec("future-ppo", "c"),
            )
        )
    with pytest.raises(ValueError, match="checkpoint SHA-256"):
        league.add_snapshot(
            HistoricalPolicySnapshot(
                snapshot_id="no-digest",
                generation=1,
                style_id="no-digest",
                policy_spec=_spec("snapshot-no-digest"),
            )
        )


def test_promoting_live_policy_archives_its_checkpoint() -> None:
    league = HistoricalSelfPlayLeague(
        _spec("live-ppo", "d"),
        current_generation=4,
        snapshot_capacity=4,
    )

    retained = league.promote_current(
        _spec("live-ppo-generation-5"),
        new_generation=5,
        previous_style_id="ppo-mlp",
    )

    assert retained
    assert league.current_generation == 5
    assert league.current_policy.policy_id == "live-ppo-generation-5"
    assert league.snapshots[0].snapshot_id == "generation-4:live-ppo"
    assert league.snapshots[0].generation == 4
    assert league.snapshots[0].policy_spec.checkpoint_sha256 == f"sha256:{'d' * 64}"


def test_campaign_requires_a_complete_seat_rotation() -> None:
    with pytest.raises(ValueError, match="cover every seat"):
        _league().build_campaign(
            "incomplete-rotation",
            num_matches=3,
            seed=3,
            player_counts=(3, 4, 5, 6),
        )


def test_campaign_executes_through_arena_with_seat_local_policy_instances() -> None:
    campaign = _league().build_campaign(
        "arena-self-play-smoke",
        num_matches=sum((3, 4, 5, 6)),
        seed=2026,
        player_counts=(3, 4, 5, 6),
        max_semantic_decisions=1,
    )

    report = campaign.run()

    assert len(report.matches) == len(campaign.matches)
    assert all(match.status == "truncated" for match in report.matches)
    for match in report.matches:
        initial_state = match.replay.initial_snapshot.state
        assert isinstance(initial_state, FullGameState)
        assert initial_state.phase_id == "initial_placement"
    live_missions = {
        seat.mission_id
        for match, plan in zip(report.matches, campaign.matches, strict=True)
        for seat in match.seat_results
        if seat.player_id == plan.live_player_id
    }
    assert len(live_missions) > 1
