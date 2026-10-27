"""Historical self-play league and balanced full-game campaign schedules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from war_game_engine.objectives import PlayerColor
from war_game_engine.setup import PLAYER_COUNT_RANGE

from ..arena import (
    ArenaMatchConfig,
    ArenaReport,
    FrozenPolicySpec,
    baseline_policy_spec,
    run_arena,
)
from ..baselines import BASELINE_POLICY_IDS
from ..envs.observations import PLAYER_COLOR_IDS


@dataclass(frozen=True)
class HistoricalPolicySnapshot:
    snapshot_id: str
    generation: int
    style_id: str
    policy_spec: FrozenPolicySpec

    def __post_init__(self) -> None:
        if not self.snapshot_id.strip() or not self.style_id.strip():
            raise ValueError("Historical snapshots require IDs and a style.")
        if (
            isinstance(self.generation, bool)
            or not isinstance(self.generation, int)
            or self.generation < 0
        ):
            raise ValueError("Snapshot generation must be a non-negative integer.")
        if self.policy_spec.checkpoint_sha256 is None:
            raise ValueError("Historical snapshots require a checkpoint SHA-256.")


@dataclass(frozen=True)
class HistoricalSelfPlayMatch:
    config: ArenaMatchConfig
    live_policy_id: str
    live_player_id: str
    live_player_color: PlayerColor
    historical_snapshot_ids: tuple[str, ...]
    opponent_styles_by_player: tuple[tuple[str, str], ...]

    def __post_init__(self) -> None:
        if self.config.policy_ids_by_player.get(self.live_player_id) != (
            self.live_policy_id
        ):
            raise ValueError("Live policy assignment does not match the match config.")
        player_index = int(self.live_player_id.removeprefix("player_"))
        if PLAYER_COLOR_IDS[player_index] != self.live_player_color:
            raise ValueError("Live seat color does not match the WAR seat mapping.")
        expected_opponents = set(self.config.policy_ids_by_player) - {
            self.live_player_id
        }
        if {player_id for player_id, _ in self.opponent_styles_by_player} != (
            expected_opponents
        ):
            raise ValueError("Each opponent seat requires one style label.")
        if not self.historical_snapshot_ids:
            raise ValueError("Every self-play match must include a historical opponent.")


@dataclass(frozen=True)
class HistoricalSelfPlayCampaign:
    matches: tuple[HistoricalSelfPlayMatch, ...]
    policy_specs: tuple[FrozenPolicySpec, ...]

    def __post_init__(self) -> None:
        if not self.matches:
            raise ValueError("Self-play campaigns require at least one match.")
        if len({match.config.match_id for match in self.matches}) != len(
            self.matches
        ):
            raise ValueError("Self-play match IDs must be unique.")
        if len({spec.policy_id for spec in self.policy_specs}) != len(
            self.policy_specs
        ):
            raise ValueError("Self-play policy IDs must be unique.")
        missing = {
            policy_id
            for match in self.matches
            for policy_id in match.config.policy_ids_by_player.values()
        } - {spec.policy_id for spec in self.policy_specs}
        if missing:
            raise ValueError(f"Self-play campaign is missing policies: {sorted(missing)}.")

    @property
    def configs(self) -> tuple[ArenaMatchConfig, ...]:
        return tuple(match.config for match in self.matches)

    @property
    def policy_registry(self) -> dict[str, FrozenPolicySpec]:
        return {spec.policy_id: spec for spec in self.policy_specs}

    def run(self) -> ArenaReport:
        """Run this campaign in the full-game arena."""
        return run_arena(self.configs, self.policy_registry)


class HistoricalSelfPlayLeague:
    """Keep immutable earlier policies and schedule them against the live one."""

    def __init__(
        self,
        current_policy: FrozenPolicySpec,
        *,
        current_generation: int,
        snapshot_capacity: int = 128,
        baseline_policy_ids: tuple[str, ...] = BASELINE_POLICY_IDS,
        seed: int = 0,
    ) -> None:
        if (
            isinstance(current_generation, bool)
            or not isinstance(current_generation, int)
            or current_generation < 0
        ):
            raise ValueError("Current generation must be a non-negative integer.")
        if (
            isinstance(snapshot_capacity, bool)
            or not isinstance(snapshot_capacity, int)
            or snapshot_capacity < 1
        ):
            raise ValueError("Snapshot capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("League seed must be a non-negative integer.")
        if not baseline_policy_ids:
            raise ValueError("Self-play requires at least one baseline style.")
        if len(set(baseline_policy_ids)) != len(baseline_policy_ids):
            raise ValueError("Baseline policy IDs must be unique.")
        unknown_baselines = set(baseline_policy_ids) - set(BASELINE_POLICY_IDS)
        if unknown_baselines:
            raise ValueError(
                f"Unknown baseline styles: {sorted(unknown_baselines)}."
            )
        if current_policy.policy_id in baseline_policy_ids:
            raise ValueError("The live policy ID cannot be a baseline style ID.")

        self._current_policy = current_policy
        self._current_generation = current_generation
        self._snapshot_capacity = snapshot_capacity
        self.baseline_policy_ids = baseline_policy_ids
        self._archive_rng = np.random.default_rng(seed)
        self._snapshots: list[HistoricalPolicySnapshot] = []
        self._registered_snapshot_ids: set[str] = set()
        self._registered_policy_ids: set[str] = set()
        self._seen_snapshot_count = 0

    @property
    def current_policy(self) -> FrozenPolicySpec:
        return self._current_policy

    @property
    def current_generation(self) -> int:
        return self._current_generation

    @property
    def snapshots(self) -> tuple[HistoricalPolicySnapshot, ...]:
        return tuple(self._snapshots)

    @property
    def seen_snapshot_count(self) -> int:
        return self._seen_snapshot_count

    def add_snapshot(self, snapshot: HistoricalPolicySnapshot) -> bool:
        if snapshot.generation >= self._current_generation:
            raise ValueError("Historical snapshots must predate the live generation.")
        if snapshot.policy_spec.policy_id in (
            self._current_policy.policy_id,
            *self.baseline_policy_ids,
        ):
            raise ValueError("Historical snapshots need distinct policy IDs.")
        if snapshot.snapshot_id in self._registered_snapshot_ids:
            raise ValueError(f"Snapshot ID {snapshot.snapshot_id!r} is already known.")
        if snapshot.policy_spec.policy_id in self._registered_policy_ids:
            raise ValueError("Historical snapshot policy IDs must be unique.")

        self._registered_snapshot_ids.add(snapshot.snapshot_id)
        self._registered_policy_ids.add(snapshot.policy_spec.policy_id)
        self._seen_snapshot_count += 1
        if len(self._snapshots) < self._snapshot_capacity:
            self._snapshots.append(snapshot)
            return True

        replacement_index = int(
            self._archive_rng.integers(0, self._seen_snapshot_count)
        )
        if replacement_index >= self._snapshot_capacity:
            return False
        self._snapshots[replacement_index] = snapshot
        return True

    def promote_current(
        self,
        new_policy: FrozenPolicySpec,
        *,
        new_generation: int,
        previous_style_id: str,
    ) -> bool:
        """Archive the old current checkpoint before installing a newer policy."""
        if (
            isinstance(new_generation, bool)
            or not isinstance(new_generation, int)
            or new_generation <= self._current_generation
        ):
            raise ValueError("New generation must be greater than the current one.")
        if self._current_policy.checkpoint_sha256 is None:
            raise ValueError(
                "Current policy must be checkpointed before it can enter history."
            )
        if new_policy.policy_id in (
            self._current_policy.policy_id,
            *self.baseline_policy_ids,
            *self._registered_policy_ids,
        ):
            raise ValueError("New live policy ID must be unique.")
        archived = HistoricalPolicySnapshot(
            snapshot_id=(
                f"generation-{self._current_generation}:"
                f"{self._current_policy.policy_id}"
            ),
            generation=self._current_generation,
            style_id=previous_style_id,
            policy_spec=self._current_policy,
        )
        if archived.snapshot_id in self._registered_snapshot_ids:
            raise ValueError("Current checkpoint has already been archived.")

        self._current_policy = new_policy
        self._current_generation = new_generation
        return self.add_snapshot(archived)

    def build_campaign(
        self,
        campaign_id: str,
        *,
        num_matches: int,
        seed: int,
        player_counts: tuple[int, ...] = (3, 4, 5, 6),
        max_semantic_decisions: int = 10_000,
        decision_budget_ms: int | None = None,
    ) -> HistoricalSelfPlayCampaign:
        """Create a seat/color-balanced campaign with historical opponents."""
        if not campaign_id.strip():
            raise ValueError("Campaign IDs must not be blank.")
        if isinstance(num_matches, bool) or not isinstance(num_matches, int):
            raise TypeError("num_matches must be an integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("Campaign seed must be a non-negative integer.")
        minimum, maximum = PLAYER_COUNT_RANGE
        if not player_counts or len(set(player_counts)) != len(player_counts):
            raise ValueError("player_counts must be a non-empty unique sequence.")
        if any(
            isinstance(count, bool)
            or not isinstance(count, int)
            or not minimum <= count <= maximum
            for count in player_counts
        ):
            raise ValueError(
                f"player_counts must be between {minimum} and {maximum}."
            )
        seat_rotation = tuple(
            (player_count, seat_index)
            for player_count in player_counts
            for seat_index in range(player_count)
        )
        if num_matches < len(seat_rotation):
            raise ValueError(
                "Campaign must cover every seat at every configured table size."
            )
        if not self._snapshots:
            raise RuntimeError(
                "Register at least one historical checkpoint before self-play."
            )

        rng = np.random.default_rng(seed)
        baseline_specs = tuple(
            baseline_policy_spec(policy_id)
            for policy_id in self.baseline_policy_ids
        )
        opponent_candidates: tuple[
            HistoricalPolicySnapshot | FrozenPolicySpec,
            ...,
        ] = (
            *self._snapshots,
            *baseline_specs,
        )
        registry: dict[str, FrozenPolicySpec] = {
            self._current_policy.policy_id: self._current_policy,
            **{
                snapshot.policy_spec.policy_id: snapshot.policy_spec
                for snapshot in self._snapshots
            },
            **{spec.policy_id: spec for spec in baseline_specs},
        }
        if len(registry) != (
            1 + len(self._snapshots) + len(baseline_specs)
        ):
            raise ValueError("Live, historical, and baseline policy IDs overlap.")

        matches: list[HistoricalSelfPlayMatch] = []
        for match_index in range(num_matches):
            player_count, live_seat = seat_rotation[
                match_index % len(seat_rotation)
            ]
            player_ids = tuple(
                f"player_{index}" for index in range(player_count)
            )
            live_player_id = player_ids[live_seat]
            opponent_seats = tuple(
                player_id
                for player_id in player_ids
                if player_id != live_player_id
            )
            policy_ids_by_player = {live_player_id: self._current_policy.policy_id}
            style_by_player: dict[str, str] = {}
            historical_ids: list[str] = []

            required_snapshot = self._snapshots[
                match_index % len(self._snapshots)
            ]
            required_style_seat = opponent_seats[0]
            policy_ids_by_player[required_style_seat] = (
                required_snapshot.policy_spec.policy_id
            )
            style_by_player[required_style_seat] = required_snapshot.style_id
            historical_ids.append(required_snapshot.snapshot_id)

            required_baseline = self.baseline_policy_ids[
                match_index % len(self.baseline_policy_ids)
            ]
            baseline_seat = opponent_seats[1]
            policy_ids_by_player[baseline_seat] = required_baseline
            style_by_player[baseline_seat] = required_baseline

            for player_id in opponent_seats[2:]:
                candidate = opponent_candidates[
                    int(rng.integers(0, len(opponent_candidates)))
                ]
                if isinstance(candidate, HistoricalPolicySnapshot):
                    policy_ids_by_player[player_id] = (
                        candidate.policy_spec.policy_id
                    )
                    style_by_player[player_id] = candidate.style_id
                    historical_ids.append(candidate.snapshot_id)
                else:
                    policy_ids_by_player[player_id] = candidate.policy_id
                    style_by_player[player_id] = candidate.policy_id

            config = ArenaMatchConfig(
                match_id=f"{campaign_id}:{match_index:06d}",
                seed=seed + match_index,
                num_players=player_count,
                policy_ids_by_player=policy_ids_by_player,
                max_semantic_decisions=max_semantic_decisions,
                decision_budget_ms=decision_budget_ms,
            )
            matches.append(
                HistoricalSelfPlayMatch(
                    config=config,
                    live_policy_id=self._current_policy.policy_id,
                    live_player_id=live_player_id,
                    live_player_color=PLAYER_COLOR_IDS[live_seat],
                    historical_snapshot_ids=tuple(dict.fromkeys(historical_ids)),
                    opponent_styles_by_player=tuple(
                        (player_id, style_by_player[player_id])
                        for player_id in opponent_seats
                    ),
                )
            )

        return HistoricalSelfPlayCampaign(
            matches=tuple(matches),
            policy_specs=tuple(registry.values()),
        )


__all__ = [
    "HistoricalPolicySnapshot",
    "HistoricalSelfPlayCampaign",
    "HistoricalSelfPlayLeague",
    "HistoricalSelfPlayMatch",
]
