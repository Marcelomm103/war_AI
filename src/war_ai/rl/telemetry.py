"""Append-only PPO training metrics and episode-level reward accounting."""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Mapping

import torch
from pydantic import BaseModel, ConfigDict, Field, model_validator
from torch import Tensor, nn
from torch.optim import Optimizer

from ..reproducibility import ArtifactReference, ArtifactStore
from .ppo import PPOLoss

try:
    import resource
except ImportError:  # pragma: no cover - resource is unavailable on Windows
    resource = None  # type: ignore[assignment]


class _TelemetryModel(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class TrainingEpisodeRecord(_TelemetryModel):
    event_type: Literal["episode"] = "episode"
    run_id: str = Field(min_length=1)
    episode_index: int = Field(ge=0)
    seed: int = Field(ge=0)
    objective_id: str = Field(min_length=1)
    table_size: int = Field(ge=3, le=6)
    official_reward: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    shaped_reward: float = Field(allow_inf_nan=False)
    decision_count: int = Field(ge=1)
    semantic_decisions: int = Field(ge=1)
    terminated: bool
    truncated: bool
    elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_episode_end(self) -> TrainingEpisodeRecord:
        if self.terminated == self.truncated:
            raise ValueError("An episode record must be terminated or truncated.")
        if self.official_reward not in (0.0, 1.0):
            raise ValueError("Official WAR reward must be zero or one.")
        if self.truncated and self.official_reward != 0.0:
            raise ValueError("Truncated episodes cannot record an official win.")
        return self


class PPOUpdateRecord(_TelemetryModel):
    event_type: Literal["ppo_update"] = "ppo_update"
    run_id: str = Field(min_length=1)
    update_index: int = Field(ge=1)
    environment_steps: int = Field(ge=0)
    total_loss: float = Field(allow_inf_nan=False)
    policy_loss: float = Field(allow_inf_nan=False)
    value_loss: float = Field(ge=0.0, allow_inf_nan=False)
    entropy: float = Field(ge=0.0, allow_inf_nan=False)
    approximate_kl: float = Field(allow_inf_nan=False)
    clip_fraction: float = Field(ge=0.0, le=1.0, allow_inf_nan=False)
    gradient_norm: float = Field(ge=0.0, allow_inf_nan=False)
    learning_rate: float = Field(ge=0.0, allow_inf_nan=False)
    update_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    total_elapsed_seconds: float = Field(ge=0.0, allow_inf_nan=False)
    updates_per_second: float = Field(ge=0.0, allow_inf_nan=False)
    environment_steps_per_second: float = Field(
        ge=0.0,
        allow_inf_nan=False,
    )
    cpu_max_rss_bytes: int | None = Field(default=None, ge=0)
    cuda_peak_memory_bytes: int | None = Field(default=None, ge=0)


@dataclass(frozen=True)
class TrainingCurve:
    objective_id: str
    table_size: int
    episodes: int
    completed_episodes: int
    official_wins: int
    official_win_rate: float | None
    shaped_reward_total: float
    shaped_reward_mean: float
    mean_decisions: float
    mean_semantic_decisions: float
    truncations: int
    truncation_rate: float


@dataclass
class _EpisodeAggregate:
    episodes: int = 0
    completed_episodes: int = 0
    official_wins: int = 0
    shaped_reward_total: float = 0.0
    decision_count_total: int = 0
    semantic_decisions_total: int = 0
    truncations: int = 0

    def add(self, record: TrainingEpisodeRecord) -> None:
        self.episodes += 1
        self.completed_episodes += int(not record.truncated)
        self.official_wins += int(
            not record.truncated and record.official_reward == 1.0
        )
        self.shaped_reward_total += record.shaped_reward
        self.decision_count_total += record.decision_count
        self.semantic_decisions_total += record.semantic_decisions
        self.truncations += int(record.truncated)

    def curve(self, objective_id: str, table_size: int) -> TrainingCurve:
        return TrainingCurve(
            objective_id=objective_id,
            table_size=table_size,
            episodes=self.episodes,
            completed_episodes=self.completed_episodes,
            official_wins=self.official_wins,
            official_win_rate=(
                self.official_wins / self.completed_episodes
                if self.completed_episodes
                else None
            ),
            shaped_reward_total=self.shaped_reward_total,
            shaped_reward_mean=self.shaped_reward_total / self.episodes,
            mean_decisions=self.decision_count_total / self.episodes,
            mean_semantic_decisions=(
                self.semantic_decisions_total / self.episodes
            ),
            truncations=self.truncations,
            truncation_rate=self.truncations / self.episodes,
        )


class EpisodeAccumulator:
    """Keep canonical win reward separate from optional shaped reward."""

    def __init__(
        self,
        *,
        run_id: str,
        episode_index: int,
        seed: int,
        reset_info: Mapping[str, object],
    ) -> None:
        objective_id = reset_info.get("objective_id")
        table_size = reset_info.get("num_players")
        if not isinstance(objective_id, str) or not objective_id.strip():
            raise ValueError("Reset info must contain the focal objective_id.")
        if isinstance(table_size, bool) or not isinstance(table_size, int):
            raise ValueError("Reset info must contain the integer num_players.")
        if not 3 <= table_size <= 6:
            raise ValueError("num_players must be between three and six.")
        if not run_id.strip():
            raise ValueError("run_id must not be blank.")
        if (
            isinstance(episode_index, bool)
            or not isinstance(episode_index, int)
            or episode_index < 0
        ):
            raise ValueError("episode_index must be a non-negative integer.")
        if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
            raise ValueError("seed must be a non-negative integer.")

        self.run_id = run_id
        self.episode_index = episode_index
        self.seed = seed
        self.objective_id = objective_id
        self.table_size = table_size
        self._started_at = time.perf_counter()
        self._elapsed_offset = 0.0
        self._official_reward = 0.0
        self._shaped_reward = 0.0
        self._decision_count = 0
        self._semantic_decisions = 0
        self._terminated = False
        self._truncated = False
        self._closed = False

    def add_transition(
        self,
        reward: float,
        terminated: bool,
        truncated: bool,
        info: Mapping[str, object],
    ) -> TrainingEpisodeRecord | None:
        if self._closed:
            raise RuntimeError("Episode accumulator is already closed.")
        if not isinstance(terminated, bool) or not isinstance(truncated, bool):
            raise TypeError("terminated and truncated must be booleans.")
        if terminated and truncated:
            raise ValueError("A transition cannot be terminated and truncated.")
        returned_reward = _finite_number(reward, "reward")
        official_reward = _finite_number(
            info.get("official_reward"),
            "info['official_reward']",
        )
        shaping_reward = _finite_number(
            info.get("shaping_reward"),
            "info['shaping_reward']",
        )
        if official_reward not in (0.0, 1.0):
            raise ValueError("Official WAR reward must be zero or one.")
        if not math.isclose(
            returned_reward,
            official_reward + shaping_reward,
            rel_tol=1e-8,
            abs_tol=1e-8,
        ):
            raise ValueError(
                "Environment reward must equal official plus shaping reward."
            )
        duration = info.get("semantic_duration")
        if (
            isinstance(duration, bool)
            or not isinstance(duration, int)
            or duration < 1
        ):
            raise ValueError("info['semantic_duration'] must be a positive integer.")
        if official_reward == 1.0 and not terminated:
            raise ValueError("An official win reward requires episode termination.")

        self._official_reward += official_reward
        self._shaped_reward += shaping_reward
        self._decision_count += 1
        self._semantic_decisions += duration
        self._terminated = terminated
        self._truncated = truncated
        if not terminated and not truncated:
            return None
        if self._official_reward not in (0.0, 1.0):
            raise ValueError("An episode cannot contain multiple official win rewards.")

        self._closed = True
        return TrainingEpisodeRecord(
            run_id=self.run_id,
            episode_index=self.episode_index,
            seed=self.seed,
            objective_id=self.objective_id,
            table_size=self.table_size,
            official_reward=self._official_reward,
            shaped_reward=self._shaped_reward,
            decision_count=self._decision_count,
            semantic_decisions=self._semantic_decisions,
            terminated=self._terminated,
            truncated=self._truncated,
            elapsed_seconds=self._elapsed_offset
            + time.perf_counter()
            - self._started_at,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "run_id": self.run_id,
            "episode_index": self.episode_index,
            "seed": self.seed,
            "objective_id": self.objective_id,
            "table_size": self.table_size,
            "elapsed_seconds": (
                self._elapsed_offset + time.perf_counter() - self._started_at
            ),
            "official_reward": self._official_reward,
            "shaped_reward": self._shaped_reward,
            "decision_count": self._decision_count,
            "semantic_decisions": self._semantic_decisions,
            "terminated": self._terminated,
            "truncated": self._truncated,
            "closed": self._closed,
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> EpisodeAccumulator:
        if state.get("schema_version") != 1:
            raise ValueError("Unsupported episode accumulator state version.")
        reset_info = {
            "objective_id": state.get("objective_id"),
            "num_players": state.get("table_size"),
        }
        accumulator = cls(
            run_id=_required_string(state.get("run_id"), "run_id"),
            episode_index=_required_int(state.get("episode_index"), "episode_index"),
            seed=_required_int(state.get("seed"), "seed"),
            reset_info=reset_info,
        )
        accumulator._elapsed_offset = _finite_number(
            state.get("elapsed_seconds"),
            "elapsed_seconds",
        )
        if accumulator._elapsed_offset < 0.0:
            raise ValueError("Restored episode elapsed time cannot be negative.")
        accumulator._official_reward = _finite_number(
            state.get("official_reward"),
            "official_reward",
        )
        accumulator._shaped_reward = _finite_number(
            state.get("shaped_reward"),
            "shaped_reward",
        )
        accumulator._decision_count = _required_int(
            state.get("decision_count"),
            "decision_count",
        )
        accumulator._semantic_decisions = _required_int(
            state.get("semantic_decisions"),
            "semantic_decisions",
        )
        accumulator._terminated = _required_bool(
            state.get("terminated"),
            "terminated",
        )
        accumulator._truncated = _required_bool(
            state.get("truncated"),
            "truncated",
        )
        accumulator._closed = _required_bool(state.get("closed"), "closed")
        accumulator._started_at = time.perf_counter()
        if accumulator._official_reward not in (0.0, 1.0):
            raise ValueError("Restored official reward must be zero or one.")
        if accumulator._decision_count < 0 or accumulator._semantic_decisions < 0:
            raise ValueError("Restored episode lengths cannot be negative.")
        return accumulator


@dataclass(frozen=True)
class _EpisodeGroup:
    objective_id: str
    table_size: int
    aggregate: _EpisodeAggregate


class TrainingRunJournal:
    """Persist episode and PPO-update telemetry as append-only JSONL records."""

    def __init__(self, path: str | Path, *, run_id: str) -> None:
        if not run_id.strip():
            raise ValueError("run_id must not be blank.")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id
        self._episode_groups: dict[tuple[str, int], _EpisodeAggregate] = {}
        self._episode_count = 0
        self._update_count = 0
        self._last_episode_index = -1
        self._last_update_index = 0
        self._last_environment_steps = 0
        self._elapsed_offset = 0.0
        if self.path.exists():
            self._load_existing()
        self._started_at = time.perf_counter()

    @property
    def episode_count(self) -> int:
        return self._episode_count

    @property
    def update_count(self) -> int:
        return self._update_count

    @property
    def last_update_index(self) -> int:
        return self._last_update_index

    @property
    def environment_steps(self) -> int:
        return self._last_environment_steps

    @property
    def curves_by_objective_and_table(self) -> tuple[TrainingCurve, ...]:
        return tuple(
            self._episode_groups[key].curve(*key)
            for key in sorted(self._episode_groups)
        )

    def record_episode(self, record: TrainingEpisodeRecord) -> None:
        self._validate_run_id(record.run_id)
        if record.episode_index <= self._last_episode_index:
            raise ValueError("Episode indexes must increase monotonically.")
        self._append_record(record)
        self._ingest_episode(record)

    def store_artifact(self, artifact_store: ArtifactStore) -> ArtifactReference:
        """Store the current JSONL journal as a content-addressed run artifact."""
        return artifact_store.put_file(
            self.path,
            media_type="application/x-ndjson",
        )

    def record_ppo_update(
        self,
        *,
        update_index: int,
        environment_steps: int,
        loss: PPOLoss,
        gradient_norm: float,
        learning_rate: float,
        update_seconds: float,
    ) -> PPOUpdateRecord:
        if isinstance(update_index, bool) or not isinstance(update_index, int):
            raise TypeError("update_index must be an integer.")
        if update_index <= self._last_update_index:
            raise ValueError("Update indexes must increase monotonically.")
        if isinstance(environment_steps, bool) or not isinstance(
            environment_steps,
            int,
        ):
            raise TypeError("environment_steps must be an integer.")
        if environment_steps < self._last_environment_steps:
            raise ValueError("Environment step count cannot move backwards.")

        elapsed = max(
            self._elapsed_offset + time.perf_counter() - self._started_at,
            1e-9,
        )
        record = PPOUpdateRecord(
            run_id=self.run_id,
            update_index=update_index,
            environment_steps=environment_steps,
            total_loss=_tensor_scalar(loss.total_loss, "total_loss"),
            policy_loss=_tensor_scalar(loss.policy_loss, "policy_loss"),
            value_loss=_tensor_scalar(loss.value_loss, "value_loss"),
            entropy=_tensor_scalar(loss.entropy, "entropy"),
            approximate_kl=_tensor_scalar(loss.approximate_kl, "approximate_kl"),
            clip_fraction=_tensor_scalar(loss.clip_fraction, "clip_fraction"),
            gradient_norm=_finite_number(gradient_norm, "gradient_norm"),
            learning_rate=_finite_number(learning_rate, "learning_rate"),
            update_seconds=_finite_number(update_seconds, "update_seconds"),
            total_elapsed_seconds=elapsed,
            updates_per_second=update_index / elapsed,
            environment_steps_per_second=environment_steps / elapsed,
            cpu_max_rss_bytes=_cpu_max_rss_bytes(),
            cuda_peak_memory_bytes=(
                torch.cuda.max_memory_allocated()
                if torch.cuda.is_available()
                else None
            ),
        )
        self._append_record(record)
        self._ingest_update(record)
        return record

    def _append_record(
        self,
        record: TrainingEpisodeRecord | PPOUpdateRecord,
    ) -> None:
        with self.path.open("a", encoding="utf-8") as journal:
            journal.write(record.model_dump_json())
            journal.write("\n")
            journal.flush()

    def _load_existing(self) -> None:
        with self.path.open(encoding="utf-8") as journal:
            for line_number, line in enumerate(journal, start=1):
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError as error:
                    raise ValueError(
                        f"Invalid telemetry JSON at line {line_number}."
                    ) from error
                if not isinstance(payload, dict):
                    raise ValueError(
                        f"Telemetry line {line_number} must contain a JSON object."
                    )
                event_type = payload.get("event_type")
                if event_type == "episode":
                    episode_record = TrainingEpisodeRecord.model_validate(payload)
                    self._validate_run_id(episode_record.run_id)
                    self._ingest_episode(episode_record)
                elif event_type == "ppo_update":
                    update_record = PPOUpdateRecord.model_validate(payload)
                    self._validate_run_id(update_record.run_id)
                    self._ingest_update(update_record)
                    self._elapsed_offset = max(
                        self._elapsed_offset,
                        update_record.total_elapsed_seconds,
                    )
                else:
                    raise ValueError(
                        f"Unknown telemetry event at line {line_number}."
                    )

    def _validate_run_id(self, run_id: str) -> None:
        if run_id != self.run_id:
            raise ValueError("Telemetry record belongs to a different run.")

    def _ingest_episode(self, record: TrainingEpisodeRecord) -> None:
        if record.episode_index <= self._last_episode_index:
            raise ValueError("Journal episode indexes are not strictly increasing.")
        key = (record.objective_id, record.table_size)
        aggregate = self._episode_groups.setdefault(key, _EpisodeAggregate())
        aggregate.add(record)
        self._episode_count += 1
        self._last_episode_index = record.episode_index

    def _ingest_update(self, record: PPOUpdateRecord) -> None:
        if record.update_index <= self._last_update_index:
            raise ValueError("Journal update indexes are not strictly increasing.")
        if record.environment_steps < self._last_environment_steps:
            raise ValueError("Journal environment steps cannot move backwards.")
        self._update_count += 1
        self._last_update_index = record.update_index
        self._last_environment_steps = record.environment_steps


def apply_ppo_update(
    model: nn.Module,
    optimizer: Optimizer,
    loss: PPOLoss,
    *,
    max_gradient_norm: float = 0.5,
) -> float:
    """Apply one PPO update and return the pre-clipping gradient norm."""
    if (
        isinstance(max_gradient_norm, bool)
        or not math.isfinite(max_gradient_norm)
        or max_gradient_norm <= 0.0
    ):
        raise ValueError("max_gradient_norm must be finite and positive.")
    if loss.total_loss.numel() != 1 or not torch.isfinite(loss.total_loss).item():
        raise FloatingPointError("PPO total loss must be a finite scalar.")
    optimizer.zero_grad(set_to_none=True)
    loss.total_loss.backward()  # type: ignore[no-untyped-call]
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm=max_gradient_norm,
    )
    if not torch.isfinite(gradient_norm).item():
        raise FloatingPointError("PPO gradient norm is not finite.")
    optimizer.step()
    return float(gradient_norm.detach().cpu().item())


def _tensor_scalar(value: Tensor, label: str) -> float:
    if value.numel() != 1:
        raise ValueError(f"{label} must be a scalar tensor.")
    scalar = float(value.detach().cpu().item())
    if not math.isfinite(scalar):
        raise FloatingPointError(f"{label} must be finite.")
    return scalar


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a real number.")
    scalar = float(value)
    if not math.isfinite(scalar):
        raise ValueError(f"{label} must be finite.")
    return scalar


def _required_int(value: object, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer.")
    return value


def _required_bool(value: object, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean.")
    return value


def _required_string(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-blank string.")
    return value


def _cpu_max_rss_bytes() -> int | None:
    try:
        import resource
    except ImportError:  # pragma: no cover - resource is unavailable on Windows
        return None
    max_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return max_rss if sys.platform == "darwin" else max_rss * 1024


__all__ = [
    "EpisodeAccumulator",
    "PPOUpdateRecord",
    "TrainingCurve",
    "TrainingEpisodeRecord",
    "TrainingRunJournal",
    "apply_ppo_update",
]
