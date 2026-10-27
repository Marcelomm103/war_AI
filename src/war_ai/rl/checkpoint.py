"""Content-addressed PPO checkpoints with optimizer, environment, and RNG state."""

from __future__ import annotations

import copy
import io
import random
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer

from ..reproducibility import ArtifactReference, ArtifactStore

_CHECKPOINT_SCHEMA_VERSION = 1
_PYTORCH_MEDIA_TYPE = "application/vnd.war-ai.pytorch-checkpoint"


@dataclass(frozen=True)
class PPOResumeState:
    """Non-model state required to continue the same training trajectory."""

    run_id: str
    update_index: int
    episode_count: int
    environment_steps: int
    environment_state: Mapping[str, object]
    rollout_state: Mapping[str, object]
    recurrent_state: Mapping[str, object]
    curriculum_state: Mapping[str, object]
    journal_state: Mapping[str, object]

    def __post_init__(self) -> None:
        if not self.run_id.strip():
            raise ValueError("Checkpoint run_id must not be blank.")
        for label, counter_value in (
            ("update_index", self.update_index),
            ("episode_count", self.episode_count),
            ("environment_steps", self.environment_steps),
        ):
            if (
                isinstance(counter_value, bool)
                or not isinstance(counter_value, int)
                or counter_value < 0
            ):
                raise ValueError(f"{label} must be a non-negative integer.")
        for label, mapping_value in (
            ("environment_state", self.environment_state),
            ("rollout_state", self.rollout_state),
            ("recurrent_state", self.recurrent_state),
            ("curriculum_state", self.curriculum_state),
            ("journal_state", self.journal_state),
        ):
            if not isinstance(mapping_value, Mapping):
                raise TypeError(f"{label} must be a state mapping.")


@dataclass(frozen=True)
class PPOCheckpointReferences:
    weights_artifact: ArtifactReference
    optimizer_state_artifact: ArtifactReference
    training_state_artifact: ArtifactReference


def save_ppo_checkpoint(
    artifact_store: ArtifactStore,
    model: nn.Module,
    optimizer: Optimizer,
    resume_state: PPOResumeState,
    *,
    numpy_generators: Mapping[str, np.random.Generator],
    python_generators: Mapping[str, random.Random],
) -> PPOCheckpointReferences:
    """Store model, optimizer, trajectory state, and all registered RNG states.

    Checkpoints use PyTorch serialization and must only be restored from a
    trusted local artifact store.
    """
    _validate_generators(numpy_generators, python_generators)
    weights = artifact_store.put_bytes(
        _torch_bytes(model.state_dict()),
        media_type=_PYTORCH_MEDIA_TYPE,
    )
    optimizer_state = artifact_store.put_bytes(
        _torch_bytes(optimizer.state_dict()),
        media_type=_PYTORCH_MEDIA_TYPE,
    )
    training_state = {
        "schema_version": _CHECKPOINT_SCHEMA_VERSION,
        "resume_state": resume_state,
        "rng_state": {
            "python_global": random.getstate(),
            "python_generators": {
                name: generator.getstate()
                for name, generator in python_generators.items()
            },
            "numpy_global": np.random.get_state(),
            "numpy_generators": {
                name: copy.deepcopy(generator.bit_generator.state)
                for name, generator in numpy_generators.items()
            },
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": (
                tuple(torch.cuda.get_rng_state_all())
                if torch.cuda.is_available()
                else ()
            ),
        },
    }
    training_reference = artifact_store.put_bytes(
        _torch_bytes(training_state),
        media_type=_PYTORCH_MEDIA_TYPE,
    )
    return PPOCheckpointReferences(
        weights_artifact=weights,
        optimizer_state_artifact=optimizer_state,
        training_state_artifact=training_reference,
    )


def restore_ppo_checkpoint(
    artifact_store: ArtifactStore,
    references: PPOCheckpointReferences,
    model: nn.Module,
    optimizer: Optimizer,
    *,
    numpy_generators: Mapping[str, np.random.Generator],
    python_generators: Mapping[str, random.Random],
    map_location: str | torch.device = "cpu",
) -> PPOResumeState:
    """Restore a checkpoint and its registered randomness exactly."""
    _validate_generators(numpy_generators, python_generators)
    weights_path = artifact_store.verify(references.weights_artifact)
    optimizer_path = artifact_store.verify(references.optimizer_state_artifact)
    training_path = artifact_store.verify(references.training_state_artifact)
    weights = _torch_load(weights_path, map_location)
    optimizer_state = _torch_load(optimizer_path, map_location)
    training_payload = _torch_load(training_path, map_location)

    if not isinstance(training_payload, dict):
        raise ValueError("Training checkpoint payload must be a dictionary.")
    if training_payload.get("schema_version") != _CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported training checkpoint schema version.")
    resume_state = training_payload.get("resume_state")
    rng_state = training_payload.get("rng_state")
    if not isinstance(resume_state, PPOResumeState):
        raise TypeError("Training checkpoint has an invalid resume state.")
    if not isinstance(rng_state, Mapping):
        raise TypeError("Training checkpoint has no RNG state.")

    saved_numpy_generators = rng_state.get("numpy_generators")
    saved_python_generators = rng_state.get("python_generators")
    if not isinstance(saved_numpy_generators, Mapping) or set(
        saved_numpy_generators
    ) != set(numpy_generators):
        raise ValueError("Checkpoint NumPy generator names do not match.")
    if not isinstance(saved_python_generators, Mapping) or set(
        saved_python_generators
    ) != set(python_generators):
        raise ValueError("Checkpoint Python generator names do not match.")
    python_global_state = rng_state.get("python_global")
    numpy_global_state = rng_state.get("numpy_global")
    torch_cpu_state = rng_state.get("torch_cpu")
    torch_cuda_states = rng_state.get("torch_cuda")
    if not isinstance(python_global_state, tuple):
        raise TypeError("Checkpoint global Python RNG state is invalid.")
    if not isinstance(numpy_global_state, tuple):
        raise TypeError("Checkpoint global NumPy RNG state is invalid.")
    if not isinstance(torch_cpu_state, Tensor):
        raise TypeError("Checkpoint CPU torch RNG state is invalid.")
    if not isinstance(torch_cuda_states, (tuple, list)) or any(
        not isinstance(state, Tensor) for state in torch_cuda_states
    ):
        raise TypeError("Checkpoint CUDA torch RNG states are invalid.")
    if torch.cuda.is_available():
        if len(torch_cuda_states) != torch.cuda.device_count():
            raise ValueError("Checkpoint CUDA device count does not match.")
    elif torch_cuda_states:
        raise ValueError("Cannot restore CUDA RNG state without CUDA devices.")

    model.load_state_dict(weights)
    optimizer.load_state_dict(optimizer_state)
    random.setstate(python_global_state)
    np.random.set_state(numpy_global_state)
    torch.set_rng_state(torch_cpu_state.cpu())
    for name, numpy_generator in numpy_generators.items():
        numpy_generator.bit_generator.state = copy.deepcopy(
            saved_numpy_generators[name]
        )
    for name, python_generator in python_generators.items():
        python_generator.setstate(saved_python_generators[name])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(list(torch_cuda_states))
    return resume_state


def _torch_bytes(payload: object) -> bytes:
    output = io.BytesIO()
    torch.save(payload, output)
    return output.getvalue()


def _torch_load(path: str | Path, map_location: str | torch.device) -> Any:
    return torch.load(
        path,
        map_location=map_location,
        weights_only=False,
    )


def _validate_generators(
    numpy_generators: Mapping[str, np.random.Generator],
    python_generators: Mapping[str, random.Random],
) -> None:
    if any(not name.strip() for name in (*numpy_generators, *python_generators)):
        raise ValueError("Registered RNG names must not be blank.")
    if any(
        not isinstance(generator, np.random.Generator)
        for generator in numpy_generators.values()
    ):
        raise TypeError("numpy_generators must contain NumPy Generator instances.")
    if any(
        not isinstance(generator, random.Random)
        for generator in python_generators.values()
    ):
        raise TypeError("python_generators must contain random.Random instances.")


__all__ = [
    "PPOCheckpointReferences",
    "PPOResumeState",
    "restore_ppo_checkpoint",
    "save_ppo_checkpoint",
]
