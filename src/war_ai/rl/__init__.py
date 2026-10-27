"""Reinforcement-learning losses and per-player recurrent state."""

from .checkpoint import (
    PPOCheckpointReferences,
    PPOResumeState,
    restore_ppo_checkpoint,
    save_ppo_checkpoint,
)
from .losses import (
    joint_action_entropy,
    joint_action_kl_divergence,
    joint_action_log_prob,
    masked_categorical_entropy,
    masked_categorical_kl_divergence,
    masked_log_softmax,
)
from .memory import PerPlayerRecurrentMemory, RecurrentState
from .ppo import (
    PPOActionSample,
    PPOActorCritic,
    PPOActorCriticOutput,
    PPOLoss,
    compute_gae,
    compute_masked_ppo_loss,
    normalize_advantages,
)
from .telemetry import (
    EpisodeAccumulator,
    PPOUpdateRecord,
    TrainingCurve,
    TrainingEpisodeRecord,
    TrainingRunJournal,
    apply_ppo_update,
)

__all__ = [
    "EpisodeAccumulator",
    "PPOActionSample",
    "PPOActorCritic",
    "PPOActorCriticOutput",
    "PPOCheckpointReferences",
    "PPOLoss",
    "PPOResumeState",
    "PPOUpdateRecord",
    "PerPlayerRecurrentMemory",
    "RecurrentState",
    "TrainingCurve",
    "TrainingEpisodeRecord",
    "TrainingRunJournal",
    "apply_ppo_update",
    "compute_gae",
    "compute_masked_ppo_loss",
    "joint_action_entropy",
    "joint_action_kl_divergence",
    "joint_action_log_prob",
    "masked_categorical_entropy",
    "masked_categorical_kl_divergence",
    "masked_log_softmax",
    "normalize_advantages",
    "restore_ppo_checkpoint",
    "save_ppo_checkpoint",
]
