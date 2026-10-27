"""Recurrent hidden-state storage isolated by game and player seat."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

import torch
from torch import Tensor

RecurrentState: TypeAlias = Tensor | tuple[Tensor, Tensor]


@dataclass(frozen=True)
class PerPlayerRecurrentMemory:
    """Batch-major memory with shape ``(games, players, layers, hidden)``."""

    hidden: Tensor
    cell: Tensor | None = None

    def __post_init__(self) -> None:
        if self.hidden.ndim != 4 or any(size < 1 for size in self.hidden.shape):
            raise ValueError("Hidden memory must have shape (games, players, layers, hidden).")
        if not self.hidden.is_floating_point():
            raise TypeError("Recurrent hidden memory must be floating point.")
        if self.cell is not None:
            if self.cell.shape != self.hidden.shape:
                raise ValueError("LSTM cell memory must match hidden-memory shape.")
            if (
                self.cell.device != self.hidden.device
                or self.cell.dtype != self.hidden.dtype
            ):
                raise ValueError("LSTM hidden and cell memory must share device and dtype.")

    @classmethod
    def zeros(
        cls,
        *,
        batch_size: int,
        num_players: int,
        num_layers: int,
        hidden_size: int,
        with_cell_state: bool = False,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> PerPlayerRecurrentMemory:
        dimensions = (batch_size, num_players, num_layers, hidden_size)
        if any(isinstance(size, bool) or size < 1 for size in dimensions):
            raise ValueError("All recurrent-memory dimensions must be positive.")
        hidden = torch.zeros(dimensions, device=device, dtype=dtype)
        cell = torch.zeros_like(hidden) if with_cell_state else None
        return cls(hidden=hidden, cell=cell)

    @property
    def batch_size(self) -> int:
        return self.hidden.shape[0]

    @property
    def num_players(self) -> int:
        return self.hidden.shape[1]

    @property
    def num_layers(self) -> int:
        return self.hidden.shape[2]

    @property
    def hidden_size(self) -> int:
        return self.hidden.shape[3]

    def select(
        self,
        player_indices: Tensor,
        *,
        active_mask: Tensor | None = None,
    ) -> RecurrentState:
        indices, active = self._validated_indices(player_indices, active_mask)
        batch_indices = torch.arange(self.batch_size, device=self.hidden.device)
        safe_indices = torch.where(active, indices, torch.zeros_like(indices))
        hidden = self.hidden[batch_indices, safe_indices].permute(1, 0, 2)
        hidden = hidden.masked_fill(~active[None, :, None], 0.0)
        if self.cell is None:
            return hidden
        cell = self.cell[batch_indices, safe_indices].permute(1, 0, 2)
        cell = cell.masked_fill(~active[None, :, None], 0.0)
        return hidden, cell

    def update(
        self,
        player_indices: Tensor,
        recurrent_state: RecurrentState,
        *,
        active_mask: Tensor | None = None,
    ) -> PerPlayerRecurrentMemory:
        indices, active = self._validated_indices(player_indices, active_mask)
        if isinstance(recurrent_state, tuple):
            next_hidden, next_cell = recurrent_state
        else:
            next_hidden, next_cell = recurrent_state, None
        expected_shape = (self.num_layers, self.batch_size, self.hidden_size)
        if next_hidden.shape != expected_shape:
            raise ValueError(
                f"Next hidden state must have shape {expected_shape}, "
                f"got {tuple(next_hidden.shape)}."
            )
        if (
            next_hidden.device != self.hidden.device
            or next_hidden.dtype != self.hidden.dtype
        ):
            raise ValueError("Next hidden state must match memory device and dtype.")
        if self.cell is None and next_cell is not None:
            raise ValueError("GRU memory does not accept an LSTM cell state.")
        if self.cell is not None:
            if next_cell is None or next_cell.shape != expected_shape:
                raise ValueError("LSTM memory requires a matching next cell state.")
            if (
                next_cell.device != self.cell.device
                or next_cell.dtype != self.cell.dtype
            ):
                raise ValueError("Next cell state must match memory device and dtype.")

        updated_hidden = self.hidden.clone()
        batch_indices = torch.arange(self.batch_size, device=self.hidden.device)
        if torch.any(active).item():
            updated_hidden[batch_indices[active], indices[active]] = (
                next_hidden.permute(1, 0, 2)[active]
            )
        updated_cell = None
        if self.cell is not None and next_cell is not None:
            updated_cell = self.cell.clone()
            if torch.any(active).item():
                updated_cell[batch_indices[active], indices[active]] = (
                    next_cell.permute(1, 0, 2)[active]
                )
        return PerPlayerRecurrentMemory(hidden=updated_hidden, cell=updated_cell)

    def reset_players(
        self,
        player_indices: Tensor,
        *,
        reset_mask: Tensor | None = None,
    ) -> PerPlayerRecurrentMemory:
        indices, reset = self._validated_indices(player_indices, reset_mask)
        batch_indices = torch.arange(self.batch_size, device=self.hidden.device)
        updated_hidden = self.hidden.clone()
        if torch.any(reset).item():
            updated_hidden[batch_indices[reset], indices[reset]] = 0.0
        updated_cell = None
        if self.cell is not None:
            updated_cell = self.cell.clone()
            if torch.any(reset).item():
                updated_cell[batch_indices[reset], indices[reset]] = 0.0
        return PerPlayerRecurrentMemory(hidden=updated_hidden, cell=updated_cell)

    def detach(self) -> PerPlayerRecurrentMemory:
        return PerPlayerRecurrentMemory(
            hidden=self.hidden.detach(),
            cell=None if self.cell is None else self.cell.detach(),
        )

    def _validated_indices(
        self,
        player_indices: Tensor,
        active_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        if player_indices.shape != (self.batch_size,):
            raise ValueError("Player indices must contain one seat per game.")
        if player_indices.dtype not in (torch.int32, torch.int64):
            raise TypeError("Player indices must use an integer tensor dtype.")
        if player_indices.device != self.hidden.device:
            raise ValueError("Player indices and memory must be on the same device.")
        if active_mask is None:
            active = torch.ones(
                self.batch_size,
                dtype=torch.bool,
                device=self.hidden.device,
            )
        else:
            if active_mask.shape != (self.batch_size,):
                raise ValueError("Active mask must contain one value per game.")
            if active_mask.device != self.hidden.device:
                raise ValueError("Active mask and memory must be on the same device.")
            active = _binary_mask(active_mask, "active_mask")
        indices = player_indices.to(dtype=torch.long)
        invalid = (indices < 0) | (indices >= self.num_players)
        if torch.any(active & invalid).item():
            raise IndexError("An active recurrent-memory seat is outside the table.")
        return indices, active


def _binary_mask(mask: Tensor, label: str) -> Tensor:
    if mask.dtype == torch.bool:
        return mask
    if torch.any((mask != 0) & (mask != 1)).item():
        raise ValueError(f"{label} may only contain zero and one.")
    return mask.to(dtype=torch.bool)


__all__ = ["PerPlayerRecurrentMemory", "RecurrentState"]
