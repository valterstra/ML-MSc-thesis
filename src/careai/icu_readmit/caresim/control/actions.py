"""Action encoding helpers for binary ICU drug combinations."""

from __future__ import annotations

import torch
import numpy as np

from ..dataset import ACTION_COLS


def build_action_grid(action_dim: int) -> np.ndarray:
    return np.array(
        [[(action_id >> bit) & 1 for bit in range(action_dim)] for action_id in range(2 ** action_dim)],
        dtype=np.float32,
    )


ACTION_GRID = build_action_grid(len(ACTION_COLS))


def decode_action_id(action_id: int, action_dim: int | None = None) -> np.ndarray:
    """Map integer action id to binary action vector."""
    grid = ACTION_GRID if action_dim is None or action_dim == ACTION_GRID.shape[1] else build_action_grid(action_dim)
    return grid[int(action_id)].copy()


def encode_action_vec(action_vec: np.ndarray | list[float], action_dim: int | None = None) -> int:
    """Map binary action vector back to integer action id."""
    arr = np.asarray(action_vec, dtype=np.float32).reshape(-1)
    expected_dim = action_dim or len(ACTION_COLS)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"Expected action vector of length {expected_dim}, got {arr.shape[0]}")
    bits = (arr > 0.5).astype(np.int64)
    return int(sum(int(bit) << idx for idx, bit in enumerate(bits)))


def action_tensor(action_id: int, device: torch.device | None = None, action_dim: int | None = None) -> torch.Tensor:
    """Return a batched torch action tensor shaped (1, action_dim)."""
    return torch.tensor(decode_action_id(action_id, action_dim=action_dim), dtype=torch.float32, device=device).unsqueeze(0)
