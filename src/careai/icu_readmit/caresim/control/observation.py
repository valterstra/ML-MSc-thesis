"""Observation builder for CARE-Sim control policies."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from ..dataset import ACTION_COLS, STATE_COLS


@dataclass
class ObservationBuilder:
    """Fixed-length recent history window of state/action pairs."""

    window_len: int = 5
    state_dim: int = len(STATE_COLS)
    action_dim: int = len(ACTION_COLS)
    _steps: deque[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._steps = deque(maxlen=self.window_len)

    @property
    def obs_dim(self) -> int:
        return self.window_len * (self.state_dim + self.action_dim)

    def reset(self, seed_states: np.ndarray, seed_actions: np.ndarray) -> np.ndarray:
        """Initialize the recent window from full seeded history."""
        self._steps.clear()
        states = np.asarray(seed_states, dtype=np.float32)
        actions = np.asarray(seed_actions, dtype=np.float32)
        if states.ndim != 2 or actions.ndim != 2:
            raise ValueError("seed_states and seed_actions must be 2D arrays")
        if states.shape[0] != actions.shape[0]:
            raise ValueError("seed_states and seed_actions must have matching sequence lengths")
        self.state_dim = int(states.shape[1])
        self.action_dim = int(actions.shape[1])
        for state, action in zip(states, actions):
            self._steps.append(np.concatenate([state, action]).astype(np.float32, copy=False))
        return self.as_vector()

    def append(self, next_state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Append a new simulated state-action pair and return the current observation."""
        state_arr = np.asarray(next_state, dtype=np.float32).reshape(-1)
        action_arr = np.asarray(action, dtype=np.float32).reshape(-1)
        self._steps.append(np.concatenate([state_arr, action_arr]).astype(np.float32, copy=False))
        return self.as_vector()

    def as_vector(self) -> np.ndarray:
        """Return the flattened padded observation vector."""
        if not self._steps:
            return np.zeros(self.obs_dim, dtype=np.float32)
        stacked = np.stack(list(self._steps), axis=0)
        padded = np.zeros((self.window_len, self.state_dim + self.action_dim), dtype=np.float32)
        padded[-stacked.shape[0]:] = stacked[-self.window_len:]
        return padded.reshape(-1)
