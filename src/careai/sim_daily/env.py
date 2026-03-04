"""Gym-like simulation environment for daily hospital stays."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .features import (
    ACTION_COLS,
    INFECTION_CONTEXT,
    INPUT_COLS,
    MEASURED_FLAGS,
    STATE_BINARY,
    STATE_CONTINUOUS,
    STATIC_FEATURES,
)
from .transition import TransitionModel, predict_next


class DailySimEnv:
    """Step a virtual patient forward one calendar day at a time.

    Interface:
        reset(rng) -> state_dict
        step(action_dict) -> (next_state_dict, reward, done, info)
    """

    def __init__(
        self,
        transition_model: TransitionModel,
        initial_states_df: pd.DataFrame,
        max_days: int = 60,
        done_threshold: float = 0.5,
    ) -> None:
        self.model = transition_model
        self.initial_states = initial_states_df.reset_index(drop=True)
        self.max_days = max_days
        self.done_threshold = done_threshold

        self._state: dict[str, float] = {}
        self._day: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reset(self, rng: np.random.Generator | None = None) -> dict[str, float]:
        """Sample a random day-0 row and return the initial state dict."""
        if rng is None:
            rng = np.random.default_rng()

        idx = rng.integers(0, len(self.initial_states))
        row = self.initial_states.iloc[idx]

        state: dict[str, float] = {}
        for c in INPUT_COLS:
            if c in ACTION_COLS:
                continue  # actions are not part of state
            val = row.get(c, np.nan)
            state[c] = float(val) if pd.notna(val) else np.nan

        self._state = state
        self._day = 0
        return dict(self._state)

    def step(
        self, action_dict: dict[str, float]
    ) -> tuple[dict[str, float], float, bool, dict[str, Any]]:
        """Apply *action_dict* and advance one day.

        Returns (next_state, reward, done, info).
        """
        # Build full input: current state + actions
        full_input = dict(self._state)
        for c in ACTION_COLS:
            full_input[c] = action_dict.get(c, 0.0)
        # Ensure infection context is present
        for c in INFECTION_CONTEXT:
            if c not in full_input:
                full_input[c] = 0.0

        next_pred, done_prob = predict_next(self.model, full_input, action_dict)

        # --- Build next state ---
        next_state: dict[str, float] = {}

        # Predicted outputs (labs + binary states)
        for c in STATE_CONTINUOUS + STATE_BINARY:
            next_state[c] = next_pred[c]

        # Static features: carry forward
        for c in STATIC_FEATURES:
            if c == "day_of_stay":
                next_state[c] = self._state.get(c, 0.0) + 1.0
            elif c == "days_in_current_unit":
                prev_icu = self._state.get("is_icu", 0.0)
                curr_icu = next_pred.get("is_icu", 0.0)
                if curr_icu == prev_icu:
                    next_state[c] = self._state.get(c, 0.0) + 1.0
                else:
                    next_state[c] = 0.0  # unit changed, reset counter
            else:
                next_state[c] = self._state.get(c, 0.0)

        # Measured flags: simulation assumes all labs available
        for c in MEASURED_FLAGS:
            next_state[c] = 1.0

        # Infection context: reset per step (no persistent signal)
        for c in INFECTION_CONTEXT:
            next_state[c] = 0.0

        self._day += 1
        done = done_prob >= self.done_threshold or self._day >= self.max_days

        info: dict[str, Any] = {"done_prob": done_prob, "day": self._day}

        self._state = next_state
        reward = 0.0  # placeholder for future RL

        return dict(self._state), reward, done, info
