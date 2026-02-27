"""One-step contextual bandit environment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .outcome_model import OutcomeModel


@dataclass(frozen=True)
class StepResult:
    reward: float
    done: bool
    y: int
    p_hat: float
    action: str
    action_cost: float


class BanditEnvV1:
    def __init__(self, eval_df: pd.DataFrame, outcome_model: OutcomeModel, cfg: dict[str, Any], seed: int) -> None:
        self.eval_df = eval_df.reset_index(drop=True)
        self.model = outcome_model
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)
        self.state_cols_num = list(cfg["model"]["features_numeric"])
        self.state_cols_cat = list(cfg["model"]["features_categorical"])
        self.state_cols = self.state_cols_num + self.state_cols_cat
        self._state: pd.Series | None = None

    def reset(self) -> pd.Series:
        idx = int(self.rng.integers(0, len(self.eval_df)))
        self._state = self.eval_df.iloc[idx].copy()
        return self._state[self.state_cols].copy()

    def _action_cost(self, action: str) -> float:
        if action == "A_HIGH_SUPPORT":
            return float(self.cfg["reward"]["cost_high"])
        return float(self.cfg["reward"]["cost_low"])

    def step(self, action: str) -> StepResult:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")
        s_df = pd.DataFrame([self._state[self.state_cols].to_dict()])
        p_hat = float(self.model.predict_readmit_prob(s_df, action=action)[0])
        y = int(self.rng.binomial(1, p_hat))
        action_cost = self._action_cost(action)
        reward = float(-y - action_cost)
        return StepResult(
            reward=reward,
            done=True,
            y=y,
            p_hat=p_hat,
            action=action,
            action_cost=action_cost,
        )

