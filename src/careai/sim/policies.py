"""Baseline policies for Sim v1."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from .outcome_model import OutcomeModel


class BasePolicy:
    name: str

    def choose_action(self, state: pd.Series, model: OutcomeModel) -> str:
        raise NotImplementedError


@dataclass(frozen=True)
class AlwaysLowPolicy(BasePolicy):
    name: str = "always_low"

    def choose_action(self, state: pd.Series, model: OutcomeModel) -> str:
        return "A_LOW_SUPPORT"


@dataclass(frozen=True)
class AlwaysHighPolicy(BasePolicy):
    name: str = "always_high"

    def choose_action(self, state: pd.Series, model: OutcomeModel) -> str:
        return "A_HIGH_SUPPORT"


@dataclass(frozen=True)
class ObservedPolicy(BasePolicy):
    name: str = "observed_policy"

    def choose_action(self, state: pd.Series, model: OutcomeModel) -> str:
        a = str(state.get("a_t", "A_LOW_SUPPORT"))
        return a if a in {"A_LOW_SUPPORT", "A_HIGH_SUPPORT"} else "A_LOW_SUPPORT"


@dataclass(frozen=True)
class RiskThresholdPolicy(BasePolicy):
    threshold: float
    name: str = "risk_threshold"

    def choose_action(self, state: pd.Series, model: OutcomeModel) -> str:
        s_df = pd.DataFrame([state[model.num_cols + model.cat_cols].to_dict()])
        p_low = float(model.predict_readmit_prob(s_df, action="A_LOW_SUPPORT")[0])
        if p_low >= float(self.threshold):
            return "A_HIGH_SUPPORT"
        return "A_LOW_SUPPORT"

