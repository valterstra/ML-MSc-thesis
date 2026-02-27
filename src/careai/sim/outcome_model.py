"""Outcome model for Sim v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def _encode(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str], columns_ref: list[str] | None = None) -> pd.DataFrame:
    x_num = df[num_cols].copy()
    x_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False, dtype=float)
    x = pd.concat([x_num, x_cat], axis=1)
    if columns_ref is None:
        return x
    return x.reindex(columns=columns_ref, fill_value=0.0)


@dataclass(frozen=True)
class OutcomeModel:
    model: LogisticRegression
    feature_columns: list[str]
    num_cols: list[str]
    cat_cols: list[str]

    def predict_readmit_prob(self, states: pd.DataFrame, action: str) -> np.ndarray:
        d = states.copy()
        d["action_is_high"] = 1 if action == "A_HIGH_SUPPORT" else 0
        x = _encode(d, self.num_cols + ["action_is_high"], self.cat_cols, self.feature_columns)
        return self.model.predict_proba(x)[:, 1]


def fit_outcome_model(train_df: pd.DataFrame, cfg: dict[str, Any], sample_weight: np.ndarray | None = None) -> OutcomeModel:
    model_cfg = cfg["model"]
    num_cols = list(model_cfg["features_numeric"])
    cat_cols = list(model_cfg["features_categorical"])
    d = train_df.copy()
    d["action_is_high"] = (d["a_t"] == "A_HIGH_SUPPORT").astype(int)
    x = _encode(d, num_cols + ["action_is_high"], cat_cols)
    y = pd.to_numeric(d["within_30d_next_admit"], errors="coerce").fillna(0).astype(int)

    model = LogisticRegression(
        max_iter=int(model_cfg.get("max_iter", 2000)),
        C=float(model_cfg.get("C", 1.0)),
        solver="lbfgs",
    )
    if sample_weight is not None:
        model.fit(x, y, sample_weight=sample_weight)
    else:
        model.fit(x, y)
    return OutcomeModel(
        model=model,
        feature_columns=list(x.columns),
        num_cols=num_cols,
        cat_cols=cat_cols,
    )
