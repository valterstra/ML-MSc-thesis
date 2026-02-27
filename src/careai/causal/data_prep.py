"""Data preparation for causal effect estimation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PreparedData:
    df: pd.DataFrame
    X: pd.DataFrame
    t: pd.Series
    y: pd.Series
    feature_names: list[str]


def load_and_prepare(df: pd.DataFrame, cfg: dict[str, Any]) -> PreparedData:
    analysis = cfg["analysis"]
    include_actions = set(analysis["include_actions"])
    outcome_col = analysis["outcome_col"]
    high_label = analysis["treatment_high_label"]
    low_label = analysis["treatment_low_label"]

    work = df[df["a_t"].isin(include_actions)].copy()
    work = work.dropna(subset=[outcome_col])
    work = work[(work["a_t"] == high_label) | (work["a_t"] == low_label)].copy()

    num_cols = list(analysis["covariates_numeric"])
    cat_cols = list(analysis["covariates_categorical"])

    for col in num_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        med = work[col].median()
        if np.isnan(med):
            med = 0.0
        work[col] = work[col].fillna(med)

    for col in cat_cols:
        work[col] = work[col].fillna("UNKNOWN_CAT").astype(str)

    X_num = work[num_cols].copy()
    X_cat = pd.get_dummies(work[cat_cols], prefix=cat_cols, drop_first=False, dtype=float)
    X = pd.concat([X_num, X_cat], axis=1)

    t = (work["a_t"] == high_label).astype(int)
    y = pd.to_numeric(work[outcome_col], errors="coerce").fillna(0).astype(int)

    return PreparedData(
        df=work.reset_index(drop=True),
        X=X.reset_index(drop=True),
        t=t.reset_index(drop=True),
        y=y.reset_index(drop=True),
        feature_names=list(X.columns),
    )

