"""Input preparation for Sim v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SimPreparedData:
    train_df: pd.DataFrame
    eval_df: pd.DataFrame
    feature_names: list[str]


def prepare_sim_data(df: pd.DataFrame, cfg: dict[str, Any]) -> SimPreparedData:
    allowed_actions = set(cfg["data"]["allowed_actions"])
    eval_split = str(cfg["data"]["eval_split"])
    model_cfg = cfg["model"]
    num_cols = list(model_cfg["features_numeric"])
    cat_cols = list(model_cfg["features_categorical"])

    required = ["a_t", "within_30d_next_admit", "split"] + num_cols + cat_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns for sim_v1: {missing}")

    work = df[df["a_t"].isin(allowed_actions)].copy()
    work = work.dropna(subset=["within_30d_next_admit", "split"]).copy()
    work["within_30d_next_admit"] = pd.to_numeric(work["within_30d_next_admit"], errors="coerce").fillna(0).astype(int)

    for col in num_cols:
        work[col] = pd.to_numeric(work[col], errors="coerce")
        med = work[col].median()
        if pd.isna(med):
            med = 0.0
        work[col] = work[col].fillna(float(med))
    for col in cat_cols:
        work[col] = work[col].fillna("UNKNOWN_CAT").astype(str)

    train_df = work[work["split"] == "train"].copy()
    eval_df = work[work["split"] == eval_split].copy()
    if train_df.empty or eval_df.empty:
        raise ValueError(f"train/eval split empty after filtering. train={len(train_df)} eval={len(eval_df)}")

    feature_names = num_cols + [f"{c}__cat" for c in cat_cols]
    return SimPreparedData(
        train_df=train_df.reset_index(drop=True),
        eval_df=eval_df.reset_index(drop=True),
        feature_names=feature_names,
    )

