from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class HourlyData:
    transitions: pd.DataFrame
    train: pd.DataFrame
    eval_starts: pd.DataFrame
    state_cols: list[str]
    action_cols: list[str]


def prepare_hourly_data(df: pd.DataFrame, cfg: dict[str, Any]) -> HourlyData:
    state_cols = list(cfg["state"]["features_numeric"])
    action_cols = list(cfg["actions"]["components"])
    required = ["episode_id", "split", "t", "done"] + state_cols + action_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Transitions file missing required columns: {missing}")

    work = df.copy()
    for c in state_cols + action_cols + ["t", "done"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.sort_values(["episode_id", "t"]).reset_index(drop=True)
    train = work[work["split"] == "train"].copy()
    eval_starts = work[(work["split"] == "test") & (work["t"] == 0)].copy()
    if eval_starts.empty:
        eval_starts = work[work["t"] == 0].copy()
    if eval_starts.empty:
        raise ValueError("No episode start rows found (t == 0).")
    return HourlyData(transitions=work, train=train, eval_starts=eval_starts, state_cols=state_cols, action_cols=action_cols)

