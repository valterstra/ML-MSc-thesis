"""Load CSV and build one-step transition frames for sim_daily."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .features import (
    ACTION_COLS,
    INFECTION_CONTEXT,
    INPUT_COLS,
    MEASURED_FLAGS,
    OUTPUT_BINARY,
    OUTPUT_CONTINUOUS,
    STATIC_FEATURES,
    STATE_BINARY,
    STATE_CONTINUOUS,
)

# Columns that should be numeric
_NUMERIC_COLS = (
    STATE_CONTINUOUS + STATE_BINARY + STATIC_FEATURES + MEASURED_FLAGS
    + ACTION_COLS + INFECTION_CONTEXT
    + ["day_of_stay", "days_in_current_unit", "is_last_day"]
)


@dataclass(frozen=True)
class DailyData:
    raw: pd.DataFrame
    one_step_train: pd.DataFrame
    one_step_valid: pd.DataFrame
    one_step_test: pd.DataFrame
    initial_states: pd.DataFrame


def prepare_daily_data(csv_path: str | Path) -> DailyData:
    """Read the hosp_daily CSV and build per-split one-step transition frames."""
    df = pd.read_csv(csv_path)

    # Encode gender as binary
    df["gender_M"] = (df["gender"] == "M").astype(int)

    # Coerce numeric columns
    for c in _NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.sort_values(["hadm_id", "day_of_stay"]).reset_index(drop=True)

    # Build one-step frames per split
    splits: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "valid", "test"):
        sub = df[df["split"] == split_name].copy()
        splits[split_name] = _build_one_step_frame(sub)

    # Initial states: day_of_stay == 0 rows
    initial_states = df[df["day_of_stay"] == 0].copy()

    return DailyData(
        raw=df,
        one_step_train=splits["train"],
        one_step_valid=splits["valid"],
        one_step_test=splits["test"],
        initial_states=initial_states,
    )


def _build_one_step_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorised one-step transition builder using shift(-1) within groups."""
    output_cols = OUTPUT_CONTINUOUS + OUTPUT_BINARY

    df = df.sort_values(["hadm_id", "day_of_stay"]).reset_index(drop=True)
    grp = df.groupby("hadm_id", sort=False)

    # Create next-state columns via shift
    for c in output_cols:
        df[f"next_{c}"] = grp[c].shift(-1)

    # done_next: the *next* row is the last day of the episode
    df["done_next"] = grp["is_last_day"].shift(-1).fillna(0).astype(int)

    # Drop terminal rows (is_last_day==1) — no next state exists
    df = df[df["is_last_day"] != 1].copy()

    # Also drop rows where all next-state columns are NaN (safety)
    next_cols = [f"next_{c}" for c in output_cols]
    df = df.dropna(subset=next_cols, how="all")

    return df.reset_index(drop=True)
