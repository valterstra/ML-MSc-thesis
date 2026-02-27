"""Validation for transition datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from careai.actions.action_discharge_v1 import ACTION_SOURCE, ALLOWED_ACTIONS, A_TERMINAL, TERMINAL_LABELS

from .transition_schema import REQUIRED_TRANSITION_COLUMNS


def validate_transition_contract(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []

    missing = [c for c in REQUIRED_TRANSITION_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if "transition_id" in df.columns and df["transition_id"].duplicated().any():
        errors.append("transition_id is not unique")

    if "done" in df.columns and not (df["done"] == 1).all():
        errors.append("done must be 1 for all rows in v1")

    if "t" in df.columns and not (df["t"] == 0).all():
        errors.append("t must be 0 for all rows in v1")

    if "within_30d_next_admit" in df.columns:
        bad = ~df["within_30d_next_admit"].isin([0, 1])
        if bad.any():
            errors.append("within_30d_next_admit must be in {0,1}")

    if "split" in df.columns:
        bad = ~df["split"].isin(["train", "valid", "test"])
        if bad.any():
            errors.append("split must be one of train|valid|test")

    if "a_t" in df.columns:
        bad = ~df["a_t"].isin(ALLOWED_ACTIONS)
        if bad.any():
            errors.append(f"a_t contains invalid values. Allowed={sorted(ALLOWED_ACTIONS)}")

    if "a_t_source" in df.columns:
        bad = df["a_t_source"] != ACTION_SOURCE
        if bad.any():
            errors.append(f"a_t_source must be {ACTION_SOURCE} for all rows")

    if "a_t" in df.columns and "a_t_raw_discharge_location" in df.columns:
        is_terminal = df["a_t"] == A_TERMINAL
        if is_terminal.any():
            raw = df.loc[is_terminal, "a_t_raw_discharge_location"].fillna("")
            bad = ~raw.isin(TERMINAL_LABELS)
            if bad.any():
                errors.append("A_TERMINAL rows must have raw discharge in {'DIED','HOSPICE'}")

    if "delta_days_to_next_admit" in df.columns and "next_hadm_id" in df.columns:
        bad = df["next_hadm_id"].notna() & df["delta_days_to_next_admit"].isna()
        if bad.any():
            errors.append("delta_days_to_next_admit cannot be null when next_hadm_id exists")

    return {
        "ok": len(errors) == 0,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "errors": errors,
    }
