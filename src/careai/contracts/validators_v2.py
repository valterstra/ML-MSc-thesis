"""Validation for transition v2 multi-step datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from careai.actions.action_discharge_v1 import ACTION_SOURCE, ALLOWED_ACTIONS

from .transition_schema_v2 import REQUIRED_TRANSITION_V2_COLUMNS


def validate_transition_v2_contract(df: pd.DataFrame) -> dict[str, Any]:
    errors: list[str] = []

    missing = [c for c in REQUIRED_TRANSITION_V2_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if "transition_id" in df.columns and df["transition_id"].duplicated().any():
        errors.append("transition_id is not unique")

    if "within_30d_next_admit" in df.columns:
        bad = ~df["within_30d_next_admit"].isin([0, 1])
        if bad.any():
            errors.append("within_30d_next_admit must be in {0,1}")

    if "y_t1" in df.columns:
        bad = ~df["y_t1"].isin([0, 1])
        if bad.any():
            errors.append("y_t1 must be in {0,1}")
        if "within_30d_next_admit" in df.columns and not (df["y_t1"] == df["within_30d_next_admit"]).all():
            errors.append("y_t1 must equal within_30d_next_admit")

    if "done" in df.columns:
        bad = ~df["done"].isin([0, 1])
        if bad.any():
            errors.append("done must be in {0,1}")

    if "a_t" in df.columns:
        bad = ~df["a_t"].isin(ALLOWED_ACTIONS)
        if bad.any():
            errors.append(f"a_t contains invalid values. Allowed={sorted(ALLOWED_ACTIONS)}")

    if "a_t_source" in df.columns:
        bad = df["a_t_source"] != ACTION_SOURCE
        if bad.any():
            errors.append(f"a_t_source must be {ACTION_SOURCE}")

    if "split" in df.columns:
        bad = ~df["split"].isin(["train", "valid", "test"])
        if bad.any():
            errors.append("split must be one of train|valid|test")

    next_state_cols = ["s_t1_length", "s_t1_acuity", "s_t1_comorbidity", "s_t1_lace"]
    if all(c in df.columns for c in next_state_cols) and "done" in df.columns:
        bad = (df["done"] == 1) & df[next_state_cols].notna().any(axis=1)
        if bad.any():
            errors.append("done=1 rows must have null s_t1_* columns")
        bad2 = (df["within_30d_next_admit"] == 1) & df[next_state_cols].isna().any(axis=1)
        if bad2.any():
            errors.append("within_30d_next_admit=1 rows must have non-null s_t1_* columns")

    if all(c in df.columns for c in ["episode_id", "episode_step"]):
        grouped = df.sort_values(["episode_id", "episode_step"]).groupby("episode_id", sort=False)["episode_step"]
        bad_step = False
        for _, s in grouped:
            vals = s.to_list()
            if not vals:
                continue
            expected = list(range(len(vals)))
            if vals != expected:
                bad_step = True
                break
        if bad_step:
            errors.append("episode_step must start at 0 and increment by 1 within each episode_id")

    return {
        "ok": len(errors) == 0,
        "rows": int(len(df)),
        "cols": int(len(df.columns)),
        "errors": errors,
    }

