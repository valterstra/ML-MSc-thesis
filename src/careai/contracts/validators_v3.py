"""Validation for transition v3 hourly datasets."""

from __future__ import annotations

from typing import Any

import pandas as pd

from .transition_schema_v3 import REQUIRED_TRANSITION_V3_COLUMNS


ALLOWED_ACTIONS_V3 = {
    "A_NONE",
    "A_VASO",
    "A_VENT",
    "A_CRRT",
    "A_VASO_VENT",
    "A_VASO_CRRT",
    "A_VENT_CRRT",
    "A_VASO_VENT_CRRT",
}


def validate_transition_v3_contract(df: pd.DataFrame, sofa_jump_threshold: int = 2) -> dict[str, Any]:
    errors: list[str] = []

    missing = [c for c in REQUIRED_TRANSITION_V3_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"Missing required columns: {missing}")

    if "transition_id" in df.columns and df["transition_id"].duplicated().any():
        errors.append("transition_id is not unique")

    if "a_t" in df.columns and (~df["a_t"].isin(ALLOWED_ACTIONS_V3)).any():
        errors.append(f"a_t contains invalid values. Allowed={sorted(ALLOWED_ACTIONS_V3)}")

    for col in ["a_t_vaso", "a_t_vent", "a_t_crrt", "y_t1", "done"]:
        if col in df.columns and (~df[col].isin([0, 1])).any():
            errors.append(f"{col} must be in {{0,1}}")

    if all(c in df.columns for c in ["done", "s_t1_sofa"]):
        bad = (df["done"] == 1) & df["s_t1_sofa"].notna()
        if bad.any():
            errors.append("done=1 rows must have null s_t1_* columns")

    if all(c in df.columns for c in ["s_t_sofa", "s_t1_sofa", "y_t1"]):
        expected = ((df["s_t1_sofa"] - df["s_t_sofa"] >= float(sofa_jump_threshold)) & df["s_t1_sofa"].notna()).astype(int)
        if not (expected == df["y_t1"]).all():
            errors.append("y_t1 does not match sofa jump rule")

    if all(c in df.columns for c in ["episode_id", "episode_step"]):
        grouped = df.sort_values(["episode_id", "episode_step"]).groupby("episode_id", sort=False)["episode_step"]
        for _, s in grouped:
            vals = s.to_list()
            if vals != list(range(len(vals))):
                errors.append("episode_step must start at 0 and increment by 1 within each episode_id")
                break

    if "split" in df.columns and (~df["split"].isin(["train", "valid", "test"])).any():
        errors.append("split must be one of train|valid|test")

    return {"ok": len(errors) == 0, "rows": int(len(df)), "cols": int(len(df.columns)), "errors": errors}

