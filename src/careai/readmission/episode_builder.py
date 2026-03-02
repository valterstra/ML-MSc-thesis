"""Episode-level aggregation from hourly transitions."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


STATE_COLS = [
    "s_t_sofa",
    "s_t_mbp",
    "s_t_heart_rate",
    "s_t_resp_rate",
    "s_t_spo2",
    "s_t_gcs",
    "s_t_urine_output_rate",
    "s_t_oxygen_delivery",
    "s_t_creatinine",
    "s_t_bun",
    "s_t_age",
    "s_t_charlson",
]


def _safe_numeric(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out


def build_episode_table(transitions_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    required = {
        "episode_id",
        "patient_id",
        "index_hadm_id",
        "split",
        "index_admittime",
        "index_dischtime",
        "t",
        "a_t",
        "a_t_vaso",
        "a_t_vent",
        "a_t_crrt",
        *STATE_COLS,
    }
    missing = sorted(c for c in required if c not in transitions_df.columns)
    if missing:
        raise ValueError(f"Transitions missing required columns: {missing}")

    df = transitions_df.copy()
    df = _safe_numeric(df, ["t", "a_t_vaso", "a_t_vent", "a_t_crrt", *STATE_COLS])
    df["index_admittime"] = pd.to_datetime(df["index_admittime"], errors="coerce")
    df["index_dischtime"] = pd.to_datetime(df["index_dischtime"], errors="coerce")
    df = df.sort_values(["episode_id", "t"], kind="stable").reset_index(drop=True)

    g = df.groupby("episode_id", sort=False)

    first = g.first().reset_index()[["episode_id", "patient_id", "index_hadm_id", "split", "index_admittime", "index_dischtime"]]
    size = g.size().rename("episode_hours").reset_index()

    mean_df = g[STATE_COLS].mean().rename(columns={c: f"mean_{c}" for c in STATE_COLS}).reset_index()
    minmax_df = pd.DataFrame(
        {
            "episode_id": g.size().index,
            "min_s_t_mbp": g["s_t_mbp"].min().values,
            "max_s_t_sofa": g["s_t_sofa"].max().values,
        }
    )

    first_state = g[STATE_COLS].first().reset_index()
    last_state = g[STATE_COLS].last().rename(columns={c: f"last_{c}" for c in STATE_COLS}).reset_index()

    delta = first_state.merge(last_state, on="episode_id", how="inner")
    delta["delta_s_t_sofa"] = delta["last_s_t_sofa"] - delta["s_t_sofa"]
    delta["delta_s_t_mbp"] = delta["last_s_t_mbp"] - delta["s_t_mbp"]
    delta = delta[["episode_id", "delta_s_t_sofa", "delta_s_t_mbp"]]

    exposure = g[["a_t_vaso", "a_t_vent", "a_t_crrt"]].mean().rename(
        columns={"a_t_vaso": "frac_vaso", "a_t_vent": "frac_vent", "a_t_crrt": "frac_crrt"}
    ).reset_index()
    anyexp = g[["a_t_vaso", "a_t_vent", "a_t_crrt"]].max().rename(
        columns={"a_t_vaso": "any_vaso", "a_t_vent": "any_vent", "a_t_crrt": "any_crrt"}
    ).reset_index()

    epi = first.merge(size, on="episode_id", how="left")
    for part in [mean_df, minmax_df, last_state, delta, exposure, anyexp]:
        epi = epi.merge(part, on="episode_id", how="left")

    labels = labels_df.copy()
    labels["patient_id"] = pd.to_numeric(labels["patient_id"], errors="coerce")
    labels["index_hadm_id"] = pd.to_numeric(labels["index_hadm_id"], errors="coerce")
    labels["readmit_30d"] = pd.to_numeric(labels["readmit_30d"], errors="coerce").fillna(0).astype(int)
    labels["days_to_next_admit"] = pd.to_numeric(labels["days_to_next_admit"], errors="coerce")
    labels = labels[["patient_id", "index_hadm_id", "days_to_next_admit", "readmit_30d"]]

    epi["patient_id"] = pd.to_numeric(epi["patient_id"], errors="coerce")
    epi["index_hadm_id"] = pd.to_numeric(epi["index_hadm_id"], errors="coerce")
    out = epi.merge(labels, on=["patient_id", "index_hadm_id"], how="left")
    out["readmit_30d"] = out["readmit_30d"].fillna(0).astype(int)
    return out
