"""Build multi-step event-driven transitions (v2)."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd

from careai.actions.action_discharge_v1 import derive_action_from_discharge
from careai.state.admission_state_v1 import add_state_columns

from .episode_chain import assign_episode_ids_and_steps


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["subject_id"] = pd.to_numeric(out["subject_id"], errors="coerce")
    out["hadm_id"] = pd.to_numeric(out["hadm_id"], errors="coerce")
    out = out.dropna(subset=["subject_id", "hadm_id", "admittime", "dischtime"]).copy()
    out["subject_id"] = out["subject_id"].astype("int64")
    out["hadm_id"] = out["hadm_id"].astype("int64")
    out["admittime"] = pd.to_datetime(out["admittime"], errors="coerce")
    out["dischtime"] = pd.to_datetime(out["dischtime"], errors="coerce")
    out = out.dropna(subset=["admittime", "dischtime"]).copy()
    out = out.sort_values(["subject_id", "admittime", "hadm_id"])
    out = add_state_columns(out)
    return out


def build_transitions_v2(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    days = int(cfg["horizon"]["readmit_days"])
    schema_version = str(cfg["schema"]["version"])
    source_dataset = str(cfg["schema"]["source_dataset"])
    sample_tag = str(cfg.get("sample_tag", "full"))
    ts = datetime.now(timezone.utc).isoformat()

    base = _prepare(df)
    g = base.groupby("subject_id", sort=False)

    base["next_hadm_id"] = g["hadm_id"].shift(-1)
    base["next_admittime"] = g["admittime"].shift(-1)
    base["delta_days_to_next_admit"] = (
        (base["next_admittime"] - base["dischtime"]).dt.total_seconds() / 86400.0
    )
    base["within_30d_next_admit"] = (
        base["delta_days_to_next_admit"].notna()
        & (base["delta_days_to_next_admit"] > 0)
        & (base["delta_days_to_next_admit"] <= float(days))
    ).astype(int)
    base["y_t1"] = base["within_30d_next_admit"].astype(int)
    base["done"] = (1 - base["within_30d_next_admit"]).astype(int)

    base["next_length"] = g["s_t_length"].shift(-1)
    base["next_acuity"] = g["s_t_acuity"].shift(-1)
    base["next_comorbidity"] = g["s_t_comorbidity"].shift(-1)
    base["next_lace"] = g["s_t_lace"].shift(-1)
    has_within30 = base["within_30d_next_admit"] == 1
    base["s_t1_length"] = base["next_length"].where(has_within30)
    base["s_t1_acuity"] = base["next_acuity"].where(has_within30)
    base["s_t1_comorbidity"] = base["next_comorbidity"].where(has_within30)
    base["s_t1_lace"] = base["next_lace"].where(has_within30)

    action_df = derive_action_from_discharge(base["discharge_location"])

    out = pd.DataFrame(
        {
            "transition_id": base["subject_id"].astype(str) + "_" + base["hadm_id"].astype(str),
            "patient_id": base["subject_id"],
            "index_hadm_id": base["hadm_id"],
            "index_admittime": base["admittime"],
            "index_dischtime": base["dischtime"],
            "next_hadm_id": base["next_hadm_id"],
            "next_admittime": base["next_admittime"],
            "delta_days_to_next_admit": base["delta_days_to_next_admit"],
            "within_30d_next_admit": base["within_30d_next_admit"].astype(int),
            "y_t1": base["y_t1"].astype(int),
            "done": base["done"].astype(int),
            "a_t": action_df["a_t"],
            "a_t_source": action_df["a_t_source"],
            "a_t_raw_discharge_location": action_df["a_t_raw_discharge_location"],
            "s_t_length": base["s_t_length"],
            "s_t_acuity": base["s_t_acuity"],
            "s_t_comorbidity": base["s_t_comorbidity"],
            "s_t_lace": base["s_t_lace"],
            "s_t_physical_status": base["s_t_physical_status"],
            "s_t_age": base["s_t_age"],
            "s_t_admission_type": base["s_t_admission_type"],
            "s_t_discharge_location": base["s_t_discharge_location"],
            "s_t1_length": base["s_t1_length"],
            "s_t1_acuity": base["s_t1_acuity"],
            "s_t1_comorbidity": base["s_t1_comorbidity"],
            "s_t1_lace": base["s_t1_lace"],
            "sample_tag": sample_tag,
            "source_dataset": source_dataset,
            "schema_version": schema_version,
            "build_timestamp_utc": ts,
        }
    )
    out = assign_episode_ids_and_steps(out)
    ordered_cols = [
        "episode_id",
        "episode_step",
        "transition_id",
        "patient_id",
        "index_hadm_id",
        "t",
        "index_admittime",
        "index_dischtime",
        "next_hadm_id",
        "next_admittime",
        "delta_days_to_next_admit",
        "within_30d_next_admit",
        "y_t1",
        "done",
        "a_t",
        "a_t_source",
        "a_t_raw_discharge_location",
        "s_t_length",
        "s_t_acuity",
        "s_t_comorbidity",
        "s_t_lace",
        "s_t_physical_status",
        "s_t_age",
        "s_t_admission_type",
        "s_t_discharge_location",
        "s_t1_length",
        "s_t1_acuity",
        "s_t1_comorbidity",
        "s_t1_lace",
        "split",
        "sample_tag",
        "source_dataset",
        "schema_version",
        "build_timestamp_utc",
    ]
    # split is assigned in pipeline runner; placeholder for schema consistency
    if "split" not in out.columns:
        out["split"] = "train"
    out = out[ordered_cols]
    return out

