"""Build hourly treatment-aware transitions (v3) from extracted MIMIC hourly rows."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pandas as pd


def _build_action_label(df: pd.DataFrame, classes: dict[str, list[int]]) -> pd.Series:
    reverse = {tuple(int(v) for v in bits): label for label, bits in classes.items()}
    keys = list(zip(df["a_t_vaso"].astype(int), df["a_t_vent"].astype(int), df["a_t_crrt"].astype(int)))
    out = pd.Series(keys, index=df.index).map(reverse)
    if out.isna().any():
        bad = pd.Series(keys, index=df.index)[out.isna()].iloc[0]
        raise ValueError(f"Unmapped action tuple encountered: {bad}")
    return out.astype(str)


def build_transitions_v3_hourly(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    required = {
        "subject_id",
        "hadm_id",
        "stay_id",
        "hr",
        "starttime",
        "endtime",
        "s_t_sofa",
        "s_t_mbp",
        "s_t_heart_rate",
        "s_t_resp_rate",
        "s_t_spo2",
        "s_t_creatinine",
        "s_t_bun",
        "s_t_age",
        "s_t_charlson",
        "a_t_vaso",
        "a_t_vent",
        "a_t_crrt",
    }
    missing = sorted(c for c in required if c not in df.columns)
    if missing:
        raise ValueError(f"Missing required hourly extraction columns: {missing}")

    data = df.copy()
    data["subject_id"] = pd.to_numeric(data["subject_id"], errors="coerce")
    data["hadm_id"] = pd.to_numeric(data["hadm_id"], errors="coerce")
    data["stay_id"] = pd.to_numeric(data["stay_id"], errors="coerce")
    data["hr"] = pd.to_numeric(data["hr"], errors="coerce")
    data = data.dropna(subset=["subject_id", "hadm_id", "stay_id", "hr", "endtime"]).copy()
    data["subject_id"] = data["subject_id"].astype("int64")
    data["hadm_id"] = data["hadm_id"].astype("int64")
    data["stay_id"] = data["stay_id"].astype("int64")
    data["hr"] = data["hr"].astype("int64")

    for col in [
        "s_t_sofa",
        "s_t_mbp",
        "s_t_heart_rate",
        "s_t_resp_rate",
        "s_t_spo2",
        "s_t_creatinine",
        "s_t_bun",
        "s_t_age",
        "s_t_charlson",
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    for col in ["a_t_vaso", "a_t_vent", "a_t_crrt"]:
        data[col] = pd.to_numeric(data[col], errors="coerce").fillna(0).astype(int).clip(0, 1)

    data = data.sort_values(["stay_id", "hr"], kind="stable").reset_index(drop=True)
    g = data.groupby("stay_id", sort=False)

    threshold = int(cfg["outcome"]["sofa_jump_threshold"])
    data["s_t1_sofa"] = g["s_t_sofa"].shift(-1)
    sofa_delta = data["s_t1_sofa"] - data["s_t_sofa"]
    data["y_t1"] = ((sofa_delta >= float(threshold)) & sofa_delta.notna()).astype(int)
    data["done"] = (g["hr"].shift(-1).isna()).astype(int)

    for col in [
        "s_t_mbp",
        "s_t_heart_rate",
        "s_t_resp_rate",
        "s_t_spo2",
        "s_t_creatinine",
        "s_t_bun",
        "s_t_age",
        "s_t_charlson",
    ]:
        data[f"s_t1_{col[4:]}"] = g[col].shift(-1)

    data["a_t"] = _build_action_label(data, cfg["action"]["classes"])
    data["transition_id"] = data["stay_id"].astype(str) + "_" + data["hr"].astype(str)
    data["patient_id"] = data["subject_id"]
    data["index_hadm_id"] = data["hadm_id"]
    data["index_admittime"] = pd.to_datetime(g["starttime"].transform("min"), errors="coerce")
    data["index_dischtime"] = pd.to_datetime(g["endtime"].transform("max"), errors="coerce")
    data["a_t_source"] = "mimiccode_treatment_v3"
    data["a_t_raw_discharge_location"] = "N/A"
    data["sample_tag"] = str(cfg.get("sample_tag", "full"))
    data["source_dataset"] = str(cfg["schema"]["source_dataset"])
    data["schema_version"] = str(cfg["schema"]["version"])
    data["build_timestamp_utc"] = datetime.now(timezone.utc).isoformat()

    out = data.rename(
        columns={
            "hr": "t",
            "s_t_sofa": "s_t_sofa",
            "s_t_mbp": "s_t_mbp",
            "s_t_heart_rate": "s_t_heart_rate",
            "s_t_resp_rate": "s_t_resp_rate",
            "s_t_spo2": "s_t_spo2",
            "s_t_creatinine": "s_t_creatinine",
            "s_t_bun": "s_t_bun",
            "s_t_age": "s_t_age",
            "s_t_charlson": "s_t_charlson",
        }
    )
    out["episode_id"] = out["stay_id"].astype(str)
    out["episode_step"] = out.groupby("episode_id", sort=False).cumcount().astype(int)

    ordered = [
        "episode_id",
        "episode_step",
        "transition_id",
        "patient_id",
        "index_hadm_id",
        "t",
        "index_admittime",
        "index_dischtime",
        "y_t1",
        "done",
        "a_t",
        "a_t_vaso",
        "a_t_vent",
        "a_t_crrt",
        "a_t_source",
        "a_t_raw_discharge_location",
        "s_t_sofa",
        "s_t_mbp",
        "s_t_heart_rate",
        "s_t_resp_rate",
        "s_t_spo2",
        "s_t_creatinine",
        "s_t_bun",
        "s_t_age",
        "s_t_charlson",
        "s_t1_sofa",
        "s_t1_mbp",
        "s_t1_heart_rate",
        "s_t1_resp_rate",
        "s_t1_spo2",
        "s_t1_creatinine",
        "s_t1_bun",
        "s_t1_age",
        "s_t1_charlson",
        "split",
        "sample_tag",
        "source_dataset",
        "schema_version",
        "build_timestamp_utc",
    ]
    if "split" not in out.columns:
        out["split"] = "train"
    return out[ordered].copy()
