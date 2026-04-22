"""
Step 10a -- RL preprocessing for the broad non-causal interface.

Purpose
-------
Build the replay-style dataset used by the non-causal simulator branch from the
Step 09 non-causal interface dataset.

This side track parallels the active selected-causal step 10a, but differs in:
  - a much broader dynamic state space
  - broader static context
  - broader binary action set
  - SOFA retained as an auxiliary support / reward column rather than a default
    model input

Outputs
-------
  data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet
  data/processed/icu_readmit/step_10_noncausal/static_context_noncausal.parquet
  data/processed/icu_readmit/step_10_noncausal/scaler_params_noncausal.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SPLIT_FRACS = (0.70, 0.15, 0.15)
TERMINAL_REWARD = 15.0
Z_CLIP = 5.0

STATIC_CONTINUOUS = [
    "age",
    "Weight_kg",
    "charlson_score",
    "prior_ed_visits_6m",
]

STATIC_BINARY = [
    "gender",
    "re_admission",
]

STATIC_CATEGORICAL = [
    "race",
    "insurance",
    "marital_status",
    "admission_type",
    "admission_location",
]

DYNAMIC_CLIP_BOUNDS = {
    "HR": (0, 250),
    "SysBP": (0, 300),
    "MeanBP": (0, 200),
    "DiaBP": (0, 200),
    "NIBP_Diastolic": (0, 200),
    "Arterial_BP_Sys": (0, 300),
    "Arterial_BP_Dia": (0, 200),
    "RR": (0, 80),
    "RR_Spontaneous": (0, 80),
    "RR_Total": (0, 80),
    "SpO2": (0, 100),
    "Temp_C": (25, 45),
    "FiO2_1": (0.2, 1.0),
    "O2flow": (0, 70),
    "PEEP": (0, 40),
    "TidalVolume": (0, 1800),
    "TidalVolume_Observed": (0, 1800),
    "MinuteVentil": (0, 50),
    "PAWmean": (0, 50),
    "PAWpeak": (0, 80),
    "PAWplateau": (0, 60),
    "Pain_Level": (0, 10),
    "GCS": (3, 15),
    "Potassium": (1, 12),
    "Sodium": (95, 178),
    "Chloride": (70, 150),
    "Glucose": (1, 1000),
    "BUN": (1, 300),
    "Creatinine": (0.1, 50),
    "Magnesium": (0.5, 10),
    "Calcium": (4, 20),
    "Ionised_Ca": (0.2, 5),
    "CO2_mEqL": (5, 60),
    "SGOT": (0, 10000),
    "SGPT": (0, 10000),
    "Total_bili": (0, 100),
    "Direct_bili": (0, 70),
    "Albumin": (0.5, 7),
    "Hb": (1, 25),
    "Ht": (5, 70),
    "RBC_count": (0.5, 10),
    "WBC_count": (0, 500),
    "Platelets_count": (1, 2000),
    "PTT": (10, 200),
    "PT": (7, 200),
    "INR": (0.5, 25),
    "Arterial_pH": (6.7, 7.8),
    "paO2": (10, 700),
    "paCO2": (5, 200),
    "Arterial_BE": (-50, 40),
    "Arterial_lactate": (0.1, 30),
    "HCO3": (2, 55),
    "Phosphate": (0.5, 15),
    "Anion_Gap": (0, 40),
    "Alkaline_Phosphatase": (0, 5000),
    "Fibrinogen": (50, 1000),
    "Neuts_pct": (0, 100),
    "Lymphs_pct": (0, 100),
    "Monos_pct": (0, 100),
}

STATIC_CLIP_BOUNDS = {
    "age": (18, 100),
    "Weight_kg": (0.5, 300),
    "charlson_score": (0, 37),
    "prior_ed_visits_6m": (0, 20),
}

LOG_TRANSFORM_DYNAMIC = {
    "BUN",
    "Creatinine",
    "SGOT",
    "SGPT",
    "Total_bili",
    "Direct_bili",
    "Arterial_lactate",
    "Alkaline_Phosphatase",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "icu_readmit"
            / "step_09_noncausal_interface"
            / "noncausal_interface_dataset.parquet"
        ),
    )
    parser.add_argument(
        "--spec",
        default=str(
            PROJECT_ROOT
            / "data"
            / "processed"
            / "icu_readmit"
            / "step_09_noncausal_interface"
            / "noncausal_interface_spec.json"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "step_10_noncausal"),
    )
    parser.add_argument("--smoke", action="store_true", help="Run on first 2000 stays only")
    parser.add_argument("--log", default=str(PROJECT_ROOT / "logs" / "step_10a_noncausal.log"))
    return parser


def setup_logging(log_path: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(path, mode="w", encoding="utf-8")],
    )


def assign_splits(stay_ids: np.ndarray, fracs=(0.70, 0.15, 0.15)) -> dict[int, str]:
    ids = np.sort(stay_ids)
    n = len(ids)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    split_map: dict[int, str] = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            split_map[int(sid)] = "train"
        elif i < n_train + n_val:
            split_map[int(sid)] = "val"
        else:
            split_map[int(sid)] = "test"
    return split_map


def zscore_train_only(
    df: pd.DataFrame,
    split_col: str,
    feature: str,
    clip_bounds: tuple[float, float] | None = None,
    log_transform: bool = False,
) -> tuple[pd.Series, dict]:
    x = df[feature].astype(float).copy()
    if clip_bounds is not None:
        x = x.clip(*clip_bounds)
    if log_transform:
        x = np.log1p(x)

    train_mask = df[split_col] == "train"
    mean = float(x[train_mask].mean())
    if not np.isfinite(mean):
        mean = 0.0
    x = x.fillna(mean)
    std = float(x[train_mask].std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0

    z = ((x - mean) / std).clip(-Z_CLIP, Z_CLIP)
    return z.astype(np.float32), {"mean": mean, "std": std}


def validate_no_missing_or_inf(df: pd.DataFrame, cols: list[str], label: str) -> None:
    problems = []
    for col in cols:
        series = df[col]
        missing = int(series.isna().sum())
        infs = 0
        if pd.api.types.is_numeric_dtype(series):
            infs = int(np.isinf(series.to_numpy(dtype=np.float64, copy=False)).sum())
        if missing or infs:
            problems.append({"col": col, "missing": missing, "inf": infs})
    if problems:
        preview = ", ".join(f"{p['col']}(na={p['missing']},inf={p['inf']})" for p in problems[:10])
        raise ValueError(f"{label} contains invalid values: {preview}")


def build_action_id(df: pd.DataFrame, action_cols: list[str]) -> pd.Series:
    weights = {col: 1 << i for i, col in enumerate(action_cols)}
    return sum(df[col].astype(int) * w for col, w in weights.items()).astype(np.int32)


def build_category_codes(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, dict]:
    out = df.copy()
    maps: dict[str, dict] = {}
    for col in cols:
        categories = sorted(out[col].dropna().astype(str).unique().tolist())
        mapping = {cat: i for i, cat in enumerate(categories)}
        coded = out[col].astype("string").map(mapping)
        out[f"{col}_code"] = coded.fillna(-1).astype(np.int32)
        maps[col] = {"missing_code": -1, "categories": mapping}
    return out, maps


def build_transitions(
    df: pd.DataFrame,
    dynamic_input_cols: list[str],
    static_input_cols: list[str],
    action_cols: list[str],
) -> pd.DataFrame:
    df_s = df.sort_values(["icustayid", "bloc"]).copy()

    for feat in dynamic_input_cols:
        df_s[f"s_next_{feat}"] = df_s.groupby("icustayid")[f"s_{feat}"].shift(-1)
    for feat in static_input_cols:
        df_s[f"s_next_{feat}"] = df_s[f"s_{feat}"]

    df_s["SOFA_next"] = df_s.groupby("icustayid")["SOFA"].shift(-1)
    df_s["done"] = (
        df_s.groupby("icustayid")["bloc"].transform("max") == df_s["bloc"]
    ).astype(np.int8)
    df_s["reward_sofa"] = np.where(
        df_s["done"] == 0,
        df_s["SOFA"] - df_s["SOFA_next"],
        0.0,
    ).astype(np.float32)
    df_s["reward_terminal_readmit"] = np.where(
        df_s["done"] == 1,
        np.where(df_s["readmit_30d"] == 0, TERMINAL_REWARD, -TERMINAL_REWARD),
        0.0,
    ).astype(np.float32)
    df_s["reward_default"] = (df_s["reward_sofa"] + df_s["reward_terminal_readmit"]).astype(np.float32)

    for feat in dynamic_input_cols:
        df_s[f"s_next_{feat}"] = df_s[f"s_next_{feat}"].fillna(0.0)
    for feat in static_input_cols:
        df_s[f"s_next_{feat}"] = df_s[f"s_next_{feat}"].fillna(df_s[f"s_{feat}"])

    # Ensure action columns are compact ints.
    for col in action_cols:
        df_s[col] = df_s[col].fillna(0).astype(np.int8)

    return df_s


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(args.log)

    input_path = Path(args.input)
    spec_path = Path(args.spec)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(spec_path, "r", encoding="utf-8") as f:
        interface_spec = json.load(f)

    dynamic_cols = interface_spec["dynamic_state_cols"]
    static_cols = interface_spec["static_context_cols"]
    action_cols = interface_spec["action_cols"]
    aux_cols = interface_spec["aux_cols"]
    meta_cols = interface_spec["meta_cols"]

    read_cols = meta_cols + static_cols + dynamic_cols + action_cols + aux_cols
    read_cols = list(dict.fromkeys(read_cols))

    logging.info("Loading non-causal interface: %s", input_path)
    df = pd.read_parquet(input_path, columns=read_cols)
    logging.info("Loaded %d rows, %d stays", len(df), df["icustayid"].nunique())

    if args.smoke:
        smoke_stays = np.sort(df["icustayid"].unique())[:2000]
        df = df[df["icustayid"].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df["icustayid"].nunique())

    split_map = assign_splits(df["icustayid"].unique(), SPLIT_FRACS)
    df["split"] = df["icustayid"].map(split_map)
    logging.info(
        "Split stays -- train=%d val=%d test=%d",
        sum(v == "train" for v in split_map.values()),
        sum(v == "val" for v in split_map.values()),
        sum(v == "test" for v in split_map.values()),
    )

    scaler_params: dict[str, dict] = {
        "dynamic": {},
        "static_continuous": {},
        "static_binary": {},
        "static_categorical": {},
        "action": {"cols": action_cols},
    }

    # Dynamic preprocessing.
    for feat in dynamic_cols:
        z, stats = zscore_train_only(
            df,
            split_col="split",
            feature=feat,
            clip_bounds=DYNAMIC_CLIP_BOUNDS.get(feat),
            log_transform=feat in LOG_TRANSFORM_DYNAMIC,
        )
        df[f"s_{feat}"] = z
        scaler_params["dynamic"][feat] = {
            **stats,
            "clip": list(DYNAMIC_CLIP_BOUNDS.get(feat, (None, None))),
            "log1p": feat in LOG_TRANSFORM_DYNAMIC,
        }

    # Static continuous preprocessing.
    for feat in STATIC_CONTINUOUS:
        z, stats = zscore_train_only(
            df,
            split_col="split",
            feature=feat,
            clip_bounds=STATIC_CLIP_BOUNDS.get(feat),
            log_transform=False,
        )
        df[f"s_{feat}"] = z
        scaler_params["static_continuous"][feat] = {
            **stats,
            "clip": list(STATIC_CLIP_BOUNDS.get(feat, (None, None))),
            "log1p": False,
        }

    # Static binary kept as raw 0/1.
    for feat in STATIC_BINARY:
        df[f"s_{feat}"] = df[feat].fillna(0).astype(np.int8)
        scaler_params["static_binary"][feat] = {"encoding": "raw_binary"}

    # Static categoricals integer-coded for later embeddings.
    df, category_maps = build_category_codes(df, STATIC_CATEGORICAL)
    for feat in STATIC_CATEGORICAL:
        scaler_params["static_categorical"][feat] = {
            "encoding": "int_code",
            "n_categories": len(category_maps[feat]["categories"]),
            "mapping": category_maps[feat],
        }

    static_input_cols = (
        STATIC_CONTINUOUS
        + STATIC_BINARY
        + [f"{feat}_code" for feat in STATIC_CATEGORICAL]
    )

    # Mirror categorical encoded columns into s_* namespace for model input.
    for feat in STATIC_CATEGORICAL:
        df[f"s_{feat}_code"] = df[f"{feat}_code"].astype(np.int32)

    static_input_cols = [
        *STATIC_CONTINUOUS,
        *STATIC_BINARY,
        *[f"{feat}_code" for feat in STATIC_CATEGORICAL],
    ]

    df["a"] = build_action_id(df, action_cols)
    scaler_params["action"]["n_actions"] = int(2 ** len(action_cols))
    scaler_params["action"]["id_weights"] = {col: int(1 << i) for i, col in enumerate(action_cols)}

    df_t = build_transitions(df, dynamic_cols, static_input_cols, action_cols)

    keep_cols = (
        ["icustayid", "bloc", "timestep", "split", "done", "readmit_30d", "SOFA", "SOFA_next"]
        + [f"s_{feat}" for feat in dynamic_cols]
        + [f"s_{feat}" for feat in STATIC_CONTINUOUS]
        + [f"s_{feat}" for feat in STATIC_BINARY]
        + [f"s_{feat}_code" for feat in STATIC_CATEGORICAL]
        + ["a"]
        + action_cols
        + ["reward_sofa", "reward_terminal_readmit", "reward_default"]
        + [f"s_next_{feat}" for feat in dynamic_cols]
        + [f"s_next_{feat}" for feat in STATIC_CONTINUOUS]
        + [f"s_next_{feat}" for feat in STATIC_BINARY]
        + [f"s_next_{feat}_code" for feat in STATIC_CATEGORICAL]
    )
    out_df = df_t[keep_cols].copy()

    validate_cols = (
        [f"s_{feat}" for feat in dynamic_cols]
        + [f"s_{feat}" for feat in STATIC_CONTINUOUS]
        + [f"s_{feat}" for feat in STATIC_BINARY]
        + [f"s_{feat}_code" for feat in STATIC_CATEGORICAL]
        + [f"s_next_{feat}" for feat in dynamic_cols]
        + [f"s_next_{feat}" for feat in STATIC_CONTINUOUS]
        + [f"s_next_{feat}" for feat in STATIC_BINARY]
        + [f"s_next_{feat}_code" for feat in STATIC_CATEGORICAL]
        + action_cols
        + ["done", "readmit_30d", "reward_sofa", "reward_terminal_readmit", "reward_default"]
    )
    validate_no_missing_or_inf(out_df, validate_cols, "step_10_noncausal replay output")

    dataset_name = "rl_dataset_noncausal_smoke.parquet" if args.smoke else "rl_dataset_noncausal.parquet"
    scaler_name = "scaler_params_noncausal_smoke.json" if args.smoke else "scaler_params_noncausal.json"
    static_name = "static_context_noncausal_smoke.parquet" if args.smoke else "static_context_noncausal.parquet"

    dataset_path = out_dir / dataset_name
    scaler_path = out_dir / scaler_name
    static_path = out_dir / static_name

    out_df.to_parquet(dataset_path, index=False)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dataset": str(input_path),
                "dynamic_cols": dynamic_cols,
                "static_continuous": STATIC_CONTINUOUS,
                "static_binary": STATIC_BINARY,
                "static_categorical": STATIC_CATEGORICAL,
                "action_cols": action_cols,
                "preprocessing": scaler_params,
            },
            f,
            indent=2,
        )

    static_keep_cols = (
        ["icustayid", "split"]
        + STATIC_CONTINUOUS
        + STATIC_BINARY
        + STATIC_CATEGORICAL
        + [f"{feat}_code" for feat in STATIC_CATEGORICAL]
        + [f"s_{feat}" for feat in STATIC_CONTINUOUS]
        + [f"s_{feat}" for feat in STATIC_BINARY]
        + [f"s_{feat}_code" for feat in STATIC_CATEGORICAL]
    )
    (
        df[static_keep_cols]
        .drop_duplicates("icustayid")
        .sort_values("icustayid")
        .to_parquet(static_path, index=False)
    )

    logging.info("Saved replay dataset: %s", dataset_path)
    logging.info("Saved scalers/spec:  %s", scaler_path)
    logging.info("Saved static ctx:    %s", static_path)
    logging.info("Rows=%d, stays=%d", len(out_df), out_df["icustayid"].nunique())
    logging.info("Dynamic dim=%d, static dim=%d, action dim=%d", len(dynamic_cols), len(static_input_cols), len(action_cols))


if __name__ == "__main__":
    main()
