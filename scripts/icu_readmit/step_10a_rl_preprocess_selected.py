"""
Step 10a -- RL preprocessing: expanded Step 09-selected state/action set.

PURPOSE
-------
Build the replay-style dataset used by the next CARE-Sim track, using the
updated Step 09 recommendation anchored in:
  - step_03 state -> readmission relevance
  - step_04b robust action -> state controllability

STATE FEATURES (9 total = 6 dynamic + 3 static)
------------------------------------------------
  Dynamic:
    Hb
    BUN
    Creatinine
    Phosphate
    HR
    Chloride

  Static:
    age
    charlson_score
    prior_ed_visits_6m

ACTIONS (5 binary, 32 combos)
-----------------------------
  vasopressor_b, ivfluid_b, antibiotic_b, diuretic_b, mechvent_b

REWARD
------
  Dense:    r = SOFA_t - SOFA_{t+1}
  Terminal: r = +15 (no readmit) / -15 (readmit)

OUTPUTS
-------
  data/processed/icu_readmit/rl_dataset_selected.parquet
  data/processed/icu_readmit/scaler_params_selected.json
  data/processed/icu_readmit/static_context_selected.parquet

Usage:
    python scripts/icu_readmit/step_10a_rl_preprocess_selected.py --smoke
    python scripts/icu_readmit/step_10a_rl_preprocess_selected.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import C_BLOC, C_ICUSTAYID, C_READMIT_30D


DYNAMIC_STATE = ["Hb", "BUN", "Creatinine", "Phosphate", "HR", "Chloride"]
STATIC_STATE = ["age", "charlson_score", "prior_ed_visits_6m"]
STATE_FEATURES = DYNAMIC_STATE + STATIC_STATE

ACTIONS = [
    ("vasopressor_b", 1),
    ("ivfluid_b", 2),
    ("antibiotic_b", 4),
    ("diuretic_b", 8),
    ("mechvent_b", 16),
]
ACTION_NAMES = [name for name, _ in ACTIONS]

ACTION_SOURCES = {
    "vasopressor_b": ("vasopressor_dose", "dose"),
    "ivfluid_b": ("ivfluid_dose", "dose"),
    "antibiotic_b": ("antibiotic_active", "binary"),
    "diuretic_b": ("diuretic_active", "binary"),
    "mechvent_b": ("mechvent", "binary"),
}

STATIC_CONTEXT_EXTRA = ["gender", "race", "re_admission"]

SPLIT_FRACS = (0.70, 0.15, 0.15)
TERMINAL_REWARD = 15.0
N_ACTIONS = 32
Z_CLIP = 5.0

CLIP_BOUNDS_DYNAMIC = {
    "Hb": (1, 25),
    "BUN": (1, 200),
    "Creatinine": (0.1, 25),
    # Conservative lab bounds to suppress recording artifacts without flattening
    # ordinary ICU variation.
    "Phosphate": (0.1, 20),
    "HR": (15, 300),
    "Chloride": (70, 150),
}

LOG_TRANSFORM_DYNAMIC = ["BUN", "Creatinine"]

CLIP_BOUNDS_STATIC = {
    "age": (18, 100),
    "charlson_score": (0, 37),
    "prior_ed_visits_6m": (0, 20),
}


def assign_splits(stay_ids, fracs=(0.70, 0.15, 0.15)):
    ids = np.sort(stay_ids)
    n = len(ids)
    n_train = int(n * fracs[0])
    n_val = int(n * fracs[1])
    split_map = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            split_map[sid] = "train"
        elif i < n_train + n_val:
            split_map[sid] = "val"
        else:
            split_map[sid] = "test"
    return split_map


def build_binary_actions(df):
    df = df.copy()
    for col, (src, kind) in ACTION_SOURCES.items():
        if src not in df.columns:
            logging.warning("Action source %s not found -- %s=0", src, col)
            df[col] = 0
        elif kind == "dose":
            df[col] = (df[src] > 0).astype(int)
        else:
            df[col] = df[src].fillna(0).astype(int)
    df["a"] = sum(df[col] * weight for col, weight in ACTIONS)
    return df


def build_transitions(df):
    df_s = df.sort_values([C_ICUSTAYID, C_BLOC]).copy()

    for feat in DYNAMIC_STATE:
        df_s[f"s_next_{feat}"] = df_s.groupby(C_ICUSTAYID)[f"s_{feat}"].shift(-1)

    for feat in STATIC_STATE:
        df_s[f"s_next_{feat}"] = df_s[f"s_{feat}"]

    df_s["SOFA_next"] = df_s.groupby(C_ICUSTAYID)["SOFA"].shift(-1)
    df_s["done"] = (
        df_s.groupby(C_ICUSTAYID)[C_BLOC].transform("max") == df_s[C_BLOC]
    ).astype(int)

    df_s["r"] = np.where(
        df_s["done"] == 0,
        df_s["SOFA"] - df_s["SOFA_next"],
        np.where(df_s[C_READMIT_30D] == 0, TERMINAL_REWARD, -TERMINAL_REWARD),
    )

    for feat in DYNAMIC_STATE:
        df_s[f"s_next_{feat}"] = df_s[f"s_next_{feat}"].fillna(0.0)

    return df_s


def zscore_train_only(df, split_col, feature, clip_bounds=None, log_transform=False):
    x = df[feature].astype(float).copy()
    if clip_bounds is not None:
        x = x.clip(*clip_bounds)
    if log_transform:
        x = np.log1p(x)

    train_mask = df[split_col] == "train"
    mean = float(x[train_mask].mean())
    std = float(x[train_mask].std())
    if not np.isfinite(std) or std < 1e-8:
        std = 1.0

    z = ((x - mean) / std).clip(-Z_CLIP, Z_CLIP)
    return z.astype(np.float32), {"mean": mean, "std": std}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "ICUdataset.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit"),
    )
    parser.add_argument("--smoke", action="store_true", help="Run on first 2000 stays only")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    log_file = args.log or str(PROJECT_ROOT / "logs" / "step_10a.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10a (selected Step 09-expanded set) started.")
    logging.info("Input: %s", args.input)

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    if args.smoke:
        smoke_stays = np.sort(df[C_ICUSTAYID].unique())[:2000]
        df = df[df[C_ICUSTAYID].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    missing_state = [c for c in STATE_FEATURES if c not in df.columns]
    missing_action = [src for src, _ in ACTION_SOURCES.values() if src not in df.columns]
    if "SOFA" not in df.columns:
        logging.warning("SOFA column not found -- dense reward will be NaN")
    if missing_state:
        logging.warning("Missing state columns: %s", missing_state)
    if missing_action:
        logging.warning("Missing action columns: %s", missing_action)

    df = build_binary_actions(df)
    logging.info("Action distribution (frac stays with action active):")
    for col, _ in ACTIONS:
        logging.info("  %-20s %.3f", col, df.groupby(C_ICUSTAYID)[col].max().mean())

    action_counts = df["a"].value_counts().sort_index()
    logging.info("Unique action combos: %d / %d", len(action_counts), N_ACTIONS)

    stay_ids = df[C_ICUSTAYID].unique()
    split_map = assign_splits(stay_ids, SPLIT_FRACS)
    df["split"] = df[C_ICUSTAYID].map(split_map)
    logging.info(
        "Split stays -- train=%d val=%d test=%d",
        sum(v == "train" for v in split_map.values()),
        sum(v == "val" for v in split_map.values()),
        sum(v == "test" for v in split_map.values()),
    )

    scaler_params = {"dynamic": {}, "static": {}}

    for feat in DYNAMIC_STATE:
        z, stats = zscore_train_only(
            df,
            split_col="split",
            feature=feat,
            clip_bounds=CLIP_BOUNDS_DYNAMIC.get(feat),
            log_transform=feat in LOG_TRANSFORM_DYNAMIC,
        )
        df[f"s_{feat}"] = z
        scaler_params["dynamic"][feat] = {
            **stats,
            "clip": list(CLIP_BOUNDS_DYNAMIC.get(feat, (None, None))),
            "log1p": feat in LOG_TRANSFORM_DYNAMIC,
        }

    for feat in STATIC_STATE:
        z, stats = zscore_train_only(
            df,
            split_col="split",
            feature=feat,
            clip_bounds=CLIP_BOUNDS_STATIC.get(feat),
            log_transform=False,
        )
        df[f"s_{feat}"] = z
        scaler_params["static"][feat] = {
            **stats,
            "clip": list(CLIP_BOUNDS_STATIC.get(feat, (None, None))),
            "log1p": False,
        }

    df_t = build_transitions(df)

    keep_cols = (
        [C_ICUSTAYID, C_BLOC, "split", C_READMIT_30D, "done"]
        + [f"s_{feat}" for feat in STATE_FEATURES]
        + ["a"]
        + ACTION_NAMES
        + ["r"]
        + [f"s_next_{feat}" for feat in STATE_FEATURES]
    )
    out_df = df_t[keep_cols].copy()

    dataset_name = "rl_dataset_selected_smoke.parquet" if args.smoke else "rl_dataset_selected.parquet"
    scaler_name = "scaler_params_selected_smoke.json" if args.smoke else "scaler_params_selected.json"
    static_name = "static_context_selected_smoke.parquet" if args.smoke else "static_context_selected.parquet"

    dataset_path = os.path.join(args.out_dir, dataset_name)
    scaler_path = os.path.join(args.out_dir, scaler_name)
    static_path = os.path.join(args.out_dir, static_name)

    out_df.to_parquet(dataset_path, index=False)
    with open(scaler_path, "w", encoding="utf-8") as f:
        json.dump(scaler_params, f, indent=2)

    static_cols = [C_ICUSTAYID, "split"] + STATIC_STATE + STATIC_CONTEXT_EXTRA
    static_cols = [c for c in static_cols if c in df.columns]
    (
        df[static_cols]
        .drop_duplicates(C_ICUSTAYID)
        .sort_values(C_ICUSTAYID)
        .to_parquet(static_path, index=False)
    )

    logging.info("Saved dataset: %s", dataset_path)
    logging.info("Saved scalers: %s", scaler_path)
    logging.info("Saved static context: %s", static_path)
    logging.info("Rows=%d, stays=%d", len(out_df), out_df[C_ICUSTAYID].nunique())
    logging.info("State dim=%d, action dim=%d", len(STATE_FEATURES), len(ACTION_NAMES))
