"""
Step 10b -- Train a selected-state severity surrogate against real SOFA.

PURPOSE
-------
Learn a compact severity score using only the simulator's selected dynamic
state variables, with real SOFA as the supervision target.

This gives a state-derived severity surrogate that can later replace the
current generic reward head logic more cleanly:
  predict next_state -> derive severity(next_state) -> compute reward

DEFAULT FEATURE SET
-------------------
  Hb, BUN, Creatinine, Phosphate, HR, Chloride

MODEL
-----
  Ridge regression on the same clipped / transformed / train-fit-normalised
  features used in step_10a preprocessing.

OUTPUTS
-------
  models/icu_readmit/severity_selected/ridge_sofa_surrogate.joblib
  models/icu_readmit/severity_selected/severity_surrogate_config.json
  reports/icu_readmit/severity_selected/severity_surrogate_metrics.json
  reports/icu_readmit/severity_selected/severity_surrogate_coefficients.csv

Usage:
  python scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py
  python scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import C_ICUSTAYID


FEATURES = ["Hb", "BUN", "Creatinine", "Phosphate", "HR", "Chloride"]
TARGET = "SOFA"
SPLIT_FRACS = (0.70, 0.15, 0.15)
SOFA_RANGE = (0.0, 24.0)

CLIP_BOUNDS = {
    "Hb": (1, 25),
    "BUN": (1, 200),
    "Creatinine": (0.1, 25),
    "Phosphate": (0.1, 20),
    "HR": (15, 300),
    "Chloride": (70, 150),
}
LOG_TRANSFORM = {"BUN", "Creatinine"}


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


def transform_feature(series: pd.Series, feature: str) -> pd.Series:
    x = series.astype(float).clip(*CLIP_BOUNDS[feature])
    if feature in LOG_TRANSFORM:
        x = np.log1p(x)
    return x


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    split_map = assign_splits(df[C_ICUSTAYID].unique(), SPLIT_FRACS)
    work = df[[C_ICUSTAYID, TARGET, *FEATURES]].copy()
    work["split"] = work[C_ICUSTAYID].map(split_map)
    work = work.dropna(subset=[TARGET, *FEATURES]).reset_index(drop=True)

    scaler_params: dict[str, dict] = {}
    for feature in FEATURES:
        transformed = transform_feature(work[feature], feature)
        train_mask = work["split"] == "train"
        mean = float(transformed[train_mask].mean())
        std = float(transformed[train_mask].std())
        if not np.isfinite(std) or std < 1e-8:
            std = 1.0
        work[f"x_{feature}"] = ((transformed - mean) / std).astype(np.float32)
        scaler_params[feature] = {
            "mean": mean,
            "std": std,
            "clip": list(CLIP_BOUNDS[feature]),
            "log1p": feature in LOG_TRANSFORM,
        }
    return work, scaler_params


def evaluate(model, df: pd.DataFrame, feature_cols: list[str]) -> dict:
    x = df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_true = df[TARGET].to_numpy(dtype=np.float32, copy=True)
    y_pred = np.clip(model.predict(x), *SOFA_RANGE)
    rho = spearmanr(y_true, y_pred).statistic
    return {
        "n_rows": int(len(df)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "spearman": float(0.0 if np.isnan(rho) else rho),
        "y_true_mean": float(np.mean(y_true)),
        "y_pred_mean": float(np.mean(y_pred)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "ICUdataset.csv"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "icu_readmit" / "severity_selected"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "severity_selected"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    log_file = args.log or str(PROJECT_ROOT / "logs" / "step_10b.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10b (selected-state severity surrogate) started.")
    logging.info("Input: %s", args.input)

    use_cols = [C_ICUSTAYID, TARGET, *FEATURES]
    df = pd.read_csv(args.input, usecols=use_cols)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    if args.smoke:
        smoke_stays = np.sort(df[C_ICUSTAYID].unique())[:4000]
        df = df[df[C_ICUSTAYID].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    work, scaler_params = build_feature_frame(df)
    feature_cols = [f"x_{feature}" for feature in FEATURES]

    train_df = work[work["split"] == "train"].copy()
    val_df = work[work["split"] == "val"].copy()
    test_df = work[work["split"] == "test"].copy()
    logging.info(
        "Rows after dropna -- train=%d val=%d test=%d",
        len(train_df), len(val_df), len(test_df),
    )

    x_train = train_df[feature_cols].to_numpy(dtype=np.float32, copy=True)
    y_train = train_df[TARGET].to_numpy(dtype=np.float32, copy=True)

    alphas = np.array([0.01, 0.1, 1.0, 10.0, 100.0], dtype=np.float32)
    model = RidgeCV(alphas=alphas)
    model.fit(x_train, y_train)
    logging.info("Selected alpha: %.4f", float(model.alpha_))

    metrics = {
        "train": evaluate(model, train_df, feature_cols),
        "val": evaluate(model, val_df, feature_cols),
        "test": evaluate(model, test_df, feature_cols),
    }

    coef_df = pd.DataFrame({
        "feature": FEATURES,
        "coefficient": model.coef_.astype(float),
        "abs_coefficient": np.abs(model.coef_).astype(float),
    }).sort_values("abs_coefficient", ascending=False)

    model_payload = {
        "model": model,
        "feature_names": FEATURES,
        "feature_columns": feature_cols,
        "scaler_params": scaler_params,
        "target": TARGET,
        "target_clip": list(SOFA_RANGE),
        "selected_alpha": float(model.alpha_),
        "intercept": float(model.intercept_),
    }

    model_path = Path(args.model_dir) / ("ridge_sofa_surrogate_smoke.joblib" if args.smoke else "ridge_sofa_surrogate.joblib")
    config_path = Path(args.model_dir) / ("severity_surrogate_config_smoke.json" if args.smoke else "severity_surrogate_config.json")
    metrics_path = Path(args.report_dir) / ("severity_surrogate_metrics_smoke.json" if args.smoke else "severity_surrogate_metrics.json")
    coef_path = Path(args.report_dir) / ("severity_surrogate_coefficients_smoke.csv" if args.smoke else "severity_surrogate_coefficients.csv")

    joblib.dump(model_payload, model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": FEATURES,
            "scaler_params": scaler_params,
            "target": TARGET,
            "target_clip": list(SOFA_RANGE),
            "selected_alpha": float(model.alpha_),
            "intercept": float(model.intercept_),
        }, f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    coef_df.to_csv(coef_path, index=False)

    logging.info("Train metrics: %s", json.dumps(metrics["train"]))
    logging.info("Val metrics:   %s", json.dumps(metrics["val"]))
    logging.info("Test metrics:  %s", json.dumps(metrics["test"]))
    logging.info("Saved model: %s", model_path)
    logging.info("Saved config: %s", config_path)
    logging.info("Saved metrics: %s", metrics_path)
    logging.info("Saved coefficients: %s", coef_path)
