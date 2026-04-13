"""
Step 10c -- Train a selected-state terminal readmission model.

Uses one terminal row per ICU stay from rl_dataset_selected.parquet and trains
an outcome model:
  terminal selected state -> readmit_30d

This model is later used as the terminal reward source in the selected CARE-Sim
control environment:
  r_terminal = +15 * (1 - p_readmit) - 15 * p_readmit
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


FEATURES = [
    "s_Hb",
    "s_BUN",
    "s_Creatinine",
    "s_Phosphate",
    "s_HR",
    "s_Chloride",
    "s_age",
    "s_charlson_score",
    "s_prior_ed_visits_6m",
]
TARGET = "readmit_30d"
DONE = "done"
SPLIT = "split"
STAY = "icustayid"

LGBM_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "n_estimators": 400,
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbose": -1,
    "n_jobs": -1,
}


def evaluate(model, df: pd.DataFrame) -> dict:
    x = df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y = df[TARGET].to_numpy(dtype=np.int32, copy=True)
    p = model.predict_proba(x)[:, 1]
    return {
        "n_rows": int(len(df)),
        "readmit_rate": float(y.mean()),
        "auc": float(roc_auc_score(y, p)),
        "auprc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "logloss": float(log_loss(y, p, labels=[0, 1])),
        "pred_mean": float(p.mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "rl_dataset_selected.parquet"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "icu_readmit" / "terminal_readmit_selected"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "terminal_readmit_selected"),
    )
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log", default=None)
    args = parser.parse_args()

    log_file = args.log or str(PROJECT_ROOT / "logs" / "step_10c.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10c (selected terminal readmission model) started.")
    logging.info("Input: %s", args.data)
    use_cols = [STAY, SPLIT, DONE, TARGET, *FEATURES]
    df = pd.read_parquet(args.data, columns=use_cols)
    df = df[df[DONE] == 1].copy()
    logging.info("Terminal rows: %d stays=%d", len(df), df[STAY].nunique())

    if args.smoke:
        keep = np.sort(df[STAY].unique())[:4000]
        df = df[df[STAY].isin(keep)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[STAY].nunique())

    train_df = df[df[SPLIT] == "train"].copy()
    val_df = df[df[SPLIT] == "val"].copy()
    test_df = df[df[SPLIT] == "test"].copy()

    x_train = train_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y_train = train_df[TARGET].to_numpy(dtype=np.int32, copy=True)
    x_val = val_df[FEATURES].to_numpy(dtype=np.float32, copy=True)
    y_val = val_df[TARGET].to_numpy(dtype=np.int32, copy=True)

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        x_train,
        y_train,
        eval_set=[(x_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )

    metrics = {
        "train": evaluate(model, train_df),
        "val": evaluate(model, val_df),
        "test": evaluate(model, test_df),
    }

    feature_importance = pd.DataFrame({
        "feature": FEATURES,
        "importance_gain": model.booster_.feature_importance(importance_type="gain").astype(float),
        "importance_split": model.booster_.feature_importance(importance_type="split").astype(float),
    }).sort_values("importance_gain", ascending=False)

    payload = {
        "model": model,
        "feature_names": FEATURES,
        "target": TARGET,
    }
    model_path = Path(args.model_dir) / ("terminal_readmit_selected_smoke.joblib" if args.smoke else "terminal_readmit_selected.joblib")
    config_path = Path(args.model_dir) / ("terminal_readmit_selected_config_smoke.json" if args.smoke else "terminal_readmit_selected_config.json")
    metrics_path = Path(args.report_dir) / ("terminal_readmit_selected_metrics_smoke.json" if args.smoke else "terminal_readmit_selected_metrics.json")
    feat_path = Path(args.report_dir) / ("terminal_readmit_selected_feature_importance_smoke.csv" if args.smoke else "terminal_readmit_selected_feature_importance.csv")

    joblib.dump(payload, model_path)
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump({"feature_names": FEATURES, "target": TARGET}, f, indent=2)
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    feature_importance.to_csv(feat_path, index=False)

    logging.info("Train metrics: %s", json.dumps(metrics["train"]))
    logging.info("Val metrics:   %s", json.dumps(metrics["val"]))
    logging.info("Test metrics:  %s", json.dumps(metrics["test"]))
    logging.info("Saved model: %s", model_path)
    logging.info("Saved config: %s", config_path)
    logging.info("Saved metrics: %s", metrics_path)
    logging.info("Saved feature importance: %s", feat_path)


if __name__ == "__main__":
    main()
