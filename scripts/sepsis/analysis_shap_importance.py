"""
SHAP-based state and action importance analysis for the sepsis RL pipeline.

Trains two LightGBM classifiers on last-timestep data:
  1. Mortality model:   predict died_in_hosp (all stays)
  2. Readmission model: predict readmit_30d  (survivors only, died_in_hosp=0)

Computes SHAP values on held-out test set for each model.

Outputs:
  reports/sepsis_rl/shap_importance.csv       -- ranked feature importance table
  reports/sepsis_rl/shap_mortality.png         -- SHAP beeswarm plot (mortality)
  reports/sepsis_rl/shap_readmit.png           -- SHAP beeswarm plot (readmission)
  reports/sepsis_rl/shap_top10_overlap.txt     -- top-10 features per model + overlap

Inputs (from sepsis_readmit variant, which contains both outcome columns):
  data/processed/sepsis_readmit/rl_train_set_original.csv
  data/processed/sepsis_readmit/rl_test_set_original.csv

Usage:
  python scripts/sepsis/analysis_shap_importance.py
"""
import argparse
import logging
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

sys.path.insert(0, ".")
from src.careai.sepsis.rl.preprocessing import STATE_FEATURES

# State features to use: all 47 STATE_FEATURES except 'bloc' (timestep counter)
# Plus vaso_input and iv_input (discretized actions) to assess action importance
FEATURE_COLS = [f for f in STATE_FEATURES if f != "bloc"] + ["vaso_input", "iv_input"]

# LightGBM params — fast, regularised, good for tabular data
LGB_PARAMS = {
    "objective":      "binary",
    "metric":         "auc",
    "n_estimators":   300,
    "learning_rate":  0.05,
    "num_leaves":     31,
    "min_child_samples": 20,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "random_state":   42,
    "verbose":        -1,
}


def last_timestep(df):
    """Return one row per ICU stay: the final timestep."""
    return df.groupby("icustayid").tail(1).reset_index(drop=True)


def train_and_shap(X_train, y_train, X_test, y_test, label):
    """Train LightGBM, evaluate on test set, compute SHAP values."""
    logging.info("Training %s model: %d train, %d test, %.1f%% positive",
                 label, len(X_train), len(X_test), 100 * y_train.mean())

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X_train, y_train)

    preds = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, preds)
    logging.info("  %s AUC: %.4f", label, auc)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    # For binary classification shap returns list [neg_class, pos_class] in older
    # versions, or a 2D array in newer versions
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=X_test.columns).sort_values(ascending=False)
    logging.info("  Top 10 features:\n%s", importance.head(10).to_string())

    return model, shap_values, importance, auc


def plot_beeswarm(shap_values, X_test, title, save_path, max_display=20):
    """SHAP beeswarm summary plot."""
    plt.figure(figsize=(10, 7))
    shap.summary_plot(
        shap_values, X_test,
        max_display=max_display,
        show=False,
        plot_size=None,
    )
    plt.title(title, fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    logging.info("Beeswarm plot saved: %s", save_path)


def main():
    parser = argparse.ArgumentParser(description="SHAP importance analysis for sepsis RL")
    parser.add_argument("--data-dir",   default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir", default="reports/sepsis_readmit/analysis/shap")
    parser.add_argument("--log",        default="logs/analysis_shap_importance.log")
    parser.add_argument("--top-n",      type=int, default=10,
                        help="Number of top features to highlight in overlap summary")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # ── Load data ─────────────────────────────────────────────────────
    logging.info("Loading data from %s", args.data_dir)
    train_df = pd.read_csv(f"{args.data_dir}/rl_train_set_original.csv")
    test_df  = pd.read_csv(f"{args.data_dir}/rl_test_set_original.csv")

    train_last = last_timestep(train_df)
    test_last  = last_timestep(test_df)
    logging.info("Last-timestep rows: train=%d, test=%d", len(train_last), len(test_last))

    # Filter feature list to columns that actually exist
    feature_cols = [c for c in FEATURE_COLS if c in train_last.columns]
    missing = [c for c in FEATURE_COLS if c not in train_last.columns]
    if missing:
        logging.warning("Features not found, skipping: %s", missing)
    logging.info("Using %d features", len(feature_cols))

    X_train = train_last[feature_cols]
    X_test  = test_last[feature_cols]

    # ── Model 1: Mortality ────────────────────────────────────────────
    y_train_mort = train_last["died_in_hosp"]
    y_test_mort  = test_last["died_in_hosp"]

    model_mort, shap_mort, imp_mort, auc_mort = train_and_shap(
        X_train, y_train_mort, X_test, y_test_mort, "mortality"
    )
    plot_beeswarm(
        shap_mort, X_test,
        f"SHAP Feature Importance -- In-Hospital Mortality (AUC={auc_mort:.3f})",
        f"{args.report_dir}/shap_mortality.png",
    )

    # ── Model 2: Readmission (survivors only) ─────────────────────────
    train_surv = train_last[train_last["died_in_hosp"] == 0].reset_index(drop=True)
    test_surv  = test_last[test_last["died_in_hosp"] == 0].reset_index(drop=True)
    logging.info("Survivors: train=%d, test=%d", len(train_surv), len(test_surv))

    X_train_surv = train_surv[feature_cols]
    X_test_surv  = test_surv[feature_cols]
    y_train_read = train_surv["readmit_30d"]
    y_test_read  = test_surv["readmit_30d"]

    model_read, shap_read, imp_read, auc_read = train_and_shap(
        X_train_surv, y_train_read, X_test_surv, y_test_read, "readmission"
    )
    plot_beeswarm(
        shap_read, X_test_surv,
        f"SHAP Feature Importance -- 30-Day Readmission (AUC={auc_read:.3f})",
        f"{args.report_dir}/shap_readmit.png",
    )

    # ── Combined importance table ─────────────────────────────────────
    importance_df = pd.DataFrame({
        "feature":        feature_cols,
        "shap_mortality": [imp_mort.get(f, 0.0) for f in feature_cols],
        "shap_readmit":   [imp_read.get(f, 0.0) for f in feature_cols],
    })
    # Rank each (1 = most important)
    importance_df["rank_mortality"] = importance_df["shap_mortality"].rank(
        ascending=False).astype(int)
    importance_df["rank_readmit"]   = importance_df["shap_readmit"].rank(
        ascending=False).astype(int)
    importance_df["mean_rank"] = (
        importance_df["rank_mortality"] + importance_df["rank_readmit"]) / 2
    importance_df = importance_df.sort_values("mean_rank")

    out_csv = f"{args.report_dir}/shap_importance.csv"
    importance_df.to_csv(out_csv, index=False, float_format="%.6f")
    logging.info("Importance table saved: %s", out_csv)

    # ── Top-N overlap summary ─────────────────────────────────────────
    top_mort = set(imp_mort.head(args.top_n).index)
    top_read = set(imp_read.head(args.top_n).index)
    overlap  = top_mort & top_read

    summary_lines = [
        f"SHAP Importance Summary (top {args.top_n} features per model)",
        f"Mortality model AUC:   {auc_mort:.4f}",
        f"Readmission model AUC: {auc_read:.4f}",
        "",
        f"Top {args.top_n} for MORTALITY:",
    ]
    for i, (feat, val) in enumerate(imp_mort.head(args.top_n).items(), 1):
        marker = " <-- also in readmit top-N" if feat in top_read else ""
        summary_lines.append(f"  {i:2d}. {feat:<30s} mean|SHAP|={val:.5f}{marker}")

    summary_lines += ["", f"Top {args.top_n} for READMISSION:"]
    for i, (feat, val) in enumerate(imp_read.head(args.top_n).items(), 1):
        marker = " <-- also in mortality top-N" if feat in top_mort else ""
        summary_lines.append(f"  {i:2d}. {feat:<30s} mean|SHAP|={val:.5f}{marker}")

    summary_lines += [
        "",
        f"Overlap ({len(overlap)} features in both top-{args.top_n}):",
    ]
    for feat in sorted(overlap):
        summary_lines.append(f"  - {feat}")

    summary_lines += [
        "",
        f"Top {args.top_n} by mean rank across both models:",
    ]
    for i, row in importance_df.head(args.top_n).iterrows():
        summary_lines.append(
            f"  {row['mean_rank']:4.1f}  {row['feature']:<30s} "
            f"mort_rank={row['rank_mortality']:2d}  readmit_rank={row['rank_readmit']:2d}"
        )

    summary_text = "\n".join(summary_lines)
    summary_path = f"{args.report_dir}/shap_top10_overlap.txt"
    with open(summary_path, "w") as f:
        f.write(summary_text)
    logging.info("Overlap summary saved: %s", summary_path)
    print("\n" + summary_text)

    logging.info("Analysis complete.")


if __name__ == "__main__":
    main()
