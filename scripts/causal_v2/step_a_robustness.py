"""Step A (robustness): Multi-model variable selection for causal discovery.

Runs four analyses on the v2 transition dataset:
  1. LightGBM split-count importances  (same as step_a_variable_selection.py)
  2. SHAP mean-|SHAP| importances      (on same LightGBM models, 5k-row sample)
  3. Random Forest MDI importances     (n_estimators=100, sqrt features)
  4. Ridge / Logistic Regression       (absolute standardized coefficients)

For each of the 29 transition targets + readmission model, all four methods are
trained. Results are aggregated into a consensus ranking table.

The consensus is the primary output: features that rank consistently high
across all four mechanistically different methods are strong candidates for
causal discovery.

Usage:
    python scripts/causal_v2/step_a_robustness.py \\
        --csv data/processed/hosp_daily_v2_transitions.csv \\
        --report-dir reports/causal_v2/variable_selection/multi_model

    # Quick test on sample:
    python scripts/causal_v2/step_a_robustness.py \\
        --csv data/processed/hosp_daily_v2_transitions_sample5k.csv \\
        --report-dir reports/causal_v2/variable_selection_sample/multi_model
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.causal_v2.variable_selection import (
    ALL_TRANSITION_TARGETS,
    prepare_transitions,
    build_feature_matrix,
    # LightGBM + SHAP
    train_transition_models,
    compute_shap_importances,
    train_readmission_model,
    compute_readmission_shap,
    # Random Forest
    train_transition_models_rf,
    train_readmission_model_rf,
    # Linear / Logistic
    train_transition_models_lr,
    train_readmission_model_lr,
    # Consensus + save
    build_consensus,
    save_results_multi,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step A robustness: multi-model variable selection."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to hosp_daily_v2_transitions.csv",
    )
    parser.add_argument(
        "--report-dir",
        default="reports/causal_v2/variable_selection/multi_model",
        help="Output directory for multi-model importance tables",
    )
    parser.add_argument(
        "--split", default="train",
        help="Which split to train on (default: train)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    report_dir = PROJECT_ROOT / args.report_dir

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading dataset: %s", args.csv)
    import pandas as pd
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded: %d rows, %d columns", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Build feature matrix (shared across all methods)
    # ------------------------------------------------------------------
    trans_df = prepare_transitions(df)
    trans_df, feature_cols = build_feature_matrix(trans_df)
    log.info("  Feature matrix: %d features", len(feature_cols))

    readmission_df, feature_cols_r = build_feature_matrix(df.copy())

    # Accumulate results — saved incrementally after each method completes
    transition_by_method = {}
    readmission_by_method = {}

    # ------------------------------------------------------------------
    # Analysis 1 + 2: LightGBM (with SHAP)
    # ------------------------------------------------------------------
    log.info("=== [1/4] LightGBM transition models (returning model objects for SHAP) ===")
    lgbm_importances, lgbm_models = train_transition_models(
        trans_df, feature_cols, split=args.split, return_models=True
    )
    log.info("  Trained %d LightGBM transition models", len(lgbm_importances))
    transition_by_method["lgbm"] = lgbm_importances

    log.info("=== [2/4] SHAP importances (TreeExplainer on LightGBM models) ===")
    shap_importances = compute_shap_importances(lgbm_models, feature_cols)
    log.info("  SHAP computed for %d models", len(shap_importances))
    transition_by_method["shap"] = shap_importances

    log.info("--- Checkpoint save after LightGBM + SHAP ---")
    save_results_multi(transition_by_method, readmission_by_method, feature_cols, report_dir)

    # ------------------------------------------------------------------
    # Analysis 3: Random Forest
    # ------------------------------------------------------------------
    log.info("=== [3/4] Random Forest transition models ===")
    rf_importances = train_transition_models_rf(
        trans_df, feature_cols, split=args.split
    )
    log.info("  Trained %d RF transition models", len(rf_importances))
    transition_by_method["rf"] = rf_importances

    log.info("--- Checkpoint save after RF ---")
    save_results_multi(transition_by_method, readmission_by_method, feature_cols, report_dir)

    # ------------------------------------------------------------------
    # Analysis 4: Ridge / Logistic Regression
    # ------------------------------------------------------------------
    log.info("=== [4/4] Ridge / Logistic Regression transition models ===")
    lr_importances = train_transition_models_lr(
        trans_df, feature_cols, split=args.split
    )
    log.info("  Trained %d LR transition models", len(lr_importances))
    transition_by_method["lr"] = lr_importances

    # ------------------------------------------------------------------
    # Readmission models (all methods)
    # ------------------------------------------------------------------
    log.info("=== Readmission models (all methods) ===")
    lgbm_readmit, lgbm_readmit_model = train_readmission_model(
        readmission_df, feature_cols_r, split=args.split, return_model=True
    )
    readmission_by_method["lgbm"] = lgbm_readmit

    shap_readmit = compute_readmission_shap(
        lgbm_readmit_model, readmission_df, feature_cols_r, split=args.split
    )
    readmission_by_method["shap"] = shap_readmit

    rf_readmit = train_readmission_model_rf(
        readmission_df, feature_cols_r, split=args.split
    )
    readmission_by_method["rf"] = rf_readmit

    lr_readmit = train_readmission_model_lr(
        readmission_df, feature_cols_r, split=args.split
    )
    readmission_by_method["lr"] = lr_readmit

    # ------------------------------------------------------------------
    # Final save (all methods + consensus)
    # ------------------------------------------------------------------
    log.info("=== Final save ===")
    save_results_multi(
        transition_by_method,
        readmission_by_method,
        feature_cols,
        report_dir,
    )

    # ------------------------------------------------------------------
    # Print consensus top-20 to terminal
    # ------------------------------------------------------------------
    log.info("=== CONSENSUS TOP 20 TRANSITION FEATURES ===")
    consensus = build_consensus(
        method_importances={
            "lgbm": lgbm_importances,
            "shap": shap_importances,
            "rf": rf_importances,
            "lr": lr_importances,
        },
        feature_cols=feature_cols,
    )
    top20 = consensus.head(20)
    method_cols = [c for c in top20.columns if c not in ("mean_rank", "mean_norm_importance", "n_methods_top10")]
    header = f"  {'feature':<35} {'mean_rank':>9} {'n_top10':>7}  " + "  ".join(f"{m:>8}" for m in method_cols)
    log.info(header)
    for feat, row in top20.iterrows():
        method_vals = "  ".join(f"{row[m]:>8.4f}" for m in method_cols)
        log.info(
            "  %-35s %9.1f %7d  %s",
            feat, row["mean_rank"], row["n_methods_top10"], method_vals,
        )

    log.info("=== CONSENSUS TOP 15 READMISSION FEATURES ===")
    readmit_df_path = report_dir / "readmission_multi.csv"
    import pandas as pd2
    if readmit_df_path.exists():
        readmit_df = pd.read_csv(readmit_df_path, index_col=0)
        for feat, row in readmit_df.head(15).iterrows():
            log.info("  %-35s %.4f", feat, row["mean_importance"])

    log.info("Step A robustness complete. Results in: %s", report_dir)


if __name__ == "__main__":
    main()
