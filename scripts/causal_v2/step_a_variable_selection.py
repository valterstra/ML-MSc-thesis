"""Step A: ML variable selection for causal discovery.

Runs two LightGBM analyses on the v2 dataset:
  1. Transition models: predict next-day state from current state + actions
  2. Readmission model: predict readmit_30d from discharge-day state

Results saved to reports/causal_v2/variable_selection/ for user review.
User and Claude then jointly decide which variables enter causal discovery.

Usage:
    python scripts/causal_v2/step_a_variable_selection.py \
        --csv data/processed/hosp_daily_v2_transitions.csv \
        --report-dir reports/causal_v2/variable_selection

    # Quick test on sample:
    python scripts/causal_v2/step_a_variable_selection.py \
        --csv data/processed/hosp_daily_v2_transitions_sample5k.csv \
        --report-dir reports/causal_v2/variable_selection_sample
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
    train_transition_models,
    train_readmission_model,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step A: ML variable selection for causal discovery."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to hosp_daily_v2_transitions.csv",
    )
    parser.add_argument(
        "--report-dir", default="reports/causal_v2/variable_selection",
        help="Output directory for importance tables",
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
    # Analysis 1: Transition models
    # ------------------------------------------------------------------
    log.info("=== Analysis 1: Transition importance ===")
    trans_df = prepare_transitions(df)
    trans_df, feature_cols = build_feature_matrix(trans_df)
    log.info("  Feature matrix: %d features", len(feature_cols))
    log.info("  Features: %s", feature_cols)

    transition_importances = train_transition_models(
        trans_df, feature_cols, split=args.split
    )
    log.info(
        "  Trained %d transition models (out of %d targets)",
        len(transition_importances), len(ALL_TRANSITION_TARGETS),
    )

    # ------------------------------------------------------------------
    # Analysis 2: Readmission model on discharge-day rows
    # ------------------------------------------------------------------
    log.info("=== Analysis 2: Readmission importance ===")
    # Use full df (not trans_df) — need is_last_day == 1 rows
    _, readmission_df = build_feature_matrix(df.copy())
    readmission_df = df  # rebuild with full df
    readmission_df, feature_cols_r = build_feature_matrix(readmission_df)

    readmission_importance = train_readmission_model(
        readmission_df, feature_cols_r, split=args.split
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    log.info("=== Saving results ===")
    save_results(transition_importances, readmission_importance, report_dir)

    # ------------------------------------------------------------------
    # Print top-line summary to terminal
    # ------------------------------------------------------------------
    log.info("=== TOP 20 FEATURES BY MEAN TRANSITION IMPORTANCE ===")
    import pandas as pd2
    wide = pd.DataFrame(transition_importances)
    wide = wide.div(wide.sum(axis=0), axis=1)
    wide["mean_importance"] = wide.mean(axis=1)
    top20 = wide["mean_importance"].sort_values(ascending=False).head(20)
    for feat, val in top20.items():
        log.info("  %-35s %.4f", feat, val)

    log.info("=== TOP 20 FEATURES FOR READMISSION ===")
    readmission_norm = readmission_importance / readmission_importance.sum()
    for feat, val in readmission_norm.head(20).items():
        log.info("  %-35s %.4f", feat, val)

    log.info("Step A complete. Results in: %s", report_dir)


if __name__ == "__main__":
    main()
