"""
Step 09 - Preprocess MKdataset for RL pipeline.

Chains from: data/processed/sepsis/MKdataset.csv (step 08 output)
Produces:
  data/processed/sepsis/rl_train_set_unscaled.csv   (z-normalized, not [0,1])
  data/processed/sepsis/rl_val_set_unscaled.csv
  data/processed/sepsis/rl_test_set_unscaled.csv
  data/processed/sepsis/rl_train_set_scaled.csv      (scaled to [0,1])
  data/processed/sepsis/rl_val_set_scaled.csv
  data/processed/sepsis/rl_test_set_scaled.csv
  data/processed/sepsis/norm_stats.json               (normalization params)
  data/processed/sepsis/action_quartiles.json          (action bin thresholds)

Ported from: sepsisrl/preprocessing/process_interventions.ipynb
             sepsisrl/preprocessing/preprocess_data.ipynb
"""
import argparse
import json
import logging
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.careai.sepsis.rl.preprocessing import (
    BINARY_FIELDS,
    LOG_FIELDS,
    NORM_FIELDS,
    STATE_FEATURES,
    add_sparse_reward,
    discretize_actions,
    normalize,
    scale_01,
    select_output_columns,
    split_by_patient,
)


def main():
    parser = argparse.ArgumentParser(description="Step 09: Preprocess MKdataset for RL")
    parser.add_argument(
        "--input", default="data/processed/sepsis/MKdataset.csv",
        help="Path to MKdataset.csv",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/sepsis",
        help="Output directory",
    )
    parser.add_argument(
        "--log", default="logs/step_09_preprocess.log",
        help="Log file path",
    )
    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    logging.info("Step 09 started. input=%s", args.input)
    t0 = time.time()

    # ── Load MKdataset ────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d columns, %d ICU stays",
                 len(df), len(df.columns), df["icustayid"].nunique())

    # ── Step 1: Discretize actions ────────────────────────────────────
    logging.info("Discretizing actions (5x5 bins)")
    df, vaso_q, iv_q = discretize_actions(df)

    # Save quartiles for reproducibility
    quartiles = {
        "vaso_quartiles": vaso_q.tolist(),
        "iv_quartiles": iv_q.tolist(),
    }
    qpath = f"{args.output_dir}/action_quartiles.json"
    with open(qpath, "w") as f:
        json.dump(quartiles, f, indent=2)
    logging.info("Action quartiles saved: %s", qpath)

    # ── Step 2: Split by patient ──────────────────────────────────────
    logging.info("Splitting 70/10/20 by patient")
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    # Keep original (unscaled) copies for reward shaping later
    train_orig = train_df.copy()
    val_orig = val_df.copy()
    test_orig = test_df.copy()

    # ── Step 3: Normalize ─────────────────────────────────────────────
    logging.info("Normalizing features (binary/zscore/log)")
    train_df, val_df, test_df, norm_stats = normalize(train_df, val_df, test_df)

    # ── Step 4: Add sparse terminal reward ────────────────────────────
    logging.info("Adding sparse terminal rewards")
    train_df = add_sparse_reward(train_df)
    val_df = add_sparse_reward(val_df)
    test_df = add_sparse_reward(test_df)

    # ── Step 5: Select output columns ─────────────────────────────────
    train_df = select_output_columns(train_df)
    val_df = select_output_columns(val_df)
    test_df = select_output_columns(test_df)

    # ── Step 6: Save unscaled (z-normalized) ──────────────────────────
    logging.info("Saving unscaled datasets")
    train_df.to_csv(f"{args.output_dir}/rl_train_set_unscaled.csv", index=False)
    val_df.to_csv(f"{args.output_dir}/rl_val_set_unscaled.csv", index=False)
    test_df.to_csv(f"{args.output_dir}/rl_test_set_unscaled.csv", index=False)
    logging.info("  train: %d rows, val: %d rows, test: %d rows",
                 len(train_df), len(val_df), len(test_df))

    # ── Step 7: Scale to [0, 1] ───────────────────────────────────────
    logging.info("Scaling features to [0,1]")
    scalable_cols = BINARY_FIELDS + NORM_FIELDS + LOG_FIELDS
    # Filter to cols actually present
    scalable_cols = [c for c in scalable_cols if c in train_df.columns]
    train_scaled, val_scaled, test_scaled, scale_stats = scale_01(
        train_df.copy(), val_df.copy(), test_df.copy(), scalable_cols
    )

    # ── Step 8: Save scaled ───────────────────────────────────────────
    logging.info("Saving scaled datasets")
    train_scaled.to_csv(f"{args.output_dir}/rl_train_set_scaled.csv", index=False)
    val_scaled.to_csv(f"{args.output_dir}/rl_val_set_scaled.csv", index=False)
    test_scaled.to_csv(f"{args.output_dir}/rl_test_set_scaled.csv", index=False)

    # ── Step 9: Save original (pre-normalization) for reward shaping ──
    # The continuous RL branch needs original SOFA/lactate for reward shaping
    logging.info("Saving original (pre-normalization) datasets for reward shaping")
    train_orig = select_output_columns(add_sparse_reward(train_orig))
    val_orig = select_output_columns(add_sparse_reward(val_orig))
    test_orig = select_output_columns(add_sparse_reward(test_orig))
    train_orig.to_csv(f"{args.output_dir}/rl_train_set_original.csv", index=False)
    val_orig.to_csv(f"{args.output_dir}/rl_val_set_original.csv", index=False)
    test_orig.to_csv(f"{args.output_dir}/rl_test_set_original.csv", index=False)

    # ── Step 10: Save normalization stats ─────────────────────────────
    # Convert numpy types for JSON serialization
    serializable_stats = {}
    for k, v in norm_stats.items():
        serializable_stats[k] = {
            kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
            for kk, vv in v.items()
        }
    for k, v in scale_stats.items():
        if k not in serializable_stats:
            serializable_stats[k] = {}
        serializable_stats[k]["scale_min"] = float(v["min"])
        serializable_stats[k]["scale_max"] = float(v["max"])

    stats_path = f"{args.output_dir}/norm_stats.json"
    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)
    logging.info("Normalization stats saved: %s", stats_path)

    # ── Summary ───────────────────────────────────────────────────────
    dt = time.time() - t0
    logging.info("Step 09 complete. %.1f sec", dt)
    logging.info("Output files in %s:", args.output_dir)
    logging.info("  rl_{train,val,test}_set_unscaled.csv  (z-normalized)")
    logging.info("  rl_{train,val,test}_set_scaled.csv    ([0,1] scaled)")
    logging.info("  rl_{train,val,test}_set_original.csv  (pre-normalization)")
    logging.info("  action_quartiles.json")
    logging.info("  norm_stats.json")


if __name__ == "__main__":
    main()
