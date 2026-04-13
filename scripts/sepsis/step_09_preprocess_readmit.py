"""
Step 09 (readmission variant) - Preprocess MKdataset for readmission RL pipeline.

Same as step_09_preprocess.py but optimizes for 30-day readmission instead of
in-hospital mortality.

Terminal reward (Option C):
  died_in_hosp = 1                       -> -100  (death is still a failure)
  died_in_hosp = 0 AND readmit_30d = 1   -> -100  (survived but readmitted)
  died_in_hosp = 0 AND readmit_30d = 0   -> +100  (survived and not readmitted)

readmit_30d is joined from data/interim/sepsis/intermediates/demog.csv since it
was not carried forward into MKdataset.csv by steps 06-08.

Chains from:
  data/processed/sepsis/MKdataset.csv                          (step 08 output)
  data/interim/sepsis/intermediates/demog.csv                  (step 01 output)

Produces:
  data/processed/sepsis_readmit/rl_train_set_unscaled.csv
  data/processed/sepsis_readmit/rl_val_set_unscaled.csv
  data/processed/sepsis_readmit/rl_test_set_unscaled.csv
  data/processed/sepsis_readmit/rl_train_set_scaled.csv
  data/processed/sepsis_readmit/rl_val_set_scaled.csv
  data/processed/sepsis_readmit/rl_test_set_scaled.csv
  data/processed/sepsis_readmit/rl_train_set_original.csv
  data/processed/sepsis_readmit/rl_val_set_original.csv
  data/processed/sepsis_readmit/rl_test_set_original.csv
  data/processed/sepsis_readmit/norm_stats.json
  data/processed/sepsis_readmit/action_quartiles.json
"""
import argparse
import json
import logging
import os
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
    add_shaped_reward,
    discretize_actions,
    normalize,
    scale_01,
    select_output_columns,
    split_by_patient,
)


def main():
    parser = argparse.ArgumentParser(
        description="Step 09 (readmission variant): Preprocess MKdataset for readmission RL"
    )
    parser.add_argument(
        "--input", default="data/processed/sepsis/MKdataset.csv",
        help="Path to MKdataset.csv",
    )
    parser.add_argument(
        "--demog", default="data/interim/sepsis/intermediates/demog.csv",
        help="Path to demog.csv (source of readmit_30d)",
    )
    parser.add_argument(
        "--output-dir", default="data/processed/sepsis_readmit",
        help="Output directory",
    )
    parser.add_argument(
        "--log", default="logs/step_09_preprocess_readmit.log",
        help="Log file path",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
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
    logging.info("Step 09 (readmission variant) started.")
    logging.info("  input=%s", args.input)
    logging.info("  demog=%s", args.demog)
    logging.info("  output-dir=%s", args.output_dir)
    t0 = time.time()

    # ── Load MKdataset ────────────────────────────────────────────────
    df = pd.read_csv(args.input)
    logging.info("Loaded MKdataset: %d rows, %d ICU stays",
                 len(df), df["icustayid"].nunique())

    # ── Join readmit_30d from demog.csv ───────────────────────────────
    # readmit_30d was extracted in step 01 but not carried forward by steps 06-08.
    # Definition: 1 if a NEW admission occurred within 30 days after THIS discharge.
    demog = pd.read_csv(args.demog, usecols=["icustayid", "readmit_30d"])
    demog = demog.drop_duplicates("icustayid")
    before = len(df)
    df = df.merge(demog, on="icustayid", how="left")
    missing = df["readmit_30d"].isna().sum()
    if missing > 0:
        logging.warning(
            "%d rows missing readmit_30d after join (%.1f%%) — filling with 0",
            missing, 100 * missing / len(df),
        )
        df["readmit_30d"] = df["readmit_30d"].fillna(0).astype(int)
    else:
        df["readmit_30d"] = df["readmit_30d"].astype(int)
    logging.info(
        "readmit_30d joined: %d rows (unchanged from %d). "
        "readmit_30d=1: %d stays (%.1f%%)",
        len(df), before,
        df.groupby("icustayid")["readmit_30d"].first().sum(),
        100 * df.groupby("icustayid")["readmit_30d"].first().mean(),
    )

    # Sanity check: patients who died cannot have a future readmission
    died_and_readmit = (
        (df.groupby("icustayid")["died_in_hosp"].first() == 1) &
        (df.groupby("icustayid")["readmit_30d"].first() == 1)
    ).sum()
    if died_and_readmit > 0:
        logging.warning(
            "%d stays have both died_in_hosp=1 and readmit_30d=1 "
            "(impossible — these will receive -100 via the death branch)",
            died_and_readmit,
        )

    # ── Step 1: Discretize actions ────────────────────────────────────
    logging.info("Discretizing actions (5x5 bins)")
    df, vaso_q, iv_q = discretize_actions(df)

    quartiles = {
        "vaso_quartiles": vaso_q.tolist(),
        "iv_quartiles": iv_q.tolist(),
    }
    qpath = f"{args.output_dir}/action_quartiles.json"
    with open(qpath, "w") as f:
        json.dump(quartiles, f, indent=2)
    logging.info("Action quartiles saved: %s", qpath)

    # ── Step 2: Split by patient ──────────────────────────────────────
    logging.info("Splitting 70/10/20 by patient (seed=42, same as original)")
    train_df, val_df, test_df = split_by_patient(df, seed=42)

    train_orig = train_df.copy()
    val_orig   = val_df.copy()
    test_orig  = test_df.copy()

    # ── Step 3: Normalize ─────────────────────────────────────────────
    logging.info("Normalizing features (binary/zscore/log)")
    train_df, val_df, test_df, norm_stats = normalize(train_df, val_df, test_df)

    # ── Step 4: Add sparse terminal reward (readmission variant) ──────
    logging.info("Adding sparse terminal rewards (outcome=readmit)")
    train_df = add_sparse_reward(train_df, outcome="readmit")
    val_df   = add_sparse_reward(val_df,   outcome="readmit")
    test_df  = add_sparse_reward(test_df,  outcome="readmit")

    # ── Step 5: Select output columns ─────────────────────────────────
    train_df = select_output_columns(train_df)
    val_df   = select_output_columns(val_df)
    test_df  = select_output_columns(test_df)

    # ── Step 6: Save unscaled (z-normalized) ──────────────────────────
    logging.info("Saving unscaled datasets")
    train_df.to_csv(f"{args.output_dir}/rl_train_set_unscaled.csv", index=False)
    val_df.to_csv(f"{args.output_dir}/rl_val_set_unscaled.csv",   index=False)
    test_df.to_csv(f"{args.output_dir}/rl_test_set_unscaled.csv", index=False)
    logging.info("  train: %d rows, val: %d rows, test: %d rows",
                 len(train_df), len(val_df), len(test_df))

    # ── Step 7: Scale to [0, 1] ───────────────────────────────────────
    logging.info("Scaling features to [0,1]")
    scalable_cols = [c for c in BINARY_FIELDS + NORM_FIELDS + LOG_FIELDS
                     if c in train_df.columns]
    train_scaled, val_scaled, test_scaled, scale_stats = scale_01(
        train_df.copy(), val_df.copy(), test_df.copy(), scalable_cols
    )

    # ── Step 8: Save scaled ───────────────────────────────────────────
    logging.info("Saving scaled datasets")
    train_scaled.to_csv(f"{args.output_dir}/rl_train_set_scaled.csv", index=False)
    val_scaled.to_csv(f"{args.output_dir}/rl_val_set_scaled.csv",   index=False)
    test_scaled.to_csv(f"{args.output_dir}/rl_test_set_scaled.csv", index=False)

    # ── Step 9: Save original (pre-normalization) for shaped reward ───
    logging.info("Saving original datasets for shaped reward computation")
    train_orig = select_output_columns(
        add_shaped_reward(train_orig, train_orig, outcome="readmit"))
    val_orig   = select_output_columns(
        add_shaped_reward(val_orig,   val_orig,   outcome="readmit"))
    test_orig  = select_output_columns(
        add_shaped_reward(test_orig,  test_orig,  outcome="readmit"))
    train_orig.to_csv(f"{args.output_dir}/rl_train_set_original.csv", index=False)
    val_orig.to_csv(f"{args.output_dir}/rl_val_set_original.csv",   index=False)
    test_orig.to_csv(f"{args.output_dir}/rl_test_set_original.csv", index=False)

    # ── Step 10: Save normalization stats ─────────────────────────────
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
    logging.info("Step 09 (readmission variant) complete. %.1f sec", dt)
    logging.info("Output files in %s:", args.output_dir)
    logging.info("  rl_{train,val,test}_set_unscaled.csv  (z-normalized)")
    logging.info("  rl_{train,val,test}_set_scaled.csv    ([0,1] scaled)")
    logging.info("  rl_{train,val,test}_set_original.csv  (shaped rewards, readmit outcome)")
    logging.info("  action_quartiles.json")
    logging.info("  norm_stats.json")


if __name__ == "__main__":
    main()
