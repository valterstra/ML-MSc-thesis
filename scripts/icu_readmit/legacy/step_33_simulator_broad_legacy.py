"""
Step 13 -- Model-based environment simulator (Raghu et al. 2018 Table 1).

Chains from: step 10b output (rl_dataset_broad.parquet)
Produces:
  models/icu_readmit/simulator/<model_type>/
    transition_model.pt
    model_config.pkl
  reports/icu_readmit/
    simulator_<model_type>_per_feature_mse.json
    simulator_<model_type>_rollout_eval.json
    simulator_comparison.json

Four architectures (Raghu 2018 Table 1):
  nn:     2 FC + ReLU + BN -- preferred model
  linear: linear regression baseline
  lstm:   LSTM on history sequence
  bnn:    Bayesian NN with variational inference

Key differences from sepsis step_13:
  - Input: rl_dataset_broad.parquet (not CSV)
  - State cols: s_* prefix (already z-scored, 51 features)
  - Action: 5-bit binary (5 drug flags decoded from integer 0-31)
    vs sepsis 2-feature [iv/4, vaso/4]
  - n_action=5

Usage:
  python scripts/icu_readmit/step_13_simulator.py --model-type nn --smoke
  python scripts/icu_readmit/step_13_simulator.py --model-type all
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src.careai.icu_readmit.legacy.rl.simulator import (
    MODEL_TYPES,
    build_history_dataset,
    evaluate_per_feature,
    evaluate_rollouts,
    train_transition_model,
)

ALL_MODEL_TYPES = list(MODEL_TYPES.keys())


def train_and_evaluate_one(model_type, train_dataset, test_dataset, test_df,
                           state_cols, args, log_every):
    """Train one model type, evaluate per-feature MSE and rollout quality."""
    n_state   = len(state_cols)
    model_dir = os.path.join(args.model_dir, model_type)
    t_model   = time.time()

    logging.info("")
    logging.info("=" * 50)
    logging.info("MODEL: %s", model_type.upper())
    logging.info("=" * 50)

    model, train_losses, val_losses = train_transition_model(
        train_dataset,
        n_state=n_state,
        n_action=5,
        n_history=args.n_history,
        hidden=args.hidden,
        lr=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        val_fraction=0.1,
        model_type=model_type,
        save_dir=model_dir,
        device=args.device,
        log_every=log_every,
    )

    # Per-feature MSE on test set
    logging.info("--- Per-feature MSE [%s] ---", model_type.upper())
    test_mse = evaluate_per_feature(model, test_dataset, state_cols, device=args.device)

    sorted_features = sorted(
        [(k, v) for k, v in test_mse.items() if k != "__overall__"],
        key=lambda x: x[1], reverse=True,
    )
    for feat, mse in sorted_features[:5]:
        logging.info("    %-30s MSE=%.6f", feat, mse)
    if len(sorted_features) > 10:
        logging.info("    ... (%d more) ...", len(sorted_features) - 10)
    for feat, mse in sorted_features[-5:]:
        logging.info("    %-30s MSE=%.6f", feat, mse)
    logging.info("    %-30s MSE=%.6f", "OVERALL", test_mse["__overall__"])

    mse_path = os.path.join(args.report_dir, f"simulator_{model_type}_per_feature_mse.json")
    with open(mse_path, "w") as f:
        json.dump(test_mse, f, indent=2)

    # Multi-step rollout evaluation
    logging.info("--- Rollout evaluation [%s] ---", model_type.upper())
    rollout_results = evaluate_rollouts(
        model, test_df, state_cols,
        n_history=args.n_history,
        n_rollout_steps=args.rollout_steps,
        n_patients=args.rollout_patients,
        seed=42,
        device=args.device,
    )

    rollout_path = os.path.join(args.report_dir, f"simulator_{model_type}_rollout_eval.json")
    with open(rollout_path, "w") as f:
        json.dump(rollout_results, f, indent=2)

    dt_model = time.time() - t_model
    logging.info("  [%s] done in %.1f sec", model_type.upper(), dt_model)

    return {
        "model_type":         model_type,
        "best_val_mse":       float(min(val_losses)),
        "test_mse":           test_mse["__overall__"],
        "rollout_mse_step1":  rollout_results["per_step_mse"][0],
        "rollout_mse_last":   rollout_results["per_step_mse"][-1],
        "n_rollout_steps":    rollout_results["n_steps"],
        "time_sec":           dt_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 13: Model-Based Simulator")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24"),
                        help="Directory containing rl_dataset_broad.parquet")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models" / "icu_readmit" / "legacy" / "step_25_33" / "simulator"),
                        help="Output directory for simulator models")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "legacy" / "step_30_33"),
                        help="Output directory for evaluation reports")
    parser.add_argument("--model-type", default="nn",
                        choices=ALL_MODEL_TYPES + ["all"],
                        help="Model architecture: nn, linear, lstm, bnn, or all")
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-history", type=int, default=4,
                        help="History timesteps (current + 3 past)")
    parser.add_argument("--rollout-steps",    type=int, default=10)
    parser.add_argument("--rollout-patients", type=int, default=200)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 5 epochs, 20 rollout patients")
    parser.add_argument("--log", default=str(PROJECT_ROOT / "logs" / "legacy" / "icu_readmit" / "step_33_simulator_broad_legacy.log"))
    args = parser.parse_args()

    if args.smoke:
        args.epochs           = 5
        args.rollout_patients = 20
        args.rollout_steps    = 5

    os.makedirs(args.model_dir,  exist_ok=True)
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
    mode_str    = "SMOKE TEST" if args.smoke else "FULL"
    model_types = ALL_MODEL_TYPES if args.model_type == "all" else [args.model_type]
    logging.info("Step 13 started. mode=%s, device=%s, models=%s",
                 mode_str, args.device, model_types)
    t0 = time.time()

    # ── Load broad parquet ────────────────────────────────────────────
    suffix = "_broad_smoke" if args.smoke else "_broad"
    parquet_path = os.path.join(args.data_dir, f"rl_dataset{suffix}.parquet")
    logging.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logging.info("  %d rows, %d cols", len(df), len(df.columns))

    state_cols = [c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")]
    n_state    = len(state_cols)
    logging.info("  %d state features", n_state)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)
    logging.info("  train=%d rows, test=%d rows", len(train_df), len(test_df))
    del df

    # ── Build history datasets (shared across all models) ─────────────
    logging.info("--- Building history datasets ---")
    t1 = time.time()
    train_dataset = build_history_dataset(train_df, state_cols, n_history=args.n_history)
    test_dataset  = build_history_dataset(test_df,  state_cols, n_history=args.n_history)
    logging.info("  Done in %.1f sec", time.time() - t1)

    log_every = 1 if args.smoke else 10

    # ── Train and evaluate each model ─────────────────────────────────
    all_results = {}
    for mt in model_types:
        result = train_and_evaluate_one(
            mt, train_dataset, test_dataset, test_df,
            state_cols, args, log_every,
        )
        all_results[mt] = result

    # ── Comparison summary ─────────────────────────────────────────────
    dt = time.time() - t0
    logging.info("")
    logging.info("=" * 65)
    logging.info("STEP 13 COMPLETE -- MODEL COMPARISON")
    logging.info("=" * 65)
    logging.info("  Mode: %s | Train: %d | Test: %d samples",
                 mode_str, len(train_dataset["history"]), len(test_dataset["history"]))
    logging.info("")
    logging.info("  %-8s  %12s  %12s  %14s  %14s  %8s",
                 "Model", "Best Val MSE", "Test MSE",
                 "Rollout MSE(1)", "Rollout MSE(N)", "Time(s)")
    logging.info("  %s", "-" * 78)
    for mt, r in all_results.items():
        logging.info("  %-8s  %12.6f  %12.6f  %14.6f  %14.6f  %8.1f",
                     mt.upper(), r["best_val_mse"], r["test_mse"],
                     r["rollout_mse_step1"], r["rollout_mse_last"], r["time_sec"])
    logging.info("")
    logging.info("  Total time: %.1f sec (%.1f min)", dt, dt / 60)
    logging.info("  Models:  %s/*/", args.model_dir)
    logging.info("  Reports: %s/simulator_*", args.report_dir)
    logging.info("=" * 65)

    summary_path = os.path.join(args.report_dir, "simulator_comparison.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info("  Comparison saved: %s", summary_path)


if __name__ == "__main__":
    main()
