"""
Step 13 - Model-based environment simulator (Raghu et al. 2018).

Chains from: step 09 outputs (rl_*_set_scaled.csv)
Produces:
  models/sepsis_rl/simulator/<model_type>/
    transition_model.pt         (trained transition model weights)
    model_config.pkl            (architecture config for reloading)
  reports/sepsis_rl/
    simulator_<model_type>_per_feature_mse.json
    simulator_<model_type>_rollout_eval.json

Four model architectures from Raghu 2018 Table 1:
  nn:     2 FC + ReLU + BN (paper's preferred model, MSE=0.171)
  linear: Linear regression baseline (MSE=0.195)
  lstm:   LSTM on history sequence (MSE=0.122, but poor at early timesteps)
  bnn:    Bayesian NN with variational inference (MSE=0.220, gives uncertainty)

Usage:
  python scripts/sepsis/step_13_simulator.py --model-type nn --smoke
  python scripts/sepsis/step_13_simulator.py --model-type all --smoke
  python scripts/sepsis/step_13_simulator.py --model-type all
"""
import argparse
import json
import logging
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, ".")
from src.careai.sepsis.rl.preprocessing import STATE_FEATURES
from src.careai.sepsis.rl.simulator import (
    MODEL_TYPES,
    build_history_dataset,
    evaluate_per_feature,
    evaluate_rollouts,
    train_transition_model,
)

RL_STATE_FEATURES = [f for f in STATE_FEATURES if f != "bloc"]

ALL_MODEL_TYPES = list(MODEL_TYPES.keys())


def train_and_evaluate_one(model_type, train_dataset, test_dataset, test_df,
                           state_cols, args, log_every):
    """Train one model type and evaluate it. Returns results dict."""
    n_state = len(state_cols)
    model_dir = f"{args.model_dir}/{model_type}"
    t_model = time.time()

    logging.info("")
    logging.info("=" * 50)
    logging.info("MODEL: %s", model_type.upper())
    logging.info("=" * 50)

    # Train
    model, train_losses, val_losses = train_transition_model(
        train_dataset,
        n_state=n_state,
        n_action=2,
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
    # Show top 5 and bottom 5
    for feat, mse in sorted_features[:5]:
        logging.info("    %-25s MSE=%.6f", feat, mse)
    if len(sorted_features) > 10:
        logging.info("    ... (%d more features) ...", len(sorted_features) - 10)
    for feat, mse in sorted_features[-5:]:
        logging.info("    %-25s MSE=%.6f", feat, mse)
    logging.info("    %-25s MSE=%.6f", "OVERALL", test_mse["__overall__"])

    mse_path = f"{args.report_dir}/simulator_{model_type}_per_feature_mse.json"
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

    rollout_path = f"{args.report_dir}/simulator_{model_type}_rollout_eval.json"
    with open(rollout_path, "w") as f:
        json.dump(rollout_results, f, indent=2)

    dt_model = time.time() - t_model
    logging.info("  [%s] done in %.1f sec", model_type.upper(), dt_model)

    return {
        "model_type": model_type,
        "best_val_mse": float(min(val_losses)),
        "test_mse": test_mse["__overall__"],
        "rollout_mse_step1": rollout_results["per_step_mse"][0],
        "rollout_mse_last": rollout_results["per_step_mse"][-1],
        "n_rollout_steps": rollout_results["n_steps"],
        "time_sec": dt_model,
    }


def main():
    parser = argparse.ArgumentParser(description="Step 13: Model-Based Simulator")
    parser.add_argument(
        "--data-dir", default="data/processed/sepsis",
        help="Directory with preprocessed data from step 09",
    )
    parser.add_argument(
        "--model-dir", default="models/sepsis_rl/simulator",
        help="Output directory for simulator models",
    )
    parser.add_argument(
        "--report-dir", default="reports/sepsis_rl",
        help="Output directory for evaluation reports",
    )
    parser.add_argument(
        "--model-type", default="nn",
        choices=ALL_MODEL_TYPES + ["all"],
        help="Model architecture: nn, linear, lstm, bnn, or all",
    )
    parser.add_argument(
        "--hidden", type=int, default=256,
        help="Hidden layer size (NN/LSTM; BNN uses 32 per paper)",
    )
    parser.add_argument(
        "--epochs", type=int, default=100,
        help="Training epochs",
    )
    parser.add_argument(
        "--batch-size", type=int, default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3,
        help="Learning rate",
    )
    parser.add_argument(
        "--n-history", type=int, default=4,
        help="Number of history timesteps (default: 4 = current + 3 past)",
    )
    parser.add_argument(
        "--rollout-steps", type=int, default=10,
        help="Number of rollout steps for evaluation",
    )
    parser.add_argument(
        "--rollout-patients", type=int, default=200,
        help="Number of patients for rollout evaluation",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test mode: 5 epochs, small rollout eval",
    )
    parser.add_argument(
        "--log", default="logs/step_13_simulator.log",
        help="Log file path",
    )
    args = parser.parse_args()

    # Smoke test overrides
    if args.smoke:
        args.epochs = 5
        args.rollout_patients = 20
        args.rollout_steps = 5

    os.makedirs(args.model_dir, exist_ok=True)
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
    mode_str = "SMOKE TEST" if args.smoke else "FULL"
    model_types = ALL_MODEL_TYPES if args.model_type == "all" else [args.model_type]
    logging.info("Step 13 started. mode=%s, device=%s, models=%s",
                 mode_str, args.device, model_types)
    t0 = time.time()

    # ── Load data ──────────────────────────────────────────────────────
    logging.info("Loading scaled datasets from %s", args.data_dir)
    import pandas as pd

    train_df = pd.read_csv(f"{args.data_dir}/rl_train_set_scaled.csv")
    test_df = pd.read_csv(f"{args.data_dir}/rl_test_set_scaled.csv")

    state_cols = [c for c in RL_STATE_FEATURES if c in train_df.columns]
    n_state = len(state_cols)
    logging.info("  %d state features, train=%d rows, test=%d rows",
                 n_state, len(train_df), len(test_df))

    # ── Build history dataset (shared across all models) ───────────────
    logging.info("--- Building history datasets ---")
    t1 = time.time()
    train_dataset = build_history_dataset(
        train_df, state_cols, n_history=args.n_history,
    )
    test_dataset = build_history_dataset(
        test_df, state_cols, n_history=args.n_history,
    )
    logging.info("  History datasets built in %.1f sec", time.time() - t1)

    log_every = 1 if args.smoke else 10

    # ── Train and evaluate each model ──────────────────────────────────
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
    logging.info("=" * 60)
    logging.info("STEP 13 COMPLETE — MODEL COMPARISON")
    logging.info("=" * 60)
    logging.info("  Mode: %s | Training samples: %d | Test samples: %d",
                 mode_str, len(train_dataset["history"]), len(test_dataset["history"]))
    logging.info("")
    logging.info("  %-8s  %12s  %12s  %14s  %14s  %8s",
                 "Model", "Best Val MSE", "Test MSE", "Rollout MSE(1)", "Rollout MSE(N)", "Time(s)")
    logging.info("  %s", "-" * 78)
    for mt, r in all_results.items():
        logging.info("  %-8s  %12.6f  %12.6f  %14.6f  %14.6f  %8.1f",
                     mt.upper(), r["best_val_mse"], r["test_mse"],
                     r["rollout_mse_step1"], r["rollout_mse_last"], r["time_sec"])
    logging.info("")
    logging.info("  Total time: %.1f sec (%.1f min)", dt, dt / 60)
    logging.info("  Models: %s/*/", args.model_dir)
    logging.info("  Reports: %s/simulator_*", args.report_dir)
    logging.info("=" * 60)

    # Save comparison summary
    summary_path = f"{args.report_dir}/simulator_comparison.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logging.info("  Comparison saved: %s", summary_path)


if __name__ == "__main__":
    main()
