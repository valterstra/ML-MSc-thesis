#!/usr/bin/env python
"""Train the daily hospital-stay transition model and save to disk."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Ensure project root is on sys.path so `src.careai` is importable
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from careai.sim_daily.data import prepare_daily_data
from careai.sim_daily.evaluate import single_step_metrics
from careai.sim_daily.transition import fit_transition_model, save_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train daily transition model")
    parser.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions_sample5k.csv"),
        help="Path to the hosp_daily CSV",
    )
    parser.add_argument(
        "--model-dir",
        default=str(_PROJECT_ROOT / "models" / "sim_daily"),
        help="Directory to save model files",
    )
    parser.add_argument(
        "--report-dir",
        default=str(_PROJECT_ROOT / "reports" / "sim_daily"),
        help="Directory to save metrics JSON",
    )
    args = parser.parse_args()

    # 1. Load and prepare data
    print(f"Loading data from {args.csv} ...")
    data = prepare_daily_data(args.csv)
    print(f"  Raw rows:       {len(data.raw):,}")
    print(f"  Train trans:    {len(data.one_step_train):,}")
    print(f"  Valid trans:    {len(data.one_step_valid):,}")
    print(f"  Test trans:     {len(data.one_step_test):,}")
    print(f"  Initial states: {len(data.initial_states):,}")

    # 2. Train transition model
    print("\nTraining transition model (LightGBM) ...")
    model = fit_transition_model(data.one_step_train, data.one_step_valid)
    print(f"  Continuous models: {len(model.continuous_models)}")
    print(f"  Binary models:     {len(model.binary_models)}")
    print(f"  Done model:        trained")

    # 3. Save model
    save_model(model, args.model_dir)
    print(f"\nModel saved to {args.model_dir}/")

    # 4. Evaluate single-step metrics
    print("\nSingle-step metrics on test set:")
    metrics = single_step_metrics(model, data.one_step_test)

    print("\n  Continuous (R² / MAE / n):")
    for col, m in metrics["continuous"].items():
        r2 = f"{m['r2']:.4f}" if m["r2"] is not None else "N/A"
        mae = f"{m['mae']:.4f}" if m["mae"] is not None else "N/A"
        print(f"    {col:30s}  R²={r2}  MAE={mae}  n={m['n']}")

    print("\n  Binary (AUC / n):")
    for col, m in metrics["binary"].items():
        auc = f"{m['auc']:.4f}" if m["auc"] is not None else "N/A"
        print(f"    {col:40s}  AUC={auc}  n={m['n']}")

    done_auc = metrics["done"].get("auc")
    print(f"\n  Done model AUC: {done_auc:.4f}" if done_auc else "\n  Done model AUC: N/A")

    # 5. Save metrics JSON
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = report_dir / "single_step_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nMetrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
