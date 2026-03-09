#!/usr/bin/env python
"""Train the readmission risk model used as the RL reward signal."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from careai.sim_daily.data import prepare_daily_data
from careai.rl_daily.readmission import fit_readmission_model, save_readmission_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train readmission reward model")
    parser.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions_sample5k.csv"),
        help="Path to the hosp_daily CSV",
    )
    parser.add_argument(
        "--model-dir",
        default=str(_PROJECT_ROOT / "models" / "rl_daily"),
        help="Directory to save model (default: models/rl_daily)",
    )
    parser.add_argument(
        "--report-dir",
        default=str(_PROJECT_ROOT / "reports" / "rl_daily"),
        help="Directory to save metrics JSON (default: reports/rl_daily)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load and prepare data
    print(f"Loading data from {args.csv} ...")
    data = prepare_daily_data(args.csv)
    print(f"  Raw rows:      {len(data.raw):,}")
    print(f"  Train rows:    {(data.raw['split'] == 'train').sum():,}")
    print(f"  Test rows:     {(data.raw['split'] == 'test').sum():,}")

    train_df = data.raw[data.raw["split"] == "train"].copy()
    test_df  = data.raw[data.raw["split"] == "test"].copy()

    # 2. Train readmission model
    print("\nTraining readmission risk model ...")
    model = fit_readmission_model(train_df, test_df)

    print(f"\nResults on test set:")
    print(f"  AUC:        {model.test_auc:.4f}")
    print(f"  Brier:      {model.test_brier:.4f}")
    print(f"  Prevalence: {model.prevalence:.4f}")
    print(f"  Features:   {len(model.feature_cols)}")

    # 3. Save model
    save_readmission_model(model, args.model_dir)
    print(f"\nModel saved to {args.model_dir}/")

    # 4. Save metrics JSON
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "test_auc": model.test_auc,
        "test_brier": model.test_brier,
        "prevalence": model.prevalence,
        "n_features": len(model.feature_cols),
        "feature_cols": model.feature_cols,
    }
    metrics_path = report_dir / "readmission_metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"Metrics saved to {metrics_path}")


if __name__ == "__main__":
    main()
