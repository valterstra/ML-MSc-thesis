"""Action sensitivity analysis for the daily transition model.

For each action (drug class), we take real patient-days from the test set,
flip that action from 0 to 1 (and 1 to 0), and measure how the predicted
next state changes. This tells us whether the model actually learned the
effect of each drug — not just correlations.

Expected clinical directions (what good results look like):
  insulin_active        -> glucose DOWN
  anticoagulant_active  -> INR UP
  diuretic_active       -> BUN UP (concentration), potassium DOWN, sodium UP
  steroid_active        -> glucose UP, WBC UP (demargination)
  antibiotic_active     -> WBC DOWN (infection clearing), over time
  opioid_active         -> less clear physiologically at daily resolution
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from careai.sim_daily.data import prepare_daily_data
from careai.sim_daily.features import ACTION_COLS, OUTPUT_BINARY, OUTPUT_CONTINUOUS
from careai.sim_daily.transition import load_model

# What we expect from each action, for the final verdict
CLINICAL_EXPECTATIONS: dict[str, dict[str, str]] = {
    "insulin_active":        {"glucose": "down", "potassium": "down"},
    "anticoagulant_active":  {"inr": "up"},
    "diuretic_active":       {"bun": "up", "potassium": "down", "magnesium": "down",
                              "bicarbonate": "up", "sodium": "up"},
    "steroid_active":        {"glucose": "up", "wbc": "up"},
    "antibiotic_active":     {"wbc": "down"},
    "opioid_active":         {},  # no strong single-step expectation
}


def run_sensitivity(
    model,
    test_df: pd.DataFrame,
    n_sample: int = 3000,
    seed: int = 42,
) -> pd.DataFrame:
    """For each action × output, compute mean predicted difference (action=1) - (action=0)."""
    rng = np.random.default_rng(seed)

    # Sample rows for speed
    idx = rng.choice(len(test_df), size=min(n_sample, len(test_df)), replace=False)
    sample = test_df.iloc[idx].reset_index(drop=True)

    X = sample[model.input_cols].copy()
    results = []

    for action in ACTION_COLS:
        # Predict with action forced ON
        X_on = X.copy()
        X_on[action] = 1.0

        # Predict with action forced OFF
        X_off = X.copy()
        X_off[action] = 0.0

        # Continuous outputs
        for col in model.output_continuous:
            pred_on  = model.continuous_models[col].predict(X_on)
            pred_off = model.continuous_models[col].predict(X_off)
            diff = pred_on - pred_off  # positive = action increases this lab

            results.append({
                "action": action,
                "output": col,
                "output_type": "continuous",
                "mean_diff": float(np.mean(diff)),
                "std_diff":  float(np.std(diff)),
                "pct_rows_affected": float(np.mean(np.abs(diff) > 1e-6)),
            })

        # Binary outputs — difference in predicted probability
        for col in model.output_binary:
            prob_on  = model.binary_models[col].predict_proba(X_on)[:, 1]
            prob_off = model.binary_models[col].predict_proba(X_off)[:, 1]
            diff = prob_on - prob_off

            results.append({
                "action": action,
                "output": col,
                "output_type": "binary_prob",
                "mean_diff": float(np.mean(diff)),
                "std_diff":  float(np.std(diff)),
                "pct_rows_affected": float(np.mean(np.abs(diff) > 1e-4)),
            })

    return pd.DataFrame(results)


def print_results(df: pd.DataFrame) -> None:
    """Print a readable table grouped by action."""
    print("\n" + "=" * 70)
    print("ACTION SENSITIVITY ANALYSIS")
    print("mean_diff = average change in predicted next-day value")
    print("when action is forced ON (=1) vs forced OFF (=0)")
    print("=" * 70)

    for action in ACTION_COLS:
        sub = df[df["action"] == action].copy()
        sub = sub.reindex(sub["mean_diff"].abs().sort_values(ascending=False).index)

        expectations = CLINICAL_EXPECTATIONS.get(action, {})
        print(f"\n{'-' * 60}")
        print(f"  {action.upper()}")
        if expectations:
            expected_str = ", ".join(f"{k} {v}" for k, v in expectations.items())
            print(f"  Expected: {expected_str}")
        print(f"  {'Output':<45} {'Mean diff':>10}  {'Direction'}")
        print(f"  {'-'*45} {'-'*10}  {'-'*10}")

        for _, row in sub.iterrows():
            diff = row["mean_diff"]
            direction = "UP  " if diff > 0.001 else ("DOWN" if diff < -0.001 else "FLAT")
            col = row["output"]

            # Check against expectation
            expected_dir = expectations.get(col)
            if expected_dir == "up" and diff > 0.001:
                verdict = "OK"
            elif expected_dir == "down" and diff < -0.001:
                verdict = "OK"
            elif expected_dir is not None:
                verdict = "WRONG"
            else:
                verdict = ""

            print(f"  {col:<45} {diff:>+10.4f}  {direction}  {verdict}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Action sensitivity analysis")
    parser.add_argument("--csv", default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions_sample5k.csv"))
    parser.add_argument("--model-dir", default=str(_PROJECT_ROOT / "models" / "sim_daily"))
    parser.add_argument("--report-dir", default=str(_PROJECT_ROOT / "reports" / "sim_daily"))
    args = parser.parse_args()

    csv_path = Path(args.csv)
    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("Loading data ...")
    data = prepare_daily_data(csv_path)

    print("Loading model ...")
    model = load_model(model_dir)

    print(f"Running sensitivity analysis on {min(3000, len(data.one_step_test))} test rows ...")
    results = run_sensitivity(model, data.one_step_test)

    print_results(results)

    # Save
    out_path = report_dir / "action_sensitivity.csv"
    results.to_csv(out_path, index=False)
    print(f"\nFull results saved to {out_path}")


if __name__ == "__main__":
    main()
