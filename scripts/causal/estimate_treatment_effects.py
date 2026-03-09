"""CLI: estimate causal treatment effects using AIPW (doubly robust).

Usage
-----
    python scripts/causal/estimate_treatment_effects.py
    python scripts/causal/estimate_treatment_effects.py --csv data/processed/hosp_daily_transitions_sample5k.csv
    python scripts/causal/estimate_treatment_effects.py --n-boot 200 --report-dir reports/causal_daily
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from repo root without installing the package
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from careai.causal_daily.evaluate import print_results_table, run_causal_analysis
from careai.causal_daily.features import TREATMENT_OUTCOME_PAIRS
from careai.sim_daily.data import prepare_daily_data

_DEFAULT_CSV = _REPO / "data/processed/hosp_daily_transitions_sample5k.csv"
_DEFAULT_REPORT_DIR = _REPO / "reports/causal_daily"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Estimate causal treatment effects (AIPW)")
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                   help="Path to hosp_daily transitions CSV")
    p.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR,
                   help="Directory to write results CSV and JSON")
    p.add_argument("--n-boot", type=int, default=500,
                   help="Number of bootstrap resamples for CIs (default: 500)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.csv} ...")
    daily = prepare_daily_data(args.csv)
    train_df = daily.one_step_train
    test_df = daily.one_step_test

    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}")
    print(f"Treatment-outcome pairs: {len(TREATMENT_OUTCOME_PAIRS)}")
    print()

    results = run_causal_analysis(
        train_df=train_df,
        test_df=test_df,
        treatment_outcome_pairs=TREATMENT_OUTCOME_PAIRS,
        n_boot=args.n_boot,
        seed=args.seed,
        verbose=True,
    )

    print_results_table(results)

    # Save outputs
    args.report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.report_dir / "treatment_effects.csv"
    results.to_csv(csv_path, index=False)
    print(f"Saved CSV:  {csv_path}")

    json_path = args.report_dir / "treatment_effects.json"
    json_path.write_text(results.to_json(orient="records", indent=2))
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
