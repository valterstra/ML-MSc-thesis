"""CLI: propensity score overlap and covariate balance diagnostics.

Usage
-----
    python scripts/causal/check_overlap.py
    python scripts/causal/check_overlap.py --csv data/processed/hosp_daily_transitions_sample5k.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from careai.causal_daily.balance import balance_table, check_overlap
from careai.causal_daily.features import ACTION_COLS
from careai.causal_daily.propensity import fit_propensity_models, predict_propensity
from careai.sim_daily.data import prepare_daily_data

_DEFAULT_CSV = _REPO / "data/processed/hosp_daily_transitions_sample5k.csv"
_DEFAULT_REPORT_DIR = _REPO / "reports/causal_daily"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Propensity overlap and covariate balance diagnostics")
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                   help="Path to hosp_daily transitions CSV")
    p.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR,
                   help="Directory to write balance CSV")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Loading data from {args.csv} ...")
    daily = prepare_daily_data(args.csv)
    train_df = daily.one_step_train
    test_df = daily.one_step_test

    print(f"Train rows: {len(train_df):,}  |  Test rows: {len(test_df):,}\n")

    # Fit propensity on train, assess on test
    print("Fitting propensity models ...")
    prop_model = fit_propensity_models(train_df)

    # ---------------------------------------------------------------------------
    # Overlap report
    # ---------------------------------------------------------------------------
    print()
    print("=" * 80)
    print("  PROPENSITY SCORE OVERLAP (test split)")
    print("=" * 80)
    hdr = f"{'Drug':<25} {'N_treated':>10} {'N_control':>10} {'PS_mean_T':>10} {'PS_mean_C':>10} {'Frac_out':>10} {'Flag':>6}"
    print(hdr)
    print("-" * 80)

    any_poor = False
    for drug in ACTION_COLS:
        ps = predict_propensity(prop_model, test_df, drug)
        info = check_overlap(test_df, ps, drug)
        flag = "POOR" if info["poor_overlap"] else "ok"
        if info["poor_overlap"]:
            any_poor = True
        print(
            f"{drug:<25} {info['n_treated']:>10,} {info['n_control']:>10,} "
            f"{info['ps_mean_treated']:>10.3f} {info['ps_mean_control']:>10.3f} "
            f"{info['frac_outside_0.1_0.9']:>10.1%} {flag:>6}"
        )

    if any_poor:
        print("\n  WARNING: Some drugs have poor overlap - ATE estimates may be unreliable.")
    else:
        print("\n  All drugs show acceptable overlap.")

    # ---------------------------------------------------------------------------
    # Covariate balance table
    # ---------------------------------------------------------------------------
    print()
    print("Computing covariate balance table (this may take a moment) ...")
    bal = balance_table(test_df, prop_model)

    # Summary: mean abs SMD before/after per drug
    print()
    print("=" * 80)
    print("  COVARIATE BALANCE SUMMARY (mean |SMD| across confounders)")
    print("=" * 80)
    print(f"{'Drug':<25} {'Raw mean|SMD|':>15} {'Weighted mean|SMD|':>20} {'Improved?':>10}")
    print("-" * 80)

    for drug in ACTION_COLS:
        sub = bal[bal["drug"] == drug]
        raw_mean = sub["raw_smd"].abs().mean()
        wt_mean = sub["weighted_smd"].abs().mean()
        improved = "yes" if wt_mean < raw_mean else "NO"
        print(f"{drug:<25} {raw_mean:>15.3f} {wt_mean:>20.3f} {improved:>10}")

    print()

    # Save full balance table
    args.report_dir.mkdir(parents=True, exist_ok=True)
    out = args.report_dir / "covariate_balance.csv"
    bal.to_csv(out, index=False)
    print(f"Saved full balance table: {out}")

    # Show top imbalanced confounders per drug (raw SMD > 0.2)
    badly = bal[bal["raw_smd"].abs() > 0.2].sort_values("raw_smd", ascending=False, key=abs)
    if not badly.empty:
        print()
        print(f"Confounders with |raw SMD| > 0.2 ({len(badly)} pairs):")
        print(badly[["drug", "confounder", "raw_smd", "weighted_smd"]].to_string(index=False))


if __name__ == "__main__":
    main()
