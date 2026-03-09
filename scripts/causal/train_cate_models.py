"""CLI: fit CausalForestDML CATE models for all treatment-outcome pairs.

Usage
-----
    python scripts/causal/train_cate_models.py
    python scripts/causal/train_cate_models.py --csv data/processed/hosp_daily_transitions_sample5k.csv
    python scripts/causal/train_cate_models.py --n-estimators 200 --max-depth 5

After fitting, the script compares population CATEs to the AIPW ATEs
(if treatment_effects.json is available) and saves a summary CSV and JSON
to --report-dir (default: reports/cate_daily/).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

import pandas as pd

from careai.causal_daily.cate import (
    CATERegistry,
    fit_cate_registry,
    load_cate_registry,
    save_cate_registry,
)
from careai.causal_daily.features import EXPECTED_DIRECTION, TREATMENT_OUTCOME_PAIRS
from careai.sim_daily.data import prepare_daily_data

_DEFAULT_CSV = _REPO / "data/processed/hosp_daily_transitions_sample5k.csv"
_DEFAULT_MODEL_DIR = _REPO / "models/cate_daily"
_DEFAULT_REPORT_DIR = _REPO / "reports/cate_daily"
_DEFAULT_ATE_JSON = _REPO / "reports/causal_daily_full/treatment_effects.json"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train CausalForestDML CATE models for all treatment-outcome pairs"
    )
    p.add_argument("--csv", type=Path, default=_DEFAULT_CSV,
                   help="Path to hosp_daily transitions CSV")
    p.add_argument("--model-dir", type=Path, default=_DEFAULT_MODEL_DIR,
                   help="Directory to save fitted CATE models")
    p.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR,
                   help="Directory to save summary reports")
    p.add_argument("--ate-json", type=Path, default=_DEFAULT_ATE_JSON,
                   help="Path to treatment_effects.json (AIPW results) for comparison")
    p.add_argument("--n-estimators", type=int, default=200,
                   help="Number of trees in CausalForestDML (default: 200)")
    p.add_argument("--max-depth", type=int, default=5,
                   help="Max tree depth in CausalForestDML (default: 5)")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    return p.parse_args()


def _load_aipw_ate(ate_json: Path) -> dict[tuple[str, str], float] | None:
    """Load AIPW ATEs from treatment_effects.json, or return None if absent."""
    if not ate_json.exists():
        return None
    entries = json.loads(ate_json.read_text())
    return {(e["treatment"], e["outcome"]): e["causal_ate"] for e in entries}


def _build_summary(registry: CATERegistry, aipw_ates: dict | None) -> pd.DataFrame:
    """Build comparison table: population CATE vs AIPW ATE vs expected direction."""
    rows = []
    for (treatment, outcome), model in registry.models.items():
        expected = EXPECTED_DIRECTION.get((treatment, outcome), "?")
        aipw_ate = aipw_ates.get((treatment, outcome)) if aipw_ates else None

        cate_sign = "+" if model.population_ate > 0 else "-"
        expected_sign = "+" if expected == "up" else ("-" if expected == "down" else "?")
        direction_match = (cate_sign == expected_sign) if expected != "?" else None

        rows.append(
            {
                "treatment": treatment,
                "outcome": outcome,
                "expected_direction": expected,
                "population_cate": model.population_ate,
                "cate_std": model.ate_std,
                "aipw_ate": aipw_ate,
                "cate_sign_correct": direction_match,
                "n_train": model.n_train,
            }
        )
    return pd.DataFrame(rows)


def _print_comparison(summary: pd.DataFrame) -> None:
    """Print a formatted comparison table."""
    print("\n=== CATE vs AIPW ATE Comparison ===\n")
    header = (
        f"{'Treatment':<28} {'Outcome':<34} "
        f"{'Pop.CATE':>10} {'±std':>8} {'AIPW ATE':>10} {'Expected':>10} {'Correct?':>9}"
    )
    print(header)
    print("-" * len(header))
    for _, row in summary.iterrows():
        aipw_str = f"{row['aipw_ate']:+.4f}" if row["aipw_ate"] is not None else "     n/a"
        correct_str = (
            "yes" if row["cate_sign_correct"] is True
            else ("NO" if row["cate_sign_correct"] is False else "?")
        )
        print(
            f"{row['treatment']:<28} {row['outcome']:<34} "
            f"{row['population_cate']:>+10.4f} {row['cate_std']:>8.4f} "
            f"{aipw_str:>10} {row['expected_direction']:>10} {correct_str:>9}"
        )
    print()


def main() -> None:
    args = parse_args()

    # 1. Load data
    print(f"Loading data from {args.csv} ...")
    daily = prepare_daily_data(args.csv)
    train_df = daily.one_step_train
    print(f"  Train rows: {len(train_df):,}")
    print(f"  Treatment-outcome pairs: {len(TREATMENT_OUTCOME_PAIRS)}")
    print()

    # 2. Fit CATE models
    print("Fitting CausalForestDML models ...")
    registry = fit_cate_registry(
        train_df=train_df,
        pairs=TREATMENT_OUTCOME_PAIRS,
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        seed=args.seed,
    )
    print(f"\nFitted {len(registry.models)} CATE models.")

    # 3. Save models
    save_cate_registry(registry, args.model_dir)

    # 4. Load AIPW ATEs for comparison (optional)
    aipw_ates = _load_aipw_ate(args.ate_json)
    if aipw_ates is None:
        print(f"\n(AIPW ATE file not found at {args.ate_json} — skipping comparison)")
    else:
        print(f"\nLoaded {len(aipw_ates)} AIPW ATEs from {args.ate_json}")

    # 5. Build and print comparison table
    summary = _build_summary(registry, aipw_ates)
    _print_comparison(summary)

    # 6. Save reports
    args.report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = args.report_dir / "cate_summary.csv"
    summary.to_csv(csv_path, index=False)
    print(f"Saved CSV:  {csv_path}")

    json_path = args.report_dir / "cate_summary.json"
    json_path.write_text(summary.to_json(orient="records", indent=2))
    print(f"Saved JSON: {json_path}")


if __name__ == "__main__":
    main()
