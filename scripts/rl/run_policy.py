#!/usr/bin/env python
"""Run causal exhaustive policy on test patients and save evaluation report.

Supports two policy types via --policy-type:
  ate  (default) — population-level ATE corrections (Option C)
  cate           — patient-specific CATE corrections (Option D)

Outputs are written to reports/rl_daily/{policy_type}/ so the two runs
don't overwrite each other and can be compared directly.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from careai.sim_daily.data import prepare_daily_data
from careai.sim_daily.transition import load_model
from careai.rl_daily.readmission import load_readmission_model
from careai.rl_daily.policy import load_ate_table, ATE_DRUGS
from careai.rl_daily.evaluate import evaluate_policy, print_policy_summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run causal RL policy (ATE or CATE-corrected) on test patients"
    )
    parser.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions_sample5k.csv"),
        help="Path to the hosp_daily CSV",
    )
    parser.add_argument(
        "--transition-model-dir",
        default=str(_PROJECT_ROOT / "models" / "sim_daily"),
        help="Directory containing the transition model",
    )
    parser.add_argument(
        "--readmission-model-dir",
        default=str(_PROJECT_ROOT / "models" / "rl_daily"),
        help="Directory containing the readmission model",
    )
    parser.add_argument(
        "--ate-json",
        default=str(_PROJECT_ROOT / "reports" / "causal_daily_full" / "treatment_effects.json"),
        help="Path to treatment_effects.json from causal_daily step",
    )
    parser.add_argument(
        "--policy-type",
        choices=["ate", "cate"],
        default="ate",
        help="Policy type: 'ate' (population ATE, default) or 'cate' (patient-specific CATE)",
    )
    parser.add_argument(
        "--cate-model-dir",
        default=str(_PROJECT_ROOT / "models" / "cate_daily"),
        help="Directory containing CATE models (used when --policy-type cate)",
    )
    parser.add_argument(
        "--report-dir",
        default=None,
        help=(
            "Directory to save policy evaluation outputs. "
            "Defaults to reports/rl_daily/{policy_type}/"
        ),
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=500,
        help="Number of test patients to evaluate (default: 500)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Resolve report directory: default is reports/rl_daily/{policy_type}/
    if args.report_dir is None:
        args.report_dir = str(_PROJECT_ROOT / "reports" / "rl_daily" / args.policy_type)

    # 1. Load data
    print(f"Loading data from {args.csv} ...")
    data = prepare_daily_data(args.csv)
    test_initial = data.initial_states[data.initial_states["split"] == "test"].copy()
    print(f"  Test initial states: {len(test_initial):,}")

    # 2. Load transition model
    print(f"\nLoading transition model from {args.transition_model_dir} ...")
    transition_model = load_model(args.transition_model_dir)

    # 3. Load readmission model
    print(f"\nLoading readmission model from {args.readmission_model_dir} ...")
    readmission_model = load_readmission_model(args.readmission_model_dir)
    print(f"  AUC: {readmission_model.test_auc:.4f}  Brier: {readmission_model.test_brier:.4f}")

    # 4. Load causal effect table (ATE or CATE)
    ate_table = None
    cate_registry = None

    if args.policy_type == "ate":
        print(f"\nLoading ATE table from {args.ate_json} ...")
        ate_table = load_ate_table(args.ate_json)
        print(f"  (treatment, outcome) pairs: {len(ate_table)}")
        print(f"  Optimising over drugs: {ATE_DRUGS}")
        print(f"  Search space: 2^{len(ATE_DRUGS)} = {2**len(ATE_DRUGS)} action combos")
    else:
        from careai.causal_daily.cate import load_cate_registry
        print(f"\nLoading CATE registry from {args.cate_model_dir} ...")
        cate_registry = load_cate_registry(args.cate_model_dir)
        print(f"  Fitted pairs: {len(cate_registry.models)}")
        print(f"  Optimising over drugs: {ATE_DRUGS}")
        print(f"  Search space: 2^{len(ATE_DRUGS)} = {2**len(ATE_DRUGS)} action combos")

    # 5. Run evaluation
    n = min(args.n_patients, len(test_initial))
    print(f"\nEvaluating {args.policy_type.upper()} policy on {n} test patients ...")

    results = evaluate_policy(
        initial_states=test_initial,
        transition_model=transition_model,
        readmission_model=readmission_model,
        ate_table=ate_table,
        cate_registry=cate_registry,
        n_patients=args.n_patients,
        seed=args.seed,
    )

    # 6. Print and save
    print_policy_summary(results)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    csv_path = report_dir / "policy_evaluation.csv"
    results.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}  ({len(results)} rows)")

    drug_cols = [c for c in results.columns if c.startswith("policy_") and c != "policy_risk"]
    n_pairs = len(ate_table) if ate_table is not None else len(cate_registry.models)
    summary = {
        "policy_type": args.policy_type,
        "n_patients": len(results),
        "optimised_drugs": ATE_DRUGS,
        "n_causal_pairs": n_pairs,
        "mean_policy_risk": float(results["policy_risk"].mean()),
        "mean_donothing_risk": float(results["donothing_risk"].mean()),
        "mean_real_risk": float(results["real_risk"].mean()),
        "policy_beats_donothing_frac": float(
            (results["policy_risk"] < results["donothing_risk"]).mean()
        ),
        "policy_beats_real_frac": float(
            (results["policy_risk"] < results["real_risk"]).mean()
        ),
        "drug_recommendation_freq": {
            col[len("policy_"):]: float(results[col].mean()) for col in drug_cols
        },
    }
    json_path = report_dir / "policy_evaluation.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON saved to {json_path}")


if __name__ == "__main__":
    main()
