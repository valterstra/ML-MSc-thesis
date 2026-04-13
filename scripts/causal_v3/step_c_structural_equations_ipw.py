"""Step C (IPW): Structural equations trained with inverse probability weighting.

PURPOSE
=======
This script tests whether IPW-weighted LightGBM training can recover the
correct interventional direction for the 3 drug->lab edges that learned
wrong signs due to confounding-by-indication in the observational data:

  insulin_active    -> next_glucose    (observational: +15.5, should be DOWN)
  steroid_active    -> next_platelets  (observational: -2.3,  should be UP)
  anticoagulant_active -> next_platelets (observational: +3.0, should be DOWN)

APPROACH vs DECONFOUNDED VARIANT
=================================
The deconfounded variant (step_c_structural_equations_deconfounded.py) removes
these 3 drug edges entirely, giving them zero effect. That is the conservative
fix: zero is wrong but safe; a flipped sign actively corrupts RL training.

This script keeps ALL drug edges in the parent sets (starting from the
NOTEARS-augmented parent sets) and instead applies stabilized IPW sample
weights when training each LightGBM model. The weights upweight rows where
drug assignment was "surprising" given the patient's observed state (quasi-
experimental observations) and downweight rows where assignment was highly
predictable (strongly confounded).

Stabilized weight for drug d at row i:
  treated   (d_i=1):  w_i = P(d=1) / P(d=1 | x_i)
  untreated (d_i=0):  w_i = P(d=0) / P(d=0 | x_i)

For models with multiple drug parents, per-drug weights are multiplied.
All weights clipped to [0.05, 20].

THREE POSSIBLE OUTCOMES (per wrong-direction edge):
  1. Sign flips to correct -> IPW solved it (data-driven deconfounding works)
  2. Magnitude shrinks but sign stays wrong -> IPW helps but unmeasured
     confounders (diabetes, HbA1c) dominate; manual override still needed
  3. No change -> confounders are truly outside the dataset; validates
     the manual deconfounding approach as necessary

OUTPUTS
=======
  models/causal_v3/structural_equations_ipw/
    {all 14 targets}.pkl
    manifest.json
    train_metrics.csv / test_metrics.csv / feature_importances.csv
    direction_validation.csv   <- key output: directions vs naive vs deconfounded
  reports/causal_v3/step_c_ipw/
    run_log.txt

USAGE
=====
  # Smoke test (fast, uses 5000 train rows)
  python scripts/causal_v3/step_c_structural_equations_ipw.py --sample-n 5000

  # Full run
  python scripts/causal_v3/step_c_structural_equations_ipw.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger(__name__)

# Edges to validate after training.
# (drug, target, expected_sign, description)
DIRECTION_CHECKS = [
    # Previously WRONG in naive NOTEARS-augmented simulator
    ("insulin_active",       "next_glucose",   -1, "DOWN - insulin lowers glucose"),
    ("steroid_active",       "next_platelets", +1, "UP   - steroids raise platelets (ITP tx)"),
    ("anticoagulant_active", "next_platelets", -1, "DOWN - anticoag lowers platelets (HIT)"),
    # Previously CORRECT - should stay correct after IPW
    ("diuretic_active",      "next_bun",       +1, "UP   - diuretics raise BUN (hemoconcentration)"),
    ("diuretic_active",      "next_potassium", -1, "DOWN - loop diuretics deplete K+"),
    ("steroid_active",       "next_glucose",   +1, "UP   - steroids raise glucose"),
    ("steroid_active",       "next_wbc",       +1, "UP   - steroids cause demargination"),
    ("anticoagulant_active", "next_inr",       +1, "UP   - anticoag extends clotting time"),
    ("antibiotic_active",    "next_hemoglobin",-1, "DOWN - antibiotic effect (PC found this)"),
    ("antibiotic_active",    "next_calcium",   -1, "DOWN - antibiotic effect (PC found this)"),
]


def _validate_directions(
    models: dict,
    parent_sets: dict,
    test_df: pd.DataFrame,
    n_sample: int = 1000,
) -> pd.DataFrame:
    """Flip each drug 0->1 and measure mean delta on test sample."""
    from careai.causal_v3.structural_equations import BINARY_TARGETS

    base = test_df.dropna(subset=list(parent_sets.values())[0]).head(n_sample)
    rows = []

    for drug, target, expected_sign, note in DIRECTION_CHECKS:
        if target not in models or models[target] is None:
            continue
        parents = parent_sets.get(target, [])
        if drug not in parents:
            rows.append({
                "drug": drug, "target": target,
                "delta": 0.0, "expected_sign": expected_sign,
                "direction": "REMOVED (zero)", "note": note,
            })
            continue

        avail = base[parents].dropna()
        if len(avail) < 50:
            continue

        b0 = avail.copy(); b0[drug] = 0
        b1 = avail.copy(); b1[drug] = 1

        var_name = target.replace("next_", "")
        m = models[target]
        if var_name in BINARY_TARGETS:
            d0 = m.predict_proba(b0)[:, 1]
            d1 = m.predict_proba(b1)[:, 1]
        else:
            d0 = m.predict(b0)
            d1 = m.predict(b1)

        delta = float((d1 - d0).mean())
        correct = (expected_sign > 0 and delta > 0) or (expected_sign < 0 and delta < 0)
        direction = "CORRECT" if correct else "WRONG"

        rows.append({
            "drug": drug, "target": target,
            "delta": round(delta, 4),
            "expected_sign": "+" if expected_sign > 0 else "-",
            "direction": direction,
            "note": note,
        })

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step C (IPW): structural equations with inverse probability weighting.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--source-model-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations_notears"),
        help="NOTEARS-augmented model dir to read parent sets from.",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations_ipw"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_c_ipw"),
    )
    parser.add_argument(
        "--sample-n", type=int, default=0,
        help="If > 0, subsample train rows for smoke test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir  = Path(args.model_dir)
    source_dir = Path(args.source_model_dir)
    report_dir = Path(args.report_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(report_dir / "run_log.txt"),
                                mode="w", encoding="utf-8"),
        ],
    )

    t_start = time.time()
    log.info("=" * 70)
    log.info("Step C (IPW): structural equations with IPW sample weights")
    log.info("  source-model-dir : %s", source_dir)
    log.info("  model-dir        : %s", model_dir)
    log.info("  sample-n         : %s", args.sample_n or "ALL (full run)")
    log.info("=" * 70)

    from careai.causal_v3.structural_equations import (
        build_parent_sets_notears_augmented,
        train_all_equations,
        evaluate_all_equations,
        save_models,
        ACTION_FEATURES,
    )
    from careai.causal_v3.propensity_v3 import (
        fit_propensity_v3,
        compute_model_weights,
    )

    # ── [1/6] Load parent sets from NOTEARS-augmented manifest ────────────
    log.info("")
    log.info("[1/6] Loading NOTEARS-augmented parent sets ...")
    import json
    with open(source_dir / "manifest.json") as f:
        source_manifest = json.load(f)
    parent_sets: dict[str, list[str]] = dict(source_manifest["parent_sets"])
    log.info("  Loaded parent sets for %d targets", len(parent_sets))
    for target, parents in parent_sets.items():
        drug_pars = [p for p in parents if p in ACTION_FEATURES]
        log.info("  %s: %d features, drugs=%s", target, len(parents), drug_pars)

    # ── [2/6] Load data ───────────────────────────────────────────────────
    log.info("")
    log.info("[2/6] Loading V3 triplet data ...")
    t0 = time.time()
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1f sec", len(df), time.time() - t0)

    train_df = df[df["split"] == "train"].copy().reset_index(drop=True)
    test_df  = df[df["split"] == "test"].copy().reset_index(drop=True)
    log.info("  Train: %d rows | Test: %d rows", len(train_df), len(test_df))

    if args.sample_n > 0 and args.sample_n < len(train_df):
        train_df = train_df.sample(n=args.sample_n, random_state=args.seed
                                   ).reset_index(drop=True)
        log.info("  Subsampled train to %d rows (smoke test)", len(train_df))

    # ── [3/6] Fit V3 propensity models ───────────────────────────────────
    log.info("")
    log.info("[3/6] Fitting V3 propensity models (logistic regression per drug) ...")
    t0 = time.time()
    prop = fit_propensity_v3(train_df)
    log.info("  Propensity models fitted in %.1f sec", time.time() - t0)

    # ── [4/6] Compute per-model IPW weights and train ────────────────────
    log.info("")
    log.info("[4/6] Training structural equations with IPW sample weights ...")

    models: dict = {}
    all_train_metrics: list[dict] = []

    for target, parents in parent_sets.items():
        drug_parents = [p for p in parents if p in ACTION_FEATURES]

        log.info("")
        log.info("  --- %s ---", target)
        log.info("    drug parents: %s", drug_parents or "(none)")

        # Compute combined IPW weight for this model
        t0 = time.time()
        sample_weight = compute_model_weights(prop, train_df, drug_parents)
        log.info("    weight computation: %.1f sec  weighted=%s",
                 time.time() - t0, sample_weight is not None)

        # Train with weights
        from careai.causal_v3.structural_equations import train_equation
        t0 = time.time()
        model, metrics = train_equation(
            train_df, target, parents, sample_weight=sample_weight,
        )
        log.info("    train: %.1f sec  metric=%.4f",
                 time.time() - t0,
                 metrics.get("train_r2", metrics.get("train_auc", float("nan"))))

        models[target] = model
        all_train_metrics.append(metrics)

    # ── [5/6] Evaluate on test split ─────────────────────────────────────
    log.info("")
    log.info("[5/6] Evaluating on test split ...")
    test_metrics = evaluate_all_equations(models, test_df, parent_sets)

    # ── [6/6] Direction validation ────────────────────────────────────────
    log.info("")
    log.info("[6/6] Direction validation (flip drug 0->1, measure mean delta) ...")
    direction_df = _validate_directions(models, parent_sets, test_df)

    log.info("")
    log.info("%-26s %-22s %10s %8s %s",
             "Drug", "Target", "Delta", "Expected", "Direction")
    log.info("-" * 80)
    for _, row in direction_df.iterrows():
        log.info("%-26s %-22s %+10.4f %8s %s",
                 row["drug"], row["target"], row["delta"],
                 row["expected_sign"], row["direction"])

    n_correct = (direction_df["direction"] == "CORRECT").sum()
    n_wrong   = (direction_df["direction"] == "WRONG").sum()
    n_removed = direction_df["direction"].str.startswith("REMOVED").sum()
    log.info("")
    log.info("Direction summary: %d CORRECT  |  %d WRONG  |  %d REMOVED",
             n_correct, n_wrong, n_removed)

    # Save
    direction_csv = model_dir / "direction_validation.csv"
    direction_df.to_csv(direction_csv, index=False)
    log.info("Direction validation saved to %s", direction_csv)

    # Compare with naive NOTEARS-augmented (load that direction from log or recompute)
    log.info("")
    log.info("KNOWN DIRECTIONS IN NAIVE NOTEARS-AUGMENTED (for comparison):")
    log.info("  insulin_active    -> next_glucose   : +15.5 (WRONG, expected DOWN)")
    log.info("  steroid_active    -> next_platelets :  -2.3 (WRONG, expected UP)")
    log.info("  anticoagulant     -> next_platelets :  +3.0 (WRONG, expected DOWN)")

    # Save models and metrics
    save_models(models, all_train_metrics, test_metrics, parent_sets, model_dir)

    elapsed = time.time() - t_start
    log.info("")
    log.info("=" * 70)
    log.info("Step C (IPW) complete in %.1f sec (%.1f min)", elapsed, elapsed / 60)
    log.info("  Models : %s", model_dir)
    log.info("  Report : %s", report_dir)
    log.info("")
    log.info("Next: run step_d_fqi_agent.py with:")
    log.info("  --structural-eq-dir models/causal_v3/structural_equations_ipw")
    log.info("  --model-dir models/fqi_v3_ipw")
    log.info("  --report-dir reports/causal_v3/step_d_ipw")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
