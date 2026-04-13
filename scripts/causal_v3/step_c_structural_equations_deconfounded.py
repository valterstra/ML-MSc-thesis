"""Step C (deconfounded): NOTEARS-augmented models with wrong-direction drug edges removed.

WHY THIS SCRIPT EXISTS
======================
After training the NOTEARS-augmented structural equations and running the FQI-V3
RL agent, we validated the direction of all drug -> next_lab edges by computing
the LightGBM partial dependence (flip drug 0->1, measure mean delta on next_lab).

We found 3 edges where the LightGBM model learned the pharmacologically WRONG
direction due to confounding-by-indication in the observational data:

  insulin_active       -> next_glucose    : model says +15.5 (should be negative)
  steroid_active       -> next_platelets  : model says  -2.3 (should be positive)
  anticoagulant_active -> next_platelets  : model says  +3.0 (should be negative)

Root cause: PC causal discovery correctly identified these edges as causally real
(they survive full 35-variable conditioning). However, LightGBM is trained on
the observational conditional distribution using only the PC parent set (4-6
variables), not all 35. Unobserved confounders not in the 35-variable set
(diabetes diagnosis, HbA1c, TPN status) dominate the insulin->glucose signal.
PC cannot adjust for variables outside the dataset (causal sufficiency violation).

THE FIX
=======
We remove the offending drug from the feature set for those specific models:
  - next_glucose  : drop insulin_active
  - next_platelets: drop steroid_active AND anticoagulant_active

This gives the simulator a ZERO effect for those pairs rather than the WRONG
SIGN. A zero effect is a conservative error; a sign-flipped effect actively
corrupts RL training by rewarding pharmacologically harmful actions.

This is defended by prior pharmacological knowledge:
  - Insulin reduces glucose (textbook endocrinology)
  - Steroids raise platelets (used to treat ITP)
  - Heparin/anticoag can lower platelets (HIT syndrome)
The observational data cannot recover these directions due to indication bias.

All other 11 edges are direction-correct in the simulator and are preserved.

SCOPE
=====
Only next_glucose and next_platelets are retrained. All other 12 models are
copied directly from models/causal_v3/structural_equations_notears/ without
modification. This is a targeted patch, not a full retraining.

OUTPUTS
=======
  models/causal_v3/structural_equations_deconfounded/
    {all 14 targets}.pkl
    manifest.json  (with corrected parent_sets for the 2 affected targets)
    train_metrics.csv / test_metrics.csv / feature_importances.csv
  reports/causal_v3/step_c_deconfounded/
    run_log.txt

USAGE
=====
  # Smoke test
  python scripts/causal_v3/step_c_structural_equations_deconfounded.py --sample-n 5000

  # Full run
  python scripts/causal_v3/step_c_structural_equations_deconfounded.py
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import shutil
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger(__name__)

# Edges to remove: (drug, target) pairs where LightGBM learned wrong direction.
# Key: target model name. Value: list of drug features to drop.
DECONFOUND_REMOVALS: dict[str, list[str]] = {
    "next_glucose":   ["insulin_active"],
    "next_platelets": ["steroid_active", "anticoagulant_active"],
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step C (deconfounded): patch wrong-direction drug edges.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--source-model-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations_notears"),
        help="NOTEARS-augmented model dir to copy unchanged models from",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations_deconfounded"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_c_deconfounded"),
    )
    parser.add_argument("--sample-n", type=int, default=0,
                        help="If > 0, subsample train/test for smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir  = Path(args.model_dir)
    source_dir = Path(args.source_model_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(report_dir / "run_log.txt"), mode="w",
                                encoding="utf-8"),
        ],
    )

    log.info("=" * 70)
    log.info("Step C (deconfounded): remove wrong-direction drug->lab edges")
    log.info("  source-model-dir: %s", source_dir)
    log.info("  model-dir:        %s", model_dir)
    log.info("  sample-n:         %s", args.sample_n or "ALL")
    log.info("")
    log.info("Removing drug features with wrong observational direction:")
    for target, drugs in DECONFOUND_REMOVALS.items():
        log.info("  %s: drop %s", target, drugs)
    log.info("=" * 70)

    from careai.causal_v3.structural_equations import (
        train_equation, evaluate_equation, save_models,
    )

    # ── [1/5] Load source manifest and parent sets ─────────────────────
    log.info("")
    log.info("[1/5] Loading source manifest from NOTEARS-augmented models ...")
    with open(source_dir / "manifest.json") as f:
        source_manifest = json.load(f)

    parent_sets: dict[str, list[str]] = dict(source_manifest["parent_sets"])
    all_targets = list(parent_sets.keys())

    # Apply the deconfounding removals to parent sets
    for target, drugs_to_drop in DECONFOUND_REMOVALS.items():
        if target not in parent_sets:
            log.warning("  %s not in parent_sets -- skipping", target)
            continue
        original = parent_sets[target]
        cleaned  = [f for f in original if f not in drugs_to_drop]
        dropped  = [f for f in original if f in drugs_to_drop]
        parent_sets[target] = cleaned
        log.info("  %s: dropped %s  (%d -> %d features)",
                 target, dropped, len(original), len(cleaned))

    # ── [2/5] Copy unchanged models from source dir ────────────────────
    log.info("")
    log.info("[2/5] Copying unchanged models from source ...")
    models: dict = {}
    targets_to_retrain = set(DECONFOUND_REMOVALS.keys())

    for target in all_targets:
        if target in targets_to_retrain:
            log.info("  [%s] will be retrained (deconfounded)", target)
            continue
        src_pkl = source_dir / f"{target}.pkl"
        if not src_pkl.exists():
            log.warning("  [%s] source pkl not found -- skipping", target)
            continue
        with open(src_pkl, "rb") as f:
            models[target] = pickle.load(f)
        log.info("  [%s] copied from source", target)

    # ── [3/5] Load data ────────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Loading V3 triplet data (only need it for retrained models) ...")
    t0 = time.time()
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1f sec", len(df), time.time() - t0)

    train_df = df[df["split"] == "train"].copy()
    test_df  = df[df["split"] == "test"].copy()
    log.info("  Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    if args.sample_n > 0:
        if args.sample_n < len(train_df):
            train_df = train_df.sample(n=args.sample_n, random_state=args.seed)
            log.info("  Subsampled train to %d rows (smoke test)", args.sample_n)
        if args.sample_n < len(test_df):
            test_df = test_df.sample(n=args.sample_n, random_state=args.seed)
            log.info("  Subsampled test to %d rows (smoke test)", args.sample_n)

    # ── [4/5] Retrain the 2 affected models ───────────────────────────
    log.info("")
    log.info("[4/5] Retraining %d deconfounded models ...", len(targets_to_retrain))
    train_metrics_new: list[dict] = []

    for target in sorted(targets_to_retrain):
        parents = parent_sets[target]
        log.info("  Retraining [%s] with features: %s", target, parents)
        t0 = time.time()
        model, metrics = train_equation(train_df, target, parents)
        log.info("  [%s] train done in %.1f sec  train_metric=%.4f",
                 target, time.time() - t0,
                 metrics.get("train_r2", metrics.get("train_auc", float("nan"))))
        models[target] = model
        train_metrics_new.append(metrics)

    # ── [5/5] Evaluate all models on test split ────────────────────────
    log.info("")
    log.info("[5/5] Evaluating all models on test split ...")
    test_metrics_all: list[dict] = []
    train_metrics_all: list[dict] = []

    # Load original metrics for unchanged models
    try:
        orig_train = pd.read_csv(source_dir / "train_metrics.csv")
        orig_test  = pd.read_csv(source_dir / "test_metrics.csv")
        for _, row in orig_train.iterrows():
            if row["target"] not in targets_to_retrain:
                train_metrics_all.append(row.to_dict())
        for _, row in orig_test.iterrows():
            if row["target"] not in targets_to_retrain:
                test_metrics_all.append(row.to_dict())
    except FileNotFoundError:
        log.warning("  Could not load source metrics CSVs -- will only have retrained metrics")

    for target in sorted(targets_to_retrain):
        parents = parent_sets[target]
        model   = models[target]
        tm = next((m for m in train_metrics_new if m["target"] == target), {})
        train_metrics_all.append(tm)

        te = evaluate_equation(model, test_df, target, parents)
        test_metrics_all.append(te)
        metric_key = "test_auc" if "test_auc" in te else "test_r2"
        log.info("  [%s] %s=%.4f", target, metric_key, te.get(metric_key, float("nan")))

    # Save everything
    log.info("")
    log.info("Saving models and metrics to %s ...", model_dir)
    save_models(models, train_metrics_all, test_metrics_all, parent_sets, model_dir)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY (DECONFOUNDED)")
    log.info("=" * 70)
    log.info("")
    log.info("Unchanged models (copied): %d", len(all_targets) - len(targets_to_retrain))
    log.info("Retrained models:          %d", len(targets_to_retrain))
    log.info("")
    log.info("Parent set changes:")
    for target, drugs in DECONFOUND_REMOVALS.items():
        orig = source_manifest["parent_sets"].get(target, [])
        new  = parent_sets.get(target, [])
        log.info("  %s: %s -> %s", target, orig, new)
    log.info("")

    # Validate direction after fix
    log.info("Validating directions on test sample (1000 rows) ...")
    import numpy as np
    base_df = test_df.head(1000)

    checks = [
        ("insulin_active",       "next_glucose",   -1, "should be negative (lowers glucose)"),
        ("steroid_active",       "next_platelets", +1, "should be positive (raises platelets)"),
        ("anticoagulant_active", "next_platelets", -1, "should be negative (HIT)"),
    ]
    log.info("")
    log.info("%-26s %-20s %10s  %s", "Drug", "Target", "Delta", "Status")
    log.info("-" * 80)
    for drug, target, expected_sign, note in checks:
        feats = parent_sets[target]
        model = models[target]
        if drug not in feats:
            log.info("%-26s %-20s %10s  REMOVED (zero effect in simulator)", drug, target, "0.000")
            continue
        b = base_df[feats].dropna().head(500)
        if len(b) == 0:
            log.info("%-26s %-20s %10s  NO DATA", drug, target, "n/a")
            continue
        b0 = b.copy(); b0[drug] = 0
        b1 = b.copy(); b1[drug] = 1
        delta = model.predict(b1) - model.predict(b0)
        sign_ok = (expected_sign > 0 and delta.mean() > 0) or \
                  (expected_sign < 0 and delta.mean() < 0)
        status = "CORRECT" if sign_ok else "STILL WRONG"
        log.info("%-26s %-20s %+10.3f  %s -- %s", drug, target, delta.mean(), status, note)

    log.info("")
    log.info("Step C (deconfounded) complete.")
    log.info("  Models:  %s", model_dir)
    log.info("  Use --structural-eq-dir models/causal_v3/structural_equations_deconfounded")
    log.info("  when running step_d_fqi_agent.py")


if __name__ == "__main__":
    main()
