"""Step C: Train structural equation models for V3 causal simulator.

Trains 14 LightGBM models (one per next-state variable) using causally-
informed parent sets. Each model uses:
  - Own lag (always)
  - All 5 drug flags (always)
  - Static features: age_at_admit, charlson_score (always)
  - PC-discovered state parents (from parent_sets.csv, next_* stripped)

Outputs:
  models/causal_v3/structural_equations/
    {next_creatinine, next_bun, ...}.pkl   -- 14 LightGBM models
    manifest.json                          -- parent sets + metadata
    train_metrics.csv                      -- train R2/AUC per model
    test_metrics.csv                       -- test R2/AUC per model
    feature_importances.csv                -- all feature importances
  reports/causal_v3/step_c/
    run_log.txt

Usage:
    # Smoke test (5k rows)
    python scripts/causal_v3/step_c_structural_equations.py --sample-n 5000

    # Full run
    python scripts/causal_v3/step_c_structural_equations.py
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step C: Train V3 structural equation models.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--parent-sets",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b" / "parent_sets.csv"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_c"),
    )
    parser.add_argument("--sample-n", type=int, default=0,
                        help="If > 0, subsample this many rows for smoke test")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

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
    log.info("Step C: Structural Equations for V3 Causal Simulator")
    log.info("  csv:         %s", args.csv)
    log.info("  parent-sets: %s", args.parent_sets)
    log.info("  model-dir:   %s", model_dir)
    log.info("  sample-n:    %s", args.sample_n or "ALL")
    log.info("=" * 70)

    from careai.causal_v3.structural_equations import (
        build_parent_sets, train_all_equations, evaluate_all_equations,
        save_models, ACTION_FEATURES,
    )

    # ── [1/5] Build parent sets ───────────────────────────────────────
    log.info("")
    log.info("[1/5] Building augmented parent sets from PC discovery ...")
    parent_sets = build_parent_sets(args.parent_sets)
    log.info("  %d targets, feature counts: %s",
             len(parent_sets),
             ", ".join("%d" % len(v) for v in parent_sets.values()))

    # ── [2/5] Load data ───────────────────────────────────────────────
    log.info("")
    log.info("[2/5] Loading V3 triplet data ...")
    t0 = time.time()
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1f sec", len(df), time.time() - t0)

    train_df = df[df["split"] == "train"].copy()
    test_df = df[df["split"] == "test"].copy()
    log.info("  Train: %d rows, Test: %d rows", len(train_df), len(test_df))

    if args.sample_n > 0:
        if args.sample_n < len(train_df):
            train_df = train_df.sample(n=args.sample_n, random_state=args.seed)
            log.info("  Subsampled train to %d rows (smoke test)", args.sample_n)
        if args.sample_n < len(test_df):
            test_df = test_df.sample(n=args.sample_n, random_state=args.seed)
            log.info("  Subsampled test to %d rows (smoke test)", args.sample_n)

    # ── [3/5] Train models ────────────────────────────────────────────
    log.info("")
    log.info("[3/5] Training 14 structural equation models ...")
    t0 = time.time()
    models, train_metrics = train_all_equations(train_df, parent_sets)
    train_time = time.time() - t0
    log.info("  Training complete in %.1f sec", train_time)

    n_trained = len([m for m in models.values() if m is not None])
    log.info("  %d models trained successfully", n_trained)

    # ── [4/5] Evaluate on test split ──────────────────────────────────
    log.info("")
    log.info("[4/5] Evaluating on test split ...")
    t0 = time.time()
    test_metrics = evaluate_all_equations(models, test_df, parent_sets)
    log.info("  Evaluation complete in %.1f sec", time.time() - t0)

    # ── [5/5] Save ────────────────────────────────────────────────────
    log.info("")
    log.info("[5/5] Saving models and metrics ...")
    save_models(models, train_metrics, test_metrics, parent_sets, model_dir)

    # ── Summary ───────────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)

    # Merge train and test metrics
    train_map = {m["target"]: m for m in train_metrics}
    test_map = {m["target"]: m for m in test_metrics}

    log.info("")
    log.info("%-20s %8s %8s %8s %6s" % ("Target", "Train", "Test", "Drug%", "Feats"))
    log.info("-" * 60)

    for target in parent_sets:
        tm = train_map.get(target, {})
        te = test_map.get(target, {})

        train_val = tm.get("train_r2", tm.get("train_auc", None))
        test_val = te.get("test_r2", te.get("test_auc", None))
        drug_pct = tm.get("drug_importance_pct", 0)
        n_feat = tm.get("n_features", 0)

        train_str = "%.4f" % train_val if train_val is not None else "N/A"
        test_str = "%.4f" % test_val if test_val is not None else "N/A"

        log.info("%-20s %8s %8s %7.1f%% %5d",
                 target, train_str, test_str, drug_pct, n_feat)

    # Drug effect summary
    log.info("")
    log.info("DRUG FEATURE IMPORTANCE (which drugs matter in which models):")
    log.info("")
    for target in parent_sets:
        tm = train_map.get(target, {})
        imps = tm.get("feature_importances", {})
        if not imps:
            continue
        total = sum(imps.values())
        if total == 0:
            continue
        drug_imps = []
        for d in ACTION_FEATURES:
            imp = imps.get(d, 0)
            pct = imp / total * 100
            if pct >= 1.0:  # only show drugs with >= 1% importance
                drug_imps.append((d.replace("_active", ""), pct))
        if drug_imps:
            parts = ", ".join("%s(%.1f%%)" % (d, p) for d, p in drug_imps)
            log.info("  %s: %s", target, parts)
        else:
            log.info("  %s: no drug with >= 1%% importance", target)

    log.info("")
    log.info("Step C complete. Models: %s | Metrics: %s", model_dir, report_dir)
    log.info("Total training time: %.1f sec", train_time)


if __name__ == "__main__":
    main()
