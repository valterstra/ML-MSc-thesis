"""Step C (NOTEARS-augmented): Structural equations with PC + NOTEARS state parents.

Starts from strict PC-only parents, then adds every state/static variable
that NOTEARS (lambda=0.01) found as a parent of each next_* target --
regardless of weight threshold.

Rationale: PC found many next→next contemporaneous links (e.g. next_anion_gap
→ next_bicarbonate) that had to be stripped because we can't use future values
at simulation time. NOTEARS recovers the usable lag-1 form of these links
(anion_gap → next_bicarbonate, weight=0.607). Drug parents still come only
from PC to keep drug-effect attribution causally clean.

Outputs:
  models/causal_v3/structural_equations_notears/
    {next_creatinine, ...}.pkl
    manifest.json / train_metrics.csv / test_metrics.csv / feature_importances.csv
  reports/causal_v3/step_c_notears/
    run_log.txt

Usage:
    # Smoke test
    python scripts/causal_v3/step_c_structural_equations_notears.py --sample-n 5000

    # Full run
    python scripts/causal_v3/step_c_structural_equations_notears.py
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
        description="Step C (NOTEARS-augmented): strict PC + NOTEARS state parents.",
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
        "--notears-edges",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b_notears"
                    / "edges_lambda_0p0100.csv"),
        help="NOTEARS edges CSV (lambda=0.01)",
    )
    parser.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations_notears"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_c_notears"),
    )
    parser.add_argument("--sample-n", type=int, default=0,
                        help="If > 0, subsample for smoke test")
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
    log.info("Step C (NOTEARS-augmented): Strict PC + NOTEARS state parents")
    log.info("  csv:           %s", args.csv)
    log.info("  parent-sets:   %s", args.parent_sets)
    log.info("  notears-edges: %s", args.notears_edges)
    log.info("  model-dir:     %s", model_dir)
    log.info("  sample-n:      %s", args.sample_n or "ALL")
    log.info("=" * 70)

    from careai.causal_v3.structural_equations import (
        build_parent_sets_notears_augmented,
        train_all_equations, evaluate_all_equations,
        save_models, ACTION_FEATURES, STATE_VARS, STATIC_FEATURES,
    )

    # ── [1/5] Build NOTEARS-augmented parent sets ─────────────────────
    log.info("")
    log.info("[1/5] Building NOTEARS-augmented parent sets ...")
    parent_sets = build_parent_sets_notears_augmented(
        args.parent_sets, args.notears_edges,
    )
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
    log.info("[3/5] Training 14 structural equation models (NOTEARS-augmented) ...")
    t0 = time.time()
    models, train_metrics = train_all_equations(train_df, parent_sets)
    train_time = time.time() - t0
    log.info("  Training complete in %.1f sec", train_time)
    log.info("  %d models trained successfully",
             len([m for m in models.values() if m is not None]))

    # ── [4/5] Evaluate ────────────────────────────────────────────────
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
    log.info("SUMMARY (NOTEARS-AUGMENTED)")
    log.info("=" * 70)

    train_map = {m["target"]: m for m in train_metrics}
    test_map = {m["target"]: m for m in test_metrics}

    log.info("")
    log.info("%-20s %8s %8s %6s %6s %6s" %
             ("Target", "Train", "Test", "Feats", "PC", "NOTEARS"))
    log.info("-" * 65)

    # Load NOTEARS edges to recompute how many features came from each source
    import pandas as _pd
    valid_sources = set(STATE_VARS) | set(STATIC_FEATURES)
    notears_df = _pd.read_csv(args.notears_edges)
    notears_map: dict[str, set[str]] = {}
    for _, row in notears_df.iterrows():
        src, tgt = str(row["source"]), str(row["target"])
        if tgt in parent_sets and not src.startswith("next_") and src in valid_sources:
            notears_map.setdefault(tgt, set()).add(src)

    from careai.causal_v3.structural_equations import build_parent_sets
    strict_sets = build_parent_sets(args.parent_sets, strict=True)

    for target in parent_sets:
        tm = train_map.get(target, {})
        te = test_map.get(target, {})
        train_val = tm.get("train_r2", tm.get("train_auc", None))
        test_val = te.get("test_r2", te.get("test_auc", None))
        train_str = "%.4f" % train_val if train_val is not None else "N/A"
        test_str = "%.4f" % test_val if test_val is not None else "N/A"

        n_pc = len(strict_sets.get(target, []))
        n_notears = len(parent_sets[target]) - n_pc

        log.info("%-20s %8s %8s %5d %5d %5d",
                 target, train_str, test_str,
                 len(parent_sets[target]), n_pc, n_notears)

    log.info("")
    log.info("Step C (NOTEARS-augmented) complete.")
    log.info("  Models:  %s", model_dir)
    log.info("  Metrics: %s", report_dir)
    log.info("  Total training time: %.1f sec", train_time)


if __name__ == "__main__":
    main()
