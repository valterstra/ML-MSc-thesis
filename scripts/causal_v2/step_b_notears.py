"""Step B (NOTEARS): Run NOTEARS linear causal discovery on v2 transitions.

Usage:
    python scripts/causal_v2/step_b_notears.py \
        --csv data/processed/hosp_daily_v2_transitions.csv \
        --report-dir reports/causal_v2/step_b_notears \
        [--sample-n 300000] \
        [--lambda1 0.1] \
        [--w-threshold 0.3]

Progress is written to <report_dir>/run.log — tail this file to monitor status.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("notears_run")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--report-dir", required=True)
    parser.add_argument("--sample-n", type=int, default=300_000)
    parser.add_argument("--lambda1", type=float, default=0.1)
    parser.add_argument("--w-threshold", type=float, default=0.3)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    log = setup_logging(report_dir / "run.log")

    log.info("=== NOTEARS causal discovery ===")
    log.info("CSV: %s", args.csv)
    log.info("Report dir: %s", report_dir)
    log.info("sample_n=%d  lambda1=%.3f  w_threshold=%.3f",
             args.sample_n, args.lambda1, args.w_threshold)

    # ------------------------------------------------------------------
    # Stage 1: Load data
    # ------------------------------------------------------------------
    log.info("[1/5] Loading dataset...")
    t0 = time.time()
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1fs", len(df), time.time() - t0)

    # ------------------------------------------------------------------
    # Stage 2: Prepare data matrix
    # ------------------------------------------------------------------
    log.info("[2/5] Preparing causal data matrix (sample_n=%d)...", args.sample_n)
    t0 = time.time()
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from careai.causal_v2.causal_discovery import prepare_causal_data
    X, col_names = prepare_causal_data(df, sample_n=args.sample_n, split="train")
    log.info("  Data matrix ready: shape=%s, elapsed=%.1fs", X.shape, time.time() - t0)

    # ------------------------------------------------------------------
    # Stage 3: Standardize (zero mean, unit variance per column)
    # ------------------------------------------------------------------
    log.info("[3/5] Standardizing columns (zero mean, unit variance)...")
    t0 = time.time()
    col_means = X.mean(axis=0)
    col_stds  = X.std(axis=0)
    col_stds[col_stds == 0] = 1.0   # avoid divide-by-zero on constant columns
    X_std = (X - col_means) / col_stds
    log.info("  Standardized in %.1fs", time.time() - t0)

    # ------------------------------------------------------------------
    # Stage 4: Run NOTEARS
    # ------------------------------------------------------------------
    log.info("[4/5] Running NOTEARS optimization (this is the slow step)...")
    log.info("  Parameters: lambda1=%.3f, w_threshold=%.3f, loss=l2",
             args.lambda1, args.w_threshold)
    log.info("  No internal progress available -- will log when complete.")
    t0 = time.time()

    from careai.causal_v2.notears_discovery import run_notears
    W = run_notears(
        X_std, col_names,
        lambda1=args.lambda1,
        w_threshold=args.w_threshold,
    )

    elapsed = time.time() - t0
    log.info("  NOTEARS complete in %.1fs (%.1f min)", elapsed, elapsed / 60)
    log.info("  Non-zero edges in W: %d", (W != 0).sum())

    # ------------------------------------------------------------------
    # Stage 5: Extract results and save
    # ------------------------------------------------------------------
    log.info("[5/5] Extracting edges and saving results...")
    t0 = time.time()
    from careai.causal_v2.notears_discovery import (
        extract_edges_from_W, extract_parent_sets, save_results, visualize_dag
    )

    edge_df   = extract_edges_from_W(W, col_names)
    parent_df = extract_parent_sets(W, col_names)
    save_results(edge_df, W, col_names, parent_df, report_dir)
    visualize_dag(edge_df, report_dir)

    log.info("  Saved in %.1fs", time.time() - t0)
    log.info("=== DONE ===")
    log.info("Total edges: %d", len(edge_df))
    log.info("Next-state parent sets:")
    for _, row in parent_df.iterrows():
        log.info("  %s <- %s", row["target"], row["parents"])


if __name__ == "__main__":
    main()
