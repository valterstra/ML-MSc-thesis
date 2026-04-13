"""Step B (FCI): Run FCI causal discovery on v2 transitions.

Usage:
    python scripts/causal_v2/step_b_fci.py \
        --csv data/processed/hosp_daily_v2_transitions.csv \
        --report-dir reports/causal_v2/step_b_fci \
        [--sample-n 300000]

Progress written to <report_dir>/run.log.
"""

import argparse
import logging
import sys
import time
from pathlib import Path


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("fci_run")
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
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-cond-set", type=int, default=4)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    log = setup_logging(report_dir / "run.log")

    log.info("=== FCI causal discovery ===")
    log.info("CSV: %s", args.csv)
    log.info("Report dir: %s", report_dir)
    log.info("sample_n=%d  alpha=%.3f  max_cond_set=%d",
             args.sample_n, args.alpha, args.max_cond_set)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    # Stage 1: Load
    log.info("[1/5] Loading dataset...")
    t0 = time.time()
    import pandas as pd
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1fs", len(df), time.time() - t0)

    # Stage 2: Prepare data matrix
    log.info("[2/5] Preparing causal data matrix (sample_n=%d)...", args.sample_n)
    t0 = time.time()
    from careai.causal_v2.causal_discovery import prepare_causal_data
    X, col_names = prepare_causal_data(df, sample_n=args.sample_n, split="train")
    log.info("  Data matrix ready: shape=%s, elapsed=%.1fs", X.shape, time.time() - t0)

    # Stage 3: Run FCI
    log.info("[3/5] Running FCI (this is the slow step)...")
    t0 = time.time()
    from careai.causal_v2.fci_discovery import run_fci
    G, edges = run_fci(X, col_names, alpha=args.alpha, max_cond_set=args.max_cond_set)
    elapsed = time.time() - t0
    log.info("  FCI complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

    # Stage 4: Extract edges
    log.info("[4/5] Extracting edges and parent sets...")
    from careai.causal_v2.fci_discovery import extract_edges, extract_parent_sets
    edge_df   = extract_edges(G, edges, col_names)
    parent_df = extract_parent_sets(edge_df)

    # Stage 5: Save
    log.info("[5/5] Saving results...")
    from careai.causal_v2.fci_discovery import save_results, visualize_dag
    save_results(edge_df, parent_df, report_dir)
    visualize_dag(edge_df, report_dir)

    log.info("=== DONE ===")
    log.info("Edge type breakdown:")
    for etype, cnt in edge_df["edge_type"].value_counts().items():
        log.info("  %s: %d", etype, cnt)
    log.info("Parent sets (directed edges only):")
    for _, row in parent_df.iterrows():
        log.info("  %s <- %s", row["target"], row["parents"])


if __name__ == "__main__":
    main()
