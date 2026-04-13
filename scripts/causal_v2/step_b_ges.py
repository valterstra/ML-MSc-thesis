"""Step B (GES): Run GES causal discovery, then fit structural equations.

Runs GES on 300k transition rows, saves parent sets, then immediately
trains structural equations (Step C) on the discovered structure.

Usage:
    python scripts/causal_v2/step_b_ges.py \
        --csv data/processed/hosp_daily_v2_transitions.csv \
        --report-dir reports/causal_v2/step_b_ges \
        --model-dir models/causal_v2/structural_equations_ges

Progress written to <report_dir>/run.log.
"""

import argparse
import logging
import sys
import time
from pathlib import Path


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("ges_run")
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
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--sample-n", type=int, default=300_000)
    parser.add_argument("--max-parents", type=int, default=8)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    model_dir  = Path(args.model_dir)
    log = setup_logging(report_dir / "run.log")

    log.info("=== GES causal discovery + structural equations ===")
    log.info("CSV: %s  sample_n=%d  max_parents=%d",
             args.csv, args.sample_n, args.max_parents)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

    # ------------------------------------------------------------------ #
    # PART 1: GES causal discovery                                        #
    # ------------------------------------------------------------------ #

    log.info("[1/6] Loading dataset...")
    t0 = time.time()
    import pandas as pd
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1fs", len(df), time.time() - t0)

    log.info("[2/6] Preparing data matrix (sample_n=%d)...", args.sample_n)
    t0 = time.time()
    from careai.causal_v2.causal_discovery import prepare_causal_data
    X, col_names = prepare_causal_data(df, sample_n=args.sample_n, split="train")
    log.info("  Shape=%s in %.1fs", X.shape, time.time() - t0)

    log.info("[3/6] Running GES...")
    t0 = time.time()
    from careai.causal_v2.ges_discovery import run_ges
    cg = run_ges(X, col_names, max_parents=args.max_parents)
    elapsed = time.time() - t0
    log.info("  GES complete in %.1fs (%.1f min)", elapsed, elapsed / 60)

    log.info("[4/6] Extracting edges and saving discovery results...")
    from careai.causal_v2.causal_discovery import extract_edges, extract_adjacency_matrix
    from careai.causal_v2.ges_discovery import (
        extract_parent_sets_from_adj, save_results, visualize_dag
    )
    edge_df   = extract_edges(cg, col_names)
    adj_df    = extract_adjacency_matrix(cg, col_names)
    parent_df = extract_parent_sets_from_adj(adj_df)
    save_results(edge_df, adj_df, parent_df, report_dir)
    visualize_dag(edge_df, report_dir)

    log.info("GES parent sets:")
    for _, row in parent_df.iterrows():
        log.info("  %s <- %s", row["target"], row["parents"])

    # ------------------------------------------------------------------ #
    # PART 2: Structural equations (Step C) on GES parent sets            #
    # ------------------------------------------------------------------ #

    log.info("[5/6] Preparing training data for structural equations...")
    t0 = time.time()
    from careai.causal_v2.structural_equations import (
        load_parent_sets, prepare_training_data,
        train_all_equations, save_models,
    )
    parent_sets = load_parent_sets(report_dir / "parent_sets.csv")
    train_df    = prepare_training_data(df, split="train")
    log.info("  Ready in %.1fs", time.time() - t0)

    log.info("[6/6] Training structural equations...")
    t0 = time.time()
    models, metrics = train_all_equations(train_df, parent_sets)
    log.info("  Training complete in %.1fs", time.time() - t0)
    save_models(models, metrics, parent_sets, model_dir)

    log.info("=== DONE ===")
    log.info("Structural equation metrics:")
    for m in metrics:
        r2 = m.get("train_r2", m.get("train_auc", "N/A"))
        log.info("  %s: %.3f  parents=%s", m["target"],
                 float(r2) if r2 != "N/A" else 0, m.get("parents", []))


if __name__ == "__main__":
    main()
