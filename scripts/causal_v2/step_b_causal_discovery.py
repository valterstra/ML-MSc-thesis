"""Step B: Causal discovery on daily hospital transitions.

Runs the PC algorithm (causal-learn) on a sample of transition rows to
discover the causal DAG over the selected variable shortlist.

The discovered DAG defines which variables at t causally influence which
variables at t+1. This graph structure + fitted structural equations (Step C)
form the causal simulator (SCM).

Outputs (reports/causal_v2/step_b/):
  edges.csv           — directed and undirected edges found by PC
  adjacency_matrix.csv — full (n x n) adjacency matrix
  parent_sets.csv     — for each next-day variable: list of discovered parents
  dag.png             — visual graph (nodes coloured by tier)
  edge_stability.csv  — bootstrap edge frequency (if --bootstrap enabled)

Usage:
    python scripts/causal_v2/step_b_causal_discovery.py \\
        --csv data/processed/hosp_daily_v2_transitions.csv \\
        --report-dir reports/causal_v2/step_b

    # With bootstrap stability check (slower):
    python scripts/causal_v2/step_b_causal_discovery.py \\
        --csv data/processed/hosp_daily_v2_transitions.csv \\
        --report-dir reports/causal_v2/step_b \\
        --bootstrap --n-bootstrap 20

    # Quick test on 5k sample:
    python scripts/causal_v2/step_b_causal_discovery.py \\
        --csv data/processed/hosp_daily_v2_transitions_sample5k.csv \\
        --report-dir reports/causal_v2/step_b_sample \\
        --sample-n 3000
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.causal_v2.causal_discovery import (
    ALL_CAUSAL_VARS,
    CAUSAL_NEXT_VARS,
    prepare_causal_data,
    run_pc,
    extract_edges,
    extract_adjacency_matrix,
    run_bootstrap_stability,
    visualize_dag,
    save_results,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step B: Causal discovery on daily hospital transitions."
    )
    parser.add_argument(
        "--csv", required=True,
        help="Path to hosp_daily_v2_transitions.csv",
    )
    parser.add_argument(
        "--report-dir", default="reports/causal_v2/step_b",
        help="Output directory for Step B results",
    )
    parser.add_argument(
        "--split", default="train",
        help="Data split to use (default: train)",
    )
    parser.add_argument(
        "--sample-n", type=int, default=100_000,
        help="Number of rows to sample for PC algorithm (default: 100000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="CI test significance level — lower = sparser graph (default: 0.01)",
    )
    parser.add_argument(
        "--max-cond-set", type=int, default=4,
        help="Max conditioning set size for PC (depth parameter, default: 4)",
    )
    parser.add_argument(
        "--bootstrap", action="store_true",
        help="Run bootstrap stability analysis after main PC run",
    )
    parser.add_argument(
        "--n-bootstrap", type=int, default=20,
        help="Number of bootstrap runs (default: 20, used only if --bootstrap)",
    )
    parser.add_argument(
        "--bootstrap-sample-n", type=int, default=50_000,
        help="Sample size per bootstrap run (default: 50000)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    log = logging.getLogger(__name__)

    report_dir = PROJECT_ROOT / args.report_dir

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    log.info("Loading dataset: %s", args.csv)
    import pandas as pd
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded: %d rows, %d columns", len(df), len(df.columns))

    # ------------------------------------------------------------------
    # Prepare data matrix
    # ------------------------------------------------------------------
    log.info("=== Preparing causal data matrix ===")
    X, col_names = prepare_causal_data(
        df,
        sample_n=args.sample_n,
        split=args.split,
    )
    log.info("  Variables entering PC: %d", len(col_names))
    log.info("  Variable list: %s", col_names)

    # ------------------------------------------------------------------
    # Run PC algorithm
    # ------------------------------------------------------------------
    log.info("=== Running PC algorithm ===")
    cg = run_pc(
        X, col_names,
        alpha=args.alpha,
        max_cond_set=args.max_cond_set,
    )

    # ------------------------------------------------------------------
    # Extract graph structure
    # ------------------------------------------------------------------
    log.info("=== Extracting graph structure ===")
    edge_df = extract_edges(cg, col_names)
    adj_df = extract_adjacency_matrix(cg, col_names)

    # ------------------------------------------------------------------
    # Bootstrap stability (optional)
    # ------------------------------------------------------------------
    stability_df = None
    if args.bootstrap:
        log.info("=== Bootstrap stability (%d runs) ===", args.n_bootstrap)
        stability_df = run_bootstrap_stability(
            df, col_names,
            n_bootstrap=args.n_bootstrap,
            sample_n=args.bootstrap_sample_n,
            alpha=args.alpha,
            max_cond_set=args.max_cond_set,
            split=args.split,
        )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    log.info("=== Saving results ===")
    save_results(edge_df, adj_df, report_dir, stability_df)
    visualize_dag(edge_df, report_dir)

    # ------------------------------------------------------------------
    # Print parent sets summary to terminal
    # ------------------------------------------------------------------
    log.info("=== DISCOVERED PARENT SETS (state_{t+1} <- parents) ===")
    next_vars_found = [c for c in col_names if c in CAUSAL_NEXT_VARS]
    for target in next_vars_found:
        if target not in adj_df.columns:
            continue
        parents = adj_df.index[adj_df[target] > 0].tolist()
        undirected = edge_df[
            (edge_df["edge_type"] == "undirected") &
            ((edge_df["source"] == target) | (edge_df["target"] == target))
        ]
        undirected_partners = []
        for _, row in undirected.iterrows():
            partner = row["target"] if row["source"] == target else row["source"]
            undirected_partners.append(partner)

        log.info(
            "  %-35s <- directed: [%s]  undirected: [%s]",
            target,
            ", ".join(parents) if parents else "none",
            ", ".join(undirected_partners) if undirected_partners else "none",
        )

    log.info("Step B complete. Results in: %s", report_dir)


if __name__ == "__main__":
    main()
