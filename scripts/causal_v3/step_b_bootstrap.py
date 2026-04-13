"""Step B Bootstrap: Subsample stability test for PC causal discovery on V3.

Runs PC N times with different random seeds (independent 100k subsamples from
the train split). Measures how often each drug->next_* edge appears across runs.

Stability score = fraction of runs in which the edge was found.
  >= 4/5  : stable
  2-3/5   : borderline
  1/5     : noise

All settings (alpha, max_cond_set, sample_n) are fixed across runs so that
only sampling variability is measured, nothing else.

Outputs (--report-dir):
  stability_drug_edges.csv  -- drug x next_* stability matrix (primary result)
  stability_all_edges.csv   -- every edge from any run, with stability score
  per_run/seed_{N}.csv      -- raw edge list for each individual run
  run_log.txt

Usage:
    # Default: 5 seeds, alpha=0.01, 100k sample
    python scripts/causal_v3/step_b_bootstrap.py

    # Custom
    python scripts/causal_v3/step_b_bootstrap.py --n-seeds 10 --sample-n 150000
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# Import shared definitions and functions from the main PC script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from step_b_pc import (
    ALL_VARS,
    TIER2_ACTION,
    TIER3_NEXT,
    build_temporal_background_knowledge,
    extract_edges,
    extract_drug_edges,
)

log = logging.getLogger(__name__)


# ── Single-seed PC run ────────────────────────────────────────────────────

def run_one_seed(data_full, seed, sample_n, alpha, max_cond_set):
    """Sample data and run PC for a single seed. Returns edge DataFrame."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    rng = np.random.default_rng(seed)
    idx = rng.choice(len(data_full), size=sample_n, replace=False)
    data = data_full[idx]

    log.info("  Seed %d: sampled %d rows", seed, len(data))

    t0 = time.time()

    # First run without BK to get node objects
    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=ALL_VARS, verbose=False,
        depth=max_cond_set,
    )
    bk = build_temporal_background_knowledge(pc_result.G.nodes, ALL_VARS)

    # Re-run with background knowledge
    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=ALL_VARS, verbose=False,
        depth=max_cond_set,
        background_knowledge=bk,
    )

    elapsed = time.time() - t0
    log.info("  Seed %d: PC done in %.1f seconds", seed, elapsed)

    edge_df = extract_edges(pc_result, ALL_VARS)
    log.info("  Seed %d: %d total edges", seed, len(edge_df))

    return edge_df, elapsed


# ── Stability aggregation ─────────────────────────────────────────────────

def aggregate_stability(all_edge_dfs, n_seeds):
    """Aggregate edge DataFrames across seeds into a stability table."""
    # Count appearances and edge types per (source, target) pair
    counts = defaultdict(int)
    directed_counts = defaultdict(int)
    type_counts = defaultdict(lambda: defaultdict(int))

    for edge_df in all_edge_dfs:
        for _, row in edge_df.iterrows():
            key = (row["source"], row["target"])
            counts[key] += 1
            type_counts[key][row["edge_type"]] += 1
            if row["edge_type"] == "directed":
                directed_counts[key] += 1

    rows = []
    for (src, tgt), count in sorted(counts.items(), key=lambda x: -x[1]):
        stability = count / n_seeds
        dominant_type = max(type_counts[(src, tgt)], key=type_counts[(src, tgt)].get)
        rows.append({
            "source": src,
            "target": tgt,
            "stability": stability,
            "count": count,
            "n_seeds": n_seeds,
            "directed_count": directed_counts[(src, tgt)],
            "dominant_type": dominant_type,
        })

    return pd.DataFrame(rows)


def build_drug_matrix(stability_df, n_seeds):
    """Build drug x next_* stability matrix (fraction format)."""
    drug_set = set(TIER2_ACTION)
    next_set = set(TIER3_NEXT)

    # Filter to drug -> next edges only (either direction)
    mask = (
        (stability_df["source"].isin(drug_set) & stability_df["target"].isin(next_set)) |
        (stability_df["source"].isin(next_set) & stability_df["target"].isin(drug_set))
    )
    drug_stability = stability_df[mask].copy()

    # Normalise so source=drug, target=next_*
    def normalise(row):
        if row["source"] in drug_set:
            return row["source"], row["target"]
        return row["target"], row["source"]

    drug_stability[["drug", "outcome"]] = drug_stability.apply(
        lambda r: pd.Series(normalise(r)), axis=1
    )

    # Pivot to matrix
    matrix = drug_stability.pivot(index="drug", columns="outcome", values="count").fillna(0)
    matrix = matrix.astype(int)

    # Add fraction labels
    frac_matrix = matrix.applymap(lambda x: f"{x}/{n_seeds}")

    return frac_matrix, drug_stability


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bootstrap stability test for V3 PC causal discovery.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
        help="Path to V3 triplets CSV",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b_bootstrap"),
        help="Output directory",
    )
    parser.add_argument(
        "--split", default="train",
        help="Data split to use (default: train)",
    )
    parser.add_argument(
        "--n-seeds", type=int, default=5,
        help="Number of independent subsamples (default: 5)",
    )
    parser.add_argument(
        "--sample-n", type=int, default=100_000,
        help="Rows per subsample (default: 100000)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="Alpha for PC independence tests (default: 0.01, fixed across all runs)",
    )
    parser.add_argument(
        "--max-cond-set", type=int, default=3,
        help="Max conditioning set size (default: 3)",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    per_run_dir = report_dir / "per_run"
    per_run_dir.mkdir(exist_ok=True)

    # Logging
    log_path = report_dir / "run_log.txt"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    log.info("=" * 70)
    log.info("Step B Bootstrap: PC stability test on V3 dataset")
    log.info("  n_seeds=%d, sample_n=%d, alpha=%.2e, max_cond_set=%d",
             args.n_seeds, args.sample_n, args.alpha, args.max_cond_set)
    log.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("Full dataset: %d rows", len(df))

    if args.split:
        df = df[df["split"] == args.split]
        log.info("After split='%s': %d rows", args.split, len(df))

    df_sub = df[ALL_VARS].dropna()
    log.info("After dropna on %d variables: %d rows", len(ALL_VARS), len(df_sub))

    if args.sample_n >= len(df_sub):
        log.warning("sample_n=%d >= available rows=%d; each seed uses all rows",
                    args.sample_n, len(df_sub))

    data_full = df_sub.values.astype(np.float64)

    # ── Run PC for each seed ─────────────────────────────────────────
    all_edge_dfs = []
    total_t0 = time.time()

    for i in range(args.n_seeds):
        seed = i  # seeds: 0, 1, 2, ... n_seeds-1
        log.info("")
        log.info("-" * 50)
        log.info("Run %d/%d (seed=%d)", i + 1, args.n_seeds, seed)
        log.info("-" * 50)

        edge_df, elapsed = run_one_seed(
            data_full, seed=seed,
            sample_n=min(args.sample_n, len(data_full)),
            alpha=args.alpha,
            max_cond_set=args.max_cond_set,
        )

        # Save per-run result
        out_path = per_run_dir / f"seed_{seed}.csv"
        edge_df.to_csv(out_path, index=False)
        log.info("  Saved: per_run/seed_%d.csv (%d edges)", seed, len(edge_df))

        drug_edges = extract_drug_edges(edge_df)
        if len(drug_edges) > 0:
            log.info("  Drug edges this run (%d):", len(drug_edges))
            for _, row in drug_edges.iterrows():
                arrow = "-->" if row["edge_type"] == "directed" else "---"
                log.info("    %s %s %s", row["source"], arrow, row["target"])
        else:
            log.info("  No drug edges found this run.")

        all_edge_dfs.append(edge_df)

    total_elapsed = time.time() - total_t0
    log.info("")
    log.info("All %d runs completed in %.1f seconds (%.1f min)",
             args.n_seeds, total_elapsed, total_elapsed / 60)

    # ── Aggregate stability ──────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("STABILITY AGGREGATION")
    log.info("=" * 70)

    stability_df = aggregate_stability(all_edge_dfs, args.n_seeds)
    stability_df.to_csv(report_dir / "stability_all_edges.csv", index=False)
    log.info("Saved: stability_all_edges.csv (%d unique edges)", len(stability_df))

    frac_matrix, drug_stability = build_drug_matrix(stability_df, args.n_seeds)
    drug_stability[["drug", "outcome", "stability", "count", "directed_count", "dominant_type"]].to_csv(
        report_dir / "stability_drug_edges.csv", index=False
    )
    log.info("Saved: stability_drug_edges.csv")

    # ── Print drug edge summary ──────────────────────────────────────
    log.info("")
    log.info("DRUG -> NEXT_* STABILITY MATRIX (%d/%d = stable)", args.n_seeds, args.n_seeds)
    log.info("")

    if len(frac_matrix) > 0:
        log.info("\n%s", frac_matrix.to_string())
    else:
        log.info("  No drug -> next edges found in any run.")

    log.info("")
    log.info("DRUG EDGE DETAILS (sorted by stability):")
    log.info("")

    drug_stability_sorted = drug_stability.sort_values("count", ascending=False)
    for _, row in drug_stability_sorted.iterrows():
        stars = "*" * row["count"]
        label = "STABLE" if row["count"] >= 4 else ("BORDERLINE" if row["count"] >= 2 else "noise")
        log.info("  %s/%d  %s  %s -> %s  [%s, directed %d/%d]",
                 row["count"], args.n_seeds, stars.ljust(args.n_seeds),
                 row["drug"], row["outcome"], label,
                 row["directed_count"], row["count"])

    # ── Sanity check: tier violations ───────────────────────────────
    log.info("")
    log.info("SANITY CHECK: temporal tier violations")
    tier_map = {}
    tiers = [
        ["age_at_admit", "charlson_score"],
        ["creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
         "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
         "magnesium", "is_icu"],
        list(TIER2_ACTION),
        list(TIER3_NEXT),
    ]
    for tidx, tvars in enumerate(tiers):
        for v in tvars:
            tier_map[v] = tidx

    violations = stability_df[
        stability_df.apply(
            lambda r: tier_map.get(r["source"], -1) > tier_map.get(r["target"], -1),
            axis=1
        )
    ]
    if len(violations) == 0:
        log.info("  No tier violations found. Background knowledge is working correctly.")
    else:
        log.warning("  %d tier violations found!", len(violations))
        for _, row in violations.iterrows():
            log.warning("    %s -> %s (appeared %d/%d runs)",
                        row["source"], row["target"], row["count"], args.n_seeds)

    log.info("")
    log.info("Step B Bootstrap complete. Results in: %s", report_dir)


if __name__ == "__main__":
    main()
