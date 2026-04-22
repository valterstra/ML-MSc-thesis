"""
Step 03b -- Robust random stability analysis (FCI-based).

PURPOSE
-------
Parallel track to step_03_random_stability.py.

This version keeps the same state -> readmission question, but strengthens each
FCI run by sampling more competing discharge-state variables per graph.

Compared with Step 03a (the original script):
  - Step 03a uses 2 sampled state variables per run
  - Step 03b uses 4 sampled state variables per run by default

This means a state must retain its edge to readmit_30d in the presence of more
other candidate states, which should make the ranking more robust.

APPROACH
--------
Each FCI run uses:

  Fixed nodes:
    age, charlson_score, prior_ed_visits_6m   <- confounders (tier 0)
    readmit_30d                               <- outcome     (tier 2)

  Random nodes:
    4 discharge-state variables sampled from the Step 02 ranking-derived pool
                                               <- state vars (tier 1)

  Total default graph size: 8 nodes.

STATE POOL
----------
Unlike Step 03a, which hardcodes the candidate pool, Step 03b builds the pool
from the Step 02 ranking CSV by default:

  reports/icu_readmit/step_09_state_action_selection/state_variable_ranking.csv

Default:
  top_k_states = 30
  exclude_last_states = ['last_PAPsys']

This reproduces the original candidate-pool logic while making the pool
selection explicit and configurable.

OUTPUTS
-------
All outputs are written with `_robust` suffixes so the original Step 03 outputs
stay untouched:

  reports/icu_readmit/step_09_state_action_selection/random_stability_results_robust.csv
  reports/icu_readmit/step_09_state_action_selection/random_stability_summary_robust.json
  reports/icu_readmit/step_09_state_action_selection/random_stability_state_pool_robust.json

Usage:
    python scripts/icu_readmit/step_09_state_action_selection/step_03b_random_stability_robust.py
    python scripts/icu_readmit/step_09_state_action_selection/step_03b_random_stability_robust.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import C_READMIT_30D

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


CONFOUNDER_COLS = ["age", "charlson_score", "prior_ed_visits_6m"]

TIER = {
    "age": 0,
    "charlson_score": 0,
    "prior_ed_visits_6m": 0,
    C_READMIT_30D: 2,
}


def load_ranked_state_pool(
    ranking_path: Path,
    top_k_states: int,
    exclude_last_states: set[str],
) -> list[str]:
    """Select the top-k ranked Step 02 states after explicit exclusions."""
    ranking = pd.read_csv(ranking_path)
    selected: list[str] = []

    for _, row in ranking.sort_values("rank").iterrows():
        last_name = str(row["variable"])
        if last_name in exclude_last_states:
            continue
        if not last_name.startswith("last_"):
            continue
        selected.append(last_name)
        if len(selected) >= top_k_states:
            break

    return selected


def build_background_knowledge(nodes, col_names: list[str], tier: dict):
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    bk = BackgroundKnowledge()
    node_map = {col_names[i]: nodes[i] for i in range(len(col_names))}
    for ni in col_names:
        for nj in col_names:
            if ni != nj and tier.get(ni, 1) > tier.get(nj, 1):
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


def run_fci_once(data: np.ndarray, col_names: list[str], tier: dict, alpha: float):
    """
    Run FCI with tier-based background knowledge.
    Two-pass pattern: first to get nodes, second with BK attached.
    """
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    try:
        g_init, _ = fci(
            data,
            fisherz,
            alpha,
            node_names=col_names,
            verbose=False,
            show_progress=False,
        )
        nodes = g_init.get_nodes()
        bk = build_background_knowledge(nodes, col_names, tier)
        g_final, _ = fci(
            data,
            fisherz,
            alpha,
            node_names=col_names,
            background_knowledge=bk,
            verbose=False,
            show_progress=False,
        )
        return g_final
    except Exception as exc:
        log.debug("FCI failed: %s", exc)
        return None


def extract_edges_to_readmit(g, col_names: list[str]) -> dict[str, str]:
    """
    Return edge type from each sampled state variable to readmit_30d.

    PAG encoding (causal-learn GeneralGraph):
      adj[i, j] = -1 -> arrowhead at j end
      adj[i, j] =  1 -> tail at j end
      adj[i, j] =  2 -> circle at j end
      adj[i, j] =  0 -> no edge
    """
    if g is None:
        return {}
    try:
        adj = g.graph
    except AttributeError:
        return {}

    readmit_idx = col_names.index(C_READMIT_30D)
    results: dict[str, str] = {}

    for i, col in enumerate(col_names):
        if not col.startswith("last_"):
            continue
        state_to_outcome = adj[i, readmit_idx]
        outcome_to_state = adj[readmit_idx, i]

        if state_to_outcome == -1 and outcome_to_state == 1:
            results[col] = "definite"
        elif state_to_outcome == -1 and outcome_to_state == 2:
            results[col] = "possible"
        elif state_to_outcome == -1 and outcome_to_state == -1:
            results[col] = "bidirected"

    return results


def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    input_path = Path(args.input)
    report_dir = Path(args.report_dir)
    ranking_path = Path(args.state_ranking_path)
    report_dir.mkdir(parents=True, exist_ok=True)

    exclude_last_states = set(args.exclude_last_states or [])
    requested_pool = load_ranked_state_pool(
        ranking_path=ranking_path,
        top_k_states=args.top_k_states,
        exclude_last_states=exclude_last_states,
    )
    if not requested_pool:
        raise ValueError("No states selected for the robust Step 03b track")

    log.info("Loading %s ...", input_path)
    df = pd.read_parquet(input_path)
    log.info("Loaded: %d stays, %d columns", len(df), len(df.columns))

    pool = [c for c in requested_pool if c in df.columns]
    missing_from_pool = [c for c in requested_pool if c not in df.columns]
    if missing_from_pool:
        log.warning("Requested pool variables not found in dataset (skipped): %s", missing_from_pool)
    if len(pool) < args.n_state_vars:
        raise ValueError(f"Only {len(pool)} usable state variables found, need {args.n_state_vars}")

    confounders = [c for c in CONFOUNDER_COLS if c in df.columns]
    if C_READMIT_30D not in df.columns:
        raise AssertionError(f"Missing column: {C_READMIT_30D}")

    log.info("State variable pool: %d variables", len(pool))
    log.info("Confounders: %s", confounders)
    log.info(
        "Config: n_runs=%d  n_sample=%d  n_state_vars=%d  alpha=%.3f  seed=%d",
        args.n_runs,
        args.n_sample,
        args.n_state_vars,
        args.alpha,
        args.seed,
    )

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    n_included = defaultdict(int)
    n_definite = defaultdict(int)
    n_possible = defaultdict(int)
    n_any = defaultdict(int)
    n_failed = 0

    scaler = StandardScaler()

    for run_idx in range(args.n_runs):
        sampled_state = rng.sample(pool, k=args.n_state_vars)
        col_names = confounders + sampled_state + [C_READMIT_30D]

        tier = dict(TIER)
        for sv in sampled_state:
            tier[sv] = 1

        n_draw = min(args.n_sample, len(df))
        subset = df[col_names].sample(n=n_draw, random_state=args.seed + run_idx).dropna()
        if len(subset) < 100:
            n_failed += 1
            continue

        data = scaler.fit_transform(subset.values.astype(np.float64))
        g = run_fci_once(data, col_names, tier, args.alpha)
        if g is None:
            n_failed += 1
            continue

        edges = extract_edges_to_readmit(g, col_names)
        for sv in sampled_state:
            n_included[sv] += 1
            edge_type = edges.get(sv)
            if edge_type == "definite":
                n_definite[sv] += 1
                n_any[sv] += 1
            elif edge_type == "possible":
                n_possible[sv] += 1
                n_any[sv] += 1

        if (run_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (run_idx + 1) / elapsed if elapsed > 0 else 0
            remaining = (args.n_runs - run_idx - 1) / rate if rate > 0 else 0
            log.info(
                "Run %4d / %d  |  elapsed %.0fs  |  est. remaining %.0fs  |  failed %d",
                run_idx + 1,
                args.n_runs,
                elapsed,
                remaining,
                n_failed,
            )

    log.info("Building results table...")
    rows = []
    for sv in pool:
        ni = n_included.get(sv, 0)
        nd = n_definite.get(sv, 0)
        np_ = n_possible.get(sv, 0)
        na = n_any.get(sv, 0)
        rows.append({
            "variable": sv,
            "n_runs_included": ni,
            "n_definite": nd,
            "n_possible": np_,
            "n_any": na,
            "freq_definite": round(nd / ni, 4) if ni > 0 else 0.0,
            "freq_any": round(na / ni, 4) if ni > 0 else 0.0,
        })

    results = pd.DataFrame(rows).sort_values("freq_definite", ascending=False).reset_index(drop=True)
    results.insert(0, "rank_definite", results.index + 1)

    rank_any_map = (
        results.sort_values("freq_any", ascending=False)
        .reset_index(drop=True)
        .assign(rank_any=lambda x: x.index + 1)
        .set_index("variable")["rank_any"]
    )
    results["rank_any"] = results["variable"].map(rank_any_map)

    out_path = report_dir / "random_stability_results_robust.csv"
    results.to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)

    total_time = time.time() - t0
    summary = {
        "config": {
            "n_runs": args.n_runs,
            "n_sample": args.n_sample,
            "n_state_vars": args.n_state_vars,
            "alpha": args.alpha,
            "seed": args.seed,
            "pool_size": len(pool),
            "expected_appearances_per_var": round(args.n_runs * args.n_state_vars / len(pool), 1),
        },
        "results": {
            "n_failed": n_failed,
            "n_completed": args.n_runs - n_failed,
            "runtime_s": round(total_time, 1),
        },
        "top_10_definite": results.head(10)[
            ["rank_definite", "variable", "freq_definite", "freq_any", "n_runs_included", "n_definite", "n_any"]
        ].to_dict("records"),
    }
    summary_path = report_dir / "random_stability_summary_robust.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved: %s", summary_path)

    pool_payload = {
        "state_ranking_path": str(ranking_path),
        "top_k_states_requested": args.top_k_states,
        "exclude_last_states": sorted(exclude_last_states),
        "selected_pool": pool,
    }
    pool_path = report_dir / "random_stability_state_pool_robust.json"
    with open(pool_path, "w", encoding="utf-8") as f:
        json.dump(pool_payload, f, indent=2)
    log.info("State pool manifest saved: %s", pool_path)

    log.info("=" * 76)
    log.info(
        "ROBUST RANDOM STABILITY RESULTS  (n_runs=%d  alpha=%.2f  n_sample=%d)",
        args.n_runs,
        args.alpha,
        args.n_sample,
    )
    log.info("Expected appearances per variable: ~%.0f runs", summary["config"]["expected_appearances_per_var"])
    log.info("%-5s %-35s %12s %10s %8s", "Rank", "Variable", "Freq_Definite", "Freq_Any", "N_runs")
    log.info("-" * 76)
    for _, row in results.iterrows():
        log.info(
            "%-5d %-35s %12.3f %10.3f %8d",
            row["rank_definite"],
            row["variable"],
            row["freq_definite"],
            row["freq_any"],
            row["n_runs_included"],
        )
    log.info("=" * 76)
    log.info(
        "Failed runs: %d / %d  (%.1f%%)",
        n_failed,
        args.n_runs,
        100 * n_failed / args.n_runs if args.n_runs > 0 else 0,
    )
    log.info("Total time: %.1f s", total_time)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input",
        default=str(
            PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "step_09_state_action_selection" / "stay_level.parquet"
        ),
        help="Path to stay_level.parquet (Step 01 output)",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "step_09_state_action_selection"),
        help="Directory for robust Step 03b outputs",
    )
    parser.add_argument(
        "--state-ranking-path",
        default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "step_09_state_action_selection" / "state_variable_ranking.csv"),
        help="Step 02 state ranking CSV used to build the candidate pool",
    )
    parser.add_argument(
        "--top-k-states",
        type=int,
        default=30,
        help="Number of top Step 02 states to keep before exclusions",
    )
    parser.add_argument(
        "--exclude-last-states",
        nargs="*",
        default=["last_PAPsys"],
        help="Explicit last_* states to exclude from the Step 03b pool",
    )
    parser.add_argument(
        "--n-runs",
        type=int,
        default=2000,
        help="Number of random graph iterations",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=5000,
        help="Stays to subsample per run",
    )
    parser.add_argument(
        "--n-state-vars",
        type=int,
        default=4,
        help="State variables to sample per run in the robust track",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="CI test significance level",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke test: 20 runs, 500 stays",
    )
    args = parser.parse_args()

    if args.smoke:
        args.n_runs = 20
        args.n_sample = 500

    main(args)
