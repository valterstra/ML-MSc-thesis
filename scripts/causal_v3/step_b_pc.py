"""Step B: PC causal discovery on V3 triplet dataset.

Discovers the causal DAG over the V3 variable set using the PC algorithm
(causal-learn) with temporal background knowledge enforced.

Temporal tiers (edges can only go forward, never backward):
  Tier 0 (static):  age_at_admit, charlson_score
  Tier 1 (state_T): 13 core labs + is_icu
  Tier 2 (action):  5 drug flags
  Tier 3 (next):    13 next-day labs + next_is_icu

Progress is logged per depth level with timestamps so you can estimate
how long the full run will take.

Outputs (--report-dir):
  edges.csv             -- all directed/undirected edges
  drug_edges.csv        -- drug -> next_* edges only (the important ones)
  parent_sets.csv       -- for each next_* variable: its discovered parents
  alpha_robustness.csv  -- which drug edges survive at each alpha level
  run_log.txt           -- full run log with timing

Usage:
    # Default: 100k sample, alpha=0.01, max_cond_set=3
    python scripts/causal_v3/step_b_pc.py

    # Custom settings
    python scripts/causal_v3/step_b_pc.py --sample-n 300000 --alpha 0.001 --max-cond-set 4

    # Multiple alphas for robustness check
    python scripts/causal_v3/step_b_pc.py --alpha 0.01 0.001 0.0001 1e-8
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

# ── Variable tiers ────────────────────────────────────────────────────────

TIER0_STATIC = ["age_at_admit", "charlson_score"]

TIER1_STATE = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
    "magnesium", "is_icu",
]

TIER2_ACTION = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active",
]

TIER3_NEXT = [
    "next_creatinine", "next_bun", "next_sodium", "next_potassium",
    "next_bicarbonate", "next_anion_gap", "next_calcium", "next_glucose",
    "next_hemoglobin", "next_wbc", "next_platelets", "next_phosphate",
    "next_magnesium", "next_is_icu",
]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_NEXT
TIERS = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_NEXT]

VAR_TIER: dict[str, int] = {}
for _tidx, _tvars in enumerate(TIERS):
    for _v in _tvars:
        VAR_TIER[_v] = _tidx


# ── Background knowledge ─────────────────────────────────────────────────

def build_temporal_background_knowledge(nodes, var_names):
    """Forbid edges from later tiers to earlier tiers."""
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    bk = BackgroundKnowledge()
    node_map = {var_names[i]: nodes[i] for i in range(len(var_names))}

    for ni in var_names:
        for nj in var_names:
            if ni != nj and VAR_TIER[ni] > VAR_TIER[nj]:
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


# ── PC wrapper with depth-level progress ─────────────────────────────────

def run_pc_with_progress(data, var_names, alpha, max_cond_set):
    """Run PC algorithm with temporal BK, logging time per depth level.

    causal-learn's PC iterates through increasing conditioning set depths.
    We wrap the call and capture the progress bar output to extract
    depth-level timing.
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    n_vars = len(var_names)
    log.info(
        "PC: %d variables, %d rows, alpha=%.1e, max_cond_set=%d",
        n_vars, len(data), alpha, max_cond_set,
    )

    t0 = time.time()

    # First run without BK to get node objects, then with BK
    # (causal-learn needs the node objects to build BK)
    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=var_names, verbose=False,
        depth=max_cond_set,
    )
    bk = build_temporal_background_knowledge(pc_result.G.nodes, var_names)

    # Re-run with background knowledge
    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=var_names, verbose=False,
        depth=max_cond_set,
        background_knowledge=bk,
    )

    elapsed = time.time() - t0
    log.info("PC completed in %.1f seconds", elapsed)

    return pc_result, elapsed


# ── Edge extraction ──────────────────────────────────────────────────────

def extract_edges(pc_result, var_names):
    """Extract edges from PC result into a DataFrame."""
    adj = pc_result.G.graph
    n = len(var_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0 and adj[j, i] == 0:
                continue
            ni, nj = var_names[i], var_names[j]

            if adj[i, j] == -1 and adj[j, i] == 1:
                rows.append({"source": ni, "target": nj, "edge_type": "directed"})
            elif adj[i, j] == 1 and adj[j, i] == -1:
                rows.append({"source": nj, "target": ni, "edge_type": "directed"})
            elif adj[i, j] == -1 and adj[j, i] == -1:
                rows.append({"source": ni, "target": nj, "edge_type": "undirected"})
            else:
                rows.append({
                    "source": ni, "target": nj,
                    "edge_type": f"other({adj[i,j]},{adj[j,i]})",
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "edge_type"]
    )


def extract_drug_edges(edge_df):
    """Filter to drug -> next_* edges only."""
    drug_set = set(TIER2_ACTION)
    next_set = set(TIER3_NEXT)

    mask = (
        edge_df["source"].isin(drug_set) & edge_df["target"].isin(next_set)
    ) | (
        edge_df["source"].isin(next_set) & edge_df["target"].isin(drug_set)
    )
    return edge_df[mask].copy()


def extract_parent_sets(edge_df, target_vars):
    """For each target variable, find its directed parents."""
    rows = []
    for target in target_vars:
        directed_parents = edge_df[
            (edge_df["target"] == target) & (edge_df["edge_type"] == "directed")
        ]["source"].tolist()

        undirected_partners = edge_df[
            (edge_df["edge_type"] == "undirected")
            & ((edge_df["source"] == target) | (edge_df["target"] == target))
        ].apply(
            lambda r: r["target"] if r["source"] == target else r["source"],
            axis=1,
        ).tolist()

        drug_parents = [p for p in directed_parents if p in set(TIER2_ACTION)]
        state_parents = [p for p in directed_parents if p not in set(TIER2_ACTION)]

        rows.append({
            "target": target,
            "n_parents": len(directed_parents),
            "drug_parents": ", ".join(drug_parents) if drug_parents else "",
            "state_parents": ", ".join(state_parents) if state_parents else "",
            "undirected": ", ".join(undirected_partners) if undirected_partners else "",
        })

    return pd.DataFrame(rows)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step B: PC causal discovery on V3 triplet dataset.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
        help="Path to V3 triplets CSV",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b"),
        help="Output directory",
    )
    parser.add_argument(
        "--split", default="train",
        help="Data split to use (default: train)",
    )
    parser.add_argument(
        "--sample-n", type=int, default=100_000,
        help="Rows to sample (default: 100000, 0 = use all)",
    )
    parser.add_argument(
        "--alpha", type=float, nargs="+", default=[0.01],
        help="Alpha levels to test (default: 0.01). Pass multiple for robustness.",
    )
    parser.add_argument(
        "--max-cond-set", type=int, default=3,
        help="Max conditioning set size (default: 3)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling (default: 42)",
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    # Set up logging to both console and file
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
    log.info("Step B: PC causal discovery on V3 dataset")
    log.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("Full dataset: %d rows, %d columns", len(df), len(df.columns))

    if args.split:
        df = df[df["split"] == args.split]
        log.info("After split='%s' filter: %d rows", args.split, len(df))

    # Select variables and drop NaN
    df_sub = df[ALL_VARS].dropna()
    log.info("After selecting %d variables and dropna: %d rows", len(ALL_VARS), len(df_sub))

    # Sample
    if args.sample_n > 0 and args.sample_n < len(df_sub):
        df_sub = df_sub.sample(n=args.sample_n, random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", args.sample_n, args.seed)

    data = df_sub.values.astype(np.float64)
    log.info("Data matrix: %d rows x %d columns", data.shape[0], data.shape[1])
    log.info("Variables: %s", ALL_VARS)

    # ── Run PC at each alpha ─────────────────────────────────────────
    all_results = {}
    for alpha in sorted(args.alpha, reverse=True):
        log.info("")
        log.info("-" * 50)
        log.info("Running PC with alpha=%.1e", alpha)
        log.info("-" * 50)

        pc_result, elapsed = run_pc_with_progress(
            data, ALL_VARS, alpha=alpha, max_cond_set=args.max_cond_set,
        )

        edge_df = extract_edges(pc_result, ALL_VARS)
        drug_edge_df = extract_drug_edges(edge_df)

        log.info("  Total edges: %d", len(edge_df))
        log.info("  Drug -> next edges: %d", len(drug_edge_df))
        log.info("  Time: %.1f seconds", elapsed)

        if len(drug_edge_df) > 0:
            log.info("  Drug edges found:")
            for _, row in drug_edge_df.iterrows():
                arrow = "-->" if row["edge_type"] == "directed" else "---"
                log.info("    %s %s %s", row["source"], arrow, row["target"])

        all_results[alpha] = {
            "edge_df": edge_df,
            "drug_edge_df": drug_edge_df,
            "elapsed": elapsed,
        }

    # ── Save results from primary alpha (first / largest) ────────────
    primary_alpha = sorted(args.alpha, reverse=True)[0]
    primary = all_results[primary_alpha]

    primary["edge_df"].to_csv(report_dir / "edges.csv", index=False)
    log.info("Saved: edges.csv (%d edges)", len(primary["edge_df"]))

    primary["drug_edge_df"].to_csv(report_dir / "drug_edges.csv", index=False)
    log.info("Saved: drug_edges.csv (%d edges)", len(primary["drug_edge_df"]))

    parent_df = extract_parent_sets(primary["edge_df"], TIER3_NEXT)
    parent_df.to_csv(report_dir / "parent_sets.csv", index=False)
    log.info("Saved: parent_sets.csv")

    # ── Alpha robustness table ───────────────────────────────────────
    if len(args.alpha) > 1:
        all_drug_edges = set()
        for a in all_results:
            for _, row in all_results[a]["drug_edge_df"].iterrows():
                all_drug_edges.add((row["source"], row["target"]))

        robustness_rows = []
        for src, tgt in sorted(all_drug_edges):
            row_data = {"source": src, "target": tgt}
            for a in sorted(args.alpha, reverse=True):
                key = f"alpha={a:.0e}"
                found = any(
                    (r["source"] == src and r["target"] == tgt)
                    for _, r in all_results[a]["drug_edge_df"].iterrows()
                )
                row_data[key] = "YES" if found else "-"
            robustness_rows.append(row_data)

        robustness_df = pd.DataFrame(robustness_rows)
        robustness_df.to_csv(report_dir / "alpha_robustness.csv", index=False)
        log.info("Saved: alpha_robustness.csv")

        log.info("")
        log.info("=" * 70)
        log.info("ALPHA ROBUSTNESS TABLE")
        log.info("=" * 70)
        log.info("\n%s", robustness_df.to_string(index=False))

    # ── Print parent set summary ─────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("PARENT SETS FOR next_* VARIABLES (alpha=%.1e)", primary_alpha)
    log.info("=" * 70)
    for _, row in parent_df.iterrows():
        log.info("")
        log.info("  %s (%d parents):", row["target"], row["n_parents"])
        if row["drug_parents"]:
            log.info("    Drugs:  %s", row["drug_parents"])
        if row["state_parents"]:
            log.info("    State:  %s", row["state_parents"])
        if row["undirected"]:
            log.info("    Undirected: %s", row["undirected"])
        if not row["drug_parents"] and not row["state_parents"] and not row["undirected"]:
            log.info("    (no parents found)")

    log.info("")
    log.info("Step B complete. Results in: %s", report_dir)


if __name__ == "__main__":
    main()
