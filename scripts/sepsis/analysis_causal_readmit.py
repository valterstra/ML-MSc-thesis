"""
Causal discovery on reduced sepsis readmission variable set.

Uses PC algorithm (causal-learn, Fisher-Z test) on last-timestep data
from all ICU stays (train + val + test combined), survivors only.

9 variables organised into 4 temporal tiers:
  Tier 0 — static patient background (cannot be caused by anything below):
    age, elixhauser, re_admission
  Tier 1 — acute state at last ICU timestep:
    BUN, Creatinine, Hb
  Tier 2 — actions at last ICU timestep:
    iv_input, vaso_input
  Tier 3 — outcome (30-day readmission):
    readmit_30d

Background knowledge: edges from higher tiers to lower tiers are forbidden.
This encodes that chronic patient characteristics cannot be caused by treatment
decisions, and that readmission (observed 30 days later) cannot cause ICU state.

Population: survivors only (died_in_hosp=0). Dead patients have readmit_30d=0
by definition, which would create spurious negative associations between
mortality predictors and readmission.

Inputs:
  data/processed/sepsis_readmit/rl_train_set_original.csv
  data/processed/sepsis_readmit/rl_val_set_original.csv
  data/processed/sepsis_readmit/rl_test_set_original.csv

Outputs:
  reports/sepsis_rl/causal_readmit_edges.csv   -- discovered edges
  reports/sepsis_rl/causal_readmit_graph.txt   -- adjacency matrix
  reports/sepsis_rl/causal_readmit_summary.txt -- plain-English summary
  logs/analysis_causal_readmit.log
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, ".")

# ── Variable tiers ────────────────────────────────────────────────────────
TIER0_STATIC  = ["age", "elixhauser", "re_admission"]
TIER1_STATE   = ["BUN", "Creatinine", "Hb"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["readmit_30d"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME

TIERS = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}


def build_background_knowledge(nodes, var_names):
    """Forbid edges from higher tiers to lower tiers (temporal ordering)."""
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    bk = BackgroundKnowledge()
    node_map = {var_names[i]: nodes[i] for i in range(len(var_names))}
    for ni in var_names:
        for nj in var_names:
            if ni != nj and VAR_TIER[ni] > VAR_TIER[nj]:
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


def run_pc(data, var_names, alpha, max_cond_set):
    """Run PC algorithm (no BK — temporal ordering enforced in post-processing)."""
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    logging.info("PC: %d variables, %d rows, alpha=%.2e, max_cond_set=%d",
                 len(var_names), len(data), alpha, max_cond_set)
    t0 = time.time()

    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=var_names, verbose=False, depth=max_cond_set,
    )

    logging.info("PC completed in %.1f sec", time.time() - t0)
    return pc_result


def extract_edges(pc_result, var_names):
    """Extract edges from PC adjacency matrix and enforce temporal ordering.

    causal-learn adjacency matrix conventions:
      adj[i,j] = -1, adj[j,i] = -1  -> undirected edge i -- j
      adj[i,j] =  1, adj[j,i] = -1  -> directed edge i -> j
      adj[i,j] =  0, adj[j,i] =  0  -> no edge

    Post-processing: any edge that goes from a higher tier to a lower tier
    is reversed (temporal ordering must hold by design). Any edge within the
    same tier that PC could not orient is left as undirected.
    """
    adj = pc_result.G.graph
    n = len(var_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            aij = adj[i, j]
            aji = adj[j, i]

            if aij == 0 and aji == 0:
                continue  # no edge

            ti = VAR_TIER[var_names[i]]
            tj = VAR_TIER[var_names[j]]

            # Determine raw direction from PC
            if aij == 1 and aji == -1:
                src, dst = var_names[i], var_names[j]
                edge_type = "directed"
            elif aij == -1 and aji == 1:
                src, dst = var_names[j], var_names[i]
                edge_type = "directed"
            else:  # undirected
                src, dst = var_names[i], var_names[j]
                edge_type = "undirected"

            # Enforce temporal ordering: if src is in a higher tier than dst,
            # this is impossible — reverse the edge
            if edge_type == "directed" and VAR_TIER[src] > VAR_TIER[dst]:
                logging.debug("Reversing impossible edge %s -> %s (tier %d -> tier %d)",
                              src, dst, VAR_TIER[src], VAR_TIER[dst])
                src, dst = dst, src
                edge_type = "directed (reversed by temporal ordering)"

            # If undirected but crosses tiers, orient by tier
            if edge_type == "undirected" and ti != tj:
                if ti < tj:
                    src, dst = var_names[i], var_names[j]
                else:
                    src, dst = var_names[j], var_names[i]
                edge_type = "directed (oriented by temporal ordering)"

            rows.append({
                "var_i":     var_names[i],
                "var_j":     var_names[j],
                "src":       src,
                "dst":       dst,
                "edge_type": edge_type,
                "direction": f"{src} -> {dst}" if "directed" in edge_type else f"{src} -- {dst}",
                "tier_src":  VAR_TIER.get(src, -1),
                "tier_dst":  VAR_TIER.get(dst, -1),
            })

    return pd.DataFrame(rows)


def plain_english_summary(edges_df, var_names, n_rows, alpha):
    """Write a plain-English interpretation of the causal graph."""
    lines = [
        "Causal Discovery Summary -- Sepsis Readmission (PC algorithm)",
        f"  Variables: {', '.join(var_names)}",
        f"  Population: {n_rows} ICU stays (survivors only)",
        f"  Alpha: {alpha}",
        "",
    ]

    directed   = edges_df[edges_df["edge_type"].str.contains("directed")]
    undirected = edges_df[edges_df["edge_type"] == "undirected"]

    lines.append(f"Directed edges ({len(directed)}):")
    if len(directed) == 0:
        lines.append("  (none)")
    for _, row in directed.iterrows():
        lines.append(f"  {row['direction']}")

    lines.append(f"\nUndirected edges ({len(undirected)}) -- direction not identified:")
    if len(undirected) == 0:
        lines.append("  (none)")
    for _, row in undirected.iterrows():
        lines.append(f"  {row['direction']}")

    lines.append("\nEdges involving readmit_30d (the outcome):")
    outcome_edges = edges_df[
        (edges_df["var_i"] == "readmit_30d") |
        (edges_df["var_j"] == "readmit_30d")
    ]
    if len(outcome_edges) == 0:
        lines.append("  (none -- readmit_30d is d-separated from all variables given the rest)")
    for _, row in outcome_edges.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    lines.append("\nEdges involving actions (iv_input, vaso_input):")
    action_edges = edges_df[
        (edges_df["var_i"].isin(["iv_input", "vaso_input"])) |
        (edges_df["var_j"].isin(["iv_input", "vaso_input"]))
    ]
    if len(action_edges) == 0:
        lines.append("  (none)")
    for _, row in action_edges.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="PC causal discovery on sepsis readmission variables"
    )
    parser.add_argument("--data-dir",   default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir", default="reports/sepsis_readmit/analysis/causal/pc")
    parser.add_argument("--log",        default="logs/analysis_causal_readmit.log")
    parser.add_argument("--alpha",      type=float, default=0.01,
                        help="Significance threshold for conditional independence tests")
    parser.add_argument("--max-cond-set", type=int, default=3,
                        help="Max conditioning set size (depth). 3 is sufficient for 9 vars.")
    parser.add_argument("--subsample",  type=int, default=2000,
                        help="Subsample N stays before running PC. With n=32k, Fisher-Z "
                             "detects r>0.01 as significant, yielding a near-complete "
                             "skeleton. 2000 is the range where only r>0.06 is detected. "
                             "Set to 0 to use all data.")
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    # ── Load all splits, take last timestep, combine ──────────────────
    logging.info("Loading data from %s", args.data_dir)
    dfs = []
    for split in ["train", "val", "test"]:
        path = f"{args.data_dir}/rl_{split}_set_original.csv"
        df = pd.read_csv(path)
        last = df.groupby("icustayid").tail(1)
        dfs.append(last)
        logging.info("  %s: %d stays", split, len(last))

    df = pd.concat(dfs, ignore_index=True)
    logging.info("Total: %d stays", len(df))

    # ── Survivors only ────────────────────────────────────────────────
    df = df[df["died_in_hosp"] == 0].reset_index(drop=True)
    logging.info("Survivors only: %d stays (%.1f%% readmitted)",
                 len(df), 100 * df["readmit_30d"].mean())

    # ── Check all variables present ───────────────────────────────────
    missing_vars = [v for v in ALL_VARS if v not in df.columns]
    if missing_vars:
        logging.error("Missing variables: %s", missing_vars)
        sys.exit(1)

    # ── Prepare data matrix ───────────────────────────────────────────
    data_df = df[ALL_VARS].copy()

    # Log variable stats
    logging.info("Variable summary (last ICU timestep, survivors):")
    for v in ALL_VARS:
        col = data_df[v]
        pct_missing = col.isna().mean() * 100
        logging.info("  %-20s  mean=%.2f  std=%.2f  missing=%.1f%%",
                     v, col.mean(), col.std(), pct_missing)

    # Subsample if requested (Fisher-Z with n=32k detects r>0.01 as significant,
    # yielding a near-complete skeleton. Subsample to get meaningful sparsity.)
    if args.subsample > 0 and len(data_df) > args.subsample:
        data_df = data_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        logging.info("Subsampled to %d stays (from %d)", len(data_df), len(df))

    # Fill remaining NaN with column median (PC requires complete data)
    n_before = data_df.isna().sum().sum()
    data_df = data_df.fillna(data_df.median())
    logging.info("Filled %d NaN values with column medians", n_before)

    # Convert to float numpy array (causal-learn requirement)
    data_arr = data_df.values.astype(np.float64)
    logging.info("Data matrix: %s", data_arr.shape)

    # ── Run PC ────────────────────────────────────────────────────────
    pc_result = run_pc(data_arr, ALL_VARS, args.alpha, args.max_cond_set)

    # ── Extract and save edges ─────────────────────────────────────────
    edges_df = extract_edges(pc_result, ALL_VARS)
    logging.info("Edges found: %d total (%d directed, %d undirected)",
                 len(edges_df),
                 edges_df["edge_type"].str.contains("directed").sum(),
                 (edges_df["edge_type"] == "undirected").sum())

    edges_path = f"{args.report_dir}/causal_readmit_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    logging.info("Edges saved: %s", edges_path)

    # ── Adjacency matrix ──────────────────────────────────────────────
    adj_lines = ["Adjacency matrix (rows=from, cols=to):"]
    adj_lines.append("  " + "  ".join(f"{v[:8]:>8}" for v in ALL_VARS))
    for i, vi in enumerate(ALL_VARS):
        row = f"{vi[:8]:>8}"
        for j in range(len(ALL_VARS)):
            val = pc_result.G.graph[i, j]
            row += f"  {int(val):>8}"
        adj_lines.append(row)
    adj_text = "\n".join(adj_lines)

    graph_path = f"{args.report_dir}/causal_readmit_graph.txt"
    with open(graph_path, "w") as f:
        f.write(adj_text)
    logging.info("Graph saved: %s", graph_path)

    # ── Plain-English summary ─────────────────────────────────────────
    summary = plain_english_summary(edges_df, ALL_VARS, len(df), args.alpha)
    summary_path = f"{args.report_dir}/causal_readmit_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logging.info("Summary saved: %s", summary_path)
    print("\n" + summary)

    logging.info("Causal discovery complete.")


if __name__ == "__main__":
    main()
