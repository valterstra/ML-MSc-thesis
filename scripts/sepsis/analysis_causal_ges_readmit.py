"""
GES causal discovery on reduced sepsis readmission variable set.

GES (Greedy Equivalence Search) is a score-based algorithm that searches
over equivalence classes of DAGs using a BIC score. It operates in two phases:
  Forward (insert edges greedily to maximise BIC score)
  Backward (remove edges greedily to maximise BIC score)

Returns a CPDAG (Completed Partially Directed Acyclic Graph) representing
the Markov equivalence class of the best-scoring DAG.

Comparison to other algorithms already run:
  PC   -- constraint-based (independence tests), returns CPDAG
  FCI  -- constraint-based, handles hidden confounders, returns PAG
  NOTEARS -- continuous optimisation (L1-regularised), returns DAG
  GES  -- score-based (BIC), returns CPDAG  <-- this script

GES is independent of significance thresholds (uses BIC penalty instead),
making it a clean complement to the hypothesis-test-based PC/FCI algorithms.

9 variables in 4 temporal tiers (same across all causal scripts):
  Tier 0 (static):  age, elixhauser, re_admission
  Tier 1 (state):   BUN, Creatinine, Hb
  Tier 2 (action):  iv_input, vaso_input
  Tier 3 (outcome): readmit_30d

Population: survivors only (died_in_hosp=0), all splits combined.
Subsampled to 2000 (same as PC/FCI) for comparability.

Inputs:
  data/processed/sepsis_readmit/rl_{train,val,test}_set_original.csv

Outputs (--report-dir):
  ges_readmit_edges.csv       -- all edges
  ges_readmit_graph.txt       -- raw adjacency matrix
  ges_readmit_summary.txt     -- plain-English summary
  run_log.txt
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")

# ── Variable tiers ────────────────────────────────────────────────────────
TIER0_STATIC  = ["age", "elixhauser", "re_admission"]
TIER1_STATE   = ["BUN", "Creatinine", "Hb"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["readmit_30d"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}

# Adjacency mark constants (causal-learn GeneralGraph convention)
TAIL  =  1
ARROW = -1


def extract_edges(adj, var_names):
    """Extract edges from GES CPDAG adjacency matrix.

    GES returns a CPDAG. Mark convention (same as causal-learn GeneralGraph):
      adj[i,j] = TAIL (1),  adj[j,i] = ARROW (-1)  =>  i -> j
      adj[i,j] = ARROW (-1), adj[j,i] = TAIL (1)   =>  i <- j
      adj[i,j] = TAIL (1),  adj[j,i] = TAIL (1)    =>  i -- j (undirected)
      adj[i,j] = 0,         adj[j,i] = 0            =>  no edge

    Post-processing enforces temporal ordering (same as all other scripts).
    """
    n    = len(var_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            mi = adj[i, j]
            mj = adj[j, i]

            if mi == 0 and mj == 0:
                continue

            ti = VAR_TIER[var_names[i]]
            tj = VAR_TIER[var_names[j]]

            if mi == TAIL and mj == ARROW:
                src, dst  = var_names[i], var_names[j]
                edge_type = "directed"
            elif mi == ARROW and mj == TAIL:
                src, dst  = var_names[j], var_names[i]
                edge_type = "directed"
            elif mi == TAIL and mj == TAIL:
                src, dst  = var_names[i], var_names[j]
                edge_type = "undirected"
            elif mi == ARROW and mj == ARROW:
                src, dst  = var_names[i], var_names[j]
                edge_type = "undirected"
            else:
                src, dst  = var_names[i], var_names[j]
                edge_type = f"other({int(mi)},{int(mj)})"

            # Enforce temporal ordering
            if edge_type == "directed" and VAR_TIER[src] > VAR_TIER[dst]:
                src, dst  = dst, src
                edge_type = "directed (reversed by temporal ordering)"
            elif edge_type == "undirected" and ti != tj:
                src, dst  = (var_names[i], var_names[j]) if ti < tj \
                             else (var_names[j], var_names[i])
                edge_type = "directed (oriented by temporal ordering)"

            symbol = (f"{src} -> {dst}" if "directed" in edge_type
                      else f"{src} -- {dst}")
            rows.append({
                "var_i":     var_names[i],
                "var_j":     var_names[j],
                "src":       src,
                "dst":       dst,
                "edge_type": edge_type,
                "direction": symbol,
                "tier_src":  VAR_TIER.get(src, -1),
                "tier_dst":  VAR_TIER.get(dst, -1),
            })

    return pd.DataFrame(rows)


def plain_english_summary(edges_df, var_names, n_rows, score):
    lines = [
        "Causal Discovery Summary -- Sepsis Readmission (GES algorithm)",
        f"  Variables: {', '.join(var_names)}",
        f"  Population: {n_rows} ICU stays (survivors only, subsampled)",
        f"  BIC score: {score:.4f}",
        "  (GES uses BIC score -- no significance threshold, unlike PC/FCI)",
        "",
    ]

    directed   = edges_df[edges_df["edge_type"].str.contains("directed")]
    undirected = edges_df[edges_df["edge_type"] == "undirected"]

    lines.append(f"Directed edges ({len(directed)}):")
    if not len(directed):
        lines.append("  (none)")
    for _, row in directed.iterrows():
        lines.append(f"  {row['direction']}")

    lines.append(f"\nUndirected edges ({len(undirected)}) -- direction not identified by GES:")
    if not len(undirected):
        lines.append("  (none)")
    for _, row in undirected.iterrows():
        lines.append(f"  {row['direction']}")

    lines.append("\nEdges involving readmit_30d (the outcome):")
    outcome = edges_df[
        (edges_df["var_i"] == "readmit_30d") |
        (edges_df["var_j"] == "readmit_30d")
    ]
    if not len(outcome):
        lines.append("  (none -- readmit_30d d-separated from all variables)")
    for _, row in outcome.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    lines.append("\nEdges involving actions (iv_input, vaso_input):")
    actions = edges_df[
        edges_df["var_i"].isin(["iv_input", "vaso_input"]) |
        edges_df["var_j"].isin(["iv_input", "vaso_input"])
    ]
    if not len(actions):
        lines.append("  (none)")
    for _, row in actions.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="GES causal discovery on sepsis readmission variables"
    )
    parser.add_argument("--data-dir",   default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir", default="reports/sepsis_readmit/analysis/causal/ges")
    parser.add_argument("--log",        default="logs/analysis_causal_ges_readmit.log")
    parser.add_argument("--subsample",  type=int, default=2000,
                        help="Subsample N stays for comparability with PC/FCI.")
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
            logging.FileHandler(
                os.path.join(args.report_dir, "run_log.txt"),
                mode="w", encoding="utf-8"
            ),
        ],
    )

    # ── Load data ─────────────────────────────────────────────────────
    logging.info("Loading data from %s", args.data_dir)
    dfs = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"{args.data_dir}/rl_{split}_set_original.csv")
        last = df.groupby("icustayid").tail(1)
        dfs.append(last)
        logging.info("  %s: %d stays", split, len(last))

    df = pd.concat(dfs, ignore_index=True)
    df = df[df["died_in_hosp"] == 0].reset_index(drop=True)
    logging.info("Survivors only: %d stays (%.1f%% readmitted)",
                 len(df), 100 * df["readmit_30d"].mean())

    missing = [v for v in ALL_VARS if v not in df.columns]
    if missing:
        logging.error("Missing variables: %s", missing)
        sys.exit(1)

    data_df = df[ALL_VARS].copy()

    if args.subsample > 0 and len(data_df) > args.subsample:
        data_df = data_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        logging.info("Subsampled to %d stays", len(data_df))

    data_df = data_df.fillna(data_df.median())
    data_arr = data_df.values.astype(np.float64)
    logging.info("Data matrix: %s", data_arr.shape)

    # ── Run GES ───────────────────────────────────────────────────────
    from causallearn.search.ScoreBased.GES import ges

    logging.info("Running GES (BIC score, %d variables, %d rows)",
                 len(ALL_VARS), len(data_arr))
    t0 = time.time()
    result = ges(data_arr, score_func="local_score_BIC", node_names=ALL_VARS)
    elapsed = time.time() - t0
    logging.info("GES completed in %.1f sec. BIC score: %.4f",
                 elapsed, result["score"])

    adj = result["G"].graph
    edges_df = extract_edges(adj, ALL_VARS)
    logging.info("Edges found: %d total (%d directed, %d undirected)",
                 len(edges_df),
                 edges_df["edge_type"].str.contains("directed").sum(),
                 (edges_df["edge_type"] == "undirected").sum())

    # Save edges
    edges_path = f"{args.report_dir}/ges_readmit_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    logging.info("Edges saved: %s", edges_path)

    # Raw adjacency matrix
    adj_lines = ["CPDAG adjacency matrix (same mark convention as causal-learn):"]
    adj_lines.append("  TAIL=1, ARROW=-1, 0=no edge")
    adj_lines.append("  " + "  ".join(f"{v[:8]:>8}" for v in ALL_VARS))
    for i, vi in enumerate(ALL_VARS):
        row = f"{vi[:8]:>8}"
        for j in range(len(ALL_VARS)):
            row += f"  {int(adj[i, j]):>8}"
        adj_lines.append(row)
    graph_path = f"{args.report_dir}/ges_readmit_graph.txt"
    with open(graph_path, "w") as f:
        f.write("\n".join(adj_lines))
    logging.info("Graph saved: %s", graph_path)

    # Plain-English summary
    summary = plain_english_summary(edges_df, ALL_VARS, len(data_arr), float(result["score"]))
    summary_path = f"{args.report_dir}/ges_readmit_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logging.info("Summary saved: %s", summary_path)
    print("\n" + summary)

    logging.info("GES analysis complete.")


if __name__ == "__main__":
    main()
