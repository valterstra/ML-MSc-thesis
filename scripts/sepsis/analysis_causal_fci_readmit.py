"""
FCI causal discovery on reduced sepsis readmission variable set.

FCI (Fast Causal Inference) extends PC to handle hidden confounders.
Returns a PAG (Partial Ancestral Graph) with three edge mark types:
  TAIL   (1)  -- definite tail end (->  or  --)
  ARROW  (-1) -- definite arrowhead
  CIRCLE (2)  -- uncertain mark (could be tail or arrowhead)

Edge types in a PAG:
  i -> j   : graph[i,j]=1,  graph[j,i]=-1  -- directed (no hidden confounder on this path)
  i <- j   : graph[i,j]=-1, graph[j,i]=1   -- directed
  i <-> j  : graph[i,j]=-1, graph[j,i]=-1  -- bidirected (hidden common cause)
  i o-> j  : graph[i,j]=2,  graph[j,i]=-1  -- partially directed
  i <-o j  : graph[i,j]=-1, graph[j,i]=2   -- partially directed
  i o-o j  : graph[i,j]=2,  graph[j,i]=2   -- non-directed (ambiguous)
  i --- j  : graph[i,j]=1,  graph[j,i]=1   -- undirected

Bidirected edges (<->) are the key FCI contribution: they indicate a latent
common cause between i and j not captured by the 9 observed variables.

Temporal ordering enforced post-hoc:
  Any edge from a higher tier to a lower tier is re-oriented.
  Bidirected edges between different tiers are flagged (they imply a hidden confounder
  that straddles the temporal boundary, which is clinically possible but unusual).

9 variables in 4 temporal tiers (same as PC and NOTEARS scripts):
  Tier 0 (static):  age, elixhauser, re_admission
  Tier 1 (state):   BUN, Creatinine, Hb
  Tier 2 (action):  iv_input, vaso_input
  Tier 3 (outcome): readmit_30d

Population: survivors only (died_in_hosp=0), all splits combined.
Subsampled to 2000 (same as PC) -- Fisher-Z with n=32k detects r>0.01.

Inputs:
  data/processed/sepsis_readmit/rl_{train,val,test}_set_original.csv

Outputs (--report-dir):
  fci_readmit_edges.csv       -- all edges with mark types
  fci_readmit_graph.txt       -- raw adjacency matrix
  fci_readmit_summary.txt     -- plain-English summary
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

# ── Variable tiers (identical to PC and NOTEARS scripts) ──────────────────
TIER0_STATIC  = ["age", "elixhauser", "re_admission"]
TIER1_STATE   = ["BUN", "Creatinine", "Hb"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["readmit_30d"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}

# PAG mark constants (causal-learn convention)
TAIL   =  1
ARROW  = -1
CIRCLE =  2


def decode_mark(m):
    """Human-readable name for a PAG endpoint mark."""
    return {TAIL: "TAIL", ARROW: "ARROW", CIRCLE: "CIRCLE"}.get(int(m), f"?{m}")


def extract_edges(pag, var_names):
    """Extract edges from FCI PAG adjacency matrix.

    causal-learn PAG adjacency encoding (graph[i,j] = mark at i's end):
      TAIL  (1)  at i, ARROW (-1) at j  =>  i -> j
      ARROW (-1) at i, TAIL  (1)  at j  =>  i <- j
      ARROW (-1) at i, ARROW (-1) at j  =>  i <-> j  (latent confounder)
      CIRCLE(2)  at i, ARROW (-1) at j  =>  i o-> j
      ARROW (-1) at i, CIRCLE(2)  at j  =>  i <-o j
      CIRCLE(2)  at i, CIRCLE(2)  at j  =>  i o-o j
      TAIL  (1)  at i, TAIL  (1)  at j  =>  i --- j

    Post-processing enforces temporal ordering:
      - Directed edges from higher to lower tier are reversed.
      - Cross-tier non-directed/circle edges are oriented by tier.
      - Bidirected edges across tiers are kept but flagged.
    """
    adj = pag.graph
    n   = len(var_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            mi = adj[i, j]   # mark at i's end
            mj = adj[j, i]   # mark at j's end

            if mi == 0 and mj == 0:
                continue  # no edge

            ti = VAR_TIER[var_names[i]]
            tj = VAR_TIER[var_names[j]]

            # Determine raw edge type from PAG marks
            if mi == TAIL and mj == ARROW:
                src, dst = var_names[i], var_names[j]
                edge_type = "directed"
                symbol    = f"{src} -> {dst}"
            elif mi == ARROW and mj == TAIL:
                src, dst = var_names[j], var_names[i]
                edge_type = "directed"
                symbol    = f"{src} -> {dst}"
            elif mi == ARROW and mj == ARROW:
                src, dst = var_names[i], var_names[j]
                edge_type = "bidirected"
                symbol    = f"{src} <-> {dst}"
            elif mi == CIRCLE and mj == ARROW:
                src, dst = var_names[i], var_names[j]
                edge_type = "circle_arrow"
                symbol    = f"{src} o-> {dst}"
            elif mi == ARROW and mj == CIRCLE:
                src, dst = var_names[j], var_names[i]
                edge_type = "circle_arrow"
                symbol    = f"{src} o-> {dst}"
            elif mi == CIRCLE and mj == CIRCLE:
                src, dst = var_names[i], var_names[j]
                edge_type = "circle_circle"
                symbol    = f"{src} o-o {dst}"
            elif mi == TAIL and mj == TAIL:
                src, dst = var_names[i], var_names[j]
                edge_type = "undirected"
                symbol    = f"{src} --- {dst}"
            else:
                src, dst = var_names[i], var_names[j]
                edge_type = f"unknown({decode_mark(mi)},{decode_mark(mj)})"
                symbol    = f"{src} ?-? {dst}"

            # ── Enforce temporal ordering ──────────────────────────────
            if edge_type == "directed":
                if VAR_TIER[src] > VAR_TIER[dst]:
                    logging.debug("Reversing impossible directed edge %s -> %s "
                                  "(tier %d -> %d)", src, dst,
                                  VAR_TIER[src], VAR_TIER[dst])
                    src, dst  = dst, src
                    edge_type = "directed (reversed by temporal ordering)"
                    symbol    = f"{src} -> {dst}"

            elif edge_type in ("circle_arrow", "circle_circle", "undirected"):
                # Orient cross-tier ambiguous edges by tier
                if ti != tj:
                    if ti < tj:
                        src, dst = var_names[i], var_names[j]
                    else:
                        src, dst = var_names[j], var_names[i]
                    edge_type = "directed (oriented by temporal ordering)"
                    symbol    = f"{src} -> {dst}"

            elif edge_type == "bidirected":
                if ti != tj:
                    edge_type = "bidirected (cross-tier hidden confounder)"
                    symbol    = f"{var_names[i]} <-> {var_names[j]}"

            rows.append({
                "var_i":     var_names[i],
                "var_j":     var_names[j],
                "src":       src,
                "dst":       dst,
                "edge_type": edge_type,
                "direction": symbol,
                "tier_i":    ti,
                "tier_j":    tj,
                "mark_i":    decode_mark(mi),
                "mark_j":    decode_mark(mj),
            })

    return pd.DataFrame(rows)


def plain_english_summary(edges_df, var_names, n_rows, alpha):
    lines = [
        "Causal Discovery Summary -- Sepsis Readmission (FCI algorithm)",
        f"  Variables: {', '.join(var_names)}",
        f"  Population: {n_rows} ICU stays (survivors only, subsampled for FCI)",
        f"  Alpha: {alpha}",
        "",
        "FCI returns a PAG. Edge types:",
        "  ->  : directed (no latent confounder on this edge)",
        "  <-> : bidirected (latent common cause)",
        "  o-> : partially directed (direction uncertain)",
        "  o-o : non-directed (fully ambiguous)",
        "  --- : undirected",
        "",
    ]

    # Group by edge type
    directed   = edges_df[edges_df["edge_type"].str.startswith("directed")]
    bidir      = edges_df[edges_df["edge_type"].str.startswith("bidirected")]
    partial    = edges_df[edges_df["edge_type"].str.startswith("circle")]
    undirected = edges_df[edges_df["edge_type"].isin(["undirected", "circle_circle"])]

    lines.append(f"Directed edges ({len(directed)}):")
    lines.append("  (no latent confounder on this causal path)")
    if len(directed) == 0:
        lines.append("  (none)")
    for _, row in directed.iterrows():
        lines.append(f"  {row['direction']}")

    lines.append(f"\nBidirected edges ({len(bidir)}):")
    lines.append("  (<-> = latent common cause not among the 9 variables)")
    if len(bidir) == 0:
        lines.append("  (none)")
    for _, row in bidir.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    lines.append(f"\nPartially directed / ambiguous edges ({len(partial) + len(undirected)}):")
    if len(partial) + len(undirected) == 0:
        lines.append("  (none)")
    for _, row in pd.concat([partial, undirected]).iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    lines.append("\nEdges involving readmit_30d (the outcome):")
    outcome = edges_df[
        (edges_df["var_i"] == "readmit_30d") |
        (edges_df["var_j"] == "readmit_30d")
    ]
    if len(outcome) == 0:
        lines.append("  (none -- readmit_30d is d-separated from all variables)")
    for _, row in outcome.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    lines.append("\nEdges involving actions (iv_input, vaso_input):")
    actions = edges_df[
        edges_df["var_i"].isin(["iv_input", "vaso_input"]) |
        edges_df["var_j"].isin(["iv_input", "vaso_input"])
    ]
    if len(actions) == 0:
        lines.append("  (none)")
    for _, row in actions.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    lines.append("\nBidirected edges (latent confounders):")
    bidir_all = edges_df[edges_df["edge_type"].str.startswith("bidirected")]
    if len(bidir_all) == 0:
        lines.append("  (none -- no evidence of hidden common causes among 9 variables)")
    for _, row in bidir_all.iterrows():
        lines.append(f"  {row['direction']}  [{row['edge_type']}]")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="FCI causal discovery on sepsis readmission variables"
    )
    parser.add_argument("--data-dir",    default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir",  default="reports/sepsis_readmit/analysis/causal/fci")
    parser.add_argument("--log",         default="logs/analysis_causal_fci_readmit.log")
    parser.add_argument("--alpha",       type=float, default=0.01,
                        help="Significance threshold for conditional independence tests")
    parser.add_argument("--max-cond-set", type=int, default=3,
                        help="Max conditioning set size (depth). 3 is sufficient for 9 vars.")
    parser.add_argument("--subsample",   type=int, default=2000,
                        help="Subsample N stays before running FCI. Same rationale as PC: "
                             "Fisher-Z with n=32k detects r>0.01 as significant.")
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

    # ── Load all splits, last timestep, survivors only ────────────────
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

    # Log variable stats
    logging.info("Variable summary (last ICU timestep, survivors):")
    for v in ALL_VARS:
        col = data_df[v]
        logging.info("  %-20s  mean=%.2f  std=%.2f  missing=%.1f%%",
                     v, col.mean(), col.std(), col.isna().mean() * 100)

    # Subsample (same rationale as PC)
    if args.subsample > 0 and len(data_df) > args.subsample:
        data_df = data_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        logging.info("Subsampled to %d stays (from %d)", len(data_df), len(df))

    # Fill NaN with column median
    n_nan = data_df.isna().sum().sum()
    data_df = data_df.fillna(data_df.median())
    logging.info("Filled %d NaN values with column medians", n_nan)

    data_arr = data_df.values.astype(np.float64)
    logging.info("Data matrix: %s", data_arr.shape)

    # ── Run FCI ───────────────────────────────────────────────────────
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    logging.info("Running FCI: %d variables, %d rows, alpha=%.2e, max_cond_set=%d",
                 len(ALL_VARS), len(data_arr), args.alpha, args.max_cond_set)
    t0 = time.time()

    pag, edges = fci(
        data_arr,
        independence_test_method=fisherz,
        alpha=args.alpha,
        depth=args.max_cond_set,
        node_names=ALL_VARS,
        verbose=False,
    )

    elapsed = time.time() - t0
    logging.info("FCI completed in %.1f sec", elapsed)

    # ── Extract and save edges ─────────────────────────────────────────
    edges_df = extract_edges(pag, ALL_VARS)
    logging.info("Edges found: %d total", len(edges_df))
    logging.info("  Directed:   %d", edges_df["edge_type"].str.startswith("directed").sum())
    logging.info("  Bidirected: %d", edges_df["edge_type"].str.startswith("bidirected").sum())
    logging.info("  Ambiguous:  %d",
                 (~edges_df["edge_type"].str.startswith("directed") &
                  ~edges_df["edge_type"].str.startswith("bidirected")).sum())

    edges_path = f"{args.report_dir}/fci_readmit_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    logging.info("Edges saved: %s", edges_path)

    # ── Raw adjacency matrix ───────────────────────────────────────────
    adj = pag.graph
    adj_lines = ["PAG adjacency matrix (graph[i,j] = mark at i's end):"]
    adj_lines.append("  marks: TAIL=1, ARROW=-1, CIRCLE=2, 0=no edge")
    adj_lines.append("  " + "  ".join(f"{v[:8]:>8}" for v in ALL_VARS))
    for i, vi in enumerate(ALL_VARS):
        row = f"{vi[:8]:>8}"
        for j in range(len(ALL_VARS)):
            row += f"  {int(adj[i, j]):>8}"
        adj_lines.append(row)
    adj_text = "\n".join(adj_lines)

    graph_path = f"{args.report_dir}/fci_readmit_graph.txt"
    with open(graph_path, "w") as f:
        f.write(adj_text)
    logging.info("Graph saved: %s", graph_path)

    # ── Plain-English summary ─────────────────────────────────────────
    summary = plain_english_summary(edges_df, ALL_VARS, len(data_arr), args.alpha)
    summary_path = f"{args.report_dir}/fci_readmit_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logging.info("Summary saved: %s", summary_path)
    print("\n" + summary)

    logging.info("FCI analysis complete.")


if __name__ == "__main__":
    main()
