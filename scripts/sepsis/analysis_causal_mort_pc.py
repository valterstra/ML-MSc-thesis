"""
PC causal discovery on sepsis mortality variable set.

Same methodology as analysis_causal_readmit.py but targeting died_in_hosp.

9 variables in 4 temporal tiers:
  Tier 0 (static):  age, elixhauser
  Tier 1 (state):   FiO2_1, GCS, BUN     (top SHAP mortality predictors)
  Tier 2 (action):  iv_input, vaso_input
  Tier 3 (outcome): died_in_hosp

Population: ALL ICU stays (train + val + test), last timestep per stay.
No survivor filter -- died_in_hosp is the outcome so all patients are valid.
Subsampled to 2000 (Fisher-Z with n=37k detects r>0.01 as significant).

Outputs (--report-dir):
  causal_mort_edges.csv
  causal_mort_graph.txt
  causal_mort_summary.txt
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

TIER0_STATIC  = ["age", "elixhauser"]
TIER1_STATE   = ["FiO2_1", "GCS", "BUN"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["died_in_hosp"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}


def extract_edges(pc_result, var_names):
    adj = pc_result.G.graph
    n   = len(var_names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            aij = adj[i, j]
            aji = adj[j, i]
            if aij == 0 and aji == 0:
                continue
            ti = VAR_TIER[var_names[i]]
            tj = VAR_TIER[var_names[j]]
            if aij == 1 and aji == -1:
                src, dst, edge_type = var_names[i], var_names[j], "directed"
            elif aij == -1 and aji == 1:
                src, dst, edge_type = var_names[j], var_names[i], "directed"
            else:
                src, dst, edge_type = var_names[i], var_names[j], "undirected"
            if edge_type == "directed" and VAR_TIER[src] > VAR_TIER[dst]:
                src, dst = dst, src
                edge_type = "directed (reversed by temporal ordering)"
            if edge_type == "undirected" and ti != tj:
                src, dst = (var_names[i], var_names[j]) if ti < tj else (var_names[j], var_names[i])
                edge_type = "directed (oriented by temporal ordering)"
            rows.append({
                "src": src, "dst": dst,
                "edge_type": edge_type,
                "direction": f"{src} -> {dst}" if "directed" in edge_type else f"{src} -- {dst}",
                "tier_src": VAR_TIER.get(src, -1), "tier_dst": VAR_TIER.get(dst, -1),
            })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",     default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir",   default="reports/sepsis_readmit/analysis/causal_mort/pc")
    parser.add_argument("--log",          default="logs/analysis_causal_mort_pc.log")
    parser.add_argument("--alpha",        type=float, default=0.01)
    parser.add_argument("--max-cond-set", type=int,   default=3)
    parser.add_argument("--subsample",    type=int,   default=2000)
    args = parser.parse_args()

    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S",
                        handlers=[logging.FileHandler(args.log, mode="w", encoding="utf-8"),
                                  logging.StreamHandler(),
                                  logging.FileHandler(os.path.join(args.report_dir, "run_log.txt"),
                                                      mode="w", encoding="utf-8")])

    dfs = []
    for split in ["train", "val", "test"]:
        df = pd.read_csv(f"{args.data_dir}/rl_{split}_set_original.csv")
        dfs.append(df.groupby("icustayid").tail(1))
    df = pd.concat(dfs, ignore_index=True)
    logging.info("All stays, last timestep: %d (%.1f%% died in hospital)",
                 len(df), 100 * df["died_in_hosp"].mean())

    missing = [v for v in ALL_VARS if v not in df.columns]
    if missing:
        logging.error("Missing variables: %s", missing)
        sys.exit(1)

    data_df = df[ALL_VARS].copy()
    logging.info("Variable stats:")
    for v in ALL_VARS:
        logging.info("  %-15s  mean=%.3f  std=%.3f  missing=%.1f%%",
                     v, data_df[v].mean(), data_df[v].std(), data_df[v].isna().mean()*100)

    if args.subsample > 0 and len(data_df) > args.subsample:
        data_df = data_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        logging.info("Subsampled to %d", len(data_df))

    data_df = data_df.fillna(data_df.median())
    data_arr = data_df.values.astype(np.float64)

    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz
    logging.info("Running PC: alpha=%.2e, max_cond_set=%d", args.alpha, args.max_cond_set)
    t0 = time.time()
    result = pc(data_arr, alpha=args.alpha, indep_test=fisherz,
                node_names=ALL_VARS, verbose=False, depth=args.max_cond_set)
    logging.info("PC done in %.1f sec", time.time() - t0)

    edges_df = extract_edges(result, ALL_VARS)
    logging.info("Edges: %d total (%d directed, %d undirected)", len(edges_df),
                 edges_df["edge_type"].str.contains("directed").sum(),
                 (edges_df["edge_type"] == "undirected").sum())

    edges_df.to_csv(f"{args.report_dir}/causal_mort_edges.csv", index=False)

    adj = result.G.graph
    adj_lines = ["Adjacency matrix:"]
    adj_lines.append("  " + "  ".join(f"{v[:8]:>8}" for v in ALL_VARS))
    for i, vi in enumerate(ALL_VARS):
        row = f"{vi[:8]:>8}" + "".join(f"  {int(adj[i,j]):>8}" for j in range(len(ALL_VARS)))
        adj_lines.append(row)
    with open(f"{args.report_dir}/causal_mort_graph.txt", "w") as f:
        f.write("\n".join(adj_lines))

    lines = [
        "Causal Discovery -- Sepsis Mortality (PC algorithm)",
        f"  Variables: {', '.join(ALL_VARS)}",
        f"  Population: {len(df)} ICU stays (ALL, last timestep), subsampled to {len(data_arr)}",
        f"  Alpha: {args.alpha}",
        "",
        f"Directed edges ({edges_df['edge_type'].str.contains('directed').sum()}):",
    ]
    for _, r in edges_df[edges_df["edge_type"].str.contains("directed")].iterrows():
        lines.append(f"  {r['direction']}")
    lines.append(f"\nUndirected edges ({(edges_df['edge_type']=='undirected').sum()}):")
    for _, r in edges_df[edges_df["edge_type"] == "undirected"].iterrows():
        lines.append(f"  {r['direction']}")
    lines.append("\nEdges involving died_in_hosp:")
    out = edges_df[(edges_df["src"]=="died_in_hosp")|(edges_df["dst"]=="died_in_hosp")]
    lines.append("  (none)" if not len(out) else "")
    for _, r in out.iterrows():
        lines.append(f"  {r['direction']}  [{r['edge_type']}]")
    lines.append("\nEdges involving actions (iv_input, vaso_input):")
    act = edges_df[edges_df["src"].isin(["iv_input","vaso_input"])|
                   edges_df["dst"].isin(["iv_input","vaso_input"])]
    lines.append("  (none)" if not len(act) else "")
    for _, r in act.iterrows():
        lines.append(f"  {r['direction']}  [{r['edge_type']}]")

    summary = "\n".join(lines)
    with open(f"{args.report_dir}/causal_mort_summary.txt", "w") as f:
        f.write(summary)
    print("\n" + summary)
    logging.info("PC mortality analysis complete.")


if __name__ == "__main__":
    main()
