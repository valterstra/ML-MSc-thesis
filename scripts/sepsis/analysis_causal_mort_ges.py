"""
GES causal discovery on sepsis mortality variable set.

Same as analysis_causal_ges_readmit.py but with died_in_hosp as outcome.
All ICU stays (no survivor filter). Subsampled to 2000 for comparability.

Tier 0 (static):  age, elixhauser
Tier 1 (state):   FiO2_1, GCS, BUN
Tier 2 (action):  iv_input, vaso_input
Tier 3 (outcome): died_in_hosp

Outputs (--report-dir):
  ges_mort_edges.csv
  ges_mort_graph.txt
  ges_mort_summary.txt
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

TAIL = 1; ARROW = -1


def extract_edges(adj, var_names):
    n = len(var_names); rows = []
    for i in range(n):
        for j in range(i + 1, n):
            mi = adj[i, j]; mj = adj[j, i]
            if mi == 0 and mj == 0:
                continue
            ti = VAR_TIER[var_names[i]]; tj = VAR_TIER[var_names[j]]
            if   mi == TAIL  and mj == ARROW: src,dst,etype = var_names[i],var_names[j],"directed"
            elif mi == ARROW and mj == TAIL:  src,dst,etype = var_names[j],var_names[i],"directed"
            elif mi == TAIL  and mj == TAIL:  src,dst,etype = var_names[i],var_names[j],"undirected"
            elif mi == ARROW and mj == ARROW: src,dst,etype = var_names[i],var_names[j],"undirected"
            else: src,dst,etype = var_names[i],var_names[j],f"other({int(mi)},{int(mj)})"
            if etype == "directed" and VAR_TIER[src] > VAR_TIER[dst]:
                src,dst = dst,src; etype = "directed (reversed by temporal ordering)"
            elif etype == "undirected" and ti != tj:
                src,dst = (var_names[i],var_names[j]) if ti<tj else (var_names[j],var_names[i])
                etype = "directed (oriented by temporal ordering)"
            sym = f"{src} -> {dst}" if "directed" in etype else f"{src} -- {dst}"
            rows.append({"src": src, "dst": dst, "edge_type": etype, "direction": sym,
                         "tier_src": VAR_TIER.get(src,-1), "tier_dst": VAR_TIER.get(dst,-1)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",   default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir", default="reports/sepsis_readmit/analysis/causal_mort/ges")
    parser.add_argument("--log",        default="logs/analysis_causal_mort_ges.log")
    parser.add_argument("--subsample",  type=int, default=2000)
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
    logging.info("All stays, last timestep: %d (%.1f%% died)", len(df),
                 100 * df["died_in_hosp"].mean())

    missing = [v for v in ALL_VARS if v not in df.columns]
    if missing:
        logging.error("Missing: %s", missing); sys.exit(1)

    data_df = df[ALL_VARS].copy()
    if args.subsample > 0 and len(data_df) > args.subsample:
        data_df = data_df.sample(n=args.subsample, random_state=42).reset_index(drop=True)
        logging.info("Subsampled to %d", len(data_df))
    data_df = data_df.fillna(data_df.median())
    data_arr = data_df.values.astype(np.float64)

    from causallearn.search.ScoreBased.GES import ges
    logging.info("Running GES (BIC, %d vars, %d rows)", len(ALL_VARS), len(data_arr))
    t0 = time.time()
    result = ges(data_arr, score_func="local_score_BIC", node_names=ALL_VARS)
    logging.info("GES done in %.1f sec. BIC score: %.4f", time.time()-t0, float(result["score"]))

    adj      = result["G"].graph
    edges_df = extract_edges(adj, ALL_VARS)
    logging.info("Edges: %d (%d directed, %d undirected)", len(edges_df),
                 edges_df["edge_type"].str.contains("directed").sum(),
                 (edges_df["edge_type"]=="undirected").sum())

    edges_df.to_csv(f"{args.report_dir}/ges_mort_edges.csv", index=False)

    adj_lines = ["CPDAG adjacency (TAIL=1, ARROW=-1):"]
    adj_lines.append("  " + "  ".join(f"{v[:8]:>8}" for v in ALL_VARS))
    for i, vi in enumerate(ALL_VARS):
        adj_lines.append(f"{vi[:8]:>8}" + "".join(f"  {int(adj[i,j]):>8}" for j in range(len(ALL_VARS))))
    with open(f"{args.report_dir}/ges_mort_graph.txt", "w") as f:
        f.write("\n".join(adj_lines))

    directed = edges_df[edges_df["edge_type"].str.contains("directed")]
    undirected = edges_df[edges_df["edge_type"]=="undirected"]
    outcome  = edges_df[(edges_df["src"]=="died_in_hosp")|(edges_df["dst"]=="died_in_hosp")]
    actions  = edges_df[edges_df["src"].isin(["iv_input","vaso_input"])|
                        edges_df["dst"].isin(["iv_input","vaso_input"])]

    lines = ["Causal Discovery -- Sepsis Mortality (GES algorithm)",
             f"  Variables: {', '.join(ALL_VARS)}",
             f"  Population: {len(df)} stays (ALL), subsampled to {len(data_arr)}",
             f"  BIC score: {float(result['score']):.4f}", "",
             f"Directed edges ({len(directed)}):"]
    lines += [f"  {r['direction']}" for _, r in directed.iterrows()] or ["  (none)"]
    lines += [f"\nUndirected edges ({len(undirected)}):"]
    lines += [f"  {r['direction']}" for _, r in undirected.iterrows()] or ["  (none)"]
    lines += ["\nEdges involving died_in_hosp:"]
    lines += [f"  {r['direction']}  [{r['edge_type']}]" for _, r in outcome.iterrows()] or ["  (none)"]
    lines += ["\nEdges involving actions (iv_input, vaso_input):"]
    lines += [f"  {r['direction']}  [{r['edge_type']}]" for _, r in actions.iterrows()] or ["  (none)"]

    summary = "\n".join(lines)
    with open(f"{args.report_dir}/ges_mort_summary.txt", "w") as f:
        f.write(summary)
    print("\n" + summary)
    logging.info("GES mortality analysis complete.")


if __name__ == "__main__":
    main()
