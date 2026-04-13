"""
NOTEARS causal discovery on sepsis mortality variable set.

Lambda sweep [0.05, 0.01, 0.005]. No subsampling (NOTEARS is optimisation).
All ICU stays, last timestep.

Tier 0 (static):  age, elixhauser
Tier 1 (state):   FiO2_1, GCS, BUN
Tier 2 (action):  iv_input, vaso_input
Tier 3 (outcome): died_in_hosp

Outputs (--report-dir):
  edges_lambda_{X}.csv
  summary_lambda_{X}.txt
  lambda_sweep.csv
  run_log.txt
"""
import argparse
import logging
import os
import sys
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

TIER0_STATIC  = ["age", "elixhauser"]
TIER1_STATE   = ["FiO2_1", "GCS", "BUN"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["died_in_hosp"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}


def build_temporal_mask(var_names):
    n    = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if VAR_TIER[vj] >= VAR_TIER[vi]:
                mask[i, j] = 0.0
    return mask


def extract_edges(W, var_names, threshold):
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= threshold:
                rows.append({"source": vj, "target": vi,
                             "direction": f"{vj} -> {vi}",
                             "weight": round(float(w), 6),
                             "abs_weight": round(abs(float(w)), 6),
                             "tier_src": VAR_TIER[vj], "tier_dst": VAR_TIER[vi]})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source","target","direction","weight","abs_weight","tier_src","tier_dst"])
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir",    default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir",  default="reports/sepsis_readmit/analysis/causal_mort/notears")
    parser.add_argument("--log",         default="logs/analysis_causal_mort_notears.log")
    parser.add_argument("--lambdas",     type=float, nargs="+", default=[0.05, 0.01, 0.005])
    parser.add_argument("--w-threshold", type=float, default=0.01)
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

    data_df = df[ALL_VARS].fillna(df[ALL_VARS].median())
    X = StandardScaler().fit_transform(data_df.values.astype(np.float64))
    logging.info("Standardized. Shape: %d x %d", X.shape[0], X.shape[1])

    temporal_mask = build_temporal_mask(ALL_VARS)
    from notears.linear import notears_linear

    sweep_rows = []
    for lam in sorted(args.lambdas, reverse=True):
        logging.info("--- Lambda=%.4f ---", lam)
        t0 = time.time()
        W_raw = notears_linear(X, lambda1=lam, loss_type="l2", w_threshold=0.0)
        elapsed = time.time() - t0
        W = W_raw * temporal_mask

        n_raw    = int((np.abs(W_raw) >= args.w_threshold).sum())
        n_masked = int((np.abs(W)     >= args.w_threshold).sum())
        logging.info("  %.1f sec. Edges: %d raw, %d after mask", elapsed, n_raw, n_masked)

        edges_df = extract_edges(W, ALL_VARS, args.w_threshold)
        lam_str  = str(lam).replace(".", "p")
        edges_df.to_csv(f"{args.report_dir}/edges_lambda_{lam_str}.csv", index=False)

        outcome_edges = edges_df[(edges_df["source"]=="died_in_hosp")|
                                 (edges_df["target"]=="died_in_hosp")]
        lines = [f"NOTEARS edges at lambda={lam} ({len(df)} stays)",
                 f"  Variables: {', '.join(ALL_VARS)}", ""]
        lines.append(f"All edges ({len(edges_df)}):")
        for _, r in edges_df.iterrows():
            lines.append(f"  {r['direction']:<40s} w={r['weight']:+.4f}")
        lines.append(f"\nEdges involving died_in_hosp ({len(outcome_edges)}):")
        lines.append("  (none)" if not len(outcome_edges) else "")
        for _, r in outcome_edges.iterrows():
            lines.append(f"  {r['direction']:<40s} w={r['weight']:+.4f}")
        action_edges = edges_df[edges_df["source"].isin(["iv_input","vaso_input"])|
                                edges_df["target"].isin(["iv_input","vaso_input"])]
        lines.append(f"\nEdges involving actions ({len(action_edges)}):")
        lines.append("  (none)" if not len(action_edges) else "")
        for _, r in action_edges.iterrows():
            lines.append(f"  {r['direction']:<40s} w={r['weight']:+.4f}")
        summary = "\n".join(lines)
        with open(f"{args.report_dir}/summary_lambda_{lam_str}.txt", "w") as f:
            f.write(summary)
        print(f"\n{summary}")

        sweep_rows.append({"lambda": lam, "n_edges": n_masked,
                           "n_outcome_edges": len(outcome_edges),
                           "outcome_sources": "|".join(
                               outcome_edges[outcome_edges["target"]=="died_in_hosp"]["source"].tolist()),
                           "elapsed_sec": round(elapsed, 1)})

    pd.DataFrame(sweep_rows).to_csv(f"{args.report_dir}/lambda_sweep.csv", index=False)
    logging.info("NOTEARS mortality analysis complete.")


if __name__ == "__main__":
    main()
