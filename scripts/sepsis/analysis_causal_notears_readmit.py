"""
NOTEARS causal discovery on reduced sepsis readmission variable set.

NOTEARS reformulates DAG learning as continuous optimisation:
  minimize  ||X - XW||^2  +  lambda * ||W||_1
  subject to  tr(e^(W o W)) - d = 0   (acyclicity constraint)

Returns a weighted adjacency matrix W where W[i,j] = effect of j on i (j -> i).
Temporal ordering enforced post-hoc by zeroing entries that violate tier ordering.

Lambda sweep: three values to show sensitivity to regularisation.
  0.05 -- sparse, only strongest effects survive
  0.01 -- moderate
  0.005 -- lenient, borderline edges appear

9 variables in 4 temporal tiers (same as PC script):
  Tier 0 (static):  age, elixhauser, re_admission
  Tier 1 (state):   BUN, Creatinine, Hb
  Tier 2 (action):  iv_input, vaso_input
  Tier 3 (outcome): readmit_30d

Population: survivors only (died_in_hosp=0), all splits combined.
No subsampling needed — NOTEARS is an optimisation, not a hypothesis test.

Inputs:
  data/processed/sepsis_readmit/rl_{train,val,test}_set_original.csv

Outputs (--report-dir):
  edges_lambda_{X}.csv        -- all edges per lambda
  lambda_sweep.csv            -- edge summary across lambdas
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

# ── Variable tiers (identical to PC script) ───────────────────────────────
TIER0_STATIC  = ["age", "elixhauser", "re_admission"]
TIER1_STATE   = ["BUN", "Creatinine", "Hb"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["readmit_30d"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}


def build_temporal_mask(var_names):
    """W[i,j] = effect of j on i. Forbid j->i when tier(j) >= tier(i)."""
    n = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if VAR_TIER[vj] >= VAR_TIER[vi]:
                mask[i, j] = 0.0
    return mask


def extract_edges(W, var_names, w_threshold):
    """Extract edges from W matrix where |w| >= threshold."""
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= w_threshold:
                rows.append({
                    "source":     vj,
                    "target":     vi,
                    "direction":  f"{vj} -> {vi}",
                    "weight":     round(float(w), 6),
                    "abs_weight": round(abs(float(w)), 6),
                    "tier_src":   VAR_TIER[vj],
                    "tier_dst":   VAR_TIER[vi],
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "direction", "weight", "abs_weight",
                 "tier_src", "tier_dst"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def plain_english_summary(edges_df, lam, var_names, n_rows):
    lines = [
        f"NOTEARS edges at lambda={lam} ({n_rows} stays)",
        f"  Variables: {', '.join(var_names)}",
        "",
    ]
    outcome_edges = edges_df[
        (edges_df["source"] == "readmit_30d") |
        (edges_df["target"] == "readmit_30d")
    ]
    action_edges = edges_df[
        edges_df["source"].isin(["iv_input", "vaso_input"]) |
        edges_df["target"].isin(["iv_input", "vaso_input"])
    ]

    lines.append(f"All edges ({len(edges_df)}):")
    for _, row in edges_df.iterrows():
        lines.append(f"  {row['direction']:<40s} w={row['weight']:+.4f}")

    lines.append(f"\nEdges involving readmit_30d ({len(outcome_edges)}):")
    if len(outcome_edges) == 0:
        lines.append("  (none)")
    for _, row in outcome_edges.iterrows():
        lines.append(f"  {row['direction']:<40s} w={row['weight']:+.4f}")

    lines.append(f"\nEdges involving actions ({len(action_edges)}):")
    if len(action_edges) == 0:
        lines.append("  (none)")
    for _, row in action_edges.iterrows():
        lines.append(f"  {row['direction']:<40s} w={row['weight']:+.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="NOTEARS causal discovery on sepsis readmission variables"
    )
    parser.add_argument("--data-dir",    default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir",  default="reports/sepsis_readmit/analysis/causal/notears")
    parser.add_argument("--log",         default="logs/analysis_causal_notears_readmit.log")
    parser.add_argument("--lambdas",     type=float, nargs="+", default=[0.05, 0.01, 0.005])
    parser.add_argument("--w-threshold", type=float, default=0.01,
                        help="Min |weight| to report an edge")
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
        dfs.append(df.groupby("icustayid").tail(1))
    df = pd.concat(dfs, ignore_index=True)
    df = df[df["died_in_hosp"] == 0].reset_index(drop=True)
    logging.info("Survivors, last timestep: %d stays (%.1f%% readmitted)",
                 len(df), 100 * df["readmit_30d"].mean())

    missing = [v for v in ALL_VARS if v not in df.columns]
    if missing:
        logging.error("Missing variables: %s", missing)
        sys.exit(1)

    data_df = df[ALL_VARS].fillna(df[ALL_VARS].median())
    logging.info("Using all %d stays (no subsampling — NOTEARS is an optimisation)",
                 len(data_df))

    # Standardize (z-score) for consistent lambda interpretation
    scaler = StandardScaler()
    X = scaler.fit_transform(data_df.values.astype(np.float64))
    logging.info("Data standardized. Shape: %d x %d", X.shape[0], X.shape[1])

    # Temporal mask
    temporal_mask = build_temporal_mask(ALL_VARS)
    n_forbidden = int((temporal_mask == 0).sum())
    logging.info("Temporal mask: %d entries forbidden", n_forbidden)

    from notears.linear import notears_linear

    # ── Lambda sweep ──────────────────────────────────────────────────
    sweep_rows = []
    total_t0 = time.time()

    for lam in sorted(args.lambdas, reverse=True):
        logging.info("--- Lambda=%.4f ---", lam)
        t0 = time.time()
        W_raw = notears_linear(X, lambda1=lam, loss_type="l2", w_threshold=0.0)
        elapsed = time.time() - t0

        W = W_raw * temporal_mask

        n_raw    = int((np.abs(W_raw) >= args.w_threshold).sum())
        n_masked = int((np.abs(W)     >= args.w_threshold).sum())
        logging.info("  Done in %.1f sec. Edges: %d total, %d after temporal mask",
                     elapsed, n_raw, n_masked)

        edges_df = extract_edges(W, ALL_VARS, args.w_threshold)

        # Save per-lambda edges
        lam_str = str(lam).replace(".", "p")
        edges_df.to_csv(f"{args.report_dir}/edges_lambda_{lam_str}.csv", index=False)

        # Plain-English summary
        summary = plain_english_summary(edges_df, lam, ALL_VARS, len(data_df))
        summary_path = f"{args.report_dir}/summary_lambda_{lam_str}.txt"
        with open(summary_path, "w") as f:
            f.write(summary)
        print(f"\n{summary}")

        # Sweep row
        outcome_edges = edges_df[
            (edges_df["source"] == "readmit_30d") |
            (edges_df["target"] == "readmit_30d")
        ]
        sweep_rows.append({
            "lambda":         lam,
            "n_edges":        n_masked,
            "n_readmit_edges": len(outcome_edges),
            "readmit_sources": "|".join(
                outcome_edges[outcome_edges["target"] == "readmit_30d"]["source"].tolist()
            ),
            "elapsed_sec":    round(elapsed, 1),
        })

    # Save sweep summary
    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(f"{args.report_dir}/lambda_sweep.csv", index=False)
    logging.info("Lambda sweep saved: %s/lambda_sweep.csv", args.report_dir)
    logging.info("Total elapsed: %.1f sec", time.time() - total_t0)
    logging.info("NOTEARS analysis complete.")


if __name__ == "__main__":
    main()
