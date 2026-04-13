"""
SDCD causal discovery on reduced sepsis readmission variable set.

SDCD (Stable Differentiable Causal Discovery) is a neural network-based
approach that learns a DAG by training a flow model with a differentiable
acyclicity constraint. It is in the same family as NOTEARS but uses a
neural network to model non-linear relationships rather than a linear model.

Two-stage training:
  Stage 1: learn a dense adjacency via continuous relaxation
  Stage 2: prune/finetune with hard masking to recover a sparse DAG

Returns a weighted adjacency matrix W where W[i,j] > 0 means j -> i.

Temporal ordering enforced post-hoc by zeroing entries that violate tier
ordering (same approach as NOTEARS script).

9 variables in 4 temporal tiers:
  Tier 0 (static):  age, elixhauser, re_admission
  Tier 1 (state):   BUN, Creatinine, Hb
  Tier 2 (action):  iv_input, vaso_input
  Tier 3 (outcome): readmit_30d

Population: survivors only (died_in_hosp=0), all splits combined.
All data used (no subsampling -- neural training benefits from more data).

Inputs:
  data/processed/sepsis_readmit/rl_{train,val,test}_set_original.csv

Outputs (--report-dir):
  sdcd_readmit_edges.csv      -- edges above threshold
  sdcd_readmit_weights.csv    -- full weighted adjacency (all pairs)
  sdcd_readmit_summary.txt    -- plain-English summary
  run_log.txt
"""
import argparse
import logging
import os
import sys
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, ".")

# Suppress wandb and pkg_resources warnings
warnings.filterwarnings("ignore")
os.environ["WANDB_MODE"] = "disabled"

# ── Variable tiers ────────────────────────────────────────────────────────
TIER0_STATIC  = ["age", "elixhauser", "re_admission"]
TIER1_STATE   = ["BUN", "Creatinine", "Hb"]
TIER2_ACTION  = ["iv_input", "vaso_input"]
TIER3_OUTCOME = ["readmit_30d"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_OUTCOME
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_OUTCOME]
VAR_TIER = {v: t for t, tier in enumerate(TIERS) for v in tier}


def build_temporal_mask(var_names):
    """W[i,j] = effect of j on i. Forbid j->i when tier(j) >= tier(i)."""
    n    = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if VAR_TIER[vj] >= VAR_TIER[vi]:
                mask[i, j] = 0.0
    return mask


def extract_edges(W, var_names, w_threshold):
    """Extract edges from weighted adjacency matrix W[i,j] = effect of j on i."""
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
        columns=["source", "target", "direction", "weight",
                 "abs_weight", "tier_src", "tier_dst"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def plain_english_summary(edges_df, n_rows, n_edges_raw, n_edges_masked):
    lines = [
        "Causal Discovery Summary -- Sepsis Readmission (SDCD algorithm)",
        f"  Variables: {', '.join(ALL_VARS)}",
        f"  Population: {n_rows} ICU stays (all survivors, no subsampling)",
        f"  Edges before temporal mask: {n_edges_raw}",
        f"  Edges after temporal mask:  {n_edges_masked}",
        "  (SDCD = neural differentiable DAG; non-linear complement to NOTEARS)",
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
    if not len(edges_df):
        lines.append("  (none above threshold)")
    for _, row in edges_df.iterrows():
        lines.append(f"  {row['direction']:<40s} w={row['weight']:+.4f}")

    lines.append(f"\nEdges involving readmit_30d ({len(outcome_edges)}):")
    if not len(outcome_edges):
        lines.append("  (none)")
    for _, row in outcome_edges.iterrows():
        lines.append(f"  {row['direction']:<40s} w={row['weight']:+.4f}")

    lines.append(f"\nEdges involving actions ({len(action_edges)}):")
    if not len(action_edges):
        lines.append("  (none)")
    for _, row in action_edges.iterrows():
        lines.append(f"  {row['direction']:<40s} w={row['weight']:+.4f}")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="SDCD causal discovery on sepsis readmission variables"
    )
    parser.add_argument("--data-dir",    default="data/processed/sepsis_readmit")
    parser.add_argument("--report-dir",  default="reports/sepsis_readmit/analysis/causal/sdcd")
    parser.add_argument("--log",         default="logs/analysis_causal_sdcd_readmit.log")
    parser.add_argument("--w-threshold", type=float, default=0.01,
                        help="Min |weight| to report an edge (same default as NOTEARS script)")
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

    # ── Load data (all survivors, no subsampling) ─────────────────────
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

    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(data_df.values.astype(np.float32))
    logging.info("Data standardized. Shape: %d x %d", X.shape[0], X.shape[1])

    # Temporal mask
    temporal_mask = build_temporal_mask(ALL_VARS)
    logging.info("Temporal mask: %d entries forbidden",
                 int((temporal_mask == 0).sum()))

    # ── Build torch Dataset ───────────────────────────────────────────
    import torch
    from torch.utils.data import TensorDataset

    X_tensor = torch.tensor(X, dtype=torch.float32)
    dataset   = TensorDataset(X_tensor)

    # ── Run SDCD ──────────────────────────────────────────────────────
    from sdcd import SDCD

    logging.info("Initialising SDCD model (standard_scale=False, already scaled)")
    model = SDCD(
        model_variance_flavor="parameter",
        standard_scale=False,
    )

    logging.info("Training SDCD (two-stage)...")
    t0 = time.time()
    model.train(
        dataset,
        log_wandb=False,
        verbose=True,
    )
    elapsed = time.time() - t0
    logging.info("SDCD training complete in %.1f sec", elapsed)

    # ── Extract adjacency ─────────────────────────────────────────────
    # get_adjacency_matrix(threshold=False) returns raw weights
    W_raw = model.get_adjacency_matrix(threshold=False)
    logging.info("Raw adjacency matrix shape: %s, range: [%.4f, %.4f]",
                 W_raw.shape, W_raw.min(), W_raw.max())

    # Apply temporal mask
    W = W_raw * temporal_mask

    n_raw    = int((np.abs(W_raw) >= args.w_threshold).sum())
    n_masked = int((np.abs(W)     >= args.w_threshold).sum())
    logging.info("Edges: %d total, %d after temporal mask (threshold=%.3f)",
                 n_raw, n_masked, args.w_threshold)

    edges_df = extract_edges(W, ALL_VARS, args.w_threshold)

    # Save edges
    edges_path = f"{args.report_dir}/sdcd_readmit_edges.csv"
    edges_df.to_csv(edges_path, index=False)
    logging.info("Edges saved: %s", edges_path)

    # Save full weight matrix
    weight_rows = []
    for i, vi in enumerate(ALL_VARS):
        for j, vj in enumerate(ALL_VARS):
            if i != j:
                weight_rows.append({
                    "source": vj, "target": vi,
                    "weight_raw":    round(float(W_raw[i, j]), 6),
                    "weight_masked": round(float(W[i, j]),     6),
                    "tier_src": VAR_TIER[vj], "tier_dst": VAR_TIER[vi],
                })
    weights_df = pd.DataFrame(weight_rows).sort_values(
        "weight_masked", key=abs, ascending=False).reset_index(drop=True)
    weights_path = f"{args.report_dir}/sdcd_readmit_weights.csv"
    weights_df.to_csv(weights_path, index=False)
    logging.info("Full weight matrix saved: %s", weights_path)

    # Plain-English summary
    summary = plain_english_summary(edges_df, len(data_df), n_raw, n_masked)
    summary_path = f"{args.report_dir}/sdcd_readmit_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary)
    logging.info("Summary saved: %s", summary_path)
    print("\n" + summary)

    logging.info("SDCD analysis complete.")


if __name__ == "__main__":
    main()
