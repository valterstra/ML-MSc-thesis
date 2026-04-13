"""Step B NOTEARS: Score-based causal discovery on V3 triplet dataset.

NOTEARS reformulates DAG learning as continuous optimisation:
  minimize  ||X - XW||^2  +  lambda * ||W||_1
  subject to  tr(e^(W o W)) - d = 0   (acyclicity)

Runs a lambda sweep to find the regularisation level where drug->next
edges appear. Temporal background knowledge is enforced POST-HOC by
zeroing out edges that violate tier ordering (NOTEARS has no built-in
BK support). The notears package is used as a black box -- no
modifications to the algorithm.

Lambda sweep rationale:
  0.05  -- moderate sparsity, only strongest effects survive
  0.01  -- what the smoke test used, most known drug edges appear
  0.005 -- lenient, brings in borderline edges (or noise)

Outputs (--report-dir):
  edges_lambda_{X}.csv      -- all edges per lambda
  drug_edges_lambda_{X}.csv -- drug -> next_* edges per lambda
  lambda_sweep.csv          -- which drug edges appear at each lambda
  run_log.txt

Usage:
    python scripts/causal_v3/step_b_notears.py

    python scripts/causal_v3/step_b_notears.py --sample-n 50000 --lambdas 0.1 0.05 0.01
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_b_pc import ALL_VARS, TIER2_ACTION, TIER3_NEXT, VAR_TIER

log = logging.getLogger(__name__)


# ── Temporal mask ─────────────────────────────────────────────────────────

def build_temporal_mask(var_names):
    """Build post-hoc mask for W matrix enforcing temporal tiers.

    W[i, j] = effect of j on i  (j -> i).
    Forbid j -> i when tier(j) >= tier(i)  (later cannot cause earlier/same).
    """
    n = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if VAR_TIER[vj] >= VAR_TIER[vi]:
                mask[i, j] = 0.0
    return mask


# ── Edge extraction ───────────────────────────────────────────────────────

def extract_edges(W, var_names, w_threshold):
    """Extract edges from W where W[i,j] = effect of j on i (j -> i)."""
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= w_threshold:
                rows.append({
                    "source": vj,
                    "target": vi,
                    "weight": round(float(w), 6),
                    "abs_weight": round(abs(float(w)), 6),
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "abs_weight"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def extract_drug_edges(edge_df):
    drug_set = set(TIER2_ACTION)
    next_set = set(TIER3_NEXT)
    mask = (
        (edge_df["source"].isin(drug_set) & edge_df["target"].isin(next_set)) |
        (edge_df["source"].isin(next_set) & edge_df["target"].isin(drug_set))
    )
    return edge_df[mask].copy()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step B NOTEARS: score-based causal discovery on V3.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b_notears"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-n", type=int, default=30_000)
    parser.add_argument(
        "--lambdas", type=float, nargs="+", default=[0.05, 0.01, 0.005],
        help="Lambda values for L1 penalty (default: 0.05 0.01 0.005)",
    )
    parser.add_argument("--w-threshold", type=float, default=0.01,
                        help="Min |weight| to report an edge (default: 0.01)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(report_dir / "run_log.txt"), mode="w", encoding="utf-8"),
        ],
    )

    log.info("=" * 70)
    log.info("Step B NOTEARS on V3 dataset")
    log.info("  sample_n=%d, lambdas=%s, w_threshold=%.3f, seed=%d",
             args.sample_n, args.lambdas, args.w_threshold, args.seed)
    log.info("=" * 70)

    # ── Load and prepare data ────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)

    if args.split:
        df = df[df["split"] == args.split]
        log.info("After split='%s': %d rows", args.split, len(df))

    df_sub = df[ALL_VARS].dropna()
    log.info("After dropna on %d variables: %d rows", len(ALL_VARS), len(df_sub))

    if args.sample_n > 0 and args.sample_n < len(df_sub):
        df_sub = df_sub.sample(n=args.sample_n, random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", args.sample_n, args.seed)

    # Standardize (z-score) for consistent lambda interpretation
    scaler = StandardScaler()
    X = scaler.fit_transform(df_sub.values.astype(np.float64))
    log.info("Data standardized (z-score). Shape: %d x %d", X.shape[0], X.shape[1])

    # Build temporal mask
    temporal_mask = build_temporal_mask(ALL_VARS)
    n_forbidden = int((temporal_mask == 0).sum())
    log.info("Temporal mask: %d entries forbidden (later tier cannot cause earlier)",
             n_forbidden)

    # Import notears (black box -- no modifications)
    from notears.linear import notears_linear

    # ── Lambda sweep ─────────────────────────────────────────────────
    all_results = {}
    total_t0 = time.time()

    for i, lam in enumerate(sorted(args.lambdas, reverse=True)):
        log.info("")
        log.info("=" * 70)
        log.info("LAMBDA = %.4f  (%d/%d)", lam, i + 1, len(args.lambdas))
        log.info("=" * 70)
        log.info("Starting NOTEARS optimisation (this may take 10-20 minutes) ...")

        t0 = time.time()
        W_raw = notears_linear(X, lambda1=lam, loss_type='l2', w_threshold=0.0)
        elapsed = time.time() - t0

        # Apply temporal mask post-hoc
        W = W_raw * temporal_mask

        # Count edges before and after masking
        n_raw = int((np.abs(W_raw) >= args.w_threshold).sum())
        n_masked = int((np.abs(W) >= args.w_threshold).sum())
        n_removed = n_raw - n_masked

        log.info("NOTEARS done in %.1f seconds (%.1f min)", elapsed, elapsed / 60)
        log.info("  Edges (|w| >= %.3f): %d total (%d removed by temporal mask)",
                 args.w_threshold, n_masked, n_removed)

        # Extract edges
        edge_df = extract_edges(W, ALL_VARS, w_threshold=args.w_threshold)
        drug_df = extract_drug_edges(edge_df)
        log.info("  Drug -> next_* edges: %d", len(drug_df))

        # Save per-lambda results
        lam_str = f"{lam:.4f}".replace(".", "p")
        edge_df.to_csv(report_dir / f"edges_lambda_{lam_str}.csv", index=False)
        drug_df.to_csv(report_dir / f"drug_edges_lambda_{lam_str}.csv", index=False)

        # Log drug edges immediately
        if len(drug_df) > 0:
            log.info("  Drug edges found:")
            for _, row in drug_df.iterrows():
                log.info("    %-25s --> %-25s  w=%.4f",
                         row["source"], row["target"], row["weight"])
        else:
            log.info("  No drug -> next_* edges found at this lambda.")

        all_results[lam] = {
            "edge_df": edge_df,
            "drug_df": drug_df,
            "elapsed": elapsed,
            "n_total_edges": n_masked,
        }

    total_elapsed = time.time() - total_t0
    log.info("")
    log.info("All lambdas completed in %.1f seconds (%.1f min)",
             total_elapsed, total_elapsed / 60)

    # ── Lambda sensitivity table for drug edges ──────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("LAMBDA SENSITIVITY TABLE (drug -> next_* edges)")
    log.info("=" * 70)

    drug_set = set(TIER2_ACTION)
    all_drug_pairs = set()
    for lam in all_results:
        for _, row in all_results[lam]["drug_df"].iterrows():
            src = row["source"] if row["source"] in drug_set else row["target"]
            tgt = row["target"] if row["source"] in drug_set else row["source"]
            all_drug_pairs.add((src, tgt))

    sweep_rows = []
    for src, tgt in sorted(all_drug_pairs):
        row_data = {"source": src, "target": tgt}
        for lam in sorted(args.lambdas, reverse=True):
            drug_df = all_results[lam]["drug_df"]
            match = drug_df[
                ((drug_df["source"] == src) & (drug_df["target"] == tgt)) |
                ((drug_df["source"] == tgt) & (drug_df["target"] == src))
            ]
            if len(match) > 0:
                w = match.iloc[0]["weight"]
                row_data[f"lam={lam}"] = f"{w:.4f}"
            else:
                row_data[f"lam={lam}"] = "-"
        sweep_rows.append(row_data)

    sweep_df = pd.DataFrame(sweep_rows)
    sweep_df.to_csv(report_dir / "lambda_sweep.csv", index=False)
    log.info("Saved: lambda_sweep.csv")
    log.info("")
    log.info("\n%s", sweep_df.to_string(index=False))

    # ── Compare with PC ──────────────────────────────────────────────
    pc_path = PROJECT_ROOT / "reports" / "causal_v3" / "step_b" / "drug_edges.csv"
    if pc_path.exists():
        pc_df = pd.read_csv(pc_path)
        pc_pairs = set(zip(pc_df["source"], pc_df["target"]))

        # Use the middle lambda for comparison
        mid_lam = sorted(args.lambdas)[len(args.lambdas) // 2]
        notears_drug = all_results[mid_lam]["drug_df"]
        notears_pairs = set()
        for _, row in notears_drug.iterrows():
            src = row["source"] if row["source"] in drug_set else row["target"]
            tgt = row["target"] if row["source"] in drug_set else row["source"]
            notears_pairs.add((src, tgt))

        in_both  = pc_pairs & notears_pairs
        pc_only  = pc_pairs - notears_pairs
        nt_only  = notears_pairs - pc_pairs

        log.info("")
        log.info("=" * 70)
        log.info("COMPARISON WITH PC (at lambda=%.4f)", mid_lam)
        log.info("=" * 70)
        log.info("  PC found: %d drug edges", len(pc_pairs))
        log.info("  NOTEARS found: %d drug edges", len(notears_pairs))
        log.info("  Agreement (%d): %s", len(in_both),
                 ", ".join(f"{s}->{t}" for s, t in sorted(in_both)) or "none")
        log.info("  PC only   (%d): %s", len(pc_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(pc_only)) or "none")
        log.info("  NOTEARS only (%d): %s", len(nt_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(nt_only)) or "none")

    log.info("")
    log.info("Step B NOTEARS complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f minutes", total_elapsed / 60)


if __name__ == "__main__":
    main()
