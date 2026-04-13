"""Step B LiNGAM: DirectLiNGAM causal discovery on V3 triplet dataset.

DirectLiNGAM uses ICA (independent component analysis) rather than conditional
independence tests. Key differences from PC:
  - Assumes LINEAR relationships with NON-GAUSSIAN noise
  - Fully orients the DAG -- no Markov equivalence class ambiguity
  - Gives linear WEIGHTS on each edge (magnitude + direction)
  - Does NOT assume Gaussianity (unlike Fisher-z in PC)

Temporal prior knowledge is enforced via the prior_knowledge matrix:
  prior_knowledge[i, j] = 0   -> edge i->j is FORBIDDEN
  prior_knowledge[i, j] = -1  -> edge i->j is unknown (default)

Tiers (same as PC):
  Tier 0 (static): age_at_admit, charlson_score
  Tier 1 (state):  13 core labs + is_icu
  Tier 2 (action): 5 drug flags
  Tier 3 (next):   13 next-day labs + next_is_icu

Outputs (--report-dir):
  edges.csv          -- all edges with linear weights
  drug_edges.csv     -- drug -> next_* edges only
  weight_matrix.csv  -- full adjacency matrix
  run_log.txt

Usage:
    python scripts/causal_v3/step_b_lingam.py

    python scripts/causal_v3/step_b_lingam.py --sample-n 300000 --w-threshold 0.05
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_b_pc import ALL_VARS, TIER2_ACTION, TIER3_NEXT, TIERS, VAR_TIER

log = logging.getLogger(__name__)


# ── Temporal prior knowledge for LiNGAM ──────────────────────────────────

def build_lingam_prior_knowledge(var_names):
    """Build prior_knowledge matrix for DirectLiNGAM.

    prior_knowledge[i, j] = 0  -> edge var[i]->var[j] is FORBIDDEN
    prior_knowledge[i, j] = -1 -> unknown (allowed)

    Rule: edges can only go from lower tier to higher tier (forward in time).
    Forbid any edge where source tier > target tier.
    Also forbid within-tier edges from action->state and next->anything.
    """
    n = len(var_names)
    pk = np.full((n, n), -1, dtype=int)  # default: unknown

    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            ti = VAR_TIER[vi]
            tj = VAR_TIER[vj]
            # Forbid edges from later tier to earlier or same tier
            # (allow only strictly forward: lower -> higher)
            if ti >= tj:
                pk[i, j] = 0  # vi -> vj is forbidden

    return pk


# ── Edge extraction ───────────────────────────────────────────────────────

def extract_edges(W, var_names, w_threshold):
    """Extract edges from adjacency matrix W.

    W[i, j] = linear causal effect of var[j] on var[i].
    So W[i,j] != 0 means j -> i with weight W[i,j].
    """
    n = len(var_names)
    rows = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= w_threshold:
                rows.append({
                    "source": var_names[j],   # j causes i
                    "target": var_names[i],
                    "weight": round(float(w), 6),
                    "abs_weight": round(abs(float(w)), 6),
                })
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "abs_weight"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def extract_drug_edges(edge_df):
    """Filter to drug -> next_* edges (either direction)."""
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
        description="Step B LiNGAM: DirectLiNGAM on V3 triplet dataset.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b_lingam"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-n", type=int, default=100_000)
    parser.add_argument("--w-threshold", type=float, default=0.1,
                        help="Min absolute weight to report an edge (default: 0.1)")
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
    log.info("Step B LiNGAM on V3 dataset")
    log.info("  sample_n=%d, w_threshold=%.2f, seed=%d",
             args.sample_n, args.w_threshold, args.seed)
    log.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
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

    data = df_sub.values.astype(np.float64)
    log.info("Data matrix: %d rows x %d columns", data.shape[0], data.shape[1])

    # ── Build temporal prior knowledge ───────────────────────────────
    pk = build_lingam_prior_knowledge(ALL_VARS)
    n_forbidden = int((pk == 0).sum())
    n_unknown   = int((pk == -1).sum())
    log.info("Prior knowledge: %d forbidden edges, %d unknown", n_forbidden, n_unknown)

    # ── Run DirectLiNGAM ─────────────────────────────────────────────
    log.info("Running DirectLiNGAM ...")
    t0 = time.time()

    import lingam
    model = lingam.DirectLiNGAM(prior_knowledge=pk, random_state=args.seed)
    model.fit(data)

    elapsed = time.time() - t0
    log.info("DirectLiNGAM complete in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    W = model.adjacency_matrix_
    log.info("Causal order: %s",
             [ALL_VARS[i] for i in model.causal_order_])

    # ── Extract and save edges ───────────────────────────────────────
    edge_df = extract_edges(W, ALL_VARS, w_threshold=args.w_threshold)
    drug_df = extract_drug_edges(edge_df)

    log.info("Total edges (|w| >= %.2f): %d", args.w_threshold, len(edge_df))
    log.info("Drug -> next_* edges: %d", len(drug_df))

    edge_df.to_csv(report_dir / "edges.csv", index=False)
    drug_df.to_csv(report_dir / "drug_edges.csv", index=False)

    # Save full weight matrix
    W_df = pd.DataFrame(W, index=ALL_VARS, columns=ALL_VARS)
    W_df.to_csv(report_dir / "weight_matrix.csv")
    log.info("Saved: edges.csv, drug_edges.csv, weight_matrix.csv")

    # ── Drug edge summary ────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("DRUG -> NEXT_* EDGES (w_threshold=%.2f)", args.w_threshold)
    log.info("=" * 70)

    if len(drug_df) == 0:
        log.info("  No drug -> next edges found at w_threshold=%.2f", args.w_threshold)
        log.info("  (This is the expected V2 result -- ICA misses small confounded effects)")
    else:
        for _, row in drug_df.iterrows():
            log.info("  %-25s --> %-25s  weight=%.4f",
                     row["source"], row["target"], row["weight"])

    # ── Sensitivity: lower threshold ─────────────────────────────────
    log.info("")
    log.info("DRUG -> NEXT_* AT LOWER THRESHOLDS (sensitivity check):")
    drug_set = set(TIER2_ACTION)
    next_set  = set(TIER3_NEXT)
    for thr in [0.05, 0.02, 0.01]:
        edge_low = extract_edges(W, ALL_VARS, w_threshold=thr)
        drug_low = extract_drug_edges(edge_low)
        log.info("  w >= %.2f: %d drug edges", thr, len(drug_low))
        for _, row in drug_low.iterrows():
            log.info("    %-25s --> %-25s  weight=%.4f",
                     row["source"], row["target"], row["weight"])

    # ── Compare with PC ──────────────────────────────────────────────
    pc_path = PROJECT_ROOT / "reports" / "causal_v3" / "step_b" / "drug_edges.csv"
    if pc_path.exists():
        pc_df   = pd.read_csv(pc_path)
        pc_pairs = set(zip(pc_df["source"], pc_df["target"]))

        lingam_pairs = set(
            (r["source"], r["target"]) for _, r in drug_df.iterrows()
            if r["source"] in drug_set
        )

        log.info("")
        log.info("=" * 70)
        log.info("COMPARISON WITH PC (drug -> next_* directed edges)")
        log.info("=" * 70)
        log.info("  PC found %d drug edges", len(pc_pairs))
        log.info("  LiNGAM found %d drug edges (w>=%.2f)", len(lingam_pairs), args.w_threshold)
        in_both  = pc_pairs & lingam_pairs
        pc_only  = pc_pairs - lingam_pairs
        lgm_only = lingam_pairs - pc_pairs
        log.info("  Agreement (%d): %s", len(in_both),
                 ", ".join(f"{s}->{t}" for s, t in sorted(in_both)) or "none")
        log.info("  PC only   (%d): %s", len(pc_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(pc_only)) or "none")
        log.info("  LiNGAM only (%d): %s", len(lgm_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(lgm_only)) or "none")

    log.info("")
    log.info("Step B LiNGAM complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
