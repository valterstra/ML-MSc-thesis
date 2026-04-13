"""Step B DYNOTEARS: Temporal causal discovery on V3 triplet dataset.

DYNOTEARS (Pamfil et al., 2020) extends NOTEARS to time-series data.
It simultaneously learns two adjacency matrices:
  W  (d x d):   contemporaneous (intra-slice) effects at time t
  A  (d*p x d): lagged (inter-slice) effects from t-1,...,t-p to t

Key advantage over our standard NOTEARS/PC approach:
  - Temporal ordering is NATIVE (A captures lags, W captures same-time)
  - No need for manual tier structure or post-hoc temporal masking
  - Can discover multi-lag effects (p=2: does drug at T affect lab at T+2?)
  - Handles variable-length patient time series (pools across admissions)

Data transformation:
  Our V3 triplets have (state_T, action_T, next_state_T+1) per row.
  For DYNOTEARS, we reconstruct per-admission time series:
    Day 1: [lab_1, ..., lab_13, is_icu_1, drug_1, ..., drug_5]
    Day 2: [lab_1, ..., lab_13, is_icu_2, drug_1, ..., drug_5]
    ...
  Using 19 time-varying variables (no static age/charlson -- they don't
  vary over time within an admission, which confuses the lag model).

  The A matrix drug->lab edges correspond directly to our PC drug->next
  edges, enabling direct cross-algorithm comparison.

Tabu edges (domain knowledge):
  W (contemporaneous): drug->lab is FORBIDDEN (in V3, labs are drawn
    BEFORE drug starts on the same day, so same-day drug cannot affect
    same-day lab).

Outputs (--report-dir):
  W_matrix.csv          -- contemporaneous adjacency (d x d)
  A_matrix.csv          -- lagged adjacency (d*p x d)
  drug_edges.csv        -- drug -> lab edges from A matrix
  all_edges.csv         -- all edges from both W and A
  run_log.txt

Usage:
    # Smoke test (500 admissions, fast)
    python scripts/causal_v3/step_b_dynotears.py --n-admissions 500

    # Full run
    python scripts/causal_v3/step_b_dynotears.py --n-admissions 0

    # Multi-lag
    python scripts/causal_v3/step_b_dynotears.py --n-admissions 0 --lag 2
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

from step_b_pc import TIER1_STATE, TIER2_ACTION

log = logging.getLogger(__name__)

# ── Variables for DYNOTEARS ──────────────────────────────────────────────
# 19 time-varying variables (no static age/charlson)
# Static variables are constant within an admission -- they would create
# degenerate lagged effects (lag value == current value).
DYNO_STATE = list(TIER1_STATE)   # 14 vars: 13 labs + is_icu
DYNO_ACTION = list(TIER2_ACTION) # 5 vars: 5 drug flags
DYNO_VARS = DYNO_STATE + DYNO_ACTION  # 19 total

STATE_IDX = {v: i for i, v in enumerate(DYNO_STATE)}
ACTION_IDX = {v: i + len(DYNO_STATE) for i, v in enumerate(DYNO_ACTION)}
VAR_IDX = {v: i for i, v in enumerate(DYNO_VARS)}


# ── Data transformation ──────────────────────────────────────────────────

def triplets_to_time_series(
    df: pd.DataFrame,
    max_admissions: int = 0,
    seed: int = 42,
) -> list[np.ndarray]:
    """Convert V3 triplet rows into per-admission time series.

    Each admission becomes a (T, 19) array where T is the number of
    consecutive days. Gaps in day_of_stay split into separate segments.

    Args:
        df: Triplet DataFrame with state, action, and day_of_stay columns.
        max_admissions: If > 0, subsample this many admissions (by hadm_id).
        seed: Random seed for subsampling.

    Returns:
        List of (T_i, 19) arrays, one per continuous segment per admission.
    """
    hadm_ids = df["hadm_id"].unique()
    if max_admissions > 0 and max_admissions < len(hadm_ids):
        rng = np.random.default_rng(seed)
        hadm_ids = rng.choice(hadm_ids, size=max_admissions, replace=False)
        df = df[df["hadm_id"].isin(hadm_ids)]

    segments = []
    n_admissions = 0
    n_gaps = 0

    for hadm_id, group in df.groupby("hadm_id"):
        group = group.sort_values("day_of_stay")
        days = group["day_of_stay"].values
        values = group[DYNO_VARS].values.astype(np.float64)

        # Split at gaps (day jumps > 1)
        if len(days) < 2:
            # Single-row admission: can still be part of a segment if we
            # reconstruct from next_* columns, but for simplicity skip.
            # DYNOTEARS needs at least p+1 rows per segment.
            continue

        gap_indices = np.where(np.diff(days) > 1)[0]
        if len(gap_indices) > 0:
            n_gaps += 1

        # Cut points
        cuts = [0] + list(gap_indices + 1) + [len(days)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            seg = values[start:end]
            if seg.shape[0] >= 2:  # need at least 2 rows for p=1
                segments.append(seg)

        n_admissions += 1

    log.info("Built %d continuous segments from %d admissions (%d had gaps)",
             len(segments), n_admissions, n_gaps)

    # Segment length distribution
    seg_lens = [s.shape[0] for s in segments]
    log.info("  Segment lengths: min=%d, median=%d, mean=%.1f, max=%d",
             min(seg_lens), int(np.median(seg_lens)),
             np.mean(seg_lens), max(seg_lens))

    return segments


def standardize_segments(
    segments: list[np.ndarray],
) -> tuple[list[np.ndarray], StandardScaler]:
    """Z-score standardize across all pooled data.

    Fits scaler on all data, then transforms each segment.
    Returns standardized segments and the fitted scaler.
    """
    all_data = np.concatenate(segments, axis=0)
    scaler = StandardScaler()
    scaler.fit(all_data)
    log.info("Standardized %d rows x %d cols (z-score)", all_data.shape[0], all_data.shape[1])

    std_segments = [scaler.transform(seg) for seg in segments]
    return std_segments, scaler


# ── Tabu edges (domain knowledge) ────────────────────────────────────────

def build_tabu_edges(var_names: list[str]) -> list[tuple[int, int, int]]:
    """Build list of forbidden edges for DYNOTEARS.

    Domain knowledge from V3 data design:
      - W (lag=0): drug -> lab is FORBIDDEN
        (labs are drawn BEFORE drug starts on the same day)

    We do NOT forbid in the A matrix -- lagged drug->lab effects are
    exactly what we want to discover.

    Returns:
        List of (lag, from_idx, to_idx) tuples.
    """
    tabu = []
    drug_indices = [i for i, v in enumerate(var_names) if v in set(DYNO_ACTION)]
    lab_indices = [i for i, v in enumerate(var_names) if v in set(DYNO_STATE)]

    for d_idx in drug_indices:
        for l_idx in lab_indices:
            tabu.append((0, d_idx, l_idx))  # drug -> lab in W is forbidden

    log.info("Tabu edges (lag=0, drug->lab): %d forbidden", len(tabu))
    return tabu


# ── Edge extraction ──────────────────────────────────────────────────────

def extract_edges(W: np.ndarray, A: np.ndarray, var_names: list[str],
                  p: int, w_threshold: float) -> pd.DataFrame:
    """Extract all edges from W and A matrices into a DataFrame.

    Convention: W[i,j] != 0 means j -> i at time t (contemporaneous).
                A[i,j] != 0 means var i_lagged -> var j at time t.

    For A with p lags, rows 0..d-1 are lag-1, rows d..2d-1 are lag-2, etc.
    """
    d = len(var_names)
    rows = []

    # Intra-slice edges (W)
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= w_threshold:
                rows.append({
                    "source": var_names[j],
                    "target": var_names[i],
                    "weight": round(float(w), 6),
                    "abs_weight": round(abs(float(w)), 6),
                    "lag": 0,
                    "matrix": "W",
                })

    # Inter-slice edges (A)
    for lag_k in range(p):
        for i in range(d):
            for j in range(d):
                a_idx = lag_k * d + i
                w = A[a_idx, j]
                if abs(w) >= w_threshold:
                    rows.append({
                        "source": f"{var_names[i]}_lag{lag_k + 1}",
                        "target": var_names[j],
                        "weight": round(float(w), 6),
                        "abs_weight": round(abs(float(w)), 6),
                        "lag": lag_k + 1,
                        "matrix": "A",
                        # Also store raw variable names for easy comparison
                        "source_var": var_names[i],
                        "target_var": var_names[j],
                    })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "abs_weight", "lag", "matrix"]
    )
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def extract_drug_lab_edges(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Filter to drug -> lab edges in the A matrix (lagged).

    These correspond to our PC drug -> next_* edges.
    """
    if edge_df.empty:
        return edge_df

    drug_set = set(DYNO_ACTION)
    lab_set = set(DYNO_STATE)

    mask = (
        (edge_df["matrix"] == "A") &
        (edge_df.get("source_var", edge_df["source"]).isin(drug_set)) &
        (edge_df.get("target_var", edge_df["target"]).isin(lab_set))
    )
    return edge_df[mask].copy()


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step B DYNOTEARS: temporal causal discovery on V3.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b_dynotears"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--n-admissions", type=int, default=0,
                        help="Max admissions to use (0 = all, 500 = smoke test)")
    parser.add_argument("--lag", type=int, default=1,
                        help="Lag order p (1 = day-to-day, 2 = 2-day lag)")
    parser.add_argument("--lambda-w", type=float, default=0.05,
                        help="L1 penalty on W (contemporaneous)")
    parser.add_argument("--lambda-a", type=float, default=0.05,
                        help="L1 penalty on A (lagged)")
    parser.add_argument("--w-threshold", type=float, default=0.01,
                        help="Min |weight| to report an edge")
    parser.add_argument("--max-iter", type=int, default=100,
                        help="Max augmented-Lagrangian iterations")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(report_dir / "run_log.txt"), mode="w",
                                encoding="utf-8"),
        ],
    )

    log.info("=" * 70)
    log.info("Step B DYNOTEARS on V3 dataset")
    log.info("  n_admissions=%s, lag=%d, lambda_w=%.3f, lambda_a=%.3f",
             args.n_admissions or "ALL", args.lag, args.lambda_w, args.lambda_a)
    log.info("  w_threshold=%.3f, max_iter=%d, seed=%d",
             args.w_threshold, args.max_iter, args.seed)
    log.info("=" * 70)

    # ── Load data ─────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)

    if args.split:
        df = df[df["split"] == args.split]
        log.info("After split='%s': %d rows", args.split, len(df))

    # Check all required columns exist
    required = DYNO_VARS + ["hadm_id", "day_of_stay"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        log.error("Missing columns: %s", missing)
        sys.exit(1)

    # Drop rows with NaN in our variables
    n_before = len(df)
    df = df.dropna(subset=DYNO_VARS)
    log.info("After dropna on %d variables: %d rows (dropped %d)",
             len(DYNO_VARS), len(df), n_before - len(df))

    # ── Build time series ─────────────────────────────────────────────
    log.info("")
    log.info("Building per-admission time series ...")
    segments = triplets_to_time_series(
        df, max_admissions=args.n_admissions, seed=args.seed,
    )

    if not segments:
        log.error("No usable segments found!")
        sys.exit(1)

    # ── Standardize ───────────────────────────────────────────────────
    std_segments, scaler = standardize_segments(segments)

    # ── Build X, Xlags ────────────────────────────────────────────────
    from careai.causal_v3.dynotears_solver import build_X_Xlags
    X, Xlags = build_X_Xlags(std_segments, p=args.lag)
    log.info("DYNOTEARS input: X=%s, Xlags=%s", X.shape, Xlags.shape)

    # ── Build tabu edges ──────────────────────────────────────────────
    tabu = build_tabu_edges(DYNO_VARS)

    # ── Run DYNOTEARS ─────────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("Running DYNOTEARS (this may take several minutes) ...")
    log.info("=" * 70)

    from careai.causal_v3.dynotears_solver import dynotears_solve

    t0 = time.time()
    W, A = dynotears_solve(
        X, Xlags,
        lambda_w=args.lambda_w,
        lambda_a=args.lambda_a,
        max_iter=args.max_iter,
        h_tol=1e-8,
        w_threshold=0.0,  # no pruning inside -- we prune post-hoc
        tabu_edges=tabu,
    )
    elapsed = time.time() - t0

    log.info("")
    log.info("DYNOTEARS completed in %.1f seconds (%.1f min)", elapsed, elapsed / 60)

    # ── Post-hoc threshold and save matrices ──────────────────────────
    W_raw_nonzero = int(np.count_nonzero(W))
    A_raw_nonzero = int(np.count_nonzero(A))
    log.info("Raw edges: W=%d non-zero, A=%d non-zero", W_raw_nonzero, A_raw_nonzero)

    # Save raw matrices
    W_df = pd.DataFrame(W, index=DYNO_VARS, columns=DYNO_VARS)
    W_df.to_csv(report_dir / "W_matrix.csv")

    a_row_names = [f"{v}_lag{k+1}" for k in range(args.lag) for v in DYNO_VARS]
    A_df = pd.DataFrame(A, index=a_row_names, columns=DYNO_VARS)
    A_df.to_csv(report_dir / "A_matrix.csv")
    log.info("Saved: W_matrix.csv, A_matrix.csv")

    # ── Extract edges ─────────────────────────────────────────────────
    edge_df = extract_edges(W, A, DYNO_VARS, p=args.lag,
                            w_threshold=args.w_threshold)
    drug_df = extract_drug_lab_edges(edge_df)

    n_w_edges = len(edge_df[edge_df["matrix"] == "W"]) if len(edge_df) > 0 else 0
    n_a_edges = len(edge_df[edge_df["matrix"] == "A"]) if len(edge_df) > 0 else 0
    log.info("Edges (|w| >= %.3f): W=%d, A=%d, total=%d",
             args.w_threshold, n_w_edges, n_a_edges, len(edge_df))
    log.info("Drug -> lab edges (from A): %d", len(drug_df))

    edge_df.to_csv(report_dir / "all_edges.csv", index=False)
    drug_df.to_csv(report_dir / "drug_edges.csv", index=False)
    log.info("Saved: all_edges.csv, drug_edges.csv")

    # ── Drug edge summary ─────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("DRUG -> LAB EDGES (A matrix, lagged)")
    log.info("=" * 70)

    if len(drug_df) == 0:
        log.info("  No drug -> lab edges found at w_threshold=%.3f", args.w_threshold)
    else:
        for _, row in drug_df.iterrows():
            src_var = row.get("source_var", row["source"])
            tgt_var = row.get("target_var", row["target"])
            log.info("  %-25s --> %-20s  w=%+.4f  (lag %d)",
                     src_var, tgt_var, row["weight"], row["lag"])

    # ── W matrix summary (contemporaneous) ────────────────────────────
    w_edges = edge_df[edge_df["matrix"] == "W"] if len(edge_df) > 0 else pd.DataFrame()
    if len(w_edges) > 0:
        log.info("")
        log.info("=" * 70)
        log.info("TOP CONTEMPORANEOUS EDGES (W matrix, same-day)")
        log.info("=" * 70)
        for _, row in w_edges.head(20).iterrows():
            log.info("  %-25s --> %-20s  w=%+.4f",
                     row["source"], row["target"], row["weight"])

    # ── Compare with PC results ───────────────────────────────────────
    pc_path = PROJECT_ROOT / "reports" / "causal_v3" / "step_b" / "drug_edges.csv"
    if pc_path.exists() and len(drug_df) > 0:
        pc_df = pd.read_csv(pc_path)
        # PC edges are (drug, next_lab) -- strip "next_" prefix for comparison
        pc_pairs = set()
        for _, row in pc_df.iterrows():
            src = row["source"]
            tgt = row["target"]
            if tgt.startswith("next_"):
                tgt = tgt[5:]  # strip "next_"
            pc_pairs.add((src, tgt))

        # DYNOTEARS drug edges: (source_var, target_var)
        dyno_pairs = set()
        for _, row in drug_df.iterrows():
            src_var = row.get("source_var", row["source"])
            tgt_var = row.get("target_var", row["target"])
            dyno_pairs.add((src_var, tgt_var))

        in_both = pc_pairs & dyno_pairs
        pc_only = pc_pairs - dyno_pairs
        dyno_only = dyno_pairs - pc_pairs

        log.info("")
        log.info("=" * 70)
        log.info("COMPARISON WITH PC (drug -> lab edges)")
        log.info("=" * 70)
        log.info("  PC found: %d drug edges", len(pc_pairs))
        log.info("  DYNOTEARS found: %d drug edges", len(dyno_pairs))
        log.info("  Agreement (%d): %s", len(in_both),
                 ", ".join(f"{s}->{t}" for s, t in sorted(in_both)) or "none")
        log.info("  PC only   (%d): %s", len(pc_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(pc_only)) or "none")
        log.info("  DYNO only (%d): %s", len(dyno_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(dyno_only)) or "none")

    # ── Also compare with NOTEARS ─────────────────────────────────────
    notears_path = PROJECT_ROOT / "reports" / "causal_v3" / "step_b_notears" / "drug_edges_lambda_0p0100.csv"
    if notears_path.exists() and len(drug_df) > 0:
        nt_df = pd.read_csv(notears_path)
        nt_pairs = set()
        for _, row in nt_df.iterrows():
            src = row["source"]
            tgt = row["target"]
            if tgt.startswith("next_"):
                tgt = tgt[5:]
            nt_pairs.add((src, tgt))

        dyno_pairs_set = set()
        for _, row in drug_df.iterrows():
            src_var = row.get("source_var", row["source"])
            tgt_var = row.get("target_var", row["target"])
            dyno_pairs_set.add((src_var, tgt_var))

        in_both_nt = nt_pairs & dyno_pairs_set
        log.info("")
        log.info("COMPARISON WITH NOTEARS (lambda=0.01):")
        log.info("  NOTEARS found: %d drug edges", len(nt_pairs))
        log.info("  Agreement (%d): %s", len(in_both_nt),
                 ", ".join(f"{s}->{t}" for s, t in sorted(in_both_nt)) or "none")

    log.info("")
    log.info("Step B DYNOTEARS complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f seconds (%.1f min)", elapsed, elapsed / 60)


if __name__ == "__main__":
    main()
