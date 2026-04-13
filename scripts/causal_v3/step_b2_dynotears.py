"""Step B2: DYNOTEARS temporal causal discovery on focused variable set.

Uses 7 time-varying variables (charlson_score excluded -- static within admission):
  day_of_stay (time-varying confounder)
  creatinine, potassium, wbc       (state)
  diuretic_active, electrolyte_active, antibiotic_active  (action)

Output: reports/causal_v3/focused_b/dynotears/
"""

from __future__ import annotations

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

from step_b2_vars import TIER1_STATE, TIER2_ACTION

log = logging.getLogger(__name__)

# charlson_score is constant within admission -- breaks lag model; use day_of_stay only
DYNO_CONFOUNDER = ["day_of_stay"]
DYNO_STATE      = list(TIER1_STATE)   # creatinine, potassium, wbc
DYNO_ACTION     = list(TIER2_ACTION)  # diuretic_active, electrolyte_active, antibiotic_active
DYNO_VARS       = DYNO_CONFOUNDER + DYNO_STATE + DYNO_ACTION  # 7 total

REPORT_DIR  = PROJECT_ROOT / "reports" / "causal_v3" / "focused_b" / "dynotears"
CSV_PATH    = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
N_ADMISSIONS = 0       # 0 = all train admissions
LAG          = 1
LAMBDA_W     = 0.05
LAMBDA_A     = 0.05
W_THRESHOLD  = 0.01
MAX_ITER     = 100
SEED         = 42


# ── Data helpers ─────────────────────────────────────────────────────────

def triplets_to_time_series(df: pd.DataFrame, max_admissions: int = 0,
                             seed: int = 42) -> list[np.ndarray]:
    hadm_ids = df["hadm_id"].unique()
    if max_admissions > 0 and max_admissions < len(hadm_ids):
        rng = np.random.default_rng(seed)
        hadm_ids = rng.choice(hadm_ids, size=max_admissions, replace=False)
        df = df[df["hadm_id"].isin(hadm_ids)]

    segments = []
    n_admissions = 0
    n_gaps = 0

    for _, group in df.groupby("hadm_id"):
        group = group.sort_values("day_of_stay")
        days = group["day_of_stay"].values
        values = group[DYNO_VARS].values.astype(np.float64)

        if len(days) < 2:
            continue

        gap_indices = np.where(np.diff(days) > 1)[0]
        if len(gap_indices) > 0:
            n_gaps += 1

        cuts = [0] + list(gap_indices + 1) + [len(days)]
        for start, end in zip(cuts[:-1], cuts[1:]):
            seg = values[start:end]
            if seg.shape[0] >= 2:
                segments.append(seg)

        n_admissions += 1

    seg_lens = [s.shape[0] for s in segments]
    log.info("Built %d segments from %d admissions (%d had gaps)",
             len(segments), n_admissions, n_gaps)
    log.info("  Lengths: min=%d, median=%d, mean=%.1f, max=%d",
             min(seg_lens), int(np.median(seg_lens)),
             np.mean(seg_lens), max(seg_lens))
    return segments


def build_tabu_edges() -> list[tuple[int, int, int]]:
    """Forbid same-day drug->lab edges (W matrix only; A is free)."""
    tabu = []
    drug_idx = [DYNO_VARS.index(v) for v in DYNO_ACTION]
    lab_idx  = [DYNO_VARS.index(v) for v in DYNO_STATE]
    for d in drug_idx:
        for l in lab_idx:
            tabu.append((0, d, l))
    log.info("Tabu edges (lag=0, drug->state): %d", len(tabu))
    return tabu


def extract_edges(W: np.ndarray, A: np.ndarray, var_names: list[str],
                  p: int, threshold: float) -> pd.DataFrame:
    d = len(var_names)
    rows = []
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            w = W[i, j]
            if abs(w) >= threshold:
                rows.append({"source": var_names[j], "target": var_names[i],
                             "weight": round(float(w), 6), "lag": 0, "matrix": "W"})
    for lag_k in range(p):
        for i in range(d):
            for j in range(d):
                w = A[lag_k * d + i, j]
                if abs(w) >= threshold:
                    rows.append({"source": var_names[i], "target": var_names[j],
                                 "source_var": var_names[i], "target_var": var_names[j],
                                 "weight": round(float(w), 6), "lag": lag_k + 1,
                                 "matrix": "A"})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "lag", "matrix"])
    return df.sort_values("weight", key=abs, ascending=False).reset_index(drop=True)


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"step_b2_dynotears_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
            logging.FileHandler(str(REPORT_DIR / "run_log.txt"), mode="w", encoding="utf-8"),
        ],
    )

    log.info("=" * 60)
    log.info("Step B2 DYNOTEARS -- focused 7-variable set")
    log.info("  vars: %s", DYNO_VARS)
    log.info("  n_admissions=%s, lag=%d, lambda_w=%.3f, lambda_a=%.3f",
             N_ADMISSIONS or "ALL", LAG, LAMBDA_W, LAMBDA_A)
    log.info("=" * 60)

    # Load
    log.info("Loading: %s", CSV_PATH)
    df = pd.read_csv(str(CSV_PATH), low_memory=False)
    df = df[df["split"] == "train"]
    log.info("Train rows: %d", len(df))

    missing = [c for c in DYNO_VARS + ["hadm_id", "day_of_stay"] if c not in df.columns]
    if missing:
        log.error("Missing columns: %s", missing)
        sys.exit(1)

    n_before = len(df)
    df = df.dropna(subset=DYNO_VARS)
    log.info("After dropna: %d rows (dropped %d)", len(df), n_before - len(df))

    # Build time series
    segments = triplets_to_time_series(df, max_admissions=N_ADMISSIONS, seed=SEED)
    if not segments:
        log.error("No usable segments found!")
        sys.exit(1)

    # Standardize
    all_data = np.concatenate(segments, axis=0)
    scaler = StandardScaler().fit(all_data)
    std_segments = [scaler.transform(s) for s in segments]
    log.info("Standardized %d rows x %d cols", all_data.shape[0], all_data.shape[1])

    # Build X, Xlags
    from careai.causal_v3.dynotears_solver import build_X_Xlags
    X, Xlags = build_X_Xlags(std_segments, p=LAG)
    log.info("DYNOTEARS input: X=%s, Xlags=%s", X.shape, Xlags.shape)

    tabu = build_tabu_edges()

    # Run
    log.info("Running DYNOTEARS ...")
    from careai.causal_v3.dynotears_solver import dynotears_solve
    t0 = time.time()
    W, A = dynotears_solve(X, Xlags,
                           lambda_w=LAMBDA_W, lambda_a=LAMBDA_A,
                           max_iter=MAX_ITER, h_tol=1e-8, w_threshold=0.0,
                           tabu_edges=tabu)
    elapsed = time.time() - t0
    log.info("Completed in %.1f s", elapsed)

    # Save matrices
    W_df = pd.DataFrame(W, index=DYNO_VARS, columns=DYNO_VARS)
    W_df.to_csv(REPORT_DIR / "W_matrix.csv")
    a_row_names = [f"{v}_lag{k+1}" for k in range(LAG) for v in DYNO_VARS]
    A_df = pd.DataFrame(A, index=a_row_names, columns=DYNO_VARS)
    A_df.to_csv(REPORT_DIR / "A_matrix.csv")
    log.info("Saved W_matrix.csv, A_matrix.csv")

    # Extract and save edges
    edge_df = extract_edges(W, A, DYNO_VARS, p=LAG, threshold=W_THRESHOLD)
    edge_df.to_csv(REPORT_DIR / "all_edges.csv", index=False)

    drug_mask = (
        (edge_df["matrix"] == "A") &
        (edge_df["source"].isin(set(DYNO_ACTION))) &
        (edge_df["target"].isin(set(DYNO_STATE)))
    )
    drug_df = edge_df[drug_mask].copy()
    drug_df.to_csv(REPORT_DIR / "drug_edges.csv", index=False)

    # Summary
    log.info("")
    log.info("=" * 60)
    log.info("DRUG -> STATE EDGES (A matrix, lagged)")
    log.info("=" * 60)
    if drug_df.empty:
        log.info("  None found at w_threshold=%.3f", W_THRESHOLD)
    else:
        for _, row in drug_df.iterrows():
            log.info("  %-25s --> %-15s  w=%+.4f  (lag %d)",
                     row["source"], row["target"], row["weight"], row["lag"])

    w_edges = edge_df[edge_df["matrix"] == "W"]
    if len(w_edges) > 0:
        log.info("")
        log.info("TOP CONTEMPORANEOUS EDGES (W):")
        for _, row in w_edges.head(10).iterrows():
            log.info("  %-20s --> %-15s  w=%+.4f", row["source"], row["target"], row["weight"])

    log.info("")
    log.info("Step B2 DYNOTEARS complete. Results in: %s", REPORT_DIR)
    log.info("Log: %s", log_path)


if __name__ == "__main__":
    main()
