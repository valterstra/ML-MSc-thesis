"""Step B3: DYNOTEARS temporal causal discovery on ultra-focused 8-variable set.

Time-varying vars (6): day_of_stay, creatinine, potassium,
                        diuretic_active, antibiotic_active
charlson_score excluded (static within admission).

Output: reports/causal_v3/focused_b3/dynotears/
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_b3_vars import TIER1_STATE, TIER2_ACTION

log = logging.getLogger(__name__)

DYNO_CONFOUNDER = ["day_of_stay"]
DYNO_STATE      = list(TIER1_STATE)   # creatinine, potassium
DYNO_ACTION     = list(TIER2_ACTION)  # diuretic_active, antibiotic_active
DYNO_VARS       = DYNO_CONFOUNDER + DYNO_STATE + DYNO_ACTION  # 5 total

REPORT_DIR   = PROJECT_ROOT / "reports" / "causal_v3" / "focused_b3" / "dynotears"
CSV_PATH     = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
N_ADMISSIONS = 0
LAG          = 1
LAMBDA_W     = 0.05
LAMBDA_A     = 0.05
W_THRESHOLD  = 0.01
MAX_ITER     = 100
SEED         = 42


def triplets_to_time_series(df, max_admissions=0, seed=42):
    hadm_ids = df["hadm_id"].unique()
    if max_admissions > 0 and max_admissions < len(hadm_ids):
        rng = np.random.default_rng(seed)
        hadm_ids = rng.choice(hadm_ids, size=max_admissions, replace=False)
        df = df[df["hadm_id"].isin(hadm_ids)]
    segments = []
    n_admissions = n_gaps = 0
    for _, group in df.groupby("hadm_id"):
        group = group.sort_values("day_of_stay")
        days   = group["day_of_stay"].values
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
             min(seg_lens), int(np.median(seg_lens)), np.mean(seg_lens), max(seg_lens))
    return segments


def build_tabu_edges():
    tabu = []
    drug_idx = [DYNO_VARS.index(v) for v in DYNO_ACTION]
    lab_idx  = [DYNO_VARS.index(v) for v in DYNO_STATE]
    for d in drug_idx:
        for l in lab_idx:
            tabu.append((0, d, l))
    log.info("Tabu edges (lag=0, drug->state): %d", len(tabu))
    return tabu


def extract_edges(W, A, var_names, p, threshold):
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
                                 "weight": round(float(w), 6), "lag": lag_k + 1, "matrix": "A"})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "lag", "matrix"])
    return df.sort_values("weight", key=abs, ascending=False).reset_index(drop=True)


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"step_b3_dynotears_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
                                  logging.FileHandler(str(REPORT_DIR / "run_log.txt"), mode="w", encoding="utf-8")])
    log.info("=" * 60)
    log.info("Step B3 DYNOTEARS  |  vars: %s", DYNO_VARS)
    log.info("  n_admissions=%s, lag=%d, lambda_w=%.3f, lambda_a=%.3f",
             N_ADMISSIONS or "ALL", LAG, LAMBDA_W, LAMBDA_A)
    log.info("=" * 60)

    df = pd.read_csv(str(CSV_PATH), low_memory=False)
    df = df[df["split"] == "train"]
    log.info("Train rows: %d", len(df))
    n_before = len(df)
    df = df.dropna(subset=DYNO_VARS)
    log.info("After dropna: %d rows (dropped %d)", len(df), n_before - len(df))

    segments = triplets_to_time_series(df, max_admissions=N_ADMISSIONS, seed=SEED)
    all_data = np.concatenate(segments, axis=0)
    scaler   = StandardScaler().fit(all_data)
    std_segs = [scaler.transform(s) for s in segments]
    log.info("Standardized %d rows x %d cols", all_data.shape[0], all_data.shape[1])

    from careai.causal_v3.dynotears_solver import build_X_Xlags, dynotears_solve
    X, Xlags = build_X_Xlags(std_segs, p=LAG)
    log.info("DYNOTEARS input: X=%s, Xlags=%s", X.shape, Xlags.shape)

    tabu = build_tabu_edges()
    log.info("Running DYNOTEARS ...")
    t0 = time.time()
    W, A = dynotears_solve(X, Xlags, lambda_w=LAMBDA_W, lambda_a=LAMBDA_A,
                           max_iter=MAX_ITER, h_tol=1e-8, w_threshold=0.0,
                           tabu_edges=tabu)
    log.info("Completed in %.1f s", time.time() - t0)

    pd.DataFrame(W, index=DYNO_VARS, columns=DYNO_VARS).to_csv(REPORT_DIR / "W_matrix.csv")
    a_rows = [f"{v}_lag{k+1}" for k in range(LAG) for v in DYNO_VARS]
    pd.DataFrame(A, index=a_rows, columns=DYNO_VARS).to_csv(REPORT_DIR / "A_matrix.csv")

    edge_df = extract_edges(W, A, DYNO_VARS, p=LAG, threshold=W_THRESHOLD)
    edge_df.to_csv(REPORT_DIR / "all_edges.csv", index=False)

    drug_mask = (edge_df["matrix"] == "A") & \
                (edge_df["source"].isin(set(DYNO_ACTION))) & \
                (edge_df["target"].isin(set(DYNO_STATE)))
    drug_df = edge_df[drug_mask].copy()
    drug_df.to_csv(REPORT_DIR / "drug_edges.csv", index=False)

    log.info("\n=== DRUG -> STATE EDGES (A matrix, lagged) ===")
    if drug_df.empty:
        log.info("  None found at w_threshold=%.3f", W_THRESHOLD)
    else:
        for _, r in drug_df.iterrows():
            log.info("  %-25s --> %-15s  w=%+.4f  (lag %d)",
                     r["source"], r["target"], r["weight"], r["lag"])

    w_edges = edge_df[edge_df["matrix"] == "W"]
    if len(w_edges) > 0:
        log.info("\nTOP CONTEMPORANEOUS EDGES (W):")
        for _, r in w_edges.head(10).iterrows():
            log.info("  %-20s --> %-15s  w=%+.4f", r["source"], r["target"], r["weight"])

    log.info("\nStep B3 DYNOTEARS complete. Results in: %s", REPORT_DIR)


if __name__ == "__main__":
    main()
