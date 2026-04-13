"""Step B3: NOTEARS causal discovery on ultra-focused 8-variable set.

Output: reports/causal_v3/focused_b3/notears/
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

from step_b3_vars import ALL_VARS, TIER2_ACTION, TIER3_NEXT, VAR_TIER
from step_b_notears import extract_edges, extract_drug_edges

log = logging.getLogger(__name__)
REPORT_DIR = PROJECT_ROOT / "reports" / "causal_v3" / "focused_b3" / "notears"
CSV_PATH   = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
LAMBDAS    = [0.05, 0.01, 0.005]
SAMPLE_N   = 100_000
W_THRESH   = 0.01
SEED       = 42


def build_temporal_mask(var_names):
    n = len(var_names)
    mask = np.ones((n, n))
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if VAR_TIER[vj] >= VAR_TIER[vi]:
                mask[i, j] = 0.0
    return mask


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(str(REPORT_DIR / "run_log.txt"), mode="w", encoding="utf-8")])
    log.info("=" * 60)
    log.info("Step B3 NOTEARS  |  variables: %s", ALL_VARS)
    log.info("=" * 60)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["split"] == "train"][ALL_VARS].dropna()
    log.info("Rows after dropna: %d", len(df))
    if SAMPLE_N > 0 and SAMPLE_N < len(df):
        df = df.sample(n=SAMPLE_N, random_state=SEED)
        log.info("Sampled: %d rows", SAMPLE_N)

    X = StandardScaler().fit_transform(df.values.astype(np.float64))
    temporal_mask = build_temporal_mask(ALL_VARS)
    log.info("Temporal mask: %d forbidden entries", int((temporal_mask == 0).sum()))

    from notears.linear import notears_linear
    all_results = {}
    for lam in sorted(LAMBDAS, reverse=True):
        log.info("\n--- lambda=%.4f ---", lam)
        t0    = time.time()
        W_raw = notears_linear(X, lambda1=lam, loss_type="l2", w_threshold=0.0)
        W     = W_raw * temporal_mask
        log.info("  Done in %.1fs", time.time() - t0)

        edge_df = extract_edges(W, ALL_VARS, W_THRESH)
        drug_df = extract_drug_edges(edge_df)
        log.info("  Total edges: %d | Drug edges: %d", len(edge_df), len(drug_df))
        for _, r in drug_df.iterrows():
            log.info("    %-25s --> %-25s  w=%+.4f", r["source"], r["target"], r["weight"])

        lam_str = f"{lam:.4f}".replace(".", "p")
        edge_df.to_csv(REPORT_DIR / f"edges_lambda_{lam_str}.csv", index=False)
        drug_df.to_csv(REPORT_DIR / f"drug_edges_lambda_{lam_str}.csv", index=False)
        all_results[lam] = {"edge_df": edge_df, "drug_df": drug_df}

    all_pairs = set()
    for lam in all_results:
        for _, r in all_results[lam]["drug_df"].iterrows():
            s = r["source"] if r["source"] in set(TIER2_ACTION) else r["target"]
            t = r["target"] if r["source"] in set(TIER2_ACTION) else r["source"]
            all_pairs.add((s, t))
    rows = []
    for src, tgt in sorted(all_pairs):
        row = {"source": src, "target": tgt}
        for lam in sorted(LAMBDAS, reverse=True):
            match = all_results[lam]["drug_df"]
            match = match[((match["source"] == src) & (match["target"] == tgt)) |
                          ((match["source"] == tgt) & (match["target"] == src))]
            row[f"lam={lam}"] = f"{match.iloc[0]['weight']:+.4f}" if len(match) else "-"
        rows.append(row)
    sweep_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    sweep_df.to_csv(REPORT_DIR / "lambda_sweep.csv", index=False)
    log.info("\n=== LAMBDA SWEEP ===\n%s",
             sweep_df.to_string(index=False) if not sweep_df.empty else "(no drug edges)")
    log.info("\nStep B3 NOTEARS complete. Results in: %s", REPORT_DIR)


if __name__ == "__main__":
    main()
