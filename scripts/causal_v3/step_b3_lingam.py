"""Step B3: DirectLiNGAM causal discovery on ultra-focused 8-variable set.

Output: reports/causal_v3/focused_b3/lingam/
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_b3_vars import ALL_VARS, TIER2_ACTION, TIER3_NEXT, VAR_TIER

log = logging.getLogger(__name__)
REPORT_DIR  = PROJECT_ROOT / "reports" / "causal_v3" / "focused_b3" / "lingam"
CSV_PATH    = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
SAMPLE_N    = 300_000
W_THRESHOLD = 0.05
SEED        = 42


def build_lingam_prior_knowledge(var_names):
    n = len(var_names)
    pk = np.full((n, n), -1, dtype=int)
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            if VAR_TIER[vi] >= VAR_TIER[vj]:
                pk[i, j] = 0
    return pk


def extract_edges(B, var_names, threshold):
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            w = B[i, j]
            if abs(w) >= threshold:
                rows.append({"source": vj, "target": vi,
                              "weight": round(float(w), 6),
                              "abs_weight": round(abs(float(w)), 6)})
    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "abs_weight"])
    return df.sort_values("abs_weight", ascending=False).reset_index(drop=True)


def extract_drug_edges(edge_df):
    drug_set = set(TIER2_ACTION)
    next_set  = set(TIER3_NEXT)
    mask = (
        (edge_df["source"].isin(drug_set) & edge_df["target"].isin(next_set)) |
        (edge_df["source"].isin(next_set) & edge_df["target"].isin(drug_set))
    )
    return edge_df[mask].copy()


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(str(REPORT_DIR / "run_log.txt"), mode="w", encoding="utf-8")])
    log.info("=" * 60)
    log.info("Step B3 LiNGAM  |  variables: %s", ALL_VARS)
    log.info("=" * 60)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["split"] == "train"][ALL_VARS].dropna()
    log.info("Rows after dropna: %d", len(df))
    if SAMPLE_N > 0 and SAMPLE_N < len(df):
        df = df.sample(n=SAMPLE_N, random_state=SEED)
        log.info("Sampled: %d rows", SAMPLE_N)

    X  = df.values.astype(np.float64)
    pk = build_lingam_prior_knowledge(ALL_VARS)
    log.info("Prior knowledge: %d forbidden edges", int((pk == 0).sum()))

    import lingam
    log.info("Running DirectLiNGAM ...")
    t0    = time.time()
    model = lingam.DirectLiNGAM(prior_knowledge=pk)
    model.fit(X)
    log.info("Done in %.1fs", time.time() - t0)

    B       = model.adjacency_matrix_
    edge_df = extract_edges(B, ALL_VARS, W_THRESHOLD)
    drug_df = extract_drug_edges(edge_df)
    log.info("Total edges: %d | Drug edges: %d", len(edge_df), len(drug_df))
    for _, r in drug_df.iterrows():
        log.info("  %-25s --> %-25s  w=%+.4f", r["source"], r["target"], r["weight"])

    edge_df.to_csv(REPORT_DIR / "edges.csv", index=False)
    drug_df.to_csv(REPORT_DIR / "drug_edges.csv", index=False)
    pd.DataFrame(B, index=ALL_VARS, columns=ALL_VARS).to_csv(REPORT_DIR / "weight_matrix.csv")
    log.info("\nStep B3 LiNGAM complete. Results in: %s", REPORT_DIR)


if __name__ == "__main__":
    main()
