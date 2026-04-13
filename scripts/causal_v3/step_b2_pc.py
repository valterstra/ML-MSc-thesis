"""Step B2: PC causal discovery on focused 11-variable set.

Identical algorithm to step_b_pc.py but uses the reduced variable set
from step_b2_vars.py (3 states + 3 actions + 2 confounders + 3 next-states).

With only 11 variables, PC runs in seconds. We use more data (full train split)
and test multiple alpha levels for robustness.

Output: reports/causal_v3/focused_b/pc/
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_b2_vars import ALL_VARS, TIER2_ACTION, TIER3_NEXT, TIERS, VAR_TIER
from step_b_pc import extract_edges, extract_parent_sets

log = logging.getLogger(__name__)

REPORT_DIR = PROJECT_ROOT / "reports" / "causal_v3" / "focused_b" / "pc"
CSV_PATH   = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
ALPHAS     = [0.05, 0.01, 0.001]
SAMPLE_N   = 0   # 0 = use full train split (fast enough at 11 vars)
SEED       = 42


def extract_drug_edges(edge_df):
    drug_set = set(TIER2_ACTION)
    next_set  = set(TIER3_NEXT)
    mask = (
        (edge_df["source"].isin(drug_set) & edge_df["target"].isin(next_set)) |
        (edge_df["source"].isin(next_set) & edge_df["target"].isin(drug_set))
    )
    return edge_df[mask].copy()


def build_bk(nodes, var_names):
    """Build temporal background knowledge using local VAR_TIER."""
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    bk = BackgroundKnowledge()
    node_map = {var_names[i]: nodes[i] for i in range(len(var_names))}
    for ni in var_names:
        for nj in var_names:
            if ni != nj and VAR_TIER[ni] > VAR_TIER[nj]:
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


def run_pc(data, var_names, alpha):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    log.info("PC: %d vars, %d rows, alpha=%.3f", len(var_names), len(data), alpha)
    t0 = time.time()
    res = pc(data, alpha=alpha, indep_test=fisherz, node_names=var_names, verbose=False)
    bk  = build_bk(res.G.nodes, var_names)
    res = pc(data, alpha=alpha, indep_test=fisherz, node_names=var_names,
             verbose=False, background_knowledge=bk)
    log.info("  Done in %.1fs", time.time() - t0)
    return res


def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(REPORT_DIR / "run_log.txt"), mode="w", encoding="utf-8"),
        ],
    )

    log.info("=" * 60)
    log.info("Step B2 PC  |  variables: %s", ALL_VARS)
    log.info("=" * 60)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["split"] == "train"]
    df_sub = df[ALL_VARS].dropna()
    log.info("Train rows after dropna: %d", len(df_sub))

    if SAMPLE_N > 0 and SAMPLE_N < len(df_sub):
        df_sub = df_sub.sample(n=SAMPLE_N, random_state=SEED)
        log.info("Sampled: %d rows", SAMPLE_N)

    data = df_sub.values.astype(np.float64)

    all_results = {}
    for alpha in ALPHAS:
        log.info("\n--- alpha=%.3f ---", alpha)
        res      = run_pc(data, ALL_VARS, alpha)
        edge_df  = extract_edges(res, ALL_VARS)
        drug_df  = extract_drug_edges(edge_df)
        log.info("  Total edges: %d | Drug edges: %d", len(edge_df), len(drug_df))
        for _, r in drug_df.iterrows():
            arrow = "-->" if r["edge_type"] == "directed" else "---"
            log.info("    %s %s %s", r["source"], arrow, r["target"])
        all_results[alpha] = {"edge_df": edge_df, "drug_df": drug_df}

    # Save primary (most lenient) alpha
    primary = all_results[ALPHAS[0]]
    primary["edge_df"].to_csv(REPORT_DIR / "edges.csv", index=False)
    primary["drug_df"].to_csv(REPORT_DIR / "drug_edges.csv", index=False)
    extract_parent_sets(primary["edge_df"], TIER3_NEXT).to_csv(
        REPORT_DIR / "parent_sets.csv", index=False)

    # Alpha robustness table
    all_pairs = set()
    for a in all_results:
        for _, r in all_results[a]["drug_df"].iterrows():
            all_pairs.add((r["source"], r["target"]))
    rows = []
    for src, tgt in sorted(all_pairs):
        row = {"source": src, "target": tgt}
        for a in ALPHAS:
            found = any(
                r["source"] == src and r["target"] == tgt
                for _, r in all_results[a]["drug_df"].iterrows()
            )
            row[f"alpha={a}"] = "YES" if found else "-"
        rows.append(row)
    rob_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    rob_df.to_csv(REPORT_DIR / "alpha_robustness.csv", index=False)

    log.info("\n=== ALPHA ROBUSTNESS ===")
    log.info("\n%s", rob_df.to_string(index=False) if not rob_df.empty else "(no drug edges found)")
    log.info("\nStep B2 PC complete. Results in: %s", REPORT_DIR)


if __name__ == "__main__":
    main()
