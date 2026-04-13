"""Step B3: FCI causal discovery on ultra-focused 8-variable set.

Output: reports/causal_v3/focused_b3/fci/
"""
from __future__ import annotations
import logging, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from step_b3_vars import ALL_VARS, TIER2_ACTION, TIER3_NEXT, TIERS, VAR_TIER

log = logging.getLogger(__name__)
REPORT_DIR = PROJECT_ROOT / "reports" / "causal_v3" / "focused_b3" / "fci"
CSV_PATH   = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
ALPHAS     = [0.05, 0.01, 0.001]
SAMPLE_N   = 0
SEED       = 42


def build_bk(nodes, var_names):
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    bk = BackgroundKnowledge()
    node_map = {var_names[i]: nodes[i] for i in range(len(var_names))}
    for ni in var_names:
        for nj in var_names:
            if ni != nj and VAR_TIER[ni] > VAR_TIER[nj]:
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


def run_fci(data, var_names, alpha):
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    log.info("FCI: %d vars, %d rows, alpha=%.3f", len(var_names), len(data), alpha)
    t0 = time.time()
    g, _ = fci(data, fisherz, alpha, node_names=var_names, verbose=False)
    bk   = build_bk(g.get_nodes(), var_names)
    g, _ = fci(data, fisherz, alpha, node_names=var_names,
               background_knowledge=bk, verbose=False)
    log.info("  Done in %.1fs", time.time() - t0)
    return g


def extract_edges(g, var_names):
    try:
        adj = g.graph
    except AttributeError:
        return pd.DataFrame(columns=["source", "target", "edge_type"])
    n = len(var_names)
    rows = []
    for i in range(n):
        for j in range(i + 1, n):
            ei, ej = adj[i, j], adj[j, i]
            if ei == 0 and ej == 0:
                continue
            ni, nj = var_names[i], var_names[j]
            if   ei == 2 and ej == 1: rows.append({"source": ni, "target": nj, "edge_type": "directed"})
            elif ei == 1 and ej == 2: rows.append({"source": nj, "target": ni, "edge_type": "directed"})
            elif ei == 2 and ej == 2: rows.append({"source": ni, "target": nj, "edge_type": "bidirected"})
            elif ei == 3 and ej == 2: rows.append({"source": ni, "target": nj, "edge_type": "possible_directed"})
            elif ei == 2 and ej == 3: rows.append({"source": nj, "target": ni, "edge_type": "possible_directed"})
            elif ei == 3 and ej == 3: rows.append({"source": ni, "target": nj, "edge_type": "undirected"})
            else: rows.append({"source": ni, "target": nj, "edge_type": f"other({ei},{ej})"})
    return pd.DataFrame(rows) if rows else pd.DataFrame(columns=["source", "target", "edge_type"])


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
    ts = time.strftime("%Y%m%d_%H%M%S")
    log_path = PROJECT_ROOT / "logs" / f"step_b3_fci_{ts}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
                                  logging.FileHandler(str(REPORT_DIR / "run_log.txt"), mode="w", encoding="utf-8")])
    log.info("=" * 60)
    log.info("Step B3 FCI  |  variables: %s", ALL_VARS)
    log.info("=" * 60)

    df = pd.read_csv(CSV_PATH, low_memory=False)
    df = df[df["split"] == "train"][ALL_VARS].dropna()
    log.info("Train rows after dropna: %d", len(df))
    if SAMPLE_N > 0 and SAMPLE_N < len(df):
        df = df.sample(n=SAMPLE_N, random_state=SEED)

    data = df.values.astype(np.float64)
    all_results = {}
    for alpha in ALPHAS:
        log.info("\n--- alpha=%.3f ---", alpha)
        g       = run_fci(data, ALL_VARS, alpha)
        edge_df = extract_edges(g, ALL_VARS)
        drug_df = extract_drug_edges(edge_df)
        log.info("  Total edges: %d | Drug edges: %d", len(edge_df), len(drug_df))
        for _, r in drug_df.iterrows():
            arrow = "-->" if r["edge_type"] == "directed" else f"~{r['edge_type']}~>"
            log.info("    %s %s %s", r["source"], arrow, r["target"])
        all_results[alpha] = {"edge_df": edge_df, "drug_df": drug_df}

    primary = all_results[ALPHAS[0]]
    primary["edge_df"].to_csv(REPORT_DIR / "edges.csv", index=False)
    primary["drug_df"].to_csv(REPORT_DIR / "drug_edges.csv", index=False)

    all_pairs = set()
    for a in all_results:
        for _, r in all_results[a]["drug_df"].iterrows():
            all_pairs.add((r["source"], r["target"]))
    rows = []
    for src, tgt in sorted(all_pairs):
        row = {"source": src, "target": tgt}
        for a in ALPHAS:
            found = any(r["source"] == src and r["target"] == tgt
                        for _, r in all_results[a]["drug_df"].iterrows())
            row[f"alpha={a}"] = "YES" if found else "-"
        rows.append(row)
    rob_df = pd.DataFrame(rows) if rows else pd.DataFrame()
    rob_df.to_csv(REPORT_DIR / "alpha_robustness.csv", index=False)
    log.info("\n=== ALPHA ROBUSTNESS ===\n%s",
             rob_df.to_string(index=False) if not rob_df.empty else "(no drug edges)")
    log.info("\nStep B3 FCI complete. Results in: %s", REPORT_DIR)


if __name__ == "__main__":
    main()
