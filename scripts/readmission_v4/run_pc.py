"""Step A: PC causal discovery on V4 readmission dataset.

Uses variable selection results (double selection) to reduce the confounder
set before running PC. Tier background knowledge enforces causal ordering.

Tiers:
  Tier 0-2 (confounders): selected by double selection — age, severity,
                           chronic flags, labs, service, etc.
  Tier 3   (actions):     8 consults + 9 discharge meds (all kept)
  Tier 4   (outcome):     readmit_30d

Background knowledge: edges can only go forward across tiers.
  - No action -> confounder
  - No outcome -> anything

Key output: which actions have a directed/undirected edge into readmit_30d
after conditioning on the full confounder set.

Usage:
    # Smoke test (5k rows, fast)
    python scripts/readmission_v4/run_pc.py --sample-only

    # Full run (342k rows, alpha=0.01, max_cond_set=3)
    python scripts/readmission_v4/run_pc.py

    # Stricter alpha
    python scripts/readmission_v4/run_pc.py --alpha 0.001

Outputs (--report-dir):
    edges.csv              -- all edges in the discovered graph
    action_edges.csv       -- action -> readmit_30d edges only
    readmit_parents.txt    -- parents of readmit_30d (the key finding)
    run_log.txt            -- full log with timing
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

# ---------------------------------------------------------------------------
# Variable tiers — loaded from variable selection JSON at runtime
# ---------------------------------------------------------------------------

OUTCOME_COL = "readmit_30d"

CATEGORICAL_COLS = [
    "gender", "insurance_cat", "race_cat", "dc_location",
    "admission_type", "first_service", "discharge_service",
]


# ---------------------------------------------------------------------------
# Background knowledge
# ---------------------------------------------------------------------------

def build_background_knowledge(nodes, var_names, confounders, actions, outcome):
    """Forbid edges that go backward across tiers.

    Tier 0-2 (confounders) -> Tier 3 (actions) -> Tier 4 (outcome)

    Forbidden:
      - action  -> confounder
      - outcome -> confounder
      - outcome -> action
    """
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge

    bk = BackgroundKnowledge()
    node_map = {var_names[i]: nodes[i] for i in range(len(var_names))}

    confounder_set = set(confounders)
    action_set     = set(actions)

    for var in var_names:
        if var not in node_map:
            continue
        # Actions cannot cause confounders
        if var in action_set:
            for conf in confounders:
                if conf in node_map:
                    bk.add_forbidden_by_node(node_map[var], node_map[conf])
        # Outcome cannot cause anything
        if var == outcome:
            for other in var_names:
                if other != outcome and other in node_map:
                    bk.add_forbidden_by_node(node_map[outcome], node_map[other])

    return bk


# ---------------------------------------------------------------------------
# PC runner
# ---------------------------------------------------------------------------

def run_pc(data, var_names, alpha, max_cond_set,
           confounders, actions, outcome):
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    n_vars = len(var_names)
    log.info(
        "PC: %d variables, %d rows, alpha=%.1e, max_cond_set=%d",
        n_vars, len(data), alpha, max_cond_set,
    )

    t0 = time.time()

    # First pass without BK to get node objects
    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=var_names, verbose=False,
        depth=max_cond_set,
    )
    bk = build_background_knowledge(
        pc_result.G.nodes, var_names, confounders, actions, outcome
    )

    # Second pass with background knowledge
    pc_result = pc(
        data, alpha=alpha, indep_test=fisherz,
        node_names=var_names, verbose=False,
        depth=max_cond_set,
        background_knowledge=bk,
    )

    elapsed = time.time() - t0
    log.info("PC completed in %.1f seconds (%.1f minutes)", elapsed, elapsed / 60)
    return pc_result, elapsed


# ---------------------------------------------------------------------------
# Edge extraction
# ---------------------------------------------------------------------------

def extract_edges(pc_result, var_names):
    adj = pc_result.G.graph
    n   = len(var_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            if adj[i, j] == 0 and adj[j, i] == 0:
                continue
            ni, nj = var_names[i], var_names[j]

            if adj[i, j] == -1 and adj[j, i] == 1:
                rows.append({"source": ni, "target": nj, "edge_type": "directed"})
            elif adj[i, j] == 1 and adj[j, i] == -1:
                rows.append({"source": nj, "target": ni, "edge_type": "directed"})
            elif adj[i, j] == -1 and adj[j, i] == -1:
                rows.append({"source": ni, "target": nj, "edge_type": "undirected"})
            else:
                rows.append({
                    "source": ni, "target": nj,
                    "edge_type": f"other({adj[i,j]},{adj[j,i]})",
                })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "edge_type"]
    )


def extract_action_edges(edge_df, actions, outcome):
    """Filter to edges involving actions <-> readmit_30d."""
    action_set = set(actions)
    mask = (
        (edge_df["source"].isin(action_set) & (edge_df["target"] == outcome))
        | ((edge_df["source"] == outcome) & edge_df["target"].isin(action_set))
        | (edge_df["source"].isin(action_set) & edge_df["target"].isin(action_set))
    )
    return edge_df[mask].copy()


def extract_readmit_parents(edge_df, outcome):
    """Directed and undirected neighbors of readmit_30d."""
    directed = edge_df[
        (edge_df["target"] == outcome) & (edge_df["edge_type"] == "directed")
    ]["source"].tolist()

    undirected = edge_df[
        (edge_df["edge_type"] == "undirected")
        & ((edge_df["source"] == outcome) | (edge_df["target"] == outcome))
    ].apply(
        lambda r: r["target"] if r["source"] == outcome else r["source"], axis=1
    ).tolist()

    return directed, undirected


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df, var_names):
    """Label-encode categoricals, impute missing, return float64 array."""
    X = df[var_names].copy()

    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category").cat.codes.astype(float)
            X.loc[X[col] == -1, col] = np.nan  # -1 = was NaN before encoding

    # Impute: median for continuous, 0 for binary flags
    for col in X.columns:
        if X[col].isna().any():
            if X[col].nunique(dropna=True) <= 2:
                X[col] = X[col].fillna(0)
            else:
                X[col] = X[col].fillna(X[col].median())

    return X.values.astype(np.float64)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="V4 PC causal discovery on readmission dataset."
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "readmission_v4_admissions.csv"),
    )
    parser.add_argument(
        "--varsel-json",
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "variable_selection.json"),
        help="Variable selection JSON from variable_selection.py",
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "readmission_v4" / "pc"),
    )
    parser.add_argument(
        "--max-confounders", type=int, default=None,
        help="Keep only top-N confounders by selection frequency (default: all)",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.01,
        help="PC independence test significance level (default: 0.01)",
    )
    parser.add_argument(
        "--max-cond-set", type=int, default=3,
        help="Max conditioning set size (default: 3)",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Smoke test: run on 5k rows only",
    )
    parser.add_argument(
        "--sample-n", type=int, default=5_000,
        help="Sample size for --sample-only (default: 5000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
    )
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    if args.sample_only:
        report_dir = report_dir.parent / "pc_smoke"
    report_dir.mkdir(parents=True, exist_ok=True)

    # Logging to file + stdout
    log_path = report_dir / "run_log.txt"
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=handlers,
    )

    log.info("=" * 70)
    log.info("V4 PC Causal Discovery")
    log.info("=" * 70)
    log.info("alpha=%.1e | max_cond_set=%d | sample_only=%s",
             args.alpha, args.max_cond_set, args.sample_only)

    # ── Load variable selection ──────────────────────────────────────
    log.info("Loading variable selection: %s", args.varsel_json)
    with open(args.varsel_json) as f:
        varsel = json.load(f)

    confounders = varsel["selected_confounders"]
    if args.max_confounders is not None:
        confounders = confounders[:args.max_confounders]
        log.info("Truncated confounders to top %d by selection frequency", args.max_confounders)
    actions     = varsel["action_cols"]
    outcome     = varsel["outcome_col"]
    var_names   = confounders + actions + [outcome]

    log.info("Confounders: %d | Actions: %d | Outcome: %s",
             len(confounders), len(actions), outcome)
    log.info("Total variables for PC: %d", len(var_names))

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows: %d", len(train))

    if args.sample_only:
        train = train.sample(n=min(args.sample_n, len(train)),
                             random_state=args.seed)
        log.info("Smoke test sample: %d rows", len(train))

    # ── Drop zero-variance columns ────────────────────────────────────
    zero_var = [c for c in var_names if c != outcome and train[c].nunique(dropna=True) <= 1]
    if zero_var:
        log.info("Dropping zero-variance columns: %s", zero_var)
        var_names = [v for v in var_names if v not in zero_var]
        actions   = [a for a in actions   if a not in zero_var]

    # ── Preprocess ───────────────────────────────────────────────────
    log.info("Preprocessing (label-encode categoricals, impute missing) ...")
    data = preprocess(train, var_names)
    log.info("Data matrix: %d rows x %d columns", data.shape[0], data.shape[1])

    # ── Run PC ───────────────────────────────────────────────────────
    pc_result, elapsed = run_pc(
        data, var_names, alpha=args.alpha,
        max_cond_set=args.max_cond_set,
        confounders=confounders, actions=actions, outcome=outcome,
    )

    # ── Extract edges ────────────────────────────────────────────────
    edge_df       = extract_edges(pc_result, var_names)
    action_edge_df = extract_action_edges(edge_df, actions, outcome)
    directed_parents, undirected_partners = extract_readmit_parents(edge_df, outcome)

    log.info("")
    log.info("=" * 70)
    log.info("RESULTS")
    log.info("=" * 70)
    log.info("Total edges in graph:   %d", len(edge_df))
    log.info("Edges involving actions: %d", len(action_edge_df))
    log.info("")
    log.info("PARENTS OF readmit_30d:")
    log.info("  Directed parents (%d):", len(directed_parents))
    for p in sorted(directed_parents):
        tier = "ACTION" if p in set(actions) else "CONFOUNDER"
        log.info("    [%s]  %s --> readmit_30d", tier, p)
    log.info("  Undirected neighbors (%d):", len(undirected_partners))
    for p in sorted(undirected_partners):
        tier = "ACTION" if p in set(actions) else "CONFOUNDER"
        log.info("    [%s]  %s --- readmit_30d", tier, p)

    # ── Save outputs ─────────────────────────────────────────────────
    edge_df.to_csv(report_dir / "edges.csv", index=False)
    log.info("Saved: edges.csv (%d edges)", len(edge_df))

    action_edge_df.to_csv(report_dir / "action_edges.csv", index=False)
    log.info("Saved: action_edges.csv (%d edges)", len(action_edge_df))

    # Human-readable readmit_30d parent summary
    txt_path = report_dir / "readmit_parents.txt"
    with open(txt_path, "w") as f:
        f.write("V4 PC CAUSAL DISCOVERY — PARENTS OF readmit_30d\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"alpha={args.alpha}  max_cond_set={args.max_cond_set}\n")
        f.write(f"n_rows={data.shape[0]}  n_variables={data.shape[1]}\n")
        f.write(f"runtime={elapsed:.1f}s ({elapsed/60:.1f} min)\n\n")

        f.write("DIRECTED PARENTS\n")
        f.write("-" * 40 + "\n")
        if directed_parents:
            for p in sorted(directed_parents):
                tier = "ACTION" if p in set(actions) else "CONFOUNDER"
                f.write(f"  [{tier}]  {p} --> readmit_30d\n")
        else:
            f.write("  (none)\n")

        f.write("\nUNDIRECTED NEIGHBORS\n")
        f.write("-" * 40 + "\n")
        if undirected_partners:
            for p in sorted(undirected_partners):
                tier = "ACTION" if p in set(actions) else "CONFOUNDER"
                f.write(f"  [{tier}]  {p} --- readmit_30d\n")
        else:
            f.write("  (none)\n")

        f.write("\nALL ACTION EDGES\n")
        f.write("-" * 40 + "\n")
        if len(action_edge_df) > 0:
            for _, row in action_edge_df.iterrows():
                arrow = "-->" if row["edge_type"] == "directed" else "---"
                f.write(f"  {row['source']} {arrow} {row['target']}\n")
        else:
            f.write("  (none)\n")

    log.info("Saved: readmit_parents.txt")
    log.info("")
    log.info("PC complete. Results in: %s", report_dir)


if __name__ == "__main__":
    main()
