"""Step B FCI: Fast Causal Inference on V3 triplet dataset.

FCI extends PC to handle hidden confounders (latent variables). Unlike PC,
FCI does NOT assume causal sufficiency -- it can detect when a drug->next
edge is confounded by an unmeasured variable (e.g. physician knowledge).

PAG edge types (what FCI returns instead of a DAG):
  i --> j   adj=-1/+1  Definitely direct effect, no hidden confounder
  i <-> j   adj=-1/-1  Hidden common cause (bidirected): serious confounding
  i o-> j   adj=+2/-1  Possibly confounded (circle = uncertainty)
  i o-o j   adj=+2/+2  Completely uncertain
  i o-  j   adj=+2/+1  Circle-tail

For drug->next edges, the key question is:
  --> (directed):  causal effect with no hidden confounder detected
  <-> (bidirected): hidden confounder -- this drug edge is confounded
  o-> (circle-arrow): direction likely correct but confounding possible

Usage:
    # Smoke test: 30k rows, depth=2 (~few minutes)
    python scripts/causal_v3/step_b_fci.py --sample-n 30000 --max-cond-set 2

    # Scaled up
    python scripts/causal_v3/step_b_fci.py --sample-n 100000 --max-cond-set 2
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

from step_b_pc import (
    ALL_VARS, TIER2_ACTION, TIER3_NEXT,
    build_temporal_background_knowledge,
)

log = logging.getLogger(__name__)

# PAG edge mark encoding in causal-learn
TAIL   =  1
ARROW  = -1
CIRCLE =  2


# ── Edge extraction (PAG-aware) ───────────────────────────────────────────

def decode_mark(m):
    if m == TAIL:   return "-"
    if m == ARROW:  return ">"
    if m == CIRCLE: return "o"
    return "?"


def edge_label(mi, mj):
    """Return human-readable edge label from PAG adjacency marks."""
    left  = "<" if mi == ARROW else ("o" if mi == CIRCLE else "-")
    right = ">" if mj == ARROW else ("o" if mj == CIRCLE else "-")
    return f"{left}--{right}"


def classify_edge(mi, mj):
    """Return semantic edge type for drug->next analysis."""
    if mi == TAIL and mj == ARROW:
        return "directed"           # -->  definite direct effect
    if mi == ARROW and mj == TAIL:
        return "directed_reverse"   # <--  reversed (should not happen with BK)
    if mi == ARROW and mj == ARROW:
        return "bidirected"         # <->  hidden common cause
    if mi == CIRCLE and mj == ARROW:
        return "circle_arrow"       # o->  possibly confounded
    if mi == ARROW and mj == CIRCLE:
        return "arrow_circle"       # <-o
    if mi == CIRCLE and mj == CIRCLE:
        return "circle_circle"      # o-o  fully uncertain
    if mi == CIRCLE and mj == TAIL:
        return "circle_tail"        # o-
    if mi == TAIL and mj == CIRCLE:
        return "tail_circle"        # -o
    return f"other({mi},{mj})"


def extract_fci_edges(G, var_names):
    """Extract all edges from FCI PAG into a DataFrame."""
    adj = G.graph
    n = len(var_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            mi = adj[i, j]   # mark at i (pointing toward j from i's side)
            mj = adj[j, i]   # mark at j (pointing toward i from j's side)

            if mi == 0 and mj == 0:
                continue

            ni, nj = var_names[i], var_names[j]
            etype = classify_edge(mi, mj)
            label = edge_label(mi, mj)

            rows.append({
                "source": ni,
                "target": nj,
                "edge_type": etype,
                "label": label,
                "mark_i": int(mi),
                "mark_j": int(mj),
            })

    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "edge_type", "label", "mark_i", "mark_j"]
    )


def extract_drug_fci_edges(edge_df):
    """Filter to edges involving any drug and any next_* variable."""
    drug_set = set(TIER2_ACTION)
    next_set = set(TIER3_NEXT)

    mask = (
        (edge_df["source"].isin(drug_set) & edge_df["target"].isin(next_set)) |
        (edge_df["source"].isin(next_set) & edge_df["target"].isin(drug_set))
    )
    return edge_df[mask].copy()


# ── FCI runner ────────────────────────────────────────────────────────────

def run_fci(data, var_names, alpha, max_cond_set):
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    log.info(
        "FCI: %d variables, %d rows, alpha=%.1e, max_cond_set=%d",
        len(var_names), len(data), alpha, max_cond_set,
    )

    t0 = time.time()

    # First run without BK to get node objects for building BK
    log.info("Pass 1/2: skeleton (no BK) ...")
    G_init, _ = fci(
        data, fisherz, alpha,
        depth=max_cond_set,
        node_names=var_names,
        verbose=False,
    )
    bk = build_temporal_background_knowledge(G_init.nodes, var_names)

    # Second run with temporal BK enforced
    log.info("Pass 2/2: FCI with temporal background knowledge ...")
    G, _ = fci(
        data, fisherz, alpha,
        depth=max_cond_set,
        node_names=var_names,
        verbose=False,
        background_knowledge=bk,
    )

    elapsed = time.time() - t0
    log.info("FCI completed in %.1f seconds", elapsed)
    return G, elapsed


# ── Main ─────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Step B FCI: Fast Causal Inference on V3 triplet dataset.",
    )
    parser.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_b_fci"),
    )
    parser.add_argument("--split", default="train")
    parser.add_argument("--sample-n", type=int, default=30_000)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--max-cond-set", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log_path = report_dir / "run_log.txt"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(log_path), mode="w", encoding="utf-8"),
        ],
    )

    log.info("=" * 70)
    log.info("Step B FCI on V3 dataset")
    log.info("  sample_n=%d, alpha=%.1e, max_cond_set=%d, seed=%d",
             args.sample_n, args.alpha, args.max_cond_set, args.seed)
    log.info("=" * 70)

    # ── Load data ────────────────────────────────────────────────────
    log.info("Loading: %s", args.csv)
    df = pd.read_csv(args.csv, low_memory=False)

    if args.split:
        df = df[df["split"] == args.split]
        log.info("After split='%s': %d rows", args.split, len(df))

    df_sub = df[ALL_VARS].dropna()
    log.info("After dropna: %d rows", len(df_sub))

    if args.sample_n > 0 and args.sample_n < len(df_sub):
        df_sub = df_sub.sample(n=args.sample_n, random_state=args.seed)
        log.info("Sampled %d rows (seed=%d)", args.sample_n, args.seed)

    data = df_sub.values.astype(np.float64)

    # ── Run FCI ──────────────────────────────────────────────────────
    G, elapsed = run_fci(data, ALL_VARS, alpha=args.alpha, max_cond_set=args.max_cond_set)

    # ── Extract edges ────────────────────────────────────────────────
    edge_df = extract_fci_edges(G, ALL_VARS)
    drug_df = extract_drug_fci_edges(edge_df)

    log.info("Total PAG edges: %d", len(edge_df))
    log.info("Drug <-> next_* edges: %d", len(drug_df))

    edge_df.to_csv(report_dir / "edges.csv", index=False)
    drug_df.to_csv(report_dir / "drug_edges.csv", index=False)
    log.info("Saved: edges.csv, drug_edges.csv")

    # ── Edge type summary ────────────────────────────────────────────
    if len(edge_df) > 0:
        log.info("")
        log.info("EDGE TYPE BREAKDOWN (all edges):")
        for etype, cnt in edge_df["edge_type"].value_counts().items():
            log.info("  %-20s %d", etype, cnt)

    # ── Drug edge summary ────────────────────────────────────────────
    log.info("")
    log.info("=" * 70)
    log.info("DRUG -> NEXT_* EDGES (FCI PAG)")
    log.info("=" * 70)
    log.info("")
    log.info("  Legend:")
    log.info("    -->  directed      : direct causal effect, no hidden confounder detected")
    log.info("    <->  bidirected    : hidden common cause (confounding by indication)")
    log.info("    o->  circle_arrow  : direct but possible hidden confounder")
    log.info("    o-o  circle_circle : fully uncertain")
    log.info("")

    if len(drug_df) == 0:
        log.info("  No drug <-> next_* edges found.")
    else:
        # Normalise so drug is always "source" for display
        drug_set = set(TIER2_ACTION)
        for _, row in drug_df.sort_values(["source", "target"]).iterrows():
            src, tgt = row["source"], row["target"]
            if src not in drug_set:
                src, tgt = tgt, src
            log.info("  %-25s %s %s   [%s]",
                     src, row["label"], tgt, row["edge_type"])

    # ── Compare with PC results ──────────────────────────────────────
    pc_path = PROJECT_ROOT / "reports" / "causal_v3" / "step_b" / "drug_edges.csv"
    if pc_path.exists():
        pc_df = pd.read_csv(pc_path)
        pc_pairs = set(zip(pc_df["source"], pc_df["target"]))

        fci_directed = drug_df[drug_df["edge_type"] == "directed"]
        fci_pairs = set()
        for _, row in fci_directed.iterrows():
            src, tgt = row["source"], row["target"]
            if src not in drug_set:
                src, tgt = tgt, src
            fci_pairs.add((src, tgt))

        in_both    = pc_pairs & fci_pairs
        pc_only    = pc_pairs - fci_pairs
        fci_only   = fci_pairs - pc_pairs

        log.info("")
        log.info("=" * 70)
        log.info("COMPARISON: PC (directed) vs FCI (directed only)")
        log.info("=" * 70)
        log.info("  In both    (%d): %s", len(in_both),
                 ", ".join(f"{s}->{t}" for s, t in sorted(in_both)))
        log.info("  PC only    (%d): %s", len(pc_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(pc_only)))
        log.info("  FCI only   (%d): %s", len(fci_only),
                 ", ".join(f"{s}->{t}" for s, t in sorted(fci_only)))

    log.info("")
    log.info("Step B FCI complete. Results in: %s", report_dir)
    log.info("Total runtime: %.1f seconds", elapsed)


if __name__ == "__main__":
    main()
