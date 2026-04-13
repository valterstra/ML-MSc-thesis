"""Step B (DirectLiNGAM variant): Causal discovery with DirectLiNGAM.

DirectLiNGAM (Shimizu et al.) uses ICA to identify causal ordering from
non-Gaussian noise. Unlike PC/FCI/GES it fully identifies the DAG (not just
equivalence class) and returns linear edge weights simultaneously.

Assumptions: linear structural equations, non-Gaussian noise.
Output: adjacency_matrix_ where W[i,j] is the linear effect of variable i on j.

Temporal tier constraints encoded via prior_knowledge matrix (0=forbidden, -1=unknown).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from careai.causal_v2.causal_discovery import (
    ALL_CAUSAL_VARS, CAUSAL_NEXT_VARS, CAUSAL_STATIC_VARS,
    CAUSAL_ACTION_VARS, _tier_of, prepare_causal_data,
)

log = logging.getLogger(__name__)


def build_prior_knowledge(col_names: list[str]) -> np.ndarray:
    """Build prior_knowledge matrix encoding temporal tier constraints.

    prior_knowledge[i, j]:
      0  = forbidden (i cannot cause j)
     -1  = unknown (allow algorithm to decide)

    Forbidden: any edge from higher tier to lower tier, and all within tier-3.
    """
    n = len(col_names)
    pk = np.full((n, n), -1, dtype=int)  # default: unknown
    tiers = [_tier_of(c) for c in col_names]

    forbidden = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                pk[i, j] = 0
                continue
            tier_i, tier_j = tiers[i], tiers[j]
            if tier_i > tier_j:
                pk[i, j] = 0
                forbidden += 1
            if tier_i == 3 and tier_j == 3:
                pk[i, j] = 0
                forbidden += 1

    log.info("  Prior knowledge: %d forbidden entries (out of %d)", forbidden, n * n)
    return pk


def run_lingam(
    X: np.ndarray,
    col_names: list[str],
    w_threshold: float = 0.1,
    random_state: int = 42,
) -> np.ndarray:
    """Run DirectLiNGAM and return thresholded weight matrix.

    Args:
        X           : data matrix (n_samples, n_vars), float64
        col_names   : variable names
        w_threshold : zero out weights below this absolute value
        random_state: for reproducibility

    Returns:
        W : (n_vars, n_vars) weight matrix, W[i,j] = effect of i on j
    """
    from lingam import DirectLiNGAM

    n, d = X.shape
    pk = build_prior_knowledge(col_names)

    log.info(
        "  Running DirectLiNGAM (w_threshold=%.2f, n=%d, p=%d)",
        w_threshold, n, d,
    )

    model = DirectLiNGAM(
        random_state=random_state,
        prior_knowledge=pk,
        apply_prior_knowledge_softly=False,
    )
    model.fit(X)

    W = model.adjacency_matrix_.copy()  # W[i,j] = effect of i on j
    raw_edges = (W != 0).sum()
    log.info("  LiNGAM raw non-zero weights: %d", raw_edges)

    # Threshold small weights
    W[np.abs(W) < w_threshold] = 0.0
    remaining = (W != 0).sum()
    log.info("  After threshold (%.2f): %d edges remain", w_threshold, remaining)

    return W


def extract_edges_from_W(W: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    rows = []
    for i in range(len(col_names)):
        for j in range(len(col_names)):
            if W[i, j] != 0:
                rows.append({
                    "source": col_names[i],
                    "target": col_names[j],
                    "weight": round(float(W[i, j]), 4),
                    "edge_type": "directed",
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "weight", "edge_type"])


def extract_parent_sets(W: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    rows = []
    for j, target in enumerate(col_names):
        if target not in CAUSAL_NEXT_VARS:
            continue
        parent_idx = np.where(W[:, j] != 0)[0]
        parents  = [col_names[i] for i in parent_idx]
        weights  = [round(float(W[i, j]), 4) for i in parent_idx]
        rows.append({
            "target": target,
            "n_parents": len(parents),
            "parents": ", ".join(parents),
            "parent_weights": ", ".join(f"{p}({w:+.3f})" for p, w in zip(parents, weights)),
        })
    return pd.DataFrame(rows)


def visualize_dag(edge_df: pd.DataFrame, report_dir: Path,
                  title: str = "Causal Discovery DAG (DirectLiNGAM)") -> None:
    try:
        import matplotlib; matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        return

    G = nx.DiGraph()
    for v in ALL_CAUSAL_VARS:
        G.add_node(v)
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=abs(row["weight"]))

    color_map = {"static": "#4e79a7", "state_t": "#59a14f",
                 "action": "#f28e2b", "next": "#e15759"}
    node_colors = [
        color_map["static"] if n in CAUSAL_STATIC_VARS else
        color_map["action"] if n in CAUSAL_ACTION_VARS else
        color_map["next"]   if n in CAUSAL_NEXT_VARS else
        color_map["state_t"] for n in G.nodes()
    ]
    ew = [G[u][v].get("weight", 0.3) for u, v in G.edges()]
    max_w = max(ew) if ew else 1.0
    widths = [1.0 + 3.0 * w / max_w for w in ew]

    fig, ax = plt.subplots(figsize=(20, 14))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_colors,
                     node_size=800, font_size=7, arrows=True,
                     arrowsize=15, edge_color="#555555", width=widths)
    from matplotlib.patches import Patch
    ax.legend(handles=[
        Patch(facecolor=color_map["static"],  label="Static confounder"),
        Patch(facecolor=color_map["state_t"], label="State at t"),
        Patch(facecolor=color_map["action"],  label="Action at t"),
        Patch(facecolor=color_map["next"],    label="State at t+1"),
    ], loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13); ax.axis("off")
    fig.savefig(report_dir / "dag.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved DAG: %s", report_dir / "dag.png")


def save_results(edge_df, W, col_names, parent_df, report_dir: Path) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(report_dir / "edges.csv", index=False)
    pd.DataFrame(W, index=col_names, columns=col_names).to_csv(
        report_dir / "weight_matrix.csv")
    parent_df.to_csv(report_dir / "parent_sets.csv", index=False)
    log.info("Saved edges.csv, weight_matrix.csv, parent_sets.csv to %s", report_dir)
