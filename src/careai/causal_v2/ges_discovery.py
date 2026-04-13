"""Step B (GES variant): Causal discovery with Greedy Equivalence Search.

GES (Chickering 2002) is a score-based algorithm that greedily maximises
the BIC score over DAG space. Unlike PC/FCI it runs no conditional
independence tests — instead it adds/removes edges based on score improvement.
This makes it much faster and uses a completely different algorithmic approach.

Assumptions: causal sufficiency (same as PC, unlike FCI).
Output: CPDAG — same format as PC, so edge extraction is identical.

BackgroundKnowledge is not supported by causal-learn's GES implementation.
Temporal tier constraints are enforced post-hoc by removing forbidden edges
from the adjacency matrix before extracting parent sets.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from careai.causal_v2.causal_discovery import (
    ALL_CAUSAL_VARS,
    CAUSAL_NEXT_VARS,
    CAUSAL_STATIC_VARS,
    CAUSAL_ACTION_VARS,
    _tier_of,
    prepare_causal_data,
    extract_edges,
    extract_adjacency_matrix,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# GES runner
# ---------------------------------------------------------------------------

def run_ges(
    X: np.ndarray,
    col_names: list[str],
    max_parents: int = 8,
) -> object:
    """Run GES on data matrix X and apply temporal tier mask.

    Args:
        X           : data matrix (n_samples, n_vars), float64
        col_names   : variable names matching X columns
        max_parents : max allowed parents per node (sparsity control)

    Returns:
        cg : causal-learn CausalGraph with forbidden edges removed
    """
    from causallearn.search.ScoreBased.GES import ges

    n, d = X.shape
    log.info(
        "  Running GES (score=BIC, max_parents=%d, n=%d, p=%d)",
        max_parents, n, d,
    )

    result = ges(X, score_func="local_score_BIC", maxP=max_parents,
                 node_names=col_names)
    cg = result["G"]

    raw_edges = int((np.abs(cg.graph) > 0).sum() // 2)
    log.info("  GES raw edges: ~%d", raw_edges)

    # Post-hoc temporal mask: zero out forbidden edges
    tiers = [_tier_of(c) for c in col_names]
    d = len(col_names)
    forbidden = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            tier_i, tier_j = tiers[i], tiers[j]
            # Remove i -> j if tier_i > tier_j (backward in time)
            if tier_i > tier_j:
                if cg.graph[i, j] != 0 or cg.graph[j, i] != 0:
                    forbidden += 1
                cg.graph[i, j] = 0
                cg.graph[j, i] = 0
            # Remove all edges within tier 3
            if tier_i == 3 and tier_j == 3:
                if cg.graph[i, j] != 0 or cg.graph[j, i] != 0:
                    forbidden += 1
                cg.graph[i, j] = 0
                cg.graph[j, i] = 0

    remaining = int((np.abs(cg.graph) > 0).sum() // 2)
    log.info(
        "  After temporal mask: %d forbidden edges removed, ~%d edges remain",
        forbidden, remaining,
    )
    return cg


# ---------------------------------------------------------------------------
# Extract parent sets (reuses PC logic — same CPDAG format)
# ---------------------------------------------------------------------------

def extract_parent_sets_from_adj(adj_df: pd.DataFrame) -> pd.DataFrame:
    """Extract parent sets for next-state variables from adjacency matrix."""
    rows = []
    for target in CAUSAL_NEXT_VARS:
        if target not in adj_df.columns:
            continue
        parents = adj_df.index[adj_df[target] > 0].tolist()
        rows.append({
            "target": target,
            "n_parents": len(parents),
            "parents": ", ".join(parents),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_dag(
    edge_df: pd.DataFrame,
    report_dir: Path,
    title: str = "Causal Discovery DAG (GES algorithm)",
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        log.warning("  matplotlib/networkx not available -- skipping")
        return

    G = nx.DiGraph()
    for v in ALL_CAUSAL_VARS:
        G.add_node(v)
    for _, row in edge_df[edge_df["edge_type"] == "directed"].iterrows():
        G.add_edge(row["source"], row["target"])

    color_map = {"static": "#4e79a7", "state_t": "#59a14f",
                 "action": "#f28e2b", "next": "#e15759"}
    node_colors = []
    for node in G.nodes():
        if node in CAUSAL_STATIC_VARS:
            node_colors.append(color_map["static"])
        elif node in CAUSAL_ACTION_VARS:
            node_colors.append(color_map["action"])
        elif node in CAUSAL_NEXT_VARS:
            node_colors.append(color_map["next"])
        else:
            node_colors.append(color_map["state_t"])

    fig, ax = plt.subplots(figsize=(20, 14))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx(G, pos=pos, ax=ax, node_color=node_colors,
                     node_size=800, font_size=7, arrows=True,
                     arrowsize=15, edge_color="#555555", width=1.5)

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=color_map["static"],  label="Static confounder"),
        Patch(facecolor=color_map["state_t"], label="State at t"),
        Patch(facecolor=color_map["action"],  label="Action at t"),
        Patch(facecolor=color_map["next"],    label="State at t+1"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.axis("off")
    out_path = report_dir / "dag.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved DAG: %s", out_path)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    edge_df: pd.DataFrame,
    adj_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    edge_df.to_csv(report_dir / "edges.csv", index=False)
    log.info("  Saved edges.csv (%d edges)", len(edge_df))
    adj_df.to_csv(report_dir / "adjacency_matrix.csv")
    log.info("  Saved adjacency_matrix.csv")
    parent_df.to_csv(report_dir / "parent_sets.csv", index=False)
    log.info("  Saved parent_sets.csv")
    log.info("All GES results saved to %s", report_dir)
