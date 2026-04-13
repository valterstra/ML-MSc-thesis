"""Step B (NOTEARS variant): Causal discovery with linear structural equations.

Uses the NOTEARS algorithm (Zheng et al. 2018) which solves:
    min_W  L(W; X) + lambda1 * ||W||_1   s.t.  h(W) = 0  (acyclicity)

Unlike PC, NOTEARS returns a weight matrix W where W[i,j] is the linear
coefficient for the edge i -> j. This gives us structure AND weights in one shot.

Limitation: linear structural equations only. If true relationships are
nonlinear, weights underfit but discovered structure may still be useful.

Temporal tier structure is enforced post-hoc by zeroing out all W[i,j] where
variable i is in a higher tier than j (backward-in-time edges). The column
ordering (static | state_t | actions | state_{t+1}) already biases NOTEARS
toward the correct temporal direction.

Variable shortlist: identical to PC run (37 nodes).
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
    CAUSAL_STATE_VARS,
    _tier_of,
    prepare_causal_data,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# NOTEARS runner
# ---------------------------------------------------------------------------

def run_notears(
    X: np.ndarray,
    col_names: list[str],
    lambda1: float = 0.1,
    loss_type: str = "l2",
    w_threshold: float = 0.3,
    max_iter: int = 100,
) -> np.ndarray:
    """Run NOTEARS linear on data matrix X and apply temporal tier mask.

    Args:
        X           : data matrix (n_samples, n_vars), float64
        col_names   : variable names matching X columns
        lambda1     : L1 sparsity penalty (higher = fewer edges)
        loss_type   : 'l2' for continuous/mixed data
        w_threshold : drop edge if |weight| < threshold
        max_iter    : max augmented Lagrangian iterations

    Returns:
        W_masked : (n_vars, n_vars) weight matrix after temporal mask applied.
                   W[i, j] != 0 means edge col_names[i] -> col_names[j].
    """
    from notears.linear import notears_linear

    n, d = X.shape
    log.info(
        "  Running NOTEARS (lambda1=%.3f, w_threshold=%.3f, loss=%s, n=%d, p=%d)",
        lambda1, w_threshold, loss_type, n, d,
    )

    W_est = notears_linear(
        X,
        lambda1=lambda1,
        loss_type=loss_type,
        max_iter=max_iter,
        w_threshold=w_threshold,
    )
    log.info("  NOTEARS raw edges: %d", (W_est != 0).sum())

    # Apply temporal tier mask: zero out backward-in-time edges
    tiers = [_tier_of(c) for c in col_names]
    W_masked = W_est.copy()
    forbidden = 0
    for i in range(d):
        for j in range(d):
            if i == j:
                continue
            # Edge i -> j is forbidden if tier_i > tier_j
            if tiers[i] > tiers[j]:
                if W_masked[i, j] != 0:
                    forbidden += 1
                W_masked[i, j] = 0.0
            # Forbid all edges within tier 3 (next-state contemporaneous)
            if tiers[i] == 3 and tiers[j] == 3:
                if W_masked[i, j] != 0:
                    forbidden += 1
                W_masked[i, j] = 0.0

    remaining = (W_masked != 0).sum()
    log.info(
        "  After temporal mask: %d forbidden edges zeroed, %d edges remain",
        forbidden, remaining,
    )
    return W_masked


# ---------------------------------------------------------------------------
# Extract edges from weight matrix
# ---------------------------------------------------------------------------

def extract_edges_from_W(W: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    """Convert W matrix to edge DataFrame with weights."""
    rows = []
    n = len(col_names)
    for i in range(n):
        for j in range(n):
            if W[i, j] != 0:
                rows.append({
                    "source": col_names[i],
                    "target": col_names[j],
                    "weight": W[i, j],
                    "edge_type": "directed",
                })
    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        log.warning("  No edges after masking — try lower lambda1 or w_threshold")
    else:
        log.info("  %d directed edges extracted", len(edge_df))
    return edge_df


def extract_parent_sets(W: np.ndarray, col_names: list[str]) -> pd.DataFrame:
    """For each next-state variable, list parents and their weights."""
    rows = []
    for j, target in enumerate(col_names):
        if target not in CAUSAL_NEXT_VARS:
            continue
        parent_indices = np.where(W[:, j] != 0)[0]
        parents = [col_names[i] for i in parent_indices]
        weights = [round(float(W[i, j]), 4) for i in parent_indices]
        parent_weight_str = ", ".join(
            f"{p}({w:+.3f})" for p, w in zip(parents, weights)
        )
        rows.append({
            "target": target,
            "n_parents": len(parents),
            "parents": ", ".join(parents),
            "parent_weights": parent_weight_str,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_dag(
    edge_df: pd.DataFrame,
    report_dir: Path,
    title: str = "Causal Discovery DAG (NOTEARS)",
) -> None:
    """Draw and save the DAG. Edge width encodes absolute weight."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        log.warning("  matplotlib/networkx not available -- skipping visualization")
        return

    G = nx.DiGraph()
    for v in ALL_CAUSAL_VARS:
        G.add_node(v)
    for _, row in edge_df.iterrows():
        G.add_edge(row["source"], row["target"], weight=abs(row["weight"]))

    color_map = {
        "static":  "#4e79a7",
        "state_t": "#59a14f",
        "action":  "#f28e2b",
        "next":    "#e15759",
    }
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

    edge_weights = [G[u][v].get("weight", 0.5) for u, v in G.edges()]
    max_w = max(edge_weights) if edge_weights else 1.0
    widths = [1.0 + 3.0 * w / max_w for w in edge_weights]

    fig, ax = plt.subplots(figsize=(20, 14))
    pos = nx.spring_layout(G, seed=42, k=2.5)
    nx.draw_networkx(
        G, pos=pos, ax=ax,
        node_color=node_colors,
        node_size=800,
        font_size=7,
        arrows=True,
        arrowsize=15,
        edge_color="#555555",
        width=widths,
    )

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
    log.info("  Saved DAG visualization: %s", out_path)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    edge_df: pd.DataFrame,
    W: np.ndarray,
    col_names: list[str],
    parent_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    """Save all NOTEARS outputs to report_dir."""
    report_dir.mkdir(parents=True, exist_ok=True)

    edge_df.to_csv(report_dir / "edges.csv", index=False)
    log.info("  Saved edges.csv (%d edges)", len(edge_df))

    weight_df = pd.DataFrame(W, index=col_names, columns=col_names)
    weight_df.to_csv(report_dir / "weight_matrix.csv")
    log.info("  Saved weight_matrix.csv (%dx%d)", len(weight_df), len(weight_df.columns))

    parent_df.to_csv(report_dir / "parent_sets.csv", index=False)
    log.info("  Saved parent_sets.csv (%d next-state targets)", len(parent_df))

    log.info("All NOTEARS results saved to %s", report_dir)
