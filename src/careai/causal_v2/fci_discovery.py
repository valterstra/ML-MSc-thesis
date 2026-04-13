"""Step B (FCI variant): Causal discovery with FCI algorithm.

FCI (Fast Causal Inference, Spirtes et al.) extends PC by relaxing the
causal sufficiency assumption — it explicitly handles unmeasured confounders.
This is the more realistic assumption for hospital data, where physician
intent (why a drug was given) is never observed.

Output is a PAG (Partial Ancestral Graph) with richer edge types than PC:
  i --> j   : definite directed (i causes j, no hidden common cause)
  i <-> j   : bidirected (hidden common cause between i and j)
  i o-> j   : i causes j OR there is a hidden common cause (uncertain)
  i o-o j   : fully uncertain

For building structural equations, we use only DEFINITE directed edges (-->).
Bidirected and circle edges are recorded separately for inspection.

Variable shortlist and temporal tier structure: identical to PC run.
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
    prepare_causal_data,
    build_background_knowledge,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FCI runner
# ---------------------------------------------------------------------------

def run_fci(
    X: np.ndarray,
    col_names: list[str],
    alpha: float = 0.01,
    max_cond_set: int = 4,
    verbose: bool = False,
):
    """Run FCI algorithm on data matrix X.

    Args:
        X             : data matrix (n_samples, n_vars)
        col_names     : variable names matching X columns
        alpha         : significance level for CI tests
        max_cond_set  : max conditioning set size (depth parameter)
        verbose       : print CI test details

    Returns:
        G     : causal-learn graph object (PAG)
        edges : list of Edge objects
    """
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz

    bk = build_background_knowledge(col_names)

    log.info(
        "  Running FCI (alpha=%.3f, max_cond_set=%d, n=%d, p=%d)",
        alpha, max_cond_set, X.shape[0], X.shape[1],
    )

    G, edges = fci(
        dataset=X,
        independence_test_method=fisherz,
        alpha=alpha,
        depth=max_cond_set,
        verbose=verbose,
        background_knowledge=bk,
        show_progress=True,
        node_names=col_names,
    )

    log.info("  FCI complete.")
    return G, edges


# ---------------------------------------------------------------------------
# Extract edges from PAG
# ---------------------------------------------------------------------------

def extract_edges(G, edges, col_names: list[str]) -> pd.DataFrame:
    """Extract all edges from FCI PAG with their types.

    Edge type classification:
      directed    : i --> j  (definite causal direction)
      bidirected  : i <-> j  (hidden common cause)
      circle_arrow: i o-> j  (uncertain: cause OR hidden confounder)
      circle_circle: i o-o j (fully uncertain)
      undirected  : i --- j

    For downstream structural equations, only 'directed' edges are used.
    """
    from causallearn.graph.Endpoint import Endpoint

    rows = []
    n = len(col_names)

    # Build node name -> index map
    node_names = [node.get_name() for node in G.get_nodes()]

    def node_idx(name):
        try:
            return node_names.index(name)
        except ValueError:
            return -1

    for edge in edges:
        node1 = edge.get_node1().get_name()
        node2 = edge.get_node2().get_name()
        ep1 = edge.get_endpoint1()  # mark at node1 end
        ep2 = edge.get_endpoint2()  # mark at node2 end

        # Classify edge type
        if ep1 == Endpoint.TAIL and ep2 == Endpoint.ARROW:
            # node1 --> node2
            edge_type = "directed"
            source, target = node1, node2
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.TAIL:
            # node2 --> node1
            edge_type = "directed"
            source, target = node2, node1
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.ARROW:
            edge_type = "bidirected"
            source, target = node1, node2
        elif ep1 == Endpoint.CIRCLE and ep2 == Endpoint.ARROW:
            edge_type = "circle_arrow"
            source, target = node1, node2
        elif ep1 == Endpoint.ARROW and ep2 == Endpoint.CIRCLE:
            edge_type = "circle_arrow"
            source, target = node2, node1
        elif ep1 == Endpoint.CIRCLE and ep2 == Endpoint.CIRCLE:
            edge_type = "circle_circle"
            source, target = node1, node2
        elif ep1 == Endpoint.TAIL and ep2 == Endpoint.TAIL:
            edge_type = "undirected"
            source, target = node1, node2
        else:
            edge_type = f"other({ep1},{ep2})"
            source, target = node1, node2

        rows.append({
            "source": source,
            "target": target,
            "edge_type": edge_type,
        })

    edge_df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=["source", "target", "edge_type"]
    )

    if edge_df.empty:
        log.warning("  No edges found — check alpha / sample size")
    else:
        counts = edge_df["edge_type"].value_counts().to_dict()
        log.info("  Edge counts: %s", counts)
        directed = (edge_df["edge_type"] == "directed").sum()
        log.info("  Definite directed edges (usable for simulator): %d", directed)

    return edge_df


def extract_parent_sets(edge_df: pd.DataFrame) -> pd.DataFrame:
    """Extract parent sets from DEFINITE directed edges only."""
    directed = edge_df[edge_df["edge_type"] == "directed"]

    rows = []
    for target in CAUSAL_NEXT_VARS:
        parents = directed[directed["target"] == target]["source"].tolist()
        rows.append({
            "target": target,
            "n_parents": len(parents),
            "parents": ", ".join(parents),
        })

    summary_df = pd.DataFrame(rows)
    return summary_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_dag(
    edge_df: pd.DataFrame,
    report_dir: Path,
    title: str = "Causal Discovery PAG (FCI algorithm)",
) -> None:
    """Draw and save the discovered PAG. Different edge types shown with different styles."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        log.warning("  matplotlib/networkx not available -- skipping visualization")
        return

    G_directed   = nx.DiGraph()
    G_uncertain  = nx.DiGraph()
    G_bidir      = nx.DiGraph()

    for v in ALL_CAUSAL_VARS:
        G_directed.add_node(v)

    for _, row in edge_df.iterrows():
        if row["edge_type"] == "directed":
            G_directed.add_edge(row["source"], row["target"])
        elif row["edge_type"] == "bidirected":
            G_bidir.add_edge(row["source"], row["target"])
        else:
            G_uncertain.add_edge(row["source"], row["target"])

    color_map = {
        "static":  "#4e79a7",
        "state_t": "#59a14f",
        "action":  "#f28e2b",
        "next":    "#e15759",
    }
    node_colors = []
    for node in G_directed.nodes():
        if node in CAUSAL_STATIC_VARS:
            node_colors.append(color_map["static"])
        elif node in CAUSAL_ACTION_VARS:
            node_colors.append(color_map["action"])
        elif node in CAUSAL_NEXT_VARS:
            node_colors.append(color_map["next"])
        else:
            node_colors.append(color_map["state_t"])

    fig, ax = plt.subplots(figsize=(22, 15))
    pos = nx.spring_layout(G_directed, seed=42, k=2.5)

    nx.draw_networkx_nodes(G_directed, pos=pos, ax=ax,
                           node_color=node_colors, node_size=800)
    nx.draw_networkx_labels(G_directed, pos=pos, ax=ax, font_size=7)
    nx.draw_networkx_edges(G_directed, pos=pos, ax=ax,
                           edge_color="#555555", width=1.5,
                           arrows=True, arrowsize=15)
    if G_bidir.edges():
        nx.draw_networkx_edges(G_bidir, pos=pos, ax=ax,
                               edge_color="#e41a1c", width=1.5,
                               style="dashed", arrows=True, arrowsize=12)
    if G_uncertain.edges():
        nx.draw_networkx_edges(G_uncertain, pos=pos, ax=ax,
                               edge_color="#aaaaaa", width=1.0,
                               style="dotted", arrows=True, arrowsize=10)

    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    legend_elements = [
        Patch(facecolor=color_map["static"],  label="Static confounder"),
        Patch(facecolor=color_map["state_t"], label="State at t"),
        Patch(facecolor=color_map["action"],  label="Action at t"),
        Patch(facecolor=color_map["next"],    label="State at t+1"),
        Line2D([0], [0], color="#555555", lw=2, label="Directed (-->)"),
        Line2D([0], [0], color="#e41a1c", lw=2, linestyle="dashed", label="Bidirected (<->)"),
        Line2D([0], [0], color="#aaaaaa", lw=1, linestyle="dotted", label="Uncertain (o-> or o-o)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_title(title, fontsize=13)
    ax.axis("off")

    out_path = report_dir / "dag.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("  Saved PAG visualization: %s", out_path)


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def save_results(
    edge_df: pd.DataFrame,
    parent_df: pd.DataFrame,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)

    edge_df.to_csv(report_dir / "edges.csv", index=False)
    log.info("  Saved edges.csv (%d edges)", len(edge_df))

    parent_df.to_csv(report_dir / "parent_sets.csv", index=False)
    log.info("  Saved parent_sets.csv")

    log.info("All FCI results saved to %s", report_dir)
