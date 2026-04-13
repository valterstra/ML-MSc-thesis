"""Step B: Causal discovery on daily hospital transitions.

Learns the causal DAG from (state_t, actions_t, state_{t+1}) rows using the
PC algorithm (causal-learn library, CMU).

Key design decisions:
  - Temporal tier structure enforces no backward-in-time edges
  - PC algorithm + Fisher's Z CI test (robust on large n, handles mixed data)
  - 100k-row sample from training split (PC is O(n) per CI test — 3M rows is wasteful)
  - Bootstrap stability (optional): repeat on N subsamples, report edge frequency

Temporal tier structure:
  Tier 0: static confounders (age_at_admit, charlson_score) — root nodes
  Tier 1: state at t (today's labs + binary flags + day_of_stay)
  Tier 2: actions at t (drug binary flags) — clinicians observe state_t then act
  Tier 3: state at t+1 (next-day labs + binary flags)

Forbidden edges (enforced via BackgroundKnowledge):
  - Any tier k -> tier j where k > j  (no backward-in-time causation)
  - Within tier 3 (no contemporaneous effects at t+1 — unobserved intra-day dynamics)

Variable shortlist (finalised from Step A multi-model consensus):
  Static:  age_at_admit, charlson_score
  State t: creatinine, bun, sodium, potassium, bicarbonate, anion_gap,
           calcium, phosphate, hemoglobin, wbc, platelets, inr, ptt,
           bilirubin, is_icu
  Actions: antibiotic_active, anticoagulant_active, diuretic_active,
           steroid_active, insulin_active
  State t+1: next_{each state variable above}
  Total: 37 nodes (2 static + 15 state_t + 5 actions + 15 next_state)
  Excluded: partial_sofa (reward signal only), glucose/ast (weak consensus),
            positive_culture_cumulative/day_of_stay (context, not dynamic state)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Variable shortlist — agreed from Step A
# ---------------------------------------------------------------------------

CAUSAL_STATIC_VARS = [
    "age_at_admit",
    "charlson_score",
]

CAUSAL_STATE_VARS = [
    # Core renal / metabolic
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "phosphate",
    # Haematology
    "hemoglobin", "wbc", "platelets",
    # Coagulation
    "inr", "ptt",
    # Liver
    "bilirubin",
    # Binary state flags
    "is_icu",
]

CAUSAL_ACTION_VARS = [
    "antibiotic_active",
    "anticoagulant_active",
    "diuretic_active",
    "steroid_active",
    "insulin_active",
]

# Next-day targets — must exist as next_{var} in the transition dataset
CAUSAL_NEXT_VARS = [f"next_{v}" for v in CAUSAL_STATE_VARS]

# Full column order fed to PC: [static | state_t | actions_t | state_{t+1}]
ALL_CAUSAL_VARS = CAUSAL_STATIC_VARS + CAUSAL_STATE_VARS + CAUSAL_ACTION_VARS + CAUSAL_NEXT_VARS

# Tier assignments (used to build BackgroundKnowledge)
TIER_STATIC  = 0   # indices 0..1   (2 vars)
TIER_STATE_T = 1   # indices 2..16  (15 vars)
TIER_ACTION  = 2   # indices 17..21 (5 vars)
TIER_NEXT    = 3   # indices 22..36 (15 vars)


def _tier_of(col: str) -> int:
    if col in CAUSAL_STATIC_VARS:
        return TIER_STATIC
    if col in CAUSAL_ACTION_VARS:
        return TIER_ACTION
    if col in CAUSAL_NEXT_VARS:
        return TIER_NEXT
    return TIER_STATE_T  # state_t variables


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_causal_data(
    df: pd.DataFrame,
    sample_n: int = 100_000,
    split: str = "train",
    random_state: int = 42,
) -> tuple[np.ndarray, list[str]]:
    """Prepare a numpy data matrix for the PC algorithm.

    Filters to training split, samples sample_n rows, selects and orders
    columns per ALL_CAUSAL_VARS, median-imputes missing values.

    Returns:
        X          : float64 array of shape (sample_n, len(ALL_CAUSAL_VARS))
        col_names  : list of column names matching X columns
    """
    train_df = df[df["split"] == split].copy()
    log.info("  Training rows available: %d", len(train_df))

    # Build next-day columns via within-admission shift (same as prepare_transitions)
    train_df = train_df.sort_values(["hadm_id", "day_of_stay"])
    for var in CAUSAL_STATE_VARS:
        if var in train_df.columns:
            train_df[f"next_{var}"] = train_df.groupby("hadm_id")[var].shift(-1)

    # Drop last day of each admission (no next-day state)
    train_df = train_df[train_df["is_last_day"] == 0].copy()
    log.info("  Transition rows (is_last_day==0): %d", len(train_df))

    # Check which columns are present
    col_names = [c for c in ALL_CAUSAL_VARS if c in train_df.columns]
    missing = [c for c in ALL_CAUSAL_VARS if c not in train_df.columns]
    if missing:
        log.warning("  %d causal variables not found in dataset: %s", len(missing), missing)

    # Sample
    n = min(sample_n, len(train_df))
    sample_df = train_df[col_names].sample(n=n, random_state=random_state)
    log.info("  Sampled %d rows for causal discovery (%d variables)", n, len(col_names))

    # Median imputation
    X = sample_df.fillna(sample_df.median(numeric_only=True)).values.astype(np.float64)
    log.info("  Data matrix shape: %s, NaN remaining: %d", X.shape, np.isnan(X).sum())

    return X, col_names


# ---------------------------------------------------------------------------
# Background knowledge (temporal tiers)
# ---------------------------------------------------------------------------

def build_background_knowledge(col_names: list[str]):
    """Build causal-learn BackgroundKnowledge enforcing temporal tier structure.

    Forbidden:
      - Any edge FROM higher-tier node TO lower-tier node
      - Any edge WITHIN tier 3 (state_{t+1} contemporaneous effects)
    """
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    from causallearn.graph.GraphNode import GraphNode

    bk = BackgroundKnowledge()
    nodes = [GraphNode(c) for c in col_names]
    tiers = [_tier_of(c) for c in col_names]
    n = len(col_names)

    forbidden_count = 0
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            tier_i = tiers[i]
            tier_j = tiers[j]

            # Forbid i -> j if tier_i > tier_j  (backward in time)
            if tier_i > tier_j:
                bk.add_forbidden_by_node(nodes[i], nodes[j])
                forbidden_count += 1

            # Forbid all edges within tier 3 (contemporaneous t+1)
            if tier_i == TIER_NEXT and tier_j == TIER_NEXT:
                bk.add_forbidden_by_node(nodes[i], nodes[j])
                forbidden_count += 1

    log.info(
        "  BackgroundKnowledge: %d forbidden edges (out of %d possible)",
        forbidden_count, n * (n - 1),
    )
    return bk


# ---------------------------------------------------------------------------
# PC algorithm
# ---------------------------------------------------------------------------

def run_pc(
    X: np.ndarray,
    col_names: list[str],
    alpha: float = 0.01,
    max_cond_set: int = 4,
    verbose: bool = False,
):
    """Run PC algorithm on data matrix X.

    Args:
        X             : data matrix (n_samples, n_vars)
        col_names     : variable names matching X columns
        alpha         : significance level for CI tests (lower = fewer edges)
        max_cond_set  : max conditioning set size (depth parameter) — controls runtime
        verbose       : print CI test details

    Returns:
        cg : causal-learn CausalGraph object
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    bk = build_background_knowledge(col_names)

    log.info(
        "  Running PC (alpha=%.3f, max_cond_set=%d, n=%d, p=%d)",
        alpha, max_cond_set, X.shape[0], X.shape[1],
    )

    cg = pc(
        data=X,
        alpha=alpha,
        indep_test=fisherz,
        stable=True,          # stable-PC: order-independent skeleton
        uc_rule=0,            # uc_sepset: standard unshielded collider orientation
        uc_priority=2,        # existing_priority: preserve existing orientations
        background_knowledge=bk,
        depth=max_cond_set,
        verbose=verbose,
        show_progress=True,
    )

    log.info("  PC complete.")
    return cg


# ---------------------------------------------------------------------------
# Extract results from CausalGraph
# ---------------------------------------------------------------------------

def extract_edges(cg, col_names: list[str]) -> pd.DataFrame:
    """Extract directed and undirected edges from a CausalGraph.

    causal-learn graph matrix convention:
      graph[i, j] =  1 and graph[j, i] = -1  =>  i -> j
      graph[i, j] = -1 and graph[j, i] = -1  =>  i -- j  (undirected)
      graph[i, j] =  1 and graph[j, i] =  1  =>  i <-> j (bidirected, FCI only)

    Returns DataFrame with columns: source, target, edge_type
    """
    G = cg.G.graph  # numpy (n, n) integer matrix
    n = len(col_names)
    rows = []

    for i in range(n):
        for j in range(i + 1, n):
            gij = G[i, j]
            gji = G[j, i]

            if gij == 0 and gji == 0:
                continue  # no edge

            if gij == 1 and gji == -1:
                edge_type = "directed"
                rows.append({"source": col_names[i], "target": col_names[j], "edge_type": edge_type})
            elif gij == -1 and gji == 1:
                edge_type = "directed"
                rows.append({"source": col_names[j], "target": col_names[i], "edge_type": edge_type})
            elif gij == -1 and gji == -1:
                edge_type = "undirected"
                rows.append({"source": col_names[i], "target": col_names[j], "edge_type": edge_type})
            else:
                edge_type = f"other({gij},{gji})"
                rows.append({"source": col_names[i], "target": col_names[j], "edge_type": edge_type})

    edge_df = pd.DataFrame(rows)
    if edge_df.empty:
        log.warning("  No edges found in graph — check alpha / sample size")
    else:
        directed = (edge_df["edge_type"] == "directed").sum()
        undirected = (edge_df["edge_type"] == "undirected").sum()
        log.info("  Edges: %d directed, %d undirected, %d total", directed, undirected, len(edge_df))

    return edge_df


def extract_adjacency_matrix(cg, col_names: list[str]) -> pd.DataFrame:
    """Return adjacency matrix as DataFrame.

    adj[i, j] = 1  means  col_names[i] -> col_names[j]  (directed)
    adj[i, j] = 0.5 means undirected edge between i and j
    adj[i, j] = 0  means no edge
    """
    G = cg.G.graph
    n = len(col_names)
    adj = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            gij = G[i, j]
            gji = G[j, i]
            if gij == 1 and gji == -1:
                adj[i, j] = 1.0   # i -> j
            elif gij == -1 and gji == -1 and i < j:
                adj[i, j] = 0.5   # undirected (mark once)
                adj[j, i] = 0.5

    return pd.DataFrame(adj, index=col_names, columns=col_names)


# ---------------------------------------------------------------------------
# Bootstrap edge stability (optional)
# ---------------------------------------------------------------------------

def run_bootstrap_stability(
    df: pd.DataFrame,
    col_names: list[str],
    n_bootstrap: int = 20,
    sample_n: int = 50_000,
    alpha: float = 0.01,
    max_cond_set: int = 4,
    split: str = "train",
) -> pd.DataFrame:
    """Run PC on n_bootstrap subsamples. Returns edge frequency matrix.

    Each cell (i, j) = fraction of bootstrap runs where edge i->j was present.
    Edges with frequency >= 0.8 are considered stable.
    """
    train_df = df[(df["split"] == split) & (df["is_last_day"] == 0)].copy()
    n = len(col_names)
    freq_matrix = np.zeros((n, n), dtype=float)

    for b in range(n_bootstrap):
        log.info("  Bootstrap run %d/%d", b + 1, n_bootstrap)
        sample = train_df[col_names].sample(
            n=min(sample_n, len(train_df)),
            random_state=b,
            replace=True,
        )
        X_b = sample.fillna(sample.median(numeric_only=True)).values.astype(np.float64)

        try:
            cg_b = run_pc(X_b, col_names, alpha=alpha, max_cond_set=max_cond_set)
            adj_b = extract_adjacency_matrix(cg_b, col_names).values
            freq_matrix += (adj_b > 0).astype(float)
        except Exception as exc:
            log.warning("  Bootstrap run %d failed: %s", b + 1, exc)

    freq_matrix /= n_bootstrap
    stability_df = pd.DataFrame(freq_matrix, index=col_names, columns=col_names)
    log.info(
        "  Bootstrap complete. Stable edges (>=0.8): %d",
        (stability_df.values >= 0.8).sum(),
    )
    return stability_df


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def visualize_dag(
    edge_df: pd.DataFrame,
    report_dir: Path,
    title: str = "Causal Discovery DAG (PC algorithm)",
) -> None:
    """Draw and save the discovered DAG using networkx + matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import networkx as nx
    except ImportError:
        log.warning("  matplotlib/networkx not available — skipping visualization")
        return

    G = nx.DiGraph()

    # Add all causal variables as nodes
    for v in ALL_CAUSAL_VARS:
        G.add_node(v)

    directed = edge_df[edge_df["edge_type"] == "directed"]
    undirected = edge_df[edge_df["edge_type"] == "undirected"]

    for _, row in directed.iterrows():
        G.add_edge(row["source"], row["target"])
    for _, row in undirected.iterrows():
        G.add_edge(row["source"], row["target"], style="dashed")

    # Color nodes by tier
    color_map = {
        "static":   "#4e79a7",   # blue
        "state_t":  "#59a14f",   # green
        "action":   "#f28e2b",   # orange
        "next":     "#e15759",   # red
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
        width=1.5,
    )

    # Legend
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
    adj_df: pd.DataFrame,
    report_dir: Path,
    stability_df: "pd.DataFrame | None" = None,
) -> None:
    """Save all Step B outputs to report_dir."""
    report_dir.mkdir(parents=True, exist_ok=True)

    edge_df.to_csv(report_dir / "edges.csv", index=False)
    log.info("  Saved edges.csv (%d edges)", len(edge_df))

    adj_df.to_csv(report_dir / "adjacency_matrix.csv")
    log.info("  Saved adjacency_matrix.csv (%dx%d)", len(adj_df), len(adj_df.columns))

    # Summary: for each next-state variable, list its discovered parents
    summary_rows = []
    next_vars_in_adj = [c for c in adj_df.columns if c in CAUSAL_NEXT_VARS]
    for target in next_vars_in_adj:
        parents = adj_df.index[adj_df[target] > 0].tolist()
        summary_rows.append({
            "target": target,
            "n_parents": len(parents),
            "parents": ", ".join(parents),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "parent_sets.csv", index=False)
    log.info("  Saved parent_sets.csv")

    if stability_df is not None:
        stability_df.to_csv(report_dir / "edge_stability.csv")
        log.info("  Saved edge_stability.csv")

    log.info("All Step B results saved to %s", report_dir)
