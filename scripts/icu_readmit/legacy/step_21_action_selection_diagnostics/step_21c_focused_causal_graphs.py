"""
Step 21c -- Legacy action-selection diagnostics: focused causal graphs.

PURPOSE
-------
Identify causal pathways from clinical actions through physiological state
changes to 30-day readmission. This step informs the final state/confounder
split for the RL pipeline and validates the action selection.

WHY FIVE SEPARATE GRAPHS
------------------------
Causal discovery algorithms (PC, NOTEARS, LiNGAM) degrade above ~10 variables:
the number of conditional independence tests grows combinatorially, and spurious
edges multiply. With 5 nodes per graph, each algorithm runs in seconds and
produces interpretable, trustworthy results.

Each graph is centred on one clinical action and its expected physiological
targets (based on pharmacological prior knowledge):

  vasopressor:  frac_vasopressor -> delta_HR, delta_SysBP, delta_Shock_Index -> readmit_30d
  ivfluid:      frac_ivfluid     -> delta_BUN, delta_Creatinine, delta_Potassium -> readmit_30d
  antibiotic:   frac_antibiotic  -> delta_WBC, delta_Temp_C, delta_RR -> readmit_30d
  sedation:     frac_sedation    -> delta_HR, delta_RR, delta_SpO2 -> readmit_30d
  diuretic:     frac_diuretic    -> delta_Potassium, delta_BUN, delta_Creatinine -> readmit_30d

WHY DELTA STATES
----------------
Using the average or last-bloc value of a state variable conflates the patient's
baseline severity with the treatment effect. A patient who arrived in shock
(high HR, low BP) will have different average vitals regardless of vasopressor
dosing -- this is confounding by indication.

The delta (last_bloc_value - first_bloc_value) captures the CHANGE in physiology
during the stay. The temporal ordering then becomes clean and defensible:

  Tier 0: frac_action   -- cumulative treatment exposure during the stay
  Tier 1: delta_state   -- physiological change as a result of treatment
  Tier 2: readmit_30d   -- outcome that occurs after discharge

This ordering is encoded as background knowledge in all three algorithms to
prevent algorithmically-discovered reverse-time edges.

WHY THREE ALGORITHMS
--------------------
Each algorithm makes different assumptions:
  PC        -- constraint-based, assumes no hidden confounders (Markov condition),
               outputs a CPDAG (some edges may be undirected)
  NOTEARS   -- continuous optimization on weighted adjacency matrix,
               assumes linear relationships
  LiNGAM    -- assumes non-Gaussian noise, fully orients all edges,
               sensitive to non-linear relationships

Agreement across all three algorithms is strong evidence for a causal edge.
An edge found by only one algorithm should be treated with caution.

OUTPUTS
-------
  reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21c_focused_causal_graphs/
    <graph_name>/
      pc/        edges.csv, adjacency.csv (per alpha level)
      notears/   edges.csv, adjacency.csv (per lambda level)
      lingam/    edges.csv, adjacency.csv
    cross_algorithm_summary.csv   -- edge agreement across algorithms x graphs
    causal_discovery.json         -- agreed edges per graph + metadata

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21c_focused_causal_graphs.py
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21c_focused_causal_graphs.py --graph vasopressor antibiotic
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21c_focused_causal_graphs.py --algorithms pc lingam
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import C_ICUSTAYID, C_BLOC, C_READMIT_30D

# ---------------------------------------------------------------------------
# Graph definitions
# Each graph: one action + three state variables + readmit_30d = 5 nodes.
# Tier encoding:
#   0 = action (frac_)
#   1 = state delta (delta_)
#   2 = outcome (readmit_30d)
# ---------------------------------------------------------------------------
GRAPHS = {
    'vasopressor': {
        'action':      'vasopressor_dose',
        'states':      ['HR', 'Arterial_BP_Sys', 'Shock_Index'],
        'description': 'Vasopressor -> hemodynamic markers -> readmission',
    },
    'ivfluid': {
        'action':      'ivfluid_dose',
        'states':      ['BUN', 'Creatinine', 'Potassium'],
        'description': 'IV fluids -> kidney/electrolyte markers -> readmission',
    },
    'antibiotic': {
        'action':      'antibiotic_active',
        'states':      ['WBC_count', 'Temp_C', 'RR'],
        'description': 'Antibiotics -> infection markers -> readmission',
    },
    'sedation': {
        'action':      'sedation_active',
        'states':      ['HR', 'RR', 'SpO2'],
        'description': 'Sedation -> cardiorespiratory markers -> readmission',
    },
    'diuretic': {
        'action':      'diuretic_active',
        'states':      ['Potassium', 'BUN', 'Creatinine'],
        'description': 'Diuretics -> electrolyte/kidney markers -> readmission',
    },
}

# Causal algorithm parameters
PC_ALPHAS      = [0.05, 0.01]        # significance levels for conditional independence
NOTEARS_LAMBDAS = [0.05, 0.01]       # L1 regularisation on edge weights
NOTEARS_W_THRESH = 0.01              # minimum edge weight to report
LINGAM_W_THRESH  = 0.05             # minimum edge weight to report


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_stay_level(df: pd.DataFrame, graph_def: dict) -> pd.DataFrame:
    """
    Build stay-level dataset for one graph:
      - frac_<action>:    fraction of blocs where drug was active / nonzero
      - delta_<state>:    last-bloc value minus first-bloc value (physiological change)
      - readmit_30d:      binary outcome

    Only stays where all variables are non-NaN are kept.
    The action column name is prefixed with 'frac_' in the output.
    Delta columns are prefixed with 'delta_'.
    """
    action_raw = graph_def['action']
    states     = graph_def['states']

    # Action: fraction of blocs where drug was active (>0 for dose cols, ==1 for binary)
    df_sorted = df.sort_values([C_ICUSTAYID, C_BLOC])

    grp = df_sorted.groupby(C_ICUSTAYID)

    if action_raw in ('vasopressor_dose', 'ivfluid_dose'):
        action_frac = (grp[action_raw].apply(lambda x: (x > 0).mean())).rename(f'frac_{action_raw}')
    else:
        action_frac = grp[action_raw].mean().rename(f'frac_{action_raw}')

    # Delta states: last - first (within stay, using non-NaN first/last values)
    delta_cols = {}
    for s in states:
        if s not in df.columns:
            continue
        first_val = grp[s].first()
        last_val  = grp[s].last()
        delta_cols[f'delta_{s}'] = last_val - first_val

    # Outcome
    outcome = grp[C_READMIT_30D].first()

    stay_df = pd.concat(
        [action_frac] + [pd.Series(v, name=k) for k, v in delta_cols.items()] + [outcome],
        axis=1
    ).reset_index()

    # Drop stays with any NaN in the graph variables
    graph_cols = [f'frac_{action_raw}'] + list(delta_cols.keys()) + [C_READMIT_30D]
    stay_df = stay_df.dropna(subset=graph_cols)

    return stay_df, graph_cols


def get_var_tiers(graph_cols: list[str]) -> dict[str, int]:
    """
    Assign temporal tier to each variable:
      0 = frac_action  (treatment exposure, temporally first)
      1 = delta_state  (physiological response, temporally second)
      2 = readmit_30d  (outcome, temporally last)
    """
    tiers = {}
    for c in graph_cols:
        if c.startswith('frac_'):
            tiers[c] = 0
        elif c.startswith('delta_'):
            tiers[c] = 1
        elif c == C_READMIT_30D:
            tiers[c] = 2
    return tiers


# ---------------------------------------------------------------------------
# Algorithm: PC
# ---------------------------------------------------------------------------

def run_pc(data: np.ndarray, var_names: list[str], tiers: dict[str, int],
           alpha: float, out_dir: Path) -> pd.DataFrame:
    """
    Run PC algorithm then post-filter any reverse-time edges.

    PC is run without background knowledge (the API for node-based constraints
    is fragile across causallearn versions). Instead, any directed edge that
    violates the temporal tier ordering (higher tier -> lower tier) is removed
    after discovery. This is equivalent in effect: we simply refuse to report
    edges that are temporally impossible.
    """
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    cg = pc(data, alpha=alpha, indep_test=fisherz,
            node_names=var_names, show_progress=False)

    edges = extract_pc_edges(cg.G, var_names)

    # Post-filter: remove directed edges that violate temporal ordering
    if len(edges) > 0:
        def temporally_valid(row):
            if row['edge_type'] == 'directed':
                return tiers.get(row['source'], 0) <= tiers.get(row['target'], 2)
            return True  # keep undirected edges as-is
        edges = edges[edges.apply(temporally_valid, axis=1)].reset_index(drop=True)

    edges['alpha'] = alpha

    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / f'edges_alpha{alpha}.csv', index=False)

    adj = pd.DataFrame(
        cg.G.graph, index=var_names, columns=var_names)
    adj.to_csv(out_dir / f'adjacency_alpha{alpha}.csv')

    return edges


def extract_pc_edges(G, var_names: list[str]) -> pd.DataFrame:
    """
    Extract directed and undirected edges from PC CPDAG.
    G.graph[i,j]=1 and G.graph[j,i]=-1 means i->j (directed).
    G.graph[i,j]=1 and G.graph[j,i]=1  means i--j (undirected).
    """
    rows = []
    n = len(var_names)
    for i in range(n):
        for j in range(i + 1, n):
            aij = G.graph[i, j]
            aji = G.graph[j, i]
            if aij == 0 and aji == 0:
                continue
            if aij == 1 and aji == -1:
                rows.append({'source': var_names[i], 'target': var_names[j],
                             'edge_type': 'directed', 'algorithm': 'pc'})
            elif aij == -1 and aji == 1:
                rows.append({'source': var_names[j], 'target': var_names[i],
                             'edge_type': 'directed', 'algorithm': 'pc'})
            elif aij == 1 and aji == 1:
                rows.append({'source': var_names[i], 'target': var_names[j],
                             'edge_type': 'undirected', 'algorithm': 'pc'})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Algorithm: NOTEARS
# ---------------------------------------------------------------------------

def run_notears(data: np.ndarray, var_names: list[str], tiers: dict[str, int],
                lam: float, out_dir: Path) -> pd.DataFrame:
    """
    Run NOTEARS with temporal mask: edges from higher tier -> lower tier
    are forced to zero after optimisation.
    """
    from notears.linear import notears_linear

    W = notears_linear(data, lambda1=lam, loss_type='l2')

    # Apply temporal mask: zero out reverse-time edges
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if tiers[vi] >= tiers[vj] and i != j:
                W[i, j] = 0.0

    edges = extract_notears_edges(W, var_names, NOTEARS_W_THRESH)
    edges['lambda'] = lam

    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / f'edges_lambda{lam}.csv', index=False)

    adj = pd.DataFrame(W, index=var_names, columns=var_names)
    adj.to_csv(out_dir / f'adjacency_lambda{lam}.csv')

    return edges


def extract_notears_edges(W: np.ndarray, var_names: list[str],
                          threshold: float) -> pd.DataFrame:
    """W[i,j] != 0 means j -> i (NOTEARS convention)."""
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i != j and abs(W[i, j]) >= threshold:
                rows.append({'source': vj, 'target': vi,
                             'weight': round(float(W[i, j]), 4),
                             'edge_type': 'directed', 'algorithm': 'notears'})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Algorithm: LiNGAM
# ---------------------------------------------------------------------------

def run_lingam(data: np.ndarray, var_names: list[str], tiers: dict[str, int],
               out_dir: Path) -> pd.DataFrame:
    """
    Run DirectLiNGAM with temporal prior knowledge.
    Prior knowledge matrix: pk[i,j]=0 forbids vi->vj.
    Forbid all edges where tier(vi) >= tier(vj) (no same-tier or reverse-time edges).
    """
    import lingam

    n = len(var_names)
    pk = np.full((n, n), -1, dtype=int)   # -1 = no constraint
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i == j:
                continue
            if tiers[vi] >= tiers[vj]:
                pk[i, j] = 0   # vi -> vj forbidden

    model = lingam.DirectLiNGAM(prior_knowledge=pk)
    model.fit(data)

    edges = extract_lingam_edges(model.adjacency_matrix_, var_names, LINGAM_W_THRESH)

    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / 'edges.csv', index=False)

    adj = pd.DataFrame(model.adjacency_matrix_, index=var_names, columns=var_names)
    adj.to_csv(out_dir / 'adjacency.csv')

    return edges


def extract_lingam_edges(B: np.ndarray, var_names: list[str],
                         threshold: float) -> pd.DataFrame:
    """B[i,j] = effect of j -> i."""
    rows = []
    for i, vi in enumerate(var_names):
        for j, vj in enumerate(var_names):
            if i != j and abs(B[i, j]) >= threshold:
                rows.append({'source': vj, 'target': vi,
                             'weight': round(float(B[i, j]), 4),
                             'edge_type': 'directed', 'algorithm': 'lingam'})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Cross-algorithm agreement
# ---------------------------------------------------------------------------

def compute_agreement(all_edges: list[pd.DataFrame]) -> pd.DataFrame:
    """
    For each (source, target) pair, count how many algorithms found it
    as a directed edge. Higher count = stronger evidence.
    """
    if not all_edges:
        return pd.DataFrame()

    directed = [e[e['edge_type'] == 'directed'][['source', 'target', 'algorithm', 'graph']]
                for e in all_edges if len(e) > 0 and 'edge_type' in e.columns]

    if not directed:
        return pd.DataFrame()

    combined = pd.concat(directed, ignore_index=True)
    agreement = (combined
                 .groupby(['graph', 'source', 'target'])
                 .agg(n_algorithms=('algorithm', 'nunique'),
                      algorithms=('algorithm', lambda x: ','.join(sorted(x.unique()))))
                 .reset_index()
                 .sort_values(['graph', 'n_algorithms'], ascending=[True, False]))
    return agreement


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'reports' / 'icu_readmit' / 'legacy' / 'step_21_action_selection_diagnostics' / 'step_21c_focused_causal_graphs'))
    parser.add_argument('--graph', nargs='+', default=list(GRAPHS.keys()),
                        choices=list(GRAPHS.keys()),
                        help='Which graphs to run (default: all 5)')
    parser.add_argument('--algorithms', nargs='+',
                        default=['pc', 'notears', 'lingam'],
                        choices=['pc', 'notears', 'lingam'],
                        help='Which algorithms to run (default: all 3)')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    out_root  = Path(args.out_dir)
    log_file  = args.log or str(PROJECT_ROOT / 'logs' / 'step_21c_focused_causal_graphs.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 21c started.")
    logging.info("Graphs: %s", args.graph)
    logging.info("Algorithms: %s", args.algorithms)

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    all_edges_collected = []
    summary_json = {}

    # -----------------------------------------------------------------------
    # Main loop: graph x algorithm
    # -----------------------------------------------------------------------
    for graph_name in args.graph:
        graph_def = GRAPHS[graph_name]
        logging.info("")
        logging.info("=== Graph: %s ===", graph_name)
        logging.info("  %s", graph_def['description'])

        # Build stay-level dataset for this graph
        stay_df, graph_cols = build_stay_level(df, graph_def)
        tiers = get_var_tiers(graph_cols)
        var_names = graph_cols  # order: frac_action, delta_state x3, readmit_30d

        logging.info("  %d stays after dropping NaN rows", len(stay_df))
        logging.info("  Variables: %s", var_names)
        logging.info("  Tiers: %s", tiers)

        # Standardise for algorithms that assume ~unit variance
        scaler = StandardScaler()
        data = scaler.fit_transform(stay_df[var_names].values.astype(float))

        graph_out = out_root / graph_name
        graph_edges = []
        summary_json[graph_name] = {'description': graph_def['description'], 'algorithms': {}}

        # ---- PC ----
        if 'pc' in args.algorithms:
            logging.info("  Running PC (alphas=%s)...", PC_ALPHAS)
            t0 = time.time()
            for alpha in PC_ALPHAS:
                try:
                    edges = run_pc(data, var_names, tiers, alpha, graph_out / 'pc')
                    edges['graph'] = graph_name
                    graph_edges.append(edges)
                    n_dir = (edges['edge_type'] == 'directed').sum() if 'edge_type' in edges.columns else 0
                    logging.info("    alpha=%.3f: %d directed edges", alpha, n_dir)
                except Exception as e:
                    logging.warning("    PC alpha=%.3f failed: %s", alpha, e)
                    import traceback; logging.debug(traceback.format_exc())
            summary_json[graph_name]['algorithms']['pc'] = {
                'alphas': PC_ALPHAS, 'runtime_s': round(time.time() - t0, 1)}

        # ---- NOTEARS ----
        if 'notears' in args.algorithms:
            logging.info("  Running NOTEARS (lambdas=%s)...", NOTEARS_LAMBDAS)
            t0 = time.time()
            for lam in NOTEARS_LAMBDAS:
                try:
                    edges = run_notears(data, var_names, tiers, lam, graph_out / 'notears')
                    edges['graph'] = graph_name
                    graph_edges.append(edges)
                    logging.info("    lambda=%.3f: %d edges (|w|>=%.3f)",
                                 lam, len(edges), NOTEARS_W_THRESH)
                except Exception as e:
                    logging.warning("    NOTEARS lambda=%.3f failed: %s", lam, e)
            summary_json[graph_name]['algorithms']['notears'] = {
                'lambdas': NOTEARS_LAMBDAS, 'runtime_s': round(time.time() - t0, 1)}

        # ---- LiNGAM ----
        if 'lingam' in args.algorithms:
            logging.info("  Running LiNGAM...")
            t0 = time.time()
            try:
                edges = run_lingam(data, var_names, tiers, graph_out / 'lingam')
                edges['graph'] = graph_name
                graph_edges.append(edges)
                logging.info("    %d edges (|w|>=%.3f)", len(edges), LINGAM_W_THRESH)
                summary_json[graph_name]['algorithms']['lingam'] = {
                    'runtime_s': round(time.time() - t0, 1)}
            except Exception as e:
                logging.warning("    LiNGAM failed: %s", e)

        all_edges_collected.extend(graph_edges)

    # -----------------------------------------------------------------------
    # Cross-algorithm agreement summary
    # -----------------------------------------------------------------------
    agreement = compute_agreement(all_edges_collected)
    if len(agreement) > 0:
        agreement.to_csv(out_root / 'cross_algorithm_summary.csv', index=False)

        # Edges agreed on by all 3 algorithms per graph
        for graph_name in args.graph:
            g_agree = agreement[
                (agreement['graph'] == graph_name) &
                (agreement['n_algorithms'] == len(args.algorithms))
            ]
            summary_json[graph_name]['agreed_edges'] = g_agree[
                ['source', 'target', 'algorithms']].to_dict(orient='records')

        logging.info("")
        logging.info("=== CROSS-ALGORITHM AGREEMENT ===")
        for graph_name in args.graph:
            agreed = [e for e in summary_json.get(graph_name, {}).get('agreed_edges', [])]
            logging.info("  %-12s  %d edges agreed by all algorithms: %s",
                         graph_name, len(agreed),
                         [(e['source'], '->', e['target']) for e in agreed])

    with open(out_root / 'causal_discovery.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    logging.info("")
    logging.info("Outputs written to %s", out_root)
    logging.info("Step 21c complete.")
