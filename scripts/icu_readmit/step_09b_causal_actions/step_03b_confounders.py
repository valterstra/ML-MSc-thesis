"""
Step 09b.3b -- Causal discovery with explicit confounders (PC only).

Same 5 focused graphs as step_03_causal_discovery, but adds three static confounders to each graph:
  age, charlson_score, prior_ed_visits_6m

These are the top predictors from step_01 Model 2 (static features predicting
readmission) and are included here because they affect both drug prescription
(confounding by indication) and readmission risk. Without conditioning on them,
PC cannot cleanly separate the treatment effect from the selection effect.

New temporal tier ordering (4 tiers):
  Tier 0: confounders      -- age, charlson_score, prior_ed_visits_6m (pre-admission, static)
  Tier 1: frac_action      -- cumulative drug exposure during stay
  Tier 2: delta_state      -- physiological change during stay
  Tier 3: readmit_30d      -- outcome after discharge

Each graph: 8 nodes (3 confounders + 1 action + 3 state deltas + readmit_30d)

Outputs: reports/icu_readmit/step_09b_causal_actions/step_03b_confounders/
  <graph_name>/pc/edges_alpha0.05.csv
  <graph_name>/pc/edges_alpha0.01.csv
  <graph_name>/pc/adjacency_alpha0.05.csv
  <graph_name>/pc/adjacency_alpha0.01.csv
  cross_algorithm_summary.csv
  causal_discovery_v2.json

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_03b_confounders.py
    python scripts/icu_readmit/step_03b_confounders.py --graph vasopressor ivfluid
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
# Confounders added in v2
# ---------------------------------------------------------------------------
CONFOUNDERS = ['age', 'charlson_score', 'prior_ed_visits_6m']

# ---------------------------------------------------------------------------
# Graph definitions (unchanged from step_09c)
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

PC_ALPHAS = [0.05, 0.01]


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def build_stay_level(df: pd.DataFrame, graph_def: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Build stay-level dataset for one graph (8 nodes):
      - age, charlson_score, prior_ed_visits_6m  (first value per stay -- static)
      - frac_<action>                             (fraction of blocs drug active)
      - delta_<state> x3                          (last_bloc - first_bloc)
      - readmit_30d                               (outcome)

    Only stays with all 8 variables non-NaN are kept.
    """
    action_raw = graph_def['action']
    states     = graph_def['states']

    df_sorted = df.sort_values([C_ICUSTAYID, C_BLOC])
    grp = df_sorted.groupby(C_ICUSTAYID)

    # Confounders: static per stay, take first value
    confounder_series = {c: grp[c].first() for c in CONFOUNDERS if c in df.columns}

    # Action fraction
    if action_raw in ('vasopressor_dose', 'ivfluid_dose'):
        action_frac = (grp[action_raw].apply(lambda x: (x > 0).mean())).rename(f'frac_{action_raw}')
    else:
        action_frac = grp[action_raw].mean().rename(f'frac_{action_raw}')

    # Delta states
    delta_cols = {}
    for s in states:
        if s not in df.columns:
            continue
        delta_cols[f'delta_{s}'] = grp[s].last() - grp[s].first()

    # Outcome
    outcome = grp[C_READMIT_30D].first()

    stay_df = pd.concat(
        [pd.Series(v, name=k) for k, v in confounder_series.items()]
        + [action_frac]
        + [pd.Series(v, name=k) for k, v in delta_cols.items()]
        + [outcome],
        axis=1
    ).reset_index()

    # Column order: confounders, frac_action, delta_states, readmit_30d
    graph_cols = (
        list(confounder_series.keys())
        + [f'frac_{action_raw}']
        + list(delta_cols.keys())
        + [C_READMIT_30D]
    )

    stay_df = stay_df.dropna(subset=graph_cols)
    return stay_df, graph_cols


def get_var_tiers(graph_cols: list[str]) -> dict[str, int]:
    """
    Assign temporal tier to each variable:
      0 = confounders         (pre-admission, static)
      1 = frac_action         (treatment exposure during stay)
      2 = delta_state         (physiological response during stay)
      3 = readmit_30d         (outcome after discharge)
    """
    tiers = {}
    for c in graph_cols:
        if c in CONFOUNDERS:
            tiers[c] = 0
        elif c.startswith('frac_'):
            tiers[c] = 1
        elif c.startswith('delta_'):
            tiers[c] = 2
        elif c == C_READMIT_30D:
            tiers[c] = 3
    return tiers


# ---------------------------------------------------------------------------
# PC algorithm (unchanged logic from step_09c)
# ---------------------------------------------------------------------------

def run_pc(data: np.ndarray, var_names: list[str], tiers: dict[str, int],
           alpha: float, out_dir: Path) -> pd.DataFrame:
    from causallearn.search.ConstraintBased.PC import pc
    from causallearn.utils.cit import fisherz

    cg = pc(data, alpha=alpha, indep_test=fisherz,
            node_names=var_names, show_progress=False)

    edges = extract_pc_edges(cg.G, var_names)

    if len(edges) > 0:
        def temporally_valid(row):
            if row['edge_type'] == 'directed':
                return tiers.get(row['source'], 0) <= tiers.get(row['target'], 3)
            return True
        edges = edges[edges.apply(temporally_valid, axis=1)].reset_index(drop=True)

    edges['alpha'] = alpha

    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / f'edges_alpha{alpha}.csv', index=False)

    adj = pd.DataFrame(cg.G.graph, index=var_names, columns=var_names)
    adj.to_csv(out_dir / f'adjacency_alpha{alpha}.csv')

    return edges


def extract_pc_edges(G, var_names: list[str]) -> pd.DataFrame:
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
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 09b.3b: Causal discovery with confounders (PC only)')
    parser.add_argument('--input', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'reports' / 'icu_readmit' / 'step_09b_causal_actions' / 'step_03b_confounders'))
    parser.add_argument('--graph', nargs='+', default=list(GRAPHS.keys()),
                        choices=list(GRAPHS.keys()),
                        help='Which graphs to run (default: all 5)')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'step_09b_step_03b_confounders.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 09b.3b started (PC only, with confounders).")
    logging.info("Confounders: %s", CONFOUNDERS)
    logging.info("Graphs: %s", args.graph)

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    all_edges_collected = []
    summary_json = {}

    for graph_name in args.graph:
        graph_def = GRAPHS[graph_name]
        logging.info("")
        logging.info("=== Graph: %s ===", graph_name)
        logging.info("  %s", graph_def['description'])

        stay_df, graph_cols = build_stay_level(df, graph_def)
        tiers = get_var_tiers(graph_cols)
        var_names = graph_cols

        logging.info("  %d stays after dropping NaN rows", len(stay_df))
        logging.info("  Variables: %s", var_names)
        logging.info("  Tiers: %s", tiers)

        scaler = StandardScaler()
        data = scaler.fit_transform(stay_df[var_names].values.astype(float))

        graph_out = out_root / graph_name
        graph_edges = []
        summary_json[graph_name] = {
            'description': graph_def['description'],
            'confounders': CONFOUNDERS,
            'nodes': var_names,
            'tiers': tiers,
            'edges': {},
        }

        logging.info("  Running PC (alphas=%s)...", PC_ALPHAS)
        t0 = time.time()
        for alpha in PC_ALPHAS:
            try:
                edges = run_pc(data, var_names, tiers, alpha, graph_out / 'pc')
                edges['graph'] = graph_name
                graph_edges.append(edges)
                n_dir = (edges['edge_type'] == 'directed').sum() if 'edge_type' in edges.columns else 0
                logging.info("    alpha=%.2f: %d directed edges", alpha, n_dir)
                if n_dir > 0:
                    dir_edges = edges[edges['edge_type'] == 'directed']
                    for _, row in dir_edges.iterrows():
                        logging.info("      %s -> %s", row['source'], row['target'])
                summary_json[graph_name]['edges'][f'alpha_{alpha}'] = (
                    edges[edges['edge_type'] == 'directed'][['source', 'target']]
                    .to_dict(orient='records') if 'edge_type' in edges.columns else []
                )
            except Exception as e:
                logging.warning("    PC alpha=%.2f failed: %s", alpha, e)
                import traceback; logging.debug(traceback.format_exc())

        summary_json[graph_name]['runtime_s'] = round(time.time() - t0, 1)
        all_edges_collected.extend(graph_edges)

    # Summary across all graphs
    logging.info("")
    logging.info("=== SUMMARY (alpha=0.05, directed edges only) ===")
    for graph_name in args.graph:
        edges_005 = summary_json[graph_name]['edges'].get('alpha_0.05', [])
        logging.info("  %-12s  %d edges: %s",
                     graph_name, len(edges_005),
                     [(e['source'], '->', e['target']) for e in edges_005])

    with open(out_root / 'causal_discovery_v2.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    logging.info("")
    logging.info("Outputs written to %s", out_root)
    logging.info("Step 09b.3b complete.")
