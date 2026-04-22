"""
Step 21f -- Legacy action-selection diagnostics: discharge state -> readmission graphs.

Five graphs mirroring the drug groupings from v3, but now asking:
does the patient's physiological state AT DISCHARGE predict 30-day readmission,
after conditioning on the same three confounders?

Uses absolute last-bloc values (not deltas) as the state representation.
At discharge, what matters is where the patient ended up, not how much they changed.

Nodes per graph (7):
  Tier 0: age, charlson_score, prior_ed_visits_6m  (static confounders)
  Tier 1: last_<state> x3                          (absolute discharge-state values)
  Tier 2: readmit_30d                               (outcome)

The five graphs use the same physiological variable groupings as the drug graphs:
  vasopressor group: last_HR, last_Arterial_BP_Sys, last_Shock_Index
  ivfluid group:     last_BUN, last_Creatinine, last_Potassium
  antibiotic group:  last_WBC_count, last_Temp_C, last_RR
  sedation group:    last_HR, last_RR, last_SpO2
  diuretic group:    last_Potassium, last_BUN, last_Creatinine

Combined argument (v3 + v4):
  v3: frac_drug -> delta_physiology  (drug causally shifts these variables)
  v4: last_physiology -> readmit_30d (discharge state predicts readmission)
  => drug treatment -> better discharge state -> lower readmission risk

Outputs: reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21f_discharge_readmission_graphs/

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21f_discharge_readmission_graphs.py
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21f_discharge_readmission_graphs.py --graph vasopressor ivfluid
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

CONFOUNDERS = ['age', 'charlson_score', 'prior_ed_visits_6m']

# Same physiological groupings as the drug graphs -- no action variable
GRAPHS = {
    'vasopressor': {
        'states': ['HR', 'Arterial_BP_Sys', 'Shock_Index'],
        'description': 'Discharge hemodynamics -> readmission',
    },
    'ivfluid': {
        'states': ['BUN', 'Creatinine', 'Potassium'],
        'description': 'Discharge renal/electrolyte state -> readmission',
    },
    'antibiotic': {
        'states': ['WBC_count', 'Temp_C', 'RR'],
        'description': 'Discharge infection markers -> readmission',
    },
    'sedation': {
        'states': ['HR', 'RR', 'SpO2'],
        'description': 'Discharge cardiorespiratory state -> readmission',
    },
    'diuretic': {
        'states': ['Potassium', 'BUN', 'Creatinine'],
        'description': 'Discharge electrolyte/kidney state -> readmission',
    },
}

PC_ALPHAS = [0.05, 0.01]


def build_stay_level(df: pd.DataFrame, graph_def: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Build stay-level dataset: confounders + last-bloc state values + readmit_30d.
    State columns use absolute last-bloc value (not delta).
    """
    states = graph_def['states']

    df_sorted = df.sort_values([C_ICUSTAYID, C_BLOC])
    grp = df_sorted.groupby(C_ICUSTAYID)

    confounder_series = {c: grp[c].first() for c in CONFOUNDERS if c in df.columns}

    last_cols = {}
    for s in states:
        if s not in df.columns:
            continue
        last_cols[f'last_{s}'] = grp[s].last()

    outcome = grp[C_READMIT_30D].first()

    stay_df = pd.concat(
        [pd.Series(v, name=k) for k, v in confounder_series.items()]
        + [pd.Series(v, name=k) for k, v in last_cols.items()]
        + [outcome],
        axis=1
    ).reset_index()

    graph_cols = (
        list(confounder_series.keys())
        + list(last_cols.keys())
        + [C_READMIT_30D]
    )

    stay_df = stay_df.dropna(subset=graph_cols)
    return stay_df, graph_cols


def get_var_tiers(graph_cols: list[str]) -> dict[str, int]:
    tiers = {}
    for c in graph_cols:
        if c in CONFOUNDERS:
            tiers[c] = 0
        elif c.startswith('last_'):
            tiers[c] = 1
        elif c == C_READMIT_30D:
            tiers[c] = 2
    return tiers


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
                return tiers.get(row['source'], 0) <= tiers.get(row['target'], 2)
            return True
        edges = edges[edges.apply(temporally_valid, axis=1)].reset_index(drop=True)

    edges['alpha'] = alpha
    out_dir.mkdir(parents=True, exist_ok=True)
    edges.to_csv(out_dir / f'edges_alpha{alpha}.csv', index=False)
    pd.DataFrame(cg.G.graph, index=var_names, columns=var_names).to_csv(
        out_dir / f'adjacency_alpha{alpha}.csv')
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Step 21f: Discharge state -> readmission causal graphs')
    parser.add_argument('--input', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'reports' / 'icu_readmit' / 'legacy' / 'step_21_action_selection_diagnostics' / 'step_21f_discharge_readmission_graphs'))
    parser.add_argument('--graph', nargs='+', default=list(GRAPHS.keys()),
                        choices=list(GRAPHS.keys()))
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'step_21f_discharge_readmission_graphs.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 21f started (discharge state -> readmission).")
    logging.info("Confounders: %s", CONFOUNDERS)
    logging.info("Graphs: %s", args.graph)

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    summary_json = {}

    for graph_name in args.graph:
        graph_def = GRAPHS[graph_name]
        logging.info("")
        logging.info("=== Graph: %s ===", graph_name)
        logging.info("  %s", graph_def['description'])

        stay_df, graph_cols = build_stay_level(df, graph_def)
        tiers = get_var_tiers(graph_cols)
        var_names = graph_cols

        logging.info("  %d stays, variables: %s", len(stay_df), var_names)
        logging.info("  Tiers: %s", tiers)

        scaler = StandardScaler()
        data = scaler.fit_transform(stay_df[var_names].values.astype(float))

        graph_out = out_root / graph_name
        summary_json[graph_name] = {
            'description': graph_def['description'],
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
                n_dir = (edges['edge_type'] == 'directed').sum() if len(edges) > 0 else 0
                logging.info("    alpha=%.2f: %d directed edges", alpha, n_dir)
                if n_dir > 0:
                    for _, row in edges[edges['edge_type'] == 'directed'].iterrows():
                        logging.info("      %s -> %s", row['source'], row['target'])
                summary_json[graph_name]['edges'][f'alpha_{alpha}'] = (
                    edges[edges['edge_type'] == 'directed'][['source', 'target']]
                    .to_dict(orient='records') if len(edges) > 0 else []
                )
            except Exception as e:
                logging.warning("    PC alpha=%.2f failed: %s", alpha, e)
        summary_json[graph_name]['runtime_s'] = round(time.time() - t0, 1)

    logging.info("")
    logging.info("=== SUMMARY (alpha=0.05, discharge state -> readmission edges only) ===")
    for graph_name in args.graph:
        edges_005 = summary_json[graph_name]['edges'].get('alpha_0.05', [])
        readmit_edges = [e for e in edges_005 if e['target'] == C_READMIT_30D]
        logging.info("  %-12s  %s", graph_name,
                     [(e['source'], '->', e['target']) for e in readmit_edges] or "no -> readmit edges")

    with open(out_root / 'causal_discovery_v4.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    logging.info("Outputs written to %s", out_root)
    logging.info("Step 21f complete.")
