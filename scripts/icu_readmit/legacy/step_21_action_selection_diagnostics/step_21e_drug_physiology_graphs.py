"""
Step 21e -- Legacy action-selection diagnostics: drug -> physiology graphs.

Same 5 graphs and 3 confounders as v2, but readmit_30d is removed entirely.
The only question asked: does this drug causally shift its target physiological
variables, after conditioning on age, charlson_score, prior_ed_visits_6m?

This is a cleaner test of the drug -> physiology causal claim. Readmission is
tested separately in step_21f_discharge_readmission_graphs using absolute discharge-state values.

Nodes per graph (7):
  Tier 0: age, charlson_score, prior_ed_visits_6m  (static confounders)
  Tier 1: frac_drug                                 (treatment intensity)
  Tier 2: delta_physiology x3                       (change during stay)

Outputs: reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21e_drug_physiology_graphs/

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21e_drug_physiology_graphs.py
    python scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21e_drug_physiology_graphs.py --graph vasopressor ivfluid
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

GRAPHS = {
    'vasopressor': {
        'action': 'vasopressor_dose',
        'states': ['HR', 'Arterial_BP_Sys', 'Shock_Index'],
    },
    'ivfluid': {
        'action': 'ivfluid_dose',
        'states': ['BUN', 'Creatinine', 'Potassium'],
    },
    'antibiotic': {
        'action': 'antibiotic_active',
        'states': ['WBC_count', 'Temp_C', 'RR'],
    },
    'sedation': {
        'action': 'sedation_active',
        'states': ['HR', 'RR', 'SpO2'],
    },
    'diuretic': {
        'action': 'diuretic_active',
        'states': ['Potassium', 'BUN', 'Creatinine'],
    },
}

PC_ALPHAS = [0.05, 0.01]


def build_stay_level(df: pd.DataFrame, graph_def: dict) -> tuple[pd.DataFrame, list[str]]:
    """
    Build stay-level dataset: confounders + frac_drug + delta_states.
    No readmission column.
    """
    action_raw = graph_def['action']
    states     = graph_def['states']

    df_sorted = df.sort_values([C_ICUSTAYID, C_BLOC])
    grp = df_sorted.groupby(C_ICUSTAYID)

    confounder_series = {c: grp[c].first() for c in CONFOUNDERS if c in df.columns}

    if action_raw in ('vasopressor_dose', 'ivfluid_dose'):
        action_frac = (grp[action_raw].apply(lambda x: (x > 0).mean())).rename(f'frac_{action_raw}')
    else:
        action_frac = grp[action_raw].mean().rename(f'frac_{action_raw}')

    delta_cols = {}
    for s in states:
        if s not in df.columns:
            continue
        delta_cols[f'delta_{s}'] = grp[s].last() - grp[s].first()

    stay_df = pd.concat(
        [pd.Series(v, name=k) for k, v in confounder_series.items()]
        + [action_frac]
        + [pd.Series(v, name=k) for k, v in delta_cols.items()],
        axis=1
    ).reset_index()

    graph_cols = (
        list(confounder_series.keys())
        + [f'frac_{action_raw}']
        + list(delta_cols.keys())
    )

    stay_df = stay_df.dropna(subset=graph_cols)
    return stay_df, graph_cols


def get_var_tiers(graph_cols: list[str]) -> dict[str, int]:
    tiers = {}
    for c in graph_cols:
        if c in CONFOUNDERS:
            tiers[c] = 0
        elif c.startswith('frac_'):
            tiers[c] = 1
        elif c.startswith('delta_'):
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
        description='Step 21e: Drug -> physiology causal graphs (no readmission)')
    parser.add_argument('--input', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'reports' / 'icu_readmit' / 'legacy' / 'step_21_action_selection_diagnostics' / 'step_21e_drug_physiology_graphs'))
    parser.add_argument('--graph', nargs='+', default=list(GRAPHS.keys()),
                        choices=list(GRAPHS.keys()))
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    out_root = Path(args.out_dir)
    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'step_21e_drug_physiology_graphs.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    out_root.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 21e started (drug -> physiology, no readmission).")
    logging.info("Confounders: %s", CONFOUNDERS)
    logging.info("Graphs: %s", args.graph)

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    summary_json = {}

    for graph_name in args.graph:
        graph_def = GRAPHS[graph_name]
        logging.info("")
        logging.info("=== Graph: %s ===", graph_name)

        stay_df, graph_cols = build_stay_level(df, graph_def)
        tiers = get_var_tiers(graph_cols)
        var_names = graph_cols

        logging.info("  %d stays, variables: %s", len(stay_df), var_names)
        logging.info("  Tiers: %s", tiers)

        scaler = StandardScaler()
        data = scaler.fit_transform(stay_df[var_names].values.astype(float))

        graph_out = out_root / graph_name
        summary_json[graph_name] = {'nodes': var_names, 'tiers': tiers, 'edges': {}}

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
    logging.info("=== SUMMARY (alpha=0.05, drug -> physiology edges only) ===")
    for graph_name in args.graph:
        drug_col = f"frac_{GRAPHS[graph_name]['action']}"
        edges_005 = summary_json[graph_name]['edges'].get('alpha_0.05', [])
        drug_edges = [e for e in edges_005 if e['source'] == drug_col]
        logging.info("  %-12s  %s", graph_name,
                     [(e['source'], '->', e['target']) for e in drug_edges] or "no drug->physiology edges")

    with open(out_root / 'causal_discovery_v3.json', 'w') as f:
        json.dump(summary_json, f, indent=2)

    logging.info("Outputs written to %s", out_root)
    logging.info("Step 21e complete.")
