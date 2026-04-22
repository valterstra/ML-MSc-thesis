"""
Step 04 -- Action -> state causal stability analysis (FCI-based).

PURPOSE
-------
Identify which drug actions causally shift the top discharge-state variables,
using the same random stability FCI approach as step_03.

APPROACH
--------
For each of 9 drug action classes, run N_RUNS_PER_DRUG FCI iterations.
Each iteration:
  1. Sample 1 state variable from the 28-variable pool
     (step_03 pool minus last_input_total, which is circular with ivfluid action)
  2. Subsample N_SAMPLE stays from the full cohort
  3. Run FCI on a 6-node graph:
       [age, charlson_score, prior_ed_visits_6m, num_blocs | frac_drug | delta_state]
       Tier 0: confounders   Tier 1: drug   Tier 2: delta_state
  4. Check for directed edge frac_drug -> delta_state in the PAG

Drug representation:
  frac_active = fraction of blocs where drug was active (dose>0 or binary flag=1).
  Gives a consistent 0-1 scale for all 9 drugs.

State representation:
  delta = last_value - first_value (physiological change over the ICU stay).
  Captures treatment effect, not absolute level.

Confounders include num_blocs (stay length) -- a critical confounder since
longer stays accumulate more drug exposure and larger physiological deltas.

PAG edge encoding (causal-learn GeneralGraph, confirmed empirically):
  adj[i,j] = -1 -> arrowhead at j end
  adj[i,j] =  1 -> tail at j end
  adj[i,j] =  2 -> circle at j end

OUTPUT
------
  reports/icu_readmit/step_09_state_action_selection/action_state_results/
    frac_<drug>_results.csv   -- ranked state list per drug
  reports/icu_readmit/step_09_state_action_selection/action_state_frequency_matrix.csv
  reports/icu_readmit/step_09_state_action_selection/action_state_summary.json

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_09_state_action_selection/step_04_action_state_stability.py --smoke
    python scripts/icu_readmit/step_09_state_action_selection/step_04_action_state_stability.py
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_BLOC,
    C_VASOPRESSOR_DOSE, C_IVFLUID_DOSE,
    C_ANTIBIOTIC_ACTIVE, C_ANTICOAGULANT_ACTIVE,
    C_DIURETIC_ACTIVE, C_STEROID_ACTIVE,
    C_INSULIN_ACTIVE, C_SEDATION_ACTIVE,
    C_MECHVENT,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

CONFOUNDER_COLS = ['age', 'charlson_score', 'prior_ed_visits_6m', 'num_blocs']

# 9 drug actions: (raw_col, frac_col, is_continuous_dose)
DRUGS = [
    (C_VASOPRESSOR_DOSE,     'frac_vasopressor',   True),
    (C_IVFLUID_DOSE,         'frac_ivfluid',        True),
    (C_ANTIBIOTIC_ACTIVE,    'frac_antibiotic',     False),
    (C_ANTICOAGULANT_ACTIVE, 'frac_anticoagulant',  False),
    (C_DIURETIC_ACTIVE,      'frac_diuretic',       False),
    (C_STEROID_ACTIVE,       'frac_steroid',        False),
    (C_INSULIN_ACTIVE,       'frac_insulin',        False),
    (C_SEDATION_ACTIVE,      'frac_sedation',       False),
    (C_MECHVENT,             'frac_mechvent',       False),
]

# State pool: step_03 pool (29 vars) minus input_total (circular with ivfluid) = 28
# Raw column names (without last_ prefix); delta = last - first
STATE_POOL_RAW = [
    'BUN', 'Hb', 'Platelets_count', 'WBC_count',
    'cumulated_balance', 'Creatinine', 'PT', 'input_4hourly_tev',
    'PTT', 'Glucose', 'output_total', 'HR', 'RR',
    'Alkaline_Phosphatase', 'Ht', 'Temp_C', 'SpO2',
    'Phosphate', 'Shock_Index', 'Chloride', 'CO2_mEqL',
    'Fibrinogen', 'SGOT', 'Pain_Level', 'Lymphs_pct',
    'Sodium', 'paCO2', 'TidalVolume_Observed',
]
DELTA_POOL = ['delta_' + s for s in STATE_POOL_RAW]


# ---------------------------------------------------------------------------
# Build stay-level feature table
# ---------------------------------------------------------------------------

def build_action_state_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bloc-level data to stay-level.

    Confounders : age, charlson_score, prior_ed_visits_6m  (first bloc)
    num_blocs   : count of blocs per stay
    frac_drug   : fraction of blocs where drug was active (dose>0 or flag=1)
    delta_state : last_value - first_value for each state variable
    """
    df = df.sort_values([C_ICUSTAYID, C_BLOC])
    grp = df.groupby(C_ICUSTAYID)

    parts = []

    # Confounders
    conf_base = ['age', 'charlson_score', 'prior_ed_visits_6m']
    present_conf = [c for c in conf_base if c in df.columns]
    parts.append(grp[present_conf].first())

    # Stay length
    parts.append(grp[C_BLOC].count().rename('num_blocs'))

    # Drug frac_active
    for raw_col, frac_col, is_continuous in DRUGS:
        if raw_col not in df.columns:
            log.warning("Drug column not found in dataset: %s", raw_col)
            continue
        if is_continuous:
            frac = grp[raw_col].apply(lambda x: (x > 0).mean())
        else:
            frac = grp[raw_col].mean()
        parts.append(frac.rename(frac_col))

    # Delta states
    present_states = [s for s in STATE_POOL_RAW if s in df.columns]
    missing_states = [s for s in STATE_POOL_RAW if s not in df.columns]
    if missing_states:
        log.warning("State columns not found (skipped): %s", missing_states)
    for s in present_states:
        delta = grp[s].last() - grp[s].first()
        parts.append(delta.rename('delta_' + s))

    stay_df = pd.concat(parts, axis=1).reset_index()
    return stay_df


# ---------------------------------------------------------------------------
# FCI helpers
# ---------------------------------------------------------------------------

def build_background_knowledge(nodes, col_names: list[str], tier: dict):
    from causallearn.utils.PCUtils.BackgroundKnowledge import BackgroundKnowledge
    bk = BackgroundKnowledge()
    node_map = {col_names[i]: nodes[i] for i in range(len(col_names))}
    for ni in col_names:
        for nj in col_names:
            if ni != nj and tier.get(ni, 1) > tier.get(nj, 1):
                bk.add_forbidden_by_node(node_map[ni], node_map[nj])
    return bk


def run_fci_once(data: np.ndarray, col_names: list[str], tier: dict, alpha: float):
    """Two-call FCI pattern: first without BK to get nodes, second with tier BK."""
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    try:
        g_init, _ = fci(data, fisherz, alpha,
                        node_names=col_names, verbose=False, show_progress=False)
        nodes = g_init.get_nodes()
        bk = build_background_knowledge(nodes, col_names, tier)
        g_final, _ = fci(data, fisherz, alpha,
                         node_names=col_names, background_knowledge=bk,
                         verbose=False, show_progress=False)
        return g_final
    except Exception as e:
        log.debug("FCI failed: %s", e)
        return None


def extract_drug_to_state_edge(g, col_names: list[str],
                                drug_col: str, state_col: str) -> str | None:
    """
    Return edge type for drug_col -> state_col, or None.

    PAG encoding (confirmed empirically):
      adj[i,j] = -1 -> arrowhead at j end
      adj[i,j] =  1 -> tail at j end
      adj[i,j] =  2 -> circle at j end

    Directed   drug -> state:  adj[drug,state]==-1 AND adj[state,drug]==1
    Possible   drug o-> state: adj[drug,state]==-1 AND adj[state,drug]==2
    Bidirected drug <-> state: adj[drug,state]==-1 AND adj[state,drug]==-1
    """
    if g is None:
        return None
    try:
        adj = g.graph
    except AttributeError:
        return None
    if drug_col not in col_names or state_col not in col_names:
        return None

    di = col_names.index(drug_col)
    si = col_names.index(state_col)
    d_s = adj[di, si]   # mark at state end
    s_d = adj[si, di]   # mark at drug end

    if d_s == -1 and s_d == 1:
        return 'definite'
    elif d_s == -1 and s_d == 2:
        return 'possible'
    elif d_s == -1 and s_d == -1:
        return 'bidirected'
    return None


# ---------------------------------------------------------------------------
# Per-drug stability loop
# ---------------------------------------------------------------------------

def run_drug_stability(
    stay_df: pd.DataFrame,
    drug_frac_col: str,
    delta_pool: list[str],
    n_runs: int,
    n_sample: int,
    alpha: float,
    rng: random.Random,
    run_offset: int,
    log_every: int = 100,
) -> dict:
    """Run N_RUNS FCI iterations for one drug. Returns raw counters."""
    n_included = defaultdict(int)
    n_definite = defaultdict(int)
    n_possible = defaultdict(int)
    n_any      = defaultdict(int)
    n_failed   = 0

    scaler = StandardScaler()

    # Tier map for this drug (confounders=0, drug=1, state deltas=2)
    tier_base = {c: 0 for c in CONFOUNDER_COLS}
    tier_base[drug_frac_col] = 1

    t_start = time.time()

    for run_idx in range(n_runs):
        # Sample one state variable
        state_col = rng.choice(delta_pool)

        # Build column list and tier map
        present_conf = [c for c in CONFOUNDER_COLS if c in stay_df.columns]
        col_names = present_conf + [drug_frac_col, state_col]
        tier = dict(tier_base)
        tier[state_col] = 2

        # Subsample stays
        n_draw = min(n_sample, len(stay_df))
        subset = stay_df[col_names].sample(n=n_draw, random_state=run_offset + run_idx)
        subset = subset.dropna()

        if len(subset) < 100:
            n_failed += 1
            continue

        # Skip if drug column has no variance (e.g. steroid_active=0% in MIMIC-IV)
        if subset[drug_frac_col].std() < 1e-6:
            n_failed += 1
            continue

        n_included[state_col] += 1

        data = scaler.fit_transform(subset.values.astype(np.float64))
        g = run_fci_once(data, col_names, tier, alpha)

        if g is None:
            n_failed += 1
            continue

        edge = extract_drug_to_state_edge(g, col_names, drug_frac_col, state_col)
        if edge == 'definite':
            n_definite[state_col] += 1
            n_any[state_col] += 1
        elif edge == 'possible':
            n_possible[state_col] += 1
            n_any[state_col] += 1
        # bidirected: not counted as causal

        if (run_idx + 1) % log_every == 0:
            elapsed = time.time() - t_start
            est_remaining = elapsed / (run_idx + 1) * (n_runs - run_idx - 1)
            log.info("  Run %4d / %d  |  elapsed %.0fs  |  est. remaining %.0fs  |  failed %d",
                     run_idx + 1, n_runs, elapsed, est_remaining, n_failed)

    return {
        'n_included': dict(n_included),
        'n_definite': dict(n_definite),
        'n_possible': dict(n_possible),
        'n_any':      dict(n_any),
        'n_failed':   n_failed,
    }


def build_drug_results_df(counters: dict, delta_pool: list[str]) -> pd.DataFrame:
    rows = []
    for sc in delta_pool:
        ni  = counters['n_included'].get(sc, 0)
        nd  = counters['n_definite'].get(sc, 0)
        np_ = counters['n_possible'].get(sc, 0)
        na  = counters['n_any'].get(sc, 0)
        rows.append({
            'state':            sc,
            'n_runs_included':  ni,
            'n_definite':       nd,
            'n_possible':       np_,
            'n_any':            na,
            'freq_definite':    round(nd / ni, 4) if ni > 0 else 0.0,
            'freq_any':         round(na / ni, 4) if ni > 0 else 0.0,
        })
    df = (pd.DataFrame(rows)
            .sort_values('freq_definite', ascending=False)
            .reset_index(drop=True))
    df.insert(0, 'rank', df.index + 1)
    return df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    input_path  = Path(args.input)
    report_dir  = Path(args.report_dir)
    results_dir = report_dir / 'action_state_results'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Only load columns we need -- reduces memory ~3.5x vs full 146-col load
    need_cols = (
        [C_ICUSTAYID, C_BLOC]
        + ['age', 'charlson_score', 'prior_ed_visits_6m']
        + [raw for raw, _, _ in DRUGS]
        + STATE_POOL_RAW
    )
    log.info("Loading %s (selected columns only) ...", input_path)
    # Read header to find which columns actually exist
    header = pd.read_csv(input_path, nrows=0).columns.tolist()
    use_cols = [c for c in need_cols if c in header]
    df_raw = pd.read_csv(input_path, usecols=use_cols, low_memory=False)
    log.info("Loaded: %d rows, %d columns (of %d total)", len(df_raw), len(df_raw.columns), len(header))

    if args.smoke:
        keep_stays = df_raw[C_ICUSTAYID].unique()[:3000]
        df_raw = df_raw[df_raw[C_ICUSTAYID].isin(keep_stays)].copy()
        log.info("Smoke: restricted to %d stays (%d rows)", len(keep_stays), len(df_raw))

    log.info("Building stay-level action+state table...")
    stay_df = build_action_state_table(df_raw)
    del df_raw
    log.info("Stay-level table: %d stays, %d columns", len(stay_df), len(stay_df.columns))

    # Filter to columns that actually exist
    delta_pool = [c for c in DELTA_POOL if c in stay_df.columns]
    drug_list  = [(raw, frac, cont) for raw, frac, cont in DRUGS if frac in stay_df.columns]
    log.info("Drugs: %d   State delta pool: %d", len(drug_list), len(delta_pool))

    n_runs   = args.n_runs_per_drug
    n_sample = args.n_sample
    alpha    = args.alpha
    seed     = args.seed
    rng      = random.Random(seed)
    np.random.seed(seed)

    log.info("Config: n_runs_per_drug=%d  n_sample=%d  alpha=%.3f  seed=%d",
             n_runs, n_sample, alpha, seed)
    log.info("Total FCI runs: %d", n_runs * len(drug_list))

    all_results: dict[str, pd.DataFrame] = {}

    for drug_idx, (raw_col, frac_col, _) in enumerate(drug_list):
        log.info("")
        log.info("=== Drug %d/%d: %s ===", drug_idx + 1, len(drug_list), frac_col)
        t_drug = time.time()

        # Use a unique random offset per drug so subsamples differ across drugs
        run_offset = drug_idx * n_runs * 13

        counters = run_drug_stability(
            stay_df, frac_col, delta_pool,
            n_runs, n_sample, alpha, rng, run_offset,
        )

        drug_df = build_drug_results_df(counters, delta_pool)
        all_results[frac_col] = drug_df

        out_path = results_dir / f'{frac_col}_results.csv'
        drug_df.to_csv(out_path, index=False)
        log.info("Saved: %s", out_path)
        log.info("Failed: %d / %d   Time: %.1fs",
                 counters['n_failed'], n_runs, time.time() - t_drug)
        log.info("Top 5 states for %s:", frac_col)
        for _, row in drug_df.head(5).iterrows():
            log.info("  %-30s  freq=%.3f  (n=%d)",
                     row['state'], row['freq_definite'], row['n_runs_included'])

    # --- Frequency matrix (drugs x states) ---
    freq_matrix = pd.DataFrame(
        {frac: {row['state']: row['freq_definite'] for _, row in df.iterrows()}
         for frac, df in all_results.items()},
        index=delta_pool,
    ).T
    freq_matrix.index.name = 'drug'
    matrix_path = report_dir / 'action_state_frequency_matrix.csv'
    freq_matrix.to_csv(matrix_path)
    log.info("Frequency matrix saved: %s", matrix_path)

    # --- Summary JSON ---
    pairs = []
    for frac_col, drug_df in all_results.items():
        for _, row in drug_df.iterrows():
            if row['freq_definite'] > 0:
                pairs.append({
                    'drug': frac_col,
                    'state': row['state'],
                    'freq_definite': row['freq_definite'],
                    'n_runs_included': row['n_runs_included'],
                })
    pairs.sort(key=lambda x: -x['freq_definite'])

    summary = {
        'n_stays':          len(stay_df),
        'n_drugs':          len(drug_list),
        'n_states':         len(delta_pool),
        'n_runs_per_drug':  n_runs,
        'n_sample':         n_sample,
        'alpha':            alpha,
        'runtime_s':        round(time.time() - t0, 1),
        'top_pairs':        pairs[:30],
    }
    summary_path = report_dir / 'action_state_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved: %s", summary_path)

    # --- Console summary ---
    log.info("")
    log.info("=" * 80)
    log.info("ACTION -> STATE STABILITY  (n_runs_per_drug=%d  alpha=%.2f)", n_runs, alpha)
    log.info("Top (drug, state) pairs by freq_definite:")
    log.info("%-25s  %-30s  %s", "Drug", "State", "Freq_Definite")
    log.info("-" * 65)
    for p in pairs[:20]:
        log.info("%-25s  %-30s  %.3f  (n=%d)",
                 p['drug'], p['state'], p['freq_definite'], p['n_runs_included'])
    log.info("=" * 80)
    log.info("Total time: %.1fs", time.time() - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--input', default=str(
            PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'),
        help='Path to ICUdataset.csv')
    parser.add_argument(
        '--report-dir', default=str(
            PROJECT_ROOT / 'reports' / 'icu_readmit' / 'step_09_state_action_selection'),
        help='Output directory')
    parser.add_argument(
        '--n-runs-per-drug', type=int, default=3000,
        help='FCI iterations per drug (default 3000)')
    parser.add_argument(
        '--n-sample', type=int, default=5000,
        help='Stays subsampled per run (default 5000)')
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='FCI significance threshold (default 0.05)')
    parser.add_argument(
        '--seed', type=int, default=42)
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: 3000 stays, 20 runs/drug, 500 sample')
    args = parser.parse_args()

    if args.smoke:
        args.n_runs_per_drug = 20
        args.n_sample = 500

    main(args)
