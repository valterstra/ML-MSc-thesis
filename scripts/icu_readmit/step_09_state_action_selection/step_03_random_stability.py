"""
Step 03 -- Random stability analysis (FCI-based).

PURPOSE
-------
Identify which discharge-state variables causally predict 30-day readmission
via a stability analysis over randomly sampled variable subsets.

THE APPROACH
------------
Instead of running a single causal discovery on all variables at once
(which degrades above ~10 variables), we run FCI thousands of times on
small random 6-node graphs:

  Fixed nodes (every run):
    age, charlson_score, prior_ed_visits_6m   <- confounders (tier 0)
    readmit_30d                               <- outcome     (tier 2)

  Random nodes (sampled fresh each run):
    2 state variables drawn without replacement from the candidate pool
                                              <- discharge state (tier 1)

  Total: 6 nodes per graph.

Each run also subsamples N_SAMPLE stays from the full dataset, so the
analysis randomises over both variables AND data. This prevents any single
outlier stay from dominating edge detection, and keeps Fisher-Z alpha
calibration reasonable.

FCI (rather than PC) is used because it explicitly allows for hidden
confounders -- it does not assume causal sufficiency. This is the right
choice for clinical observational data where unmeasured factors (genetics,
out-of-hospital care, lifestyle) always exist.

BACKGROUND KNOWLEDGE
--------------------
Tier ordering is enforced as forbidden edges in every run:
  - readmit_30d cannot cause anything (outcome is terminal)
  - state vars cannot cause confounders (physiology doesn't change age)
This prevents algorithmically discovered reverse-time edges.

EDGE TYPES TRACKED
------------------
For each state variable X per run:
  definite   : adj[X, readmit]=2 AND adj[readmit, X]=1
               -> X ---> readmit_30d (definite causal arrow, tail-arrow)
  possible   : adj[X, readmit]=3 AND adj[readmit, X]=2
               -> X o--> readmit_30d (possible directed, allows hidden cause)
  any        : definite OR possible (any forward edge to readmit_30d)

Edge matrix convention (causal-learn PAG):
  adj[i, j] = 1  -> tail at j
  adj[i, j] = 2  -> arrowhead at j
  adj[i, j] = 3  -> circle at j

OUTPUTS
-------
  reports/icu_readmit/step_09_state_action_selection/random_stability_results.csv
  reports/icu_readmit/step_09_state_action_selection/random_stability_summary.json

Columns in results CSV:
  variable          -- state variable name (last_*)
  n_runs_included   -- number of runs where this variable was in the graph
  n_definite        -- runs with definite directed edge to readmit_30d
  n_possible        -- runs with possible directed edge to readmit_30d
  n_any             -- runs with any forward edge to readmit_30d
  freq_definite     -- n_definite / n_runs_included
  freq_any          -- n_any / n_runs_included
  rank_definite     -- rank by freq_definite (1 = most frequent)
  rank_any          -- rank by freq_any

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_09_state_action_selection/step_03_random_stability.py
    python scripts/icu_readmit/step_09_state_action_selection/step_03_random_stability.py --smoke
    python scripts/icu_readmit/step_09_state_action_selection/step_03_random_stability.py \\
        --n-runs 3000 --n-sample 5000 --n-state-vars 2 --alpha 0.05
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

from careai.icu_readmit.columns import C_READMIT_30D

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Fixed and candidate configuration
# ---------------------------------------------------------------------------

CONFOUNDER_COLS = ['age', 'charlson_score', 'prior_ed_visits_6m']

# Candidate state variable pool: top 30 from step_02 minus last_PAPsys (88% missing)
STATE_POOL = [
    'last_BUN',
    'last_Hb',
    'last_Platelets_count',
    'last_WBC_count',
    'last_cumulated_balance',
    'last_input_total',
    'last_Creatinine',
    'last_PT',
    'last_input_4hourly_tev',
    'last_PTT',
    'last_Glucose',
    'last_output_total',
    'last_HR',
    'last_RR',
    'last_Alkaline_Phosphatase',
    'last_Ht',
    'last_Temp_C',
    'last_SpO2',
    # last_PAPsys excluded (88% missing)
    'last_Phosphate',
    'last_Shock_Index',
    'last_Chloride',
    'last_CO2_mEqL',
    'last_Fibrinogen',
    'last_SGOT',
    'last_Pain_Level',
    'last_Lymphs_pct',
    'last_Sodium',
    'last_paCO2',
    'last_TidalVolume_Observed',
]

# Tier assignments for background knowledge
TIER = {
    'age':                  0,
    'charlson_score':       0,
    'prior_ed_visits_6m':   0,
    C_READMIT_30D:          2,
}
# State vars get tier 1 (assigned dynamically per run)


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


def run_fci_once(data: np.ndarray, col_names: list[str],
                 tier: dict, alpha: float):
    """
    Run FCI on data with tier-based background knowledge.
    Returns the PAG graph object, or None on failure.
    Two calls: first without BK (to obtain node objects), second with BK.
    """
    from causallearn.search.ConstraintBased.FCI import fci
    from causallearn.utils.cit import fisherz
    try:
        # Pass 1: get node objects (no BK)
        g_init, _ = fci(data, fisherz, alpha,
                        node_names=col_names, verbose=False, show_progress=False)
        nodes = g_init.get_nodes()

        # Pass 2: run with tier background knowledge
        bk = build_background_knowledge(nodes, col_names, tier)
        g_final, _ = fci(data, fisherz, alpha,
                         node_names=col_names, background_knowledge=bk,
                         verbose=False, show_progress=False)
        return g_final
    except Exception as e:
        log.debug("FCI failed: %s", e)
        return None


def extract_edges_to_readmit(g, col_names: list[str]) -> dict[str, str]:
    """
    For each state variable in col_names, return the edge type toward
    readmit_30d, or None if no such edge.

    PAG adj matrix convention (causal-learn GeneralGraph, confirmed empirically):
      adj[i, j] = -1 -> arrowhead at j end
      adj[i, j] =  1 -> tail at j end
      adj[i, j] =  2 -> circle at j end
      adj[i, j] =  0 -> no edge

    Directed   X -> readmit:  adj[X, R]=-1 AND adj[R, X]=1
    Possible   X o-> readmit: adj[X, R]=-1 AND adj[R, X]=2
    Bidirected X <-> readmit: adj[X, R]=-1 AND adj[R, X]=-1
    """
    if g is None:
        return {}
    try:
        adj = g.graph
    except AttributeError:
        return {}

    readmit_idx = col_names.index(C_READMIT_30D)
    results = {}

    for i, col in enumerate(col_names):
        if not col.startswith('last_'):
            continue
        xi_r = adj[i, readmit_idx]   # mark at readmit end
        xr_i = adj[readmit_idx, i]   # mark at state-var end

        if xi_r == -1 and xr_i == 1:
            results[col] = 'definite'
        elif xi_r == -1 and xr_i == 2:
            results[col] = 'possible'
        elif xi_r == -1 and xr_i == -1:
            results[col] = 'bidirected'   # hidden common cause; tracked but not counted as causal
        # else: no edge, or reversed edge -> not counted

    return results


# ---------------------------------------------------------------------------
# Main stability loop
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    input_path = Path(args.input)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", input_path)
    df = pd.read_parquet(input_path)
    log.info("Loaded: %d stays, %d columns", len(df), len(df.columns))

    # Filter pool to columns that exist in the dataset
    pool = [c for c in STATE_POOL if c in df.columns]
    missing_from_pool = [c for c in STATE_POOL if c not in df.columns]
    if missing_from_pool:
        log.warning("Pool variables not found in dataset (skipped): %s", missing_from_pool)
    log.info("State variable pool: %d variables", len(pool))

    confounders = [c for c in CONFOUNDER_COLS if c in df.columns]
    log.info("Confounders: %s", confounders)

    # Verify readmit_30d exists
    assert C_READMIT_30D in df.columns, f"Missing column: {C_READMIT_30D}"

    n_runs       = args.n_runs
    n_sample     = args.n_sample
    n_state_vars = args.n_state_vars
    alpha        = args.alpha
    seed         = args.seed

    log.info("Config: n_runs=%d  n_sample=%d  n_state_vars=%d  alpha=%.3f  seed=%d",
             n_runs, n_sample, n_state_vars, alpha, seed)

    rng = random.Random(seed)
    np.random.seed(seed)

    # Counters per state variable
    n_included  = defaultdict(int)
    n_definite  = defaultdict(int)
    n_possible  = defaultdict(int)
    n_any       = defaultdict(int)
    n_failed    = 0

    scaler = StandardScaler()

    for run_idx in range(n_runs):
        # --- Sample state variables for this run ---
        sampled_state = rng.sample(pool, k=n_state_vars)

        # Column order: confounders | state vars | readmit_30d
        col_names = confounders + sampled_state + [C_READMIT_30D]

        # Tier map for this run (state vars = tier 1)
        tier = dict(TIER)
        for sv in sampled_state:
            tier[sv] = 1

        # --- Subsample stays ---
        n_draw = min(n_sample, len(df))
        subset = df[col_names].sample(n=n_draw, random_state=run_idx)

        # Drop rows with any NaN in this column set
        subset = subset.dropna()
        if len(subset) < 100:
            log.debug("Run %d: too few rows after dropna (%d), skipping", run_idx, len(subset))
            n_failed += 1
            continue

        # Standardise all columns (Fisher-Z is more reliable on z-scored data)
        data = scaler.fit_transform(subset.values.astype(np.float64))

        # --- Run FCI ---
        g = run_fci_once(data, col_names, tier, alpha)

        if g is None:
            n_failed += 1
            continue

        # --- Extract edges ---
        edges = extract_edges_to_readmit(g, col_names)

        for sv in sampled_state:
            n_included[sv] += 1
            etype = edges.get(sv)
            if etype == 'definite':
                n_definite[sv] += 1
                n_any[sv]      += 1
            elif etype == 'possible':
                n_possible[sv] += 1
                n_any[sv]      += 1

        # --- Progress ---
        if (run_idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (run_idx + 1) / elapsed
            remaining = (n_runs - run_idx - 1) / rate if rate > 0 else 0
            log.info("Run %4d / %d  |  elapsed %.0fs  |  est. remaining %.0fs  |  failed %d",
                     run_idx + 1, n_runs, elapsed, remaining, n_failed)

    # ---------------------------------------------------------------------------
    # Build results table
    # ---------------------------------------------------------------------------
    log.info("Building results table...")

    rows = []
    for sv in pool:
        ni  = n_included.get(sv, 0)
        nd  = n_definite.get(sv, 0)
        np_ = n_possible.get(sv, 0)
        na  = n_any.get(sv, 0)
        rows.append({
            'variable':        sv,
            'n_runs_included': ni,
            'n_definite':      nd,
            'n_possible':      np_,
            'n_any':           na,
            'freq_definite':   round(nd / ni, 4) if ni > 0 else 0.0,
            'freq_any':        round(na / ni, 4) if ni > 0 else 0.0,
        })

    results = pd.DataFrame(rows)
    results = results.sort_values('freq_definite', ascending=False).reset_index(drop=True)
    results.insert(0, 'rank_definite', results.index + 1)

    # Add rank_any (sorted by freq_any)
    rank_any_map = (
        results.sort_values('freq_any', ascending=False)
               .reset_index(drop=True)
               .assign(rank_any=lambda x: x.index + 1)
               .set_index('variable')['rank_any']
    )
    results['rank_any'] = results['variable'].map(rank_any_map)

    out_path = report_dir / 'random_stability_results.csv'
    results.to_csv(out_path, index=False)
    log.info("Saved: %s", out_path)

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    total_time = time.time() - t0
    summary = {
        'config': {
            'n_runs':       n_runs,
            'n_sample':     n_sample,
            'n_state_vars': n_state_vars,
            'alpha':        alpha,
            'seed':         seed,
            'pool_size':    len(pool),
            'expected_appearances_per_var': round(n_runs * n_state_vars / len(pool), 1),
        },
        'results': {
            'n_failed':    n_failed,
            'n_completed': n_runs - n_failed,
            'runtime_s':   round(total_time, 1),
        },
        'top_10_definite': results.head(10)[
            ['rank_definite', 'variable', 'freq_definite', 'freq_any',
             'n_runs_included', 'n_definite', 'n_any']
        ].to_dict('records'),
    }

    summary_path = report_dir / 'random_stability_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved: %s", summary_path)

    # ---------------------------------------------------------------------------
    # Print readable results
    # ---------------------------------------------------------------------------
    log.info("=" * 72)
    log.info("RANDOM STABILITY RESULTS  (n_runs=%d  alpha=%.2f  n_sample=%d)",
             n_runs, alpha, n_sample)
    log.info("Expected appearances per variable: ~%.0f runs",
             summary['config']['expected_appearances_per_var'])
    log.info("%-5s %-35s %12s %10s %8s",
             "Rank", "Variable", "Freq_Definite", "Freq_Any", "N_runs")
    log.info("-" * 72)
    for _, row in results.iterrows():
        log.info("%-5d %-35s %12.3f %10.3f %8d",
                 row['rank_definite'], row['variable'],
                 row['freq_definite'], row['freq_any'],
                 row['n_runs_included'])
    log.info("=" * 72)
    log.info("Failed runs: %d / %d  (%.1f%%)",
             n_failed, n_runs, 100 * n_failed / n_runs if n_runs > 0 else 0)
    log.info("Total time: %.1f s", total_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--input', default=str(
            PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit'
            / 'step_09_state_action_selection' / 'stay_level.parquet'),
        help='Path to stay_level.parquet (step_01 output)')
    parser.add_argument(
        '--report-dir', default=str(
            PROJECT_ROOT / 'reports' / 'icu_readmit' / 'step_09_state_action_selection'),
        help='Directory for output CSV and JSON')
    parser.add_argument(
        '--n-runs', type=int, default=2000,
        help='Number of random graph iterations (default: 2000)')
    parser.add_argument(
        '--n-sample', type=int, default=5000,
        help='Stays to subsample per run (default: 5000)')
    parser.add_argument(
        '--n-state-vars', type=int, default=2,
        help='State variables to sample per run (default: 2, giving 6-node graphs)')
    parser.add_argument(
        '--alpha', type=float, default=0.05,
        help='CI test significance level (default: 0.05)')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: 20 runs, 500 stays')
    args = parser.parse_args()

    if args.smoke:
        args.n_runs   = 20
        args.n_sample = 500

    main(args)
