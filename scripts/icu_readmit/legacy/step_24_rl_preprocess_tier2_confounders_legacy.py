"""
Step 10e -- RL preprocessing: Tier 2 -- adds static confounders to state.

Extends step_10c (Tier 2) by including the three patient-level confounders
that were fixed nodes in both FCI stability analyses (step_09 steps 3 & 4):

  age, charlson_score, prior_ed_visits_6m

These are appended to the dynamic state vector (repeated at every time step).
Rationale: drug->state causal edges were identified conditional on these
confounders being in the graph. Including them in the world model allows the
transformer to condition transition dynamics on patient heterogeneity.

STATE FEATURES (8 total = 5 dynamic + 3 static)
------------------------------------------------
  Dynamic (evolve each step):
    Hb           -- Hemoglobin (g/dL)
    BUN          -- Blood urea nitrogen
    Creatinine   -- Serum creatinine
    HR           -- Heart rate (bpm)
    Shock_Index  -- HR / Arterial_BP_Sys

  Static (same for all blocs of a stay):
    age                -- patient age at ICU admission
    charlson_score     -- Charlson comorbidity index (0-37)
    prior_ed_visits_6m -- ED visits in prior 6 months

ACTIONS (unchanged from step_10c -- 4 binary, 16 combos)
---------------------------------------------------------
  vasopressor_b, ivfluid_b, antibiotic_b, diuretic_b
  a = vasopressor_b*1 + ivfluid_b*2 + antibiotic_b*4 + diuretic_b*8

REWARD (unchanged from step_10c)
---------------------------------
  Dense:    r = SOFA_t - SOFA_{t+1}
  Terminal: r = +15 (no readmit) / -15 (readmit)

NORMALISATION
-------------
  Dynamic features: same clips + log transforms + z-score as step_10c.
  Static features:
    age:               clip [18, 100], z-score
    charlson_score:    clip [0, 37],   z-score
    prior_ed_visits_6m: clip [0, 20],  z-score
  All: post-normalisation clip to [-5, 5].

OUTPUTS
-------
  data/processed/icu_readmit/rl_dataset_tier2.parquet
  data/processed/icu_readmit/scaler_params_tier2.json
  data/processed/icu_readmit/static_context_tier2.parquet

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_10e_rl_preprocess_tier2.py --smoke
    python scripts/icu_readmit/step_10e_rl_preprocess_tier2.py 2>&1 | tee logs/step_10e.log
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import C_ICUSTAYID, C_BLOC, C_READMIT_30D

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DYNAMIC_STATE = ['Hb', 'BUN', 'Creatinine', 'HR', 'Shock_Index']
STATIC_STATE  = ['age', 'charlson_score', 'prior_ed_visits_6m']
STATE_FEATURES = DYNAMIC_STATE + STATIC_STATE   # length 8 -- matches state_dim in model

ACTIONS = [
    ('vasopressor_b',  1),
    ('ivfluid_b',      2),
    ('antibiotic_b',   4),
    ('diuretic_b',     8),
]
ACTION_NAMES = [a[0] for a in ACTIONS]

ACTION_SOURCES = {
    'vasopressor_b': ('vasopressor_dose',  'dose'),
    'ivfluid_b':     ('ivfluid_dose',      'dose'),
    'antibiotic_b':  ('antibiotic_active', 'binary'),
    'diuretic_b':    ('diuretic_active',   'binary'),
}

STATIC_CONTEXT_EXTRA = ['gender', 'race', 're_admission']   # saved but not in model state

SPLIT_FRACS    = (0.70, 0.15, 0.15)
TERMINAL_REWARD = 15.0
N_ACTIONS       = 16
Z_CLIP          = 5.0

# Clinical plausibility clips -- dynamic features
CLIP_BOUNDS_DYNAMIC = {
    'Hb':          ( 1,    25),
    'BUN':         ( 1,   200),
    'Creatinine':  ( 0.1,  25),
    'HR':          (15,   300),
    'Shock_Index': ( 0,     5),
}

# Log transform for right-skewed dynamic features
LOG_TRANSFORM_DYNAMIC = ['BUN', 'Creatinine']

# Clips for static features (applied before z-score)
CLIP_BOUNDS_STATIC = {
    'age':                 (18,  100),
    'charlson_score':      ( 0,   37),
    'prior_ed_visits_6m':  ( 0,   20),
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def assign_splits(stay_ids, fracs=(0.70, 0.15, 0.15), seed=42):
    ids = np.sort(stay_ids)
    n = len(ids)
    n_train = int(n * fracs[0])
    n_val   = int(n * fracs[1])
    split_map = {}
    for i, sid in enumerate(ids):
        if i < n_train:
            split_map[sid] = 'train'
        elif i < n_train + n_val:
            split_map[sid] = 'val'
        else:
            split_map[sid] = 'test'
    return split_map


def build_binary_actions(df):
    df = df.copy()
    for col, (src, kind) in ACTION_SOURCES.items():
        if src not in df.columns:
            logging.warning("Action source %s not found -- %s=0", src, col)
            df[col] = 0
        elif kind == 'dose':
            df[col] = (df[src] > 0).astype(int)
        else:
            df[col] = df[src].fillna(0).astype(int)
    df['a'] = sum(df[col] * weight for col, weight in ACTIONS)
    return df


def build_transitions(df):
    """Build (s, a, r, s', done) pairs.

    Dynamic next-states: shift by -1 within each stay.
    Static next-states:  copy current value (age/charlson don't change).
    """
    df_s = df.sort_values([C_ICUSTAYID, C_BLOC]).copy()

    # Dynamic next-state via shift
    for feat in DYNAMIC_STATE:
        df_s[f's_next_{feat}'] = df_s.groupby(C_ICUSTAYID)[f's_{feat}'].shift(-1)

    # Static next-state: just copy (no shift -- they don't change)
    for feat in STATIC_STATE:
        df_s[f's_next_{feat}'] = df_s[f's_{feat}']

    # Dense reward: SOFA delta
    df_s['SOFA_next'] = df_s.groupby(C_ICUSTAYID)['SOFA'].shift(-1)

    # Terminal flag
    df_s['done'] = (
        df_s.groupby(C_ICUSTAYID)[C_BLOC].transform('max') == df_s[C_BLOC]
    ).astype(int)

    # Reward
    df_s['r'] = np.where(
        df_s['done'] == 0,
        df_s['SOFA'] - df_s['SOFA_next'],
        np.where(df_s[C_READMIT_30D] == 0, TERMINAL_REWARD, -TERMINAL_REWARD),
    )

    # Fill dynamic next-state NaNs at terminal blocs with 0
    for feat in DYNAMIC_STATE:
        df_s[f's_next_{feat}'] = df_s[f's_next_{feat}'].fillna(0.0)

    return df_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'legacy' / 'step_20_24'))
    parser.add_argument('--smoke', action='store_true',
                        help='Run on first 2000 stays only')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'legacy' / 'icu_readmit' / 'step_24_rl_preprocess_tier2_confounders_legacy.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10e (Tier 2 -- with static confounders) started.")
    logging.info("Input: %s", args.input)

    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    if args.smoke:
        smoke_stays = np.sort(df[C_ICUSTAYID].unique())[:2000]
        df = df[df[C_ICUSTAYID].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    # -----------------------------------------------------------------------
    # 1. Validate columns
    # -----------------------------------------------------------------------
    missing_state   = [c for c in STATE_FEATURES if c not in df.columns]
    missing_action  = [s for s, _ in ACTION_SOURCES.values() if s not in df.columns]
    if 'SOFA' not in df.columns:
        logging.warning("SOFA column not found -- dense reward will be NaN")
    if missing_state:
        logging.warning("Missing state columns: %s", missing_state)
    if missing_action:
        logging.warning("Missing action columns: %s", missing_action)

    # -----------------------------------------------------------------------
    # 2. Binary actions
    # -----------------------------------------------------------------------
    df = build_binary_actions(df)
    logging.info("Action distribution (frac stays with drug active):")
    for col, _ in ACTIONS:
        logging.info("  %-20s %.3f", col, df.groupby(C_ICUSTAYID)[col].max().mean())

    action_counts = df['a'].value_counts().sort_index()
    logging.info("Unique action combos: %d / %d", len(action_counts), N_ACTIONS)

    # -----------------------------------------------------------------------
    # 3. Train / val / test split
    # -----------------------------------------------------------------------
    stay_ids  = df[C_ICUSTAYID].unique()
    split_map = assign_splits(stay_ids, SPLIT_FRACS)
    df['split'] = df[C_ICUSTAYID].map(split_map)
    for spl in ('train', 'val', 'test'):
        n = df[df['split'] == spl][C_ICUSTAYID].nunique()
        logging.info("Split %-5s: %d stays", spl, n)

    train_mask = df['split'] == 'train'

    # -----------------------------------------------------------------------
    # 4. Normalise DYNAMIC state features (clip -> log -> z-score)
    # -----------------------------------------------------------------------
    scaler_params = {}

    for feat, (lo, hi) in CLIP_BOUNDS_DYNAMIC.items():
        if feat in df.columns:
            n_clip = ((df[feat] < lo) | (df[feat] > hi)).sum()
            if n_clip:
                logging.info("  Clip %-20s [%.1f, %.1f]  %d rows", feat, lo, hi, n_clip)
            df[feat] = df[feat].clip(lo, hi)

    for feat in LOG_TRANSFORM_DYNAMIC:
        if feat in df.columns:
            df[feat] = np.log(0.1 + df[feat])
            logging.info("  Log-transformed %s", feat)

    for feat in DYNAMIC_STATE:
        col = f's_{feat}'
        df[col] = df[feat] if feat in df.columns else 0.0
        # Fill NaNs with train median
        if df[col].isna().any():
            med = df.loc[train_mask, col].median()
            n_fill = df[col].isna().sum()
            df[col] = df[col].fillna(med)
            logging.info("  NaN fill %-20s filled %d rows (median=%.4f)", feat, n_fill, med)
        # Re-clip non-log-transformed features
        if feat not in LOG_TRANSFORM_DYNAMIC and feat in CLIP_BOUNDS_DYNAMIC:
            lo, hi = CLIP_BOUNDS_DYNAMIC[feat]
            df[col] = df[col].clip(lo, hi)
        # Z-score (fit on train only)
        mu  = df.loc[train_mask, col].mean()
        std = df.loc[train_mask, col].std()
        std = std if std > 1e-6 else 1.0
        df[col] = ((df[col] - mu) / std).clip(-Z_CLIP, Z_CLIP)
        scaler_params[feat] = {'mean': round(float(mu), 6), 'std': round(float(std), 6)}
        logging.info("  Normalised %-20s  mean=%.4f  std=%.4f", feat, mu, std)

    # -----------------------------------------------------------------------
    # 5. Normalise STATIC state features (fill -> clip -> z-score)
    # -----------------------------------------------------------------------
    for feat in STATIC_STATE:
        col = f's_{feat}'
        lo, hi = CLIP_BOUNDS_STATIC[feat]

        if feat not in df.columns:
            logging.warning("Static feature %s not found -- setting to 0", feat)
            df[col] = 0.0
            scaler_params[feat] = {'mean': 0.0, 'std': 1.0}
            continue

        # Propagate static value across all blocs of each stay
        # (take first non-NaN value per stay, forward-fill within stay)
        df[feat] = df.groupby(C_ICUSTAYID)[feat].transform('first')

        # Fill remaining NaNs with train median
        if df[feat].isna().any():
            med = df.loc[train_mask, feat].median()
            n_fill = df[feat].isna().sum()
            df[feat] = df[feat].fillna(med)
            logging.info("  NaN fill %-20s filled %d rows (median=%.4f)", feat, n_fill, med)

        df[feat] = df[feat].clip(lo, hi)

        mu  = df.loc[train_mask, feat].mean()
        std = df.loc[train_mask, feat].std()
        std = std if std > 1e-6 else 1.0
        df[col] = ((df[feat] - mu) / std).clip(-Z_CLIP, Z_CLIP)
        scaler_params[feat] = {'mean': round(float(mu), 6), 'std': round(float(std), 6)}
        logging.info("  Normalised %-20s  mean=%.4f  std=%.4f", feat, mu, std)

    # -----------------------------------------------------------------------
    # 6. Build transitions
    # -----------------------------------------------------------------------
    logging.info("Building transitions...")
    df_trans = build_transitions(df)

    output_cols = (
        [C_ICUSTAYID, C_BLOC, 'split', C_READMIT_30D, 'done']
        + [f's_{f}' for f in STATE_FEATURES]
        + ['a'] + ACTION_NAMES
        + ['r']
        + [f's_next_{f}' for f in STATE_FEATURES]
    )
    output_cols = [c for c in output_cols if c in df_trans.columns]
    df_out = df_trans[output_cols].copy()

    logging.info("Transitions: %d rows, %d cols", len(df_out), len(df_out.columns))
    logging.info("  done=0: %d  done=1: %d", (df_out['done'] == 0).sum(), (df_out['done'] == 1).sum())

    non_term = df_out[df_out['done'] == 0]['r']
    term     = df_out[df_out['done'] == 1]['r']
    logging.info("Reward (non-terminal): mean=%.3f  std=%.3f", non_term.mean(), non_term.std())
    logging.info("Reward (terminal):     +15 count=%d  -15 count=%d",
                 (term > 0).sum(), (term < 0).sum())

    # -----------------------------------------------------------------------
    # 7. Static context table (one row per stay, saved separately)
    # -----------------------------------------------------------------------
    extra_static = [c for c in STATIC_CONTEXT_EXTRA if c in df.columns]
    static_cols  = [C_ICUSTAYID, 'split'] + STATIC_STATE + extra_static
    static_df    = df[[c for c in static_cols if c in df.columns]].groupby(C_ICUSTAYID).first().reset_index()

    # -----------------------------------------------------------------------
    # 8. Save
    # -----------------------------------------------------------------------
    suffix = '_tier2_smoke' if args.smoke else '_tier2'

    rl_path     = os.path.join(args.out_dir, f'rl_dataset{suffix}.parquet')
    static_path = os.path.join(args.out_dir, f'static_context{suffix}.parquet')
    scaler_path = os.path.join(args.out_dir, f'scaler_params{suffix}.json')

    df_out.to_parquet(rl_path, index=False)
    static_df.to_parquet(static_path, index=False)
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)

    logging.info("Saved rl_dataset     -> %s  (%d rows, %d cols)", rl_path, len(df_out), len(df_out.columns))
    logging.info("Saved static_context -> %s  (%d stays)", static_path, len(static_df))
    logging.info("Saved scaler_params  -> %s", scaler_path)

    for spl in ('train', 'val', 'test'):
        sub = df_out[df_out['split'] == spl]
        pos = sub[sub['done'] == 1][C_READMIT_30D].mean()
        logging.info("  %-5s: %d transitions, %d stays, readmit_rate=%.3f",
                     spl, len(sub), sub[C_ICUSTAYID].nunique(), pos)

    logging.info("Step 10e complete. state_dim=8 (5 dynamic + 3 static).")
    logging.info("Next: upload rl_dataset_tier2.parquet to Drive and run Colab with --state-dim 8")
