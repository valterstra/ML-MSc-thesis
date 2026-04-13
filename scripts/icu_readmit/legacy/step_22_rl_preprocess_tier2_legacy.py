"""
Step 10c -- RL preprocessing: Tier 2 FCI-guided state (s, a, r, s', done) dataset.

PURPOSE
-------
Transform ICUdataset.csv into the experience-replay format needed by the DDQN
(step 11b). This is the Tier 2 (medium) model derived from the FCI stability
analysis in step_09_causal_states.

Variable selection rationale:
  States and actions were selected by cross-referencing two FCI stability analyses:
    step_03: discharge state -> readmission (2000 runs) -- which states matter?
    step_04: drug action -> delta state (27,000 runs)   -- which drugs have causal leverage?
  Only variables appearing in BOTH analyses at high frequency are included.

STATE FEATURES (5, all continuous)
-----------------------------------
  Hb           -- Hemoglobin (g/dL)         step_03 freq=97.1%
  BUN          -- Blood urea nitrogen        step_03 freq=90.8%
  Creatinine   -- Serum creatinine           step_03 freq=51.8%
  HR           -- Heart rate (bpm)           step_03 freq=54.1%
  Shock_Index  -- HR / Arterial_BP_Sys       step_03 freq=48.9%

Excluded despite readmission signal:
  Ht:          redundant with Hb (near-perfect correlation)
  PT:          no drug in action set shifts it reliably (all step_04 freq < 0.17)
  last_input_total: circular with ivfluid action

ACTIONS (4 binary, 2^4 = 16 combinations)
------------------------------------------
  0: vasopressor_b  = vasopressor_dose > 0     step_04: HR=0.927
  1: ivfluid_b      = ivfluid_dose > 0         step_04: BUN=0.990, Creatinine=0.897
  2: antibiotic_b   = antibiotic_active        step_04: HR=0.981, Shock_Index=0.942
  3: diuretic_b     = diuretic_active          step_04: BUN=1.000, Creatinine=0.991

Action integer encoding:
  a = vasopressor_b*1 + ivfluid_b*2 + antibiotic_b*4 + diuretic_b*8
  Range: 0 (all off) to 15 (all on)

Excluded actions:
  sedation_b:  not in Tier 2 action set
  steroid_b:   0% coverage in MIMIC-IV (wrong itemids)

REWARD
------
  Dense (all non-terminal blocs):
    r = SOFA_t - SOFA_{t+1}
    Positive when SOFA improves (decreases), negative when SOFA worsens.
    SOFA is used as dense reward but is NOT part of the RL state.

  Terminal (last bloc of each stay):
    r = +15  if readmit_30d = 0  (no readmission)
    r = -15  if readmit_30d = 1  (30-day readmission)
    No SOFA delta at the terminal step (no next bloc exists).

NORMALISATION
-------------
  1. Clinical plausibility clips (before normalisation)
  2. Log transform for right-skewed features (BUN, Creatinine)
  3. Z-score normalisation fit on train split only
  4. Post-normalisation clip to [-5, 5]

SPLITS
------
70 / 15 / 15 train / val / test at stay level.
Deterministic: stays sorted by icustayid, then split by index.
Same split logic as step_10 -- stays in the same splits across both datasets.

OUTPUTS
-------
  data/processed/icu_readmit/rl_dataset_tier2.parquet
    One row per consecutive bloc pair. Columns:
      icustayid, bloc, split, readmit_30d, done
      s_Hb, s_BUN, s_Creatinine, s_HR, s_Shock_Index   (normalised)
      a                                                  (integer 0-15)
      vasopressor_b, ivfluid_b, antibiotic_b, diuretic_b
      r
      s_next_Hb, s_next_BUN, s_next_Creatinine, s_next_HR, s_next_Shock_Index

  data/processed/icu_readmit/scaler_params_tier2.json
  data/processed/icu_readmit/static_context_tier2.parquet

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_10c_rl_preprocess_tier2.py --smoke
    python scripts/icu_readmit/step_10c_rl_preprocess_tier2.py
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

# Tier 2 state: 5 continuous features, no binary features
CONTINUOUS_STATE = ['Hb', 'BUN', 'Creatinine', 'HR', 'Shock_Index']
BINARY_STATE     = []   # no binary state features in Tier 2
STATE_FEATURES   = CONTINUOUS_STATE + BINARY_STATE

# Action columns and their binary encoding weights (2^4 = 16 combos)
ACTIONS = [
    ('vasopressor_b',  1),
    ('ivfluid_b',      2),
    ('antibiotic_b',   4),
    ('diuretic_b',     8),
]
ACTION_NAMES = [a[0] for a in ACTIONS]

# Source columns for each binary action
ACTION_SOURCES = {
    'vasopressor_b': ('vasopressor_dose',   'dose'),    # > 0
    'ivfluid_b':     ('ivfluid_dose',       'dose'),    # > 0
    'antibiotic_b':  ('antibiotic_active',  'binary'),
    'diuretic_b':    ('diuretic_active',    'binary'),
}

# Static confounders stored separately (not RL state)
STATIC_CONTEXT = ['age', 'charlson_score', 'prior_ed_visits_6m',
                  'gender', 'race', 're_admission']

# Train / val / test fractions
SPLIT_FRACS = (0.70, 0.15, 0.15)

# Terminal reward magnitude
TERMINAL_REWARD = 15.0

N_ACTIONS = 16   # 2^4

# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

# Clinical plausibility clips applied BEFORE normalisation.
# BUN and Creatinine: inherited from step_10 (same values, same rationale).
# Hb and HR: new for Tier 2.
CLIP_BOUNDS = {
    'Hb':          ( 1,    25),   # below 1 g/dL = artifact; above 25 = artifact
    'BUN':         ( 1,   200),   # renal failure ceiling; same as step_10
    'Creatinine':  ( 0.1,  25),   # extreme AKI noise; same as step_10
    'HR':          (15,   300),   # below 15 or above 300 = recording artifact
    'Shock_Index': ( 0,     5),   # near-zero BP artifact; same as step_10
}

# Log transform applied AFTER clipping, BEFORE z-score.
# BUN and Creatinine are right-skewed (confirmed in step_10 analysis).
# Hb and HR are approximately symmetric -- no log transform needed.
LOG_TRANSFORM = [
    'BUN',         # skew=2.2, max_z=12
    'Creatinine',  # skew=4.1, max_z=28
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def assign_splits(stay_ids: np.ndarray, fracs=(0.70, 0.15, 0.15),
                  seed: int = 42) -> dict:
    """
    Assign each stay to train / val / test deterministically.
    Stays sorted by icustayid, divided by fraction.
    Returns dict {icustayid -> 'train'/'val'/'test'}.
    """
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


def build_binary_actions(df: pd.DataFrame) -> pd.DataFrame:
    """Add binary action columns and encode as integer 0-15."""
    df = df.copy()
    for col, (src, kind) in ACTION_SOURCES.items():
        if src not in df.columns:
            logging.warning("Action source column %s not found -- setting %s=0", src, col)
            df[col] = 0
        elif kind == 'dose':
            df[col] = (df[src] > 0).astype(int)
        else:
            df[col] = df[src].fillna(0).astype(int)

    df['a'] = sum(df[col] * weight for col, weight in ACTIONS)
    return df


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (s, a, r, s', done) pairs from bloc-level data.

    Dense reward: SOFA_t - SOFA_{t+1}  (SOFA NOT in RL state, used for reward only)
    Terminal reward: +-TERMINAL_REWARD based on readmit_30d.
    """
    df_s = df.sort_values([C_ICUSTAYID, C_BLOC]).copy()

    # Next-step state columns
    s_next_cols = {f's_next_{f}': f for f in STATE_FEATURES}
    for next_col, feat in s_next_cols.items():
        df_s[next_col] = df_s.groupby(C_ICUSTAYID)[f's_{feat}'].shift(-1)

    # Next-step SOFA for dense reward (SOFA is in ICUdataset.csv from step_07)
    df_s['SOFA_next'] = df_s.groupby(C_ICUSTAYID)['SOFA'].shift(-1)

    # Mark last bloc per stay
    df_s['done'] = (
        df_s.groupby(C_ICUSTAYID)[C_BLOC].transform('max') == df_s[C_BLOC]
    ).astype(int)

    # Reward: dense SOFA delta for non-terminal, +-15 for terminal
    df_s['r'] = np.where(
        df_s['done'] == 0,
        df_s['SOFA'] - df_s['SOFA_next'],
        np.where(
            df_s[C_READMIT_30D] == 0,
             TERMINAL_REWARD,
            -TERMINAL_REWARD,
        )
    )

    # Fill next-state with zeros at terminal blocs
    for next_col in s_next_cols.keys():
        df_s[next_col] = df_s[next_col].fillna(0.0)

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

    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'legacy' / 'icu_readmit' / 'step_22_rl_preprocess_tier2_legacy.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10c (Tier 2) started. input=%s", args.input)
    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    if args.smoke:
        smoke_stays = np.sort(df[C_ICUSTAYID].unique())[:2000]
        df = df[df[C_ICUSTAYID].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    # -----------------------------------------------------------------------
    # 1. Validate required columns
    # -----------------------------------------------------------------------
    missing_state  = [c for c in STATE_FEATURES if c not in df.columns]
    missing_action = [s for s, _ in ACTION_SOURCES.values() if s not in df.columns]
    if 'SOFA' not in df.columns:
        logging.warning("SOFA column not found -- dense reward will be NaN")
    if missing_state:
        logging.warning("Missing state columns (will be NaN): %s", missing_state)
    if missing_action:
        logging.warning("Missing action source columns: %s", missing_action)

    # -----------------------------------------------------------------------
    # 2. Binary actions + action integer
    # -----------------------------------------------------------------------
    df = build_binary_actions(df)
    logging.info("Action distribution (stay-level frac active):")
    for col, _ in ACTIONS:
        logging.info("  %-18s %.3f", col, df.groupby(C_ICUSTAYID)[col].max().mean())

    action_counts = df['a'].value_counts().sort_index()
    logging.info("Unique action combinations used: %d / %d", len(action_counts), N_ACTIONS)

    # -----------------------------------------------------------------------
    # 3. Train / val / test split
    # -----------------------------------------------------------------------
    stay_ids  = df[C_ICUSTAYID].unique()
    split_map = assign_splits(stay_ids, SPLIT_FRACS)
    df['split'] = df[C_ICUSTAYID].map(split_map)

    for spl in ('train', 'val', 'test'):
        n = df[df['split'] == spl][C_ICUSTAYID].nunique()
        logging.info("Split %-5s: %d stays", spl, n)

    # -----------------------------------------------------------------------
    # 3b. Clinical plausibility clips (before normalisation)
    # -----------------------------------------------------------------------
    for feat, (lo, hi) in CLIP_BOUNDS.items():
        if feat in df.columns:
            n_clipped = ((df[feat] < lo) | (df[feat] > hi)).sum()
            if n_clipped > 0:
                logging.info("  Clip %-20s [%.1f, %.1f]  clipped %d rows",
                             feat, lo, hi, n_clipped)
            df[feat] = df[feat].clip(lo, hi)

    # -----------------------------------------------------------------------
    # 3c. Log transform right-skewed features (after clip, before z-score)
    # -----------------------------------------------------------------------
    for feat in LOG_TRANSFORM:
        if feat in df.columns:
            df[feat] = np.log(0.1 + df[feat])
            logging.info("  Log-transformed %s", feat)

    # -----------------------------------------------------------------------
    # 4. Prefix state columns with 's_', fill NaNs, normalise
    # -----------------------------------------------------------------------
    train_mask = df['split'] == 'train'

    for feat in STATE_FEATURES:
        df[f's_{feat}'] = df[feat] if feat in df.columns else 0.0

    # Fill residual NaNs with train median before normalising
    for feat in CONTINUOUS_STATE:
        col = f's_{feat}'
        if df[col].isna().any():
            train_median = df.loc[train_mask, col].median()
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(train_median)
            logging.info("  NaN fill %-18s  filled %d rows with train median %.4f",
                         feat, n_filled, train_median)

    # Re-apply clips to s_ columns for non-log-transformed features
    for feat, (lo, hi) in CLIP_BOUNDS.items():
        if feat in LOG_TRANSFORM:
            continue
        col = f's_{feat}'
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

    # Z-score normalise (fit on train only)
    scaler_params = {}
    for feat in CONTINUOUS_STATE:
        col = f's_{feat}'
        if col not in df.columns:
            continue
        mu  = df.loc[train_mask, col].mean()
        std = df.loc[train_mask, col].std()
        std = std if std > 1e-6 else 1.0
        df[col] = (df[col] - mu) / std
        scaler_params[feat] = {'mean': round(float(mu), 6), 'std': round(float(std), 6)}
        logging.info("  Normalised %-20s  mean=%.4f  std=%.4f", feat, mu, std)

    # Post-normalisation clip to [-5, 5]
    Z_CLIP = 5.0
    for feat in CONTINUOUS_STATE:
        col = f's_{feat}'
        if col in df.columns:
            df[col] = df[col].clip(-Z_CLIP, Z_CLIP)

    # -----------------------------------------------------------------------
    # 5. Build (s, a, r, s', done) transitions
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

    logging.info("Transitions built: %d rows, %d cols", len(df_out), len(df_out.columns))
    logging.info("  done=0 (non-terminal): %d", (df_out['done'] == 0).sum())
    logging.info("  done=1 (terminal):     %d", (df_out['done'] == 1).sum())

    non_term = df_out[df_out['done'] == 0]['r']
    term     = df_out[df_out['done'] == 1]['r']
    logging.info("Reward (non-terminal): mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                 non_term.mean(), non_term.std(), non_term.min(), non_term.max())
    logging.info("Reward (terminal):     +15 count=%d  -15 count=%d",
                 (term > 0).sum(), (term < 0).sum())

    # -----------------------------------------------------------------------
    # 6. Static context table (one row per stay)
    # -----------------------------------------------------------------------
    static_cols = [C_ICUSTAYID, 'split'] + [c for c in STATIC_CONTEXT if c in df.columns]
    static_df   = df[static_cols].groupby(C_ICUSTAYID).first().reset_index()

    # -----------------------------------------------------------------------
    # 7. Save outputs
    # -----------------------------------------------------------------------
    suffix = '_tier2_smoke' if args.smoke else '_tier2'

    rl_path     = os.path.join(args.out_dir, f'rl_dataset{suffix}.parquet')
    static_path = os.path.join(args.out_dir, f'static_context{suffix}.parquet')
    scaler_path = os.path.join(args.out_dir, f'scaler_params{suffix}.json')

    df_out.to_parquet(rl_path, index=False)
    static_df.to_parquet(static_path, index=False)
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)

    logging.info("Saved rl_dataset    -> %s  (%d rows, %d cols)",
                 rl_path, len(df_out), len(df_out.columns))
    logging.info("Saved static_context -> %s  (%d stays)", static_path, len(static_df))
    logging.info("Saved scaler_params  -> %s", scaler_path)

    for spl in ('train', 'val', 'test'):
        sub = df_out[df_out['split'] == spl]
        pos = sub[sub['done'] == 1][C_READMIT_30D].mean()
        logging.info("  %-5s: %d transitions, %d stays, readmit_rate=%.3f",
                     spl, len(sub), sub[C_ICUSTAYID].nunique(), pos)

    logging.info("Step 10c complete.")
