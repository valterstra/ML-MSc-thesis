"""
Step 10 -- RL preprocessing: build (s, a, r, s', done) dataset.

PURPOSE
-------
Transform ICUdataset.csv into the experience-replay format needed by the DDQN
(step 11). This step:
  - Selects the final state features agreed upon by variable selection (09a/09b)
    and causal discovery (09c)
  - Encodes 5 binary actions as a single integer (0-31)
  - Constructs the reward signal
  - Normalises continuous state features
  - Splits stays into train / val / test
  - Outputs (s, a, r, s', done) tuples, one row per consecutive bloc pair

STATE FEATURES (15)
-------------------
  Vitals:   HR, MeanBP, RR, SpO2, Temp_C
  Labs:     Potassium, Creatinine, BUN, WBC_count, Glucose, Hb
  Derived:  SOFA, Shock_Index
  Vent:     mechvent
  Neuro:    GCS

Rationale: top outcome-relevant features (09a Model 1) that are action-responsive
(09b) and/or confirmed by causal discovery (09c PC algorithm).

Static confounders (age, charlson_score, prior_ed_visits_6m) are NOT part of the
RL state -- the agent cannot change them. They are stored in a separate
static_context table for reward model training in later steps.

ACTIONS (5 binary, 2^5 = 32 combinations)
------------------------------------------
  0: vasopressor_b  = vasopressor_dose > 0
  1: ivfluid_b      = ivfluid_dose > 0
  2: antibiotic_b   = antibiotic_active
  3: sedation_b     = sedation_active
  4: diuretic_b     = diuretic_active

Action integer encoding:
  a = vasopressor_b*1 + ivfluid_b*2 + antibiotic_b*4 + sedation_b*8 + diuretic_b*16
  Range: 0 (all off) to 31 (all on)

REWARD
------
  Dense (all non-terminal blocs):
    r = SOFA_t - SOFA_{t+1}
    Positive when SOFA improves (decreases), negative when SOFA worsens.

  Terminal (last bloc of each stay):
    r = +15  if readmit_30d = 0  (no readmission)
    r = -15  if readmit_30d = 1  (30-day readmission)
    No SOFA delta at the terminal step (no next bloc exists).

The ±15 terminal reward is intentionally smaller than the cumulative dense reward
over a typical stay to avoid the terminal signal overwhelming the SOFA learning.

NORMALISATION
-------------
Continuous state features are z-score normalised (mean=0, std=1) using statistics
fitted on the TRAIN split only. Binary features (mechvent) are left as 0/1.
Scaler parameters saved to scaler_params.json for use at inference time.

SPLITS
------
70 / 15 / 15 train / val / test at stay level (no stay crosses splits).
Deterministic: stays sorted by icustayid, then split by index.

OUTPUTS
-------
  data/processed/icu_readmit/rl_dataset.parquet
    One row per consecutive bloc pair. Columns:
      icustayid, bloc, split
      s_<feature> x15          -- current state (normalised)
      a                        -- action integer 0-31
      vasopressor_b, ivfluid_b, antibiotic_b, sedation_b, diuretic_b
      r                        -- reward
      s_next_<feature> x15     -- next state (normalised; zeros if done=1)
      done                     -- 1 if last bloc of stay
      readmit_30d              -- for evaluation reference

  data/processed/icu_readmit/scaler_params.json
    mean and std per continuous feature (for inference)

  data/processed/icu_readmit/static_context.parquet
    One row per stay: icustayid + static confounders

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_10_rl_preprocess.py
    python scripts/icu_readmit/step_10_rl_preprocess.py --smoke  # 2000 stays
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

# Continuous state features -- will be z-score normalised
CONTINUOUS_STATE = [
    'HR', 'MeanBP', 'RR', 'SpO2', 'Temp_C',
    'Potassium', 'Creatinine', 'BUN', 'WBC_count', 'Glucose', 'Hb',
    'SOFA', 'Shock_Index', 'GCS',
]

# Binary state features -- kept as 0/1, not normalised
BINARY_STATE = ['mechvent']

STATE_FEATURES = CONTINUOUS_STATE + BINARY_STATE  # 15 total

# Action columns and their binary encoding weights
ACTIONS = [
    ('vasopressor_b', 1),
    ('ivfluid_b',     2),
    ('antibiotic_b',  4),
    ('sedation_b',    8),
    ('diuretic_b',   16),
]
ACTION_NAMES = [a[0] for a in ACTIONS]

# Source columns for each binary action
ACTION_SOURCES = {
    'vasopressor_b': ('vasopressor_dose',    'dose'),    # > 0
    'ivfluid_b':     ('ivfluid_dose',        'dose'),    # > 0
    'antibiotic_b':  ('antibiotic_active',   'binary'),
    'sedation_b':    ('sedation_active',     'binary'),
    'diuretic_b':    ('diuretic_active',     'binary'),
}

# Static confounders stored separately (not RL state)
STATIC_CONTEXT = ['age', 'charlson_score', 'prior_ed_visits_6m',
                  'gender', 'race', 're_admission']

# Train / val / test fractions
SPLIT_FRACS = (0.70, 0.15, 0.15)

# Terminal reward magnitude
TERMINAL_REWARD = 15.0

# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

# Clinical plausibility clips applied BEFORE normalisation.
CLIP_BOUNDS = {
    'SpO2':        (50,   100),   # min_z=-40; below 50 incompatible with survival
    'Temp_C':      (25,    43),   # min_z=-20; below 25 artifact
    'Shock_Index': ( 0,     5),   # max_z=469; near-zero BP artifact
    'WBC_count':   ( 0.1, 200),   # max_z=55; artifact/extreme leukemia
    'Creatinine':  ( 0.1,  25),   # max_z=28; extreme AKI noise
    'BUN':         ( 1,   200),   # max_z=12; renal failure ceiling
    'Glucose':     (20,   800),   # max_z=17; DKA ceiling
    'Potassium':   ( 1.5,   9),   # fatal outside this range
    'RR':          ( 4,    60),   # max_z=12; 78 breaths/min is artifact
}

# Log transform (log(0.1 + x)) applied AFTER clipping, BEFORE z-score.
# Applied to right-skewed features with skew > 2 in the narrow feature set.
LOG_TRANSFORM = [
    'WBC_count',   # skew=9.1, max_z=55
    'Creatinine',  # skew=4.1, max_z=28
    'BUN',         # skew=2.2, max_z=12
    'Glucose',     # skew=2.0, max_z=17
]


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def assign_splits(stay_ids: np.ndarray, fracs=(0.70, 0.15, 0.15),
                  seed: int = 42) -> dict:
    """
    Assign each stay to train / val / test deterministically.
    Stays are sorted by icustayid, then divided by fraction.
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
    """Add binary action columns and encode as integer 0-31."""
    df = df.copy()
    for col, (src, kind) in ACTION_SOURCES.items():
        if src not in df.columns:
            logging.warning("Action source column %s not found -- setting %s=0", src, col)
            df[col] = 0
        elif kind == 'dose':
            df[col] = (df[src] > 0).astype(int)
        else:
            df[col] = df[src].fillna(0).astype(int)

    # Encode combination as integer
    df['a'] = sum(df[col] * weight for col, weight in ACTIONS)
    return df


def build_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build (s, a, r, s', done) pairs from bloc-level data.

    For each consecutive pair (t, t+1) within a stay:
      - Current state at t
      - Action taken at t
      - Reward: SOFA_t - SOFA_{t+1}  (dense, non-terminal)
                ±TERMINAL_REWARD      (terminal, last bloc only, no SOFA delta)
      - Next state at t+1 (zeros if done)
      - done = 1 if last bloc

    Returns one row per transition.
    """
    df_s = df.sort_values([C_ICUSTAYID, C_BLOC]).copy()

    # Next-step state columns
    s_next_cols = {f's_next_{f}': f for f in STATE_FEATURES}

    for next_col, feat in s_next_cols.items():
        df_s[next_col] = df_s.groupby(C_ICUSTAYID)[f's_{feat}'].shift(-1)

    # Next-step SOFA for dense reward
    df_s['SOFA_next'] = df_s.groupby(C_ICUSTAYID)['SOFA'].shift(-1)

    # Mark last bloc per stay
    df_s['done'] = (df_s.groupby(C_ICUSTAYID)[C_BLOC].transform('max') == df_s[C_BLOC]).astype(int)

    # Reward
    # Non-terminal: SOFA improvement (positive = SOFA went down)
    df_s['r'] = np.where(
        df_s['done'] == 0,
        df_s['SOFA'] - df_s['SOFA_next'],   # dense reward: current - next SOFA
        np.where(
            df_s[C_READMIT_30D] == 0,
            TERMINAL_REWARD,                 # no readmission: +15
            -TERMINAL_REWARD,                # readmission: -15
        )
    )

    # Fill next-state with zeros for terminal blocs
    for next_col in s_next_cols.keys():
        df_s[next_col] = df_s[next_col].fillna(0.0)

    return df_s


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=str(
        PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'legacy' / 'step_20_24'))
    parser.add_argument('--smoke', action='store_true',
                        help='Run on first 2000 stays only')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'legacy' / 'icu_readmit' / 'step_20_rl_preprocess_narrow_legacy.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10 started. input=%s", args.input)
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
    if missing_state:
        logging.warning("Missing state columns (will be NaN): %s", missing_state)
    if missing_action:
        logging.warning("Missing action source columns: %s", missing_action)

    # -----------------------------------------------------------------------
    # 2. Binary actions + action integer
    # -----------------------------------------------------------------------
    df = build_binary_actions(df)
    logging.info("Action distribution (stay-level frac):")
    for col, _ in ACTIONS:
        logging.info("  %-18s %.3f", col, df.groupby(C_ICUSTAYID)[col].max().mean())

    action_counts = df['a'].value_counts().sort_index()
    logging.info("Unique action combinations used: %d / 32", len(action_counts))

    # -----------------------------------------------------------------------
    # 3. Train / val / test split
    # -----------------------------------------------------------------------
    stay_ids   = df[C_ICUSTAYID].unique()
    split_map  = assign_splits(stay_ids, SPLIT_FRACS)
    df['split'] = df[C_ICUSTAYID].map(split_map)

    for spl in ('train', 'val', 'test'):
        n = df[df['split'] == spl][C_ICUSTAYID].nunique()
        logging.info("Split %-5s: %d stays", spl, n)

    # -----------------------------------------------------------------------
    # 3b. Clinical plausibility clips
    # -----------------------------------------------------------------------
    for feat, (lo, hi) in CLIP_BOUNDS.items():
        if feat in df.columns:
            n_clipped = ((df[feat] < lo) | (df[feat] > hi)).sum()
            if n_clipped > 0:
                logging.info("  Clip %-20s [%.1f, %.1f]  clipped %d rows", feat, lo, hi, n_clipped)
            df[feat] = df[feat].clip(lo, hi)

    # -----------------------------------------------------------------------
    # 3c. Log transform skewed features
    # -----------------------------------------------------------------------
    for feat in LOG_TRANSFORM:
        if feat in df.columns:
            df[feat] = np.log(0.1 + df[feat])

    # -----------------------------------------------------------------------
    # 4. Normalise continuous state features
    #    Fit on train, apply to all splits
    # -----------------------------------------------------------------------
    train_mask = df['split'] == 'train'

    # Prefix state columns with 's_' for clarity in output
    for feat in STATE_FEATURES:
        if feat in df.columns:
            df[f's_{feat}'] = df[feat]
        else:
            df[f's_{feat}'] = 0.0

    # Fill residual NaNs with train median
    for feat in CONTINUOUS_STATE:
        col = f's_{feat}'
        if df[col].isna().any():
            train_median = df.loc[train_mask, col].median()
            n_filled = df[col].isna().sum()
            df[col] = df[col].fillna(train_median)
            logging.info("  NaN fill %-18s  filled %d rows with train median %.4f",
                         feat, n_filled, train_median)

    # Re-apply clips to s_ columns for non-log-transformed features only.
    for feat, (lo, hi) in CLIP_BOUNDS.items():
        if feat in LOG_TRANSFORM:
            continue
        col = f's_{feat}'
        if col in df.columns:
            df[col] = df[col].clip(lo, hi)

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
        logging.info("  Normalised %-20s  mean=%.3f  std=%.3f", feat, mu, std)

    # Post-normalisation clip to [-5, 5]
    Z_CLIP = 5.0
    for feat in CONTINUOUS_STATE:
        col = f's_{feat}'
        if col in df.columns:
            df[col] = df[col].clip(-Z_CLIP, Z_CLIP)

    # Binary state features: clip to 0/1 but no normalisation
    for feat in BINARY_STATE:
        col = f's_{feat}'
        df[col] = df[col].clip(0, 1)

    # -----------------------------------------------------------------------
    # 5. Build (s, a, r, s', done) transitions
    # -----------------------------------------------------------------------
    logging.info("Building transitions...")
    df_trans = build_transitions(df)

    # Keep only non-terminal rows for training (terminal rows have done=1)
    # Actually keep all rows -- DDQN needs terminal transitions too
    output_cols = (
        [C_ICUSTAYID, C_BLOC, 'split', C_READMIT_30D, 'done']
        + [f's_{f}' for f in STATE_FEATURES]
        + ['a'] + ACTION_NAMES
        + ['r']
        + [f's_next_{f}' for f in STATE_FEATURES]
    )

    # Filter to columns that exist
    output_cols = [c for c in output_cols if c in df_trans.columns]
    df_out = df_trans[output_cols].copy()

    logging.info("Transitions built: %d rows", len(df_out))
    logging.info("  done=0 (non-terminal): %d", (df_out['done'] == 0).sum())
    logging.info("  done=1 (terminal):     %d", (df_out['done'] == 1).sum())

    # Reward summary
    non_term = df_out[df_out['done'] == 0]['r']
    term     = df_out[df_out['done'] == 1]['r']
    logging.info("Reward stats (non-terminal): mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                 non_term.mean(), non_term.std(), non_term.min(), non_term.max())
    logging.info("Reward stats (terminal):     +15 count=%d  -15 count=%d",
                 (term > 0).sum(), (term < 0).sum())

    # -----------------------------------------------------------------------
    # 6. Static context table (one row per stay)
    # -----------------------------------------------------------------------
    static_cols = [C_ICUSTAYID, 'split'] + [c for c in STATIC_CONTEXT if c in df.columns]
    static_df = df[static_cols].groupby(C_ICUSTAYID).first().reset_index()

    # -----------------------------------------------------------------------
    # 7. Save outputs
    # -----------------------------------------------------------------------
    suffix = '_smoke' if args.smoke else ''

    rl_path     = os.path.join(args.out_dir, f'rl_dataset{suffix}.parquet')
    static_path = os.path.join(args.out_dir, f'static_context{suffix}.parquet')
    scaler_path = os.path.join(args.out_dir, f'scaler_params{suffix}.json')

    df_out.to_parquet(rl_path, index=False)
    static_df.to_parquet(static_path, index=False)
    with open(scaler_path, 'w') as f:
        json.dump(scaler_params, f, indent=2)

    logging.info("Saved rl_dataset -> %s  (%d rows, %d cols)",
                 rl_path, len(df_out), len(df_out.columns))
    logging.info("Saved static_context -> %s  (%d stays)", static_path, len(static_df))
    logging.info("Saved scaler_params -> %s", scaler_path)

    # Split summary
    for spl in ('train', 'val', 'test'):
        sub = df_out[df_out['split'] == spl]
        pos = sub[sub['done'] == 1][C_READMIT_30D].mean()
        logging.info("  %-5s: %d transitions, %d stays, readmit_rate=%.3f",
                     spl,
                     len(sub),
                     sub[C_ICUSTAYID].nunique(),
                     pos)

    logging.info("Step 10 complete.")
