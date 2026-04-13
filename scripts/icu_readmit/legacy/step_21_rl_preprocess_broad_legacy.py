"""
Step 10b -- RL preprocessing (broad state): build (s, a, r, s', done) dataset.

PURPOSE
-------
Same pipeline as step_10, but uses a broad 55-feature state space following the
Raghu 2017 / AI-Clinician philosophy: include everything a bedside clinician would
routinely see in the ICU. No variable-selection filtering applied to the state.

This produces a parallel rl_dataset_broad.parquet for direct comparison against
the narrow (15-feature, causally guided) dataset from step_10.

Comparison value:
  Step 10  (narrow, 15 states) -- causally selected via 09a/09b/09c
  Step 10b (broad,  55 states) -- Raghu-style, all well-covered ICU measurements

Same actions, same reward, same split logic. Only the state representation changes.

STATE FEATURES (55)
-------------------
  Vitals (8):
    HR, MeanBP, Arterial_BP_Sys, Arterial_BP_Dia, RR, SpO2, Temp_C, FiO2_1

  Blood gas (6):
    Arterial_pH, paO2, paCO2, HCO3, Arterial_BE, Arterial_lactate

  Renal / metabolic (10):
    BUN, Creatinine, Potassium, Sodium, Chloride, Glucose,
    Magnesium, Calcium, Phosphate, Anion_Gap

  Hematology / coagulation (7):
    Hb, WBC_count, Platelets_count, PT, PTT, INR, Fibrinogen

  Liver / other labs (4):
    SGOT, SGPT, Total_bili, Albumin

  Derived scores (5):
    SOFA, SIRS, Shock_Index, PaO2_FiO2, GCS

  Fluid balance (3):
    input_total, output_total, cumulated_balance

  Ventilator (3):
    PEEP, TidalVolume, MinuteVentil

  Hemodynamic (1):
    CVP

  Neuro / sedation depth (1):
    RASS

  Static / context -- included in state vector like Raghu (3):
    age, charlson_score, prior_ed_visits_6m

  Binary state (2):
    mechvent, re_admission

Exclusions vs full dataset:
  - SysBP/DiaBP: redundant with Arterial_BP_Sys/Dia (same measurement, prefer invasive)
  - RR_Spontaneous/RR_Total: redundant with RR
  - Temp_F: redundant with Temp_C
  - TidalVolume_Observed: redundant with TidalVolume
  - GCS_Eye/Verbal/Motor: redundant with summed GCS
  - Ionised_Ca: redundant with Calcium
  - SvO2, PAPsys/dia/mean, CI, SVR: <8% coverage -- noise not signal
  - ETCO2, extubated: ~36% coverage -- too sparse
  - CRP, ACT, Total_protein, LDH: <3% coverage
  - Eos_pct, Basos_pct: 0% coverage
  - Neuts_pct/Lymphs_pct/Monos_pct: low RL utility; WBC_count captures immune state
  - Charlson flags (18 individual): replaced by charlson_score (summary)
  - drg_severity / drg_mortality: all zeros in dataset (never populated -- excluded)
  - cam_icu: all zeros in dataset (CAM-ICU never populated -- excluded)

ACTIONS (5 binary, 2^5 = 32 combinations) -- identical to step_10
---------------------------------------------------------------------
  Same 5 binary actions, same integer encoding.

REWARD -- identical to step_10
-------------------------------
  Dense:    r = SOFA_t - SOFA_{t+1}
  Terminal: r = +15 (readmit_30d=0) / -15 (readmit_30d=1)

OUTPUTS
-------
  data/processed/icu_readmit/rl_dataset_broad.parquet
  data/processed/icu_readmit/static_context_broad.parquet
  data/processed/icu_readmit/scaler_params_broad.json

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_10b_rl_preprocess_broad.py
    python scripts/icu_readmit/step_10b_rl_preprocess_broad.py --smoke
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

# Continuous state features -- z-score normalised
CONTINUOUS_STATE = [
    # Vitals
    'HR', 'MeanBP', 'Arterial_BP_Sys', 'Arterial_BP_Dia', 'RR', 'SpO2', 'Temp_C', 'FiO2_1',
    # Blood gas
    'Arterial_pH', 'paO2', 'paCO2', 'HCO3', 'Arterial_BE', 'Arterial_lactate',
    # Renal / metabolic
    'BUN', 'Creatinine', 'Potassium', 'Sodium', 'Chloride', 'Glucose',
    'Magnesium', 'Calcium', 'Phosphate', 'Anion_Gap',
    # Hematology / coagulation
    'Hb', 'WBC_count', 'Platelets_count', 'PT', 'PTT', 'INR',
    # Liver / other labs
    'SGOT', 'SGPT', 'Total_bili', 'Albumin',
    # Derived scores
    'SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'GCS',
    # Fluid balance
    'input_total', 'output_total', 'cumulated_balance',
    # Ventilator
    'PEEP', 'TidalVolume', 'MinuteVentil',
    # Neuro / sedation depth
    'RASS',
    # Static context -- in state vector (Raghu-style)
    'age', 'charlson_score', 'prior_ed_visits_6m',
]

# Binary state features -- kept as 0/1, not normalised
BINARY_STATE = ['mechvent', 're_admission']

STATE_FEATURES = CONTINUOUS_STATE + BINARY_STATE  # 55 total

# Action columns -- identical to step_10
ACTIONS = [
    ('vasopressor_b', 1),
    ('ivfluid_b',     2),
    ('antibiotic_b',  4),
    ('sedation_b',    8),
    ('diuretic_b',   16),
]
ACTION_NAMES = [a[0] for a in ACTIONS]

ACTION_SOURCES = {
    'vasopressor_b': ('vasopressor_dose',    'dose'),
    'ivfluid_b':     ('ivfluid_dose',        'dose'),
    'antibiotic_b':  ('antibiotic_active',   'binary'),
    'sedation_b':    ('sedation_active',     'binary'),
    'diuretic_b':    ('diuretic_active',     'binary'),
}

# Static context stored separately (demographic reference, not in RL state)
STATIC_CONTEXT = ['age', 'gender', 'race', 'charlson_score',
                  'prior_ed_visits_6m', 're_admission']

SPLIT_FRACS    = (0.70, 0.15, 0.15)
TERMINAL_REWARD = 15.0

# ---------------------------------------------------------------------------
# Outlier handling
# ---------------------------------------------------------------------------

# Step 1: Clinical plausibility clips applied to raw values BEFORE normalisation.
# Bounds are physiologically motivated -- values outside these ranges are either
# incompatible with survival or known MIMIC-IV data artifacts.
CLIP_BOUNDS = {
    'SpO2':            (50,   100),   # below 50 incompatible with survival
    'Temp_C':          (25,    43),   # below 25 artifact; above 43 non-survivable
    'Shock_Index':     ( 0,     5),   # max z=469 is pure artifact (near-zero BP)
    'WBC_count':       ( 0.1, 200),   # max z=55; 407 is artifact/extreme leukemia
    'Glucose':         (20,   800),   # below 20 incompatible; DKA ceiling ~800
    'Creatinine':      ( 0.1,  25),   # max z=28; clips extreme AKI noise
    'BUN':             ( 1,   200),   # renal failure ceiling
    'Potassium':       ( 1.5,   9),   # outside this range is fatal
    'Magnesium':       ( 0.5,   7),   # above 7 typically fatal; 9.8 is artifact
    'Calcium':         ( 4,    15),   # above 15 near-always artifact
    'Platelets_count': ( 1,  1500),   # max_z=14; thrombocytosis ceiling
    'Arterial_BP_Dia': ( 0,   150),   # 196 diastolic is extreme
    'RR':              ( 4,    60),   # 78 breaths/min is artifact
    'paCO2':           (10,   120),   # 181 clips extreme COPD artifacts
    'TidalVolume':     (50,  1200),   # 1800 mL non-physiological
    'MinuteVentil':    ( 0,    30),   # 50 L/min is artifact
    'INR':             ( 0.5,  15),   # 21.8 clips extreme coagulopathy
    'PT':              ( 5,   120),   # 161 seconds clips artifacts
    'cumulated_balance': (-50000, 50000),  # cap extreme long-stay fluid accumulation
}

# Step 2: Log transform applied AFTER clipping, BEFORE z-score normalisation.
# Formula: log(0.1 + x) -- the 0.1 offset safely handles zeros.
# Applied to right-skewed features where skewness > 2 or max_z > 10.
# cumulated_balance excluded (can be negative -- z-score only).
LOG_TRANSFORM = [
    'WBC_count',        # skew=9.1,  max_z=55
    'SGOT',             # skew=12.7, max_z=23
    'SGPT',             # skew=11.8, max_z=25
    'Total_bili',       # skew=7.2,  max_z=24
    'INR',              # skew=6.0,  max_z=35
    'PT',               # skew=5.7,  max_z=24
    'Creatinine',       # skew=4.1,  max_z=28
    'Arterial_lactate', # skew=3.7,  max_z=24
    'PTT',              # skew=3.0,  max_z=6
    'PaO2_FiO2',        # skew=2.4,  max_z=10
    'BUN',              # skew=2.2,  max_z=12
    'Glucose',          # skew=2.0,  max_z=17
    'input_total',      # skew=4.9,  max_z=21
    'output_total',     # skew=6.1,  max_z=24
]


# ---------------------------------------------------------------------------
# Helper functions (identical logic to step_10)
# ---------------------------------------------------------------------------

def assign_splits(stay_ids: np.ndarray, fracs=(0.70, 0.15, 0.15)) -> dict:
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
    df_s = df.sort_values([C_ICUSTAYID, C_BLOC]).copy()

    s_next_cols = {f's_next_{f}': f for f in STATE_FEATURES}
    for next_col, feat in s_next_cols.items():
        df_s[next_col] = df_s.groupby(C_ICUSTAYID)[f's_{feat}'].shift(-1)

    df_s['SOFA_next'] = df_s.groupby(C_ICUSTAYID)['SOFA'].shift(-1)
    df_s['done'] = (
        df_s.groupby(C_ICUSTAYID)[C_BLOC].transform('max') == df_s[C_BLOC]
    ).astype(int)

    df_s['r'] = np.where(
        df_s['done'] == 0,
        df_s['SOFA'] - df_s['SOFA_next'],
        np.where(
            df_s[C_READMIT_30D] == 0,
            TERMINAL_REWARD,
            -TERMINAL_REWARD,
        )
    )

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

    log_file = args.log or str(PROJECT_ROOT / 'logs' / 'legacy' / 'icu_readmit' / 'step_21_rl_preprocess_broad_legacy.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 10b (broad) started. input=%s", args.input)
    logging.info("State features: %d total (%d continuous, %d binary)",
                 len(STATE_FEATURES), len(CONTINUOUS_STATE), len(BINARY_STATE))
    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    if args.smoke:
        smoke_stays = np.sort(df[C_ICUSTAYID].unique())[:2000]
        df = df[df[C_ICUSTAYID].isin(smoke_stays)].copy()
        logging.info("Smoke mode: %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    # -----------------------------------------------------------------------
    # 1. Validate columns
    # -----------------------------------------------------------------------
    missing_state  = [c for c in STATE_FEATURES if c not in df.columns]
    missing_action = [s for s, _ in ACTION_SOURCES.values() if s not in df.columns]
    if missing_state:
        logging.warning("Missing state columns (will be zero): %s", missing_state)
    if missing_action:
        logging.warning("Missing action source columns: %s", missing_action)

    # -----------------------------------------------------------------------
    # 2. Binary actions
    # -----------------------------------------------------------------------
    df = build_binary_actions(df)
    logging.info("Action distribution (stay-level frac):")
    for col, _ in ACTIONS:
        logging.info("  %-18s %.3f", col, df.groupby(C_ICUSTAYID)[col].max().mean())
    logging.info("Unique action combinations used: %d / 32",
                 df['a'].nunique())

    # -----------------------------------------------------------------------
    # 3. Split
    # -----------------------------------------------------------------------
    stay_ids  = df[C_ICUSTAYID].unique()
    split_map = assign_splits(stay_ids, SPLIT_FRACS)
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
                logging.info("  Clip %-25s [%.1f, %.1f]  clipped %d rows", feat, lo, hi, n_clipped)
            df[feat] = df[feat].clip(lo, hi)

    # -----------------------------------------------------------------------
    # 3c. Log transform skewed features
    # -----------------------------------------------------------------------
    for feat in LOG_TRANSFORM:
        if feat in df.columns:
            df[feat] = np.log(0.1 + df[feat])

    # -----------------------------------------------------------------------
    # 4. Normalise state features
    # -----------------------------------------------------------------------
    train_mask = df['split'] == 'train'

    for feat in STATE_FEATURES:
        df[f's_{feat}'] = df[feat] if feat in df.columns else 0.0

    # Re-apply clips to s_ columns for non-log-transformed features only.
    # Log-transformed features are in log-space -- raw bounds do not apply.
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
        logging.info("  Normalised %-25s  mean=%.3f  std=%.3f", feat, mu, std)

    # Post-normalisation clip to [-5, 5] -- standard deep RL practice.
    # Catches any remaining extreme z-scores regardless of distribution shape.
    Z_CLIP = 5.0
    for feat in CONTINUOUS_STATE:
        col = f's_{feat}'
        if col in df.columns:
            df[col] = df[col].clip(-Z_CLIP, Z_CLIP)

    for feat in BINARY_STATE:
        col = f's_{feat}'
        df[col] = df[col].clip(0, 1)

    # -----------------------------------------------------------------------
    # 5. Build transitions
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

    logging.info("Transitions built: %d rows", len(df_out))
    logging.info("  done=0 (non-terminal): %d", (df_out['done'] == 0).sum())
    logging.info("  done=1 (terminal):     %d", (df_out['done'] == 1).sum())

    non_term = df_out[df_out['done'] == 0]['r']
    term     = df_out[df_out['done'] == 1]['r']
    logging.info("Reward stats (non-terminal): mean=%.3f  std=%.3f  min=%.3f  max=%.3f",
                 non_term.mean(), non_term.std(), non_term.min(), non_term.max())
    logging.info("Reward stats (terminal):     +15 count=%d  -15 count=%d",
                 (term > 0).sum(), (term < 0).sum())

    # -----------------------------------------------------------------------
    # 6. Static context
    # -----------------------------------------------------------------------
    static_cols = [C_ICUSTAYID, 'split'] + [c for c in STATIC_CONTEXT if c in df.columns]
    static_df = df[static_cols].groupby(C_ICUSTAYID).first().reset_index()

    # -----------------------------------------------------------------------
    # 7. Save
    # -----------------------------------------------------------------------
    suffix = '_broad_smoke' if args.smoke else '_broad'

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

    for spl in ('train', 'val', 'test'):
        sub = df_out[df_out['split'] == spl]
        pos = sub[sub['done'] == 1][C_READMIT_30D].mean()
        logging.info("  %-5s: %d transitions, %d stays, readmit_rate=%.3f",
                     spl, len(sub), sub[C_ICUSTAYID].nunique(), pos)

    logging.info("Step 10b complete.")
