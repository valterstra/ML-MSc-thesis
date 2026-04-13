"""
Step 02 -- Variable ranking for causal discovery.

PURPOSE
-------
Rank state and static variables by their predictive importance for 30-day
readmission, using two separate LightGBM classifiers (one per group).
Running groups separately prevents static variables (which dominate when
pooled) from drowning out state variables.

The output is two ranked lists used to build the candidate pool for the
random stability analysis in step_03.

TWO GROUPS
----------
  State variables  -- last-bloc physiological measurements (last_* columns).
                      Question: which discharge-state variables predict
                      readmission after conditioning on the patient's
                      physiology at discharge?

  Static variables -- demographics and comorbidities (age, charlson_score,
                      etc.). Question: which patient-level baseline factors
                      predict readmission?

MISSING DATA FLAG
-----------------
A variable is flagged (high_missing=True) if >20% of stays have a missing
value for that variable. High-ranked variables with high missingness should
be treated with caution: importance may reflect the pattern of missingness
(i.e. only measured in a specific patient subset) rather than the true
physiological signal.

OUTPUTS
-------
  reports/icu_readmit/step_09a_causal_states/state_variable_ranking.csv
  reports/icu_readmit/step_09a_causal_states/static_variable_ranking.csv
  reports/icu_readmit/step_09a_causal_states/variable_ranking_summary.json

Columns in each ranking CSV:
  rank            -- 1 = most important
  variable        -- column name
  importance      -- LightGBM split-gain importance (normalised to sum=1)
  pct_missing     -- % of stays with NaN for this variable
  high_missing    -- True if pct_missing > 20%

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_09a_causal_states/step_02_variable_ranking.py
    python scripts/icu_readmit/step_09a_causal_states/step_02_variable_ranking.py --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_READMIT_30D,
    C_GENDER, C_AGE, C_WEIGHT,
    C_RACE, C_INSURANCE, C_MARITAL_STATUS,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_CHARLSON, C_RE_ADMISSION, C_PRIOR_ED_VISITS,
    C_DRG_SEVERITY, C_DRG_MORTALITY, C_DISCHARGE_DISPOSITION,
    CHARLSON_FLAG_COLS,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

MISSING_FLAG_THRESHOLD = 20.0   # % missing above which high_missing=True

LGBM_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'n_estimators':     300,
    'learning_rate':    0.05,
    'num_leaves':       31,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'random_state':     42,
    'verbose':          -1,
    'n_jobs':           -1,
}

STATIC_COLS = [
    C_GENDER, C_AGE, C_WEIGHT,
    C_RACE, C_INSURANCE, C_MARITAL_STATUS,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_CHARLSON, C_RE_ADMISSION, C_PRIOR_ED_VISITS,
    C_DRG_SEVERITY, C_DRG_MORTALITY, C_DISCHARGE_DISPOSITION,
] + CHARLSON_FLAG_COLS


def compute_missing(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    """Return % missing per column."""
    return (df[cols].isna().mean() * 100).round(2)


def rank_group(
    df: pd.DataFrame,
    feature_cols: list[str],
    group_name: str,
) -> pd.DataFrame:
    """
    Train a LightGBM on feature_cols -> readmit_30d.
    Return a ranked DataFrame with importance + missing stats.
    """
    X = df[feature_cols].copy()
    y = df[C_READMIT_30D].copy()

    # Convert object columns to category dtype -- LightGBM handles these natively
    for col in X.select_dtypes(include='object').columns:
        X[col] = X[col].astype('category')

    # LightGBM handles NaN natively -- no imputation needed
    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    log.info("[%s] Training LightGBM on %d features, %d train / %d val stays...",
             group_name, len(feature_cols), len(X_tr), len(X_val))

    model = lgb.LGBMClassifier(**LGBM_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(30, verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    val_auc = model.best_score_['valid_0']['auc']
    log.info("[%s] Validation AUC: %.4f  (best iteration: %d)",
             group_name, val_auc, model.best_iteration_)

    # Importance: gain (sum of split gains) -- more robust than split count
    importances = model.booster_.feature_importance(importance_type='gain')
    total = importances.sum()
    norm_importances = importances / total if total > 0 else importances

    pct_missing = compute_missing(df, feature_cols)

    ranking = pd.DataFrame({
        'variable':    feature_cols,
        'importance':  norm_importances,
        'pct_missing': pct_missing.values,
    })
    ranking = ranking.sort_values('importance', ascending=False).reset_index(drop=True)
    ranking.insert(0, 'rank', ranking.index + 1)
    ranking['high_missing'] = ranking['pct_missing'] > MISSING_FLAG_THRESHOLD

    return ranking, val_auc


def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    input_path = Path(args.input)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", input_path)
    df = pd.read_parquet(input_path)
    log.info("Loaded: %d stays, %d columns", len(df), len(df.columns))

    if args.smoke:
        df = df.sample(n=min(5000, len(df)), random_state=42)
        log.info("Smoke test: using %d stays", len(df))

    # Identify column groups from what actually exists in the parquet
    state_cols  = [c for c in df.columns if c.startswith('last_')]
    static_cols = [c for c in STATIC_COLS if c in df.columns]

    log.info("State cols:  %d", len(state_cols))
    log.info("Static cols: %d", len(static_cols))

    # --- State variable ranking ---
    state_ranking, state_auc = rank_group(df, state_cols, 'states')

    state_path = report_dir / 'state_variable_ranking.csv'
    state_ranking.to_csv(state_path, index=False)
    log.info("Saved: %s", state_path)

    # --- Static variable ranking ---
    static_ranking, static_auc = rank_group(df, static_cols, 'statics')

    static_path = report_dir / 'static_variable_ranking.csv'
    static_ranking.to_csv(static_path, index=False)
    log.info("Saved: %s", static_path)

    # --- Summary ---
    summary = {
        'n_stays':              len(df),
        'readmit_rate_pct':     round(float(df[C_READMIT_30D].mean()) * 100, 2),
        'missing_flag_threshold_pct': MISSING_FLAG_THRESHOLD,
        'state': {
            'n_features':       len(state_cols),
            'val_auc':          round(state_auc, 4),
            'top_10':           state_ranking.head(10)[['rank','variable','importance','pct_missing','high_missing']].to_dict('records'),
            'n_high_missing':   int(state_ranking['high_missing'].sum()),
            'high_missing_top20': state_ranking.head(20).query('high_missing == True')['variable'].tolist(),
        },
        'static': {
            'n_features':       len(static_cols),
            'val_auc':          round(static_auc, 4),
            'top_10':           static_ranking.head(10)[['rank','variable','importance','pct_missing','high_missing']].to_dict('records'),
            'n_high_missing':   int(static_ranking['high_missing'].sum()),
            'high_missing_top20': static_ranking.head(20).query('high_missing == True')['variable'].tolist(),
        },
        'runtime_s': round(time.time() - t0, 1),
    }

    summary_path = report_dir / 'variable_ranking_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info("Summary saved: %s", summary_path)

    # Print readable summary to console
    log.info("=" * 60)
    log.info("STATE VARIABLES  (AUC=%.4f)", state_auc)
    log.info("%-5s %-35s %9s  %10s  %s",
             "Rank", "Variable", "Importance", "Pct_Missing", "Flag")
    log.info("-" * 70)
    for _, row in state_ranking.head(20).iterrows():
        flag = " *** HIGH MISSING" if row['high_missing'] else ""
        log.info("%-5d %-35s %9.4f  %9.1f%%%s",
                 row['rank'], row['variable'],
                 row['importance'], row['pct_missing'], flag)

    log.info("")
    log.info("STATIC VARIABLES  (AUC=%.4f)", static_auc)
    log.info("%-5s %-35s %9s  %10s  %s",
             "Rank", "Variable", "Importance", "Pct_Missing", "Flag")
    log.info("-" * 70)
    for _, row in static_ranking.iterrows():
        flag = " *** HIGH MISSING" if row['high_missing'] else ""
        log.info("%-5d %-35s %9.4f  %9.1f%%%s",
                 row['rank'], row['variable'],
                 row['importance'], row['pct_missing'], flag)

    log.info("=" * 60)
    log.info("Done in %.1fs", time.time() - t0)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--input', default=str(
            PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit'
            / 'step_09a_causal_states' / 'stay_level.parquet'),
        help='Path to stay_level.parquet (step_01 output)')
    parser.add_argument(
        '--report-dir', default=str(
            PROJECT_ROOT / 'reports' / 'icu_readmit' / 'step_09a_causal_states'),
        help='Directory for ranking CSVs and summary JSON')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: sample 5000 stays')
    args = parser.parse_args()
    main(args)
