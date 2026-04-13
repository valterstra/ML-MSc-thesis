"""
Step 09b.2 -- Variable selection: action-responsive features.

PURPOSE
-------
Identify which state features (labs, vitals, derived scores) are meaningfully
influenced by clinical actions (drugs, ventilation, fluids). These are the
features that an RL agent can actually move by choosing different actions.

This is the second half of two-stage variable selection:
  Step 09b.1: outcome relevance    -- which features predict readmission?
  Step 09b.2: transition relevance -- which features respond to actions?  (this script)

The final RL state space = features that are BOTH outcome-relevant AND action-responsive.

METHOD
------
For each time-varying state feature f, train a LightGBM predicting next-bloc f
(f at bloc t+1) from current state + current actions (at bloc t).

  X = [all time-varying state features at bloc t]
    + [static context: age, charlson_score, gender]
    + [all action columns at bloc t]
  y = f at bloc t+1

Then measure the ACTION IMPORTANCE SHARE:
  action_share = sum(importance of action columns) / sum(all feature importances)

A high action_share means actions contribute meaningfully to predicting where
this feature goes next -- i.e., the feature is responsive to treatment.
A low action_share means the feature evolves mostly on its own trajectory
regardless of what drugs are given.

METRICS
-------
  Continuous features (labs, vitals, SOFA, SIRS, etc.):  R2 on held-out blocs
  mechvent (binary):                                       AUC on held-out blocs

mechvent is included separately because initiating/weaning ventilation is a
direct clinical decision and clinically meaningful even though it is binary.

TRAINING DATA
-------------
Consecutive bloc pairs within the same stay: (bloc t, bloc t+1) for t = 1..N-1.
Last bloc per stay is dropped (no next-bloc target exists).
Stay-level 80/20 split: stays assigned to train or test by subject to avoid
a single patient's trajectory appearing in both sets.

OUTPUTS
-------
  reports/icu_readmit/step_09b_causal_actions/step_02_transition_selection/
    feature_responsiveness.csv   -- one row per feature:
                                    r2 (or auc), action_share, rank, metric_type
    action_importance_detail.csv -- per-feature breakdown of which action
                                    columns had the highest importance
    variable_selection_09b.json  -- features above action_share threshold + metadata

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_02_transition_selection.py
    python scripts/icu_readmit/step_02_transition_selection.py --features HR SysBP WBC_count
    python scripts/icu_readmit/step_02_transition_selection.py --max-rows 100000
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_BLOC,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES, CHARLSON_FLAG_COLS,
)

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

# Continuous state features -- these are the targets we model next-bloc value for
CONTINUOUS_TARGETS = (
    CHART_FIELD_NAMES
    + LAB_FIELD_NAMES
    + ['SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2']
)

# Binary target modelled separately with AUC
BINARY_TARGETS = ['mechvent']

# All action columns -- these are the features whose importance share we measure
ACTION_COLS = [
    'vasopressor_dose', 'ivfluid_dose',
    'antibiotic_active', 'anticoagulant_active', 'diuretic_active',
    'steroid_active', 'insulin_active', 'opioid_active',
    'sedation_active', 'transfusion_active', 'electrolyte_active',
]

# Static context included as predictors (they provide patient-level context
# for how labs evolve, but are not themselves targets)
STATIC_CONTEXT = ['age', 'charlson_score', 'gender']

LGB_PARAMS_REGRESSION = {
    'objective':             'regression',
    'metric':                'rmse',
    'learning_rate':         0.05,
    'num_leaves':            31,
    'min_child_samples':     20,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'n_estimators':          300,
    'early_stopping_rounds': 20,
    'verbose':               -1,
    'n_jobs':                -1,
    'random_state':          42,
}

LGB_PARAMS_BINARY = {
    'objective':             'binary',
    'metric':                'auc',
    'learning_rate':         0.05,
    'num_leaves':            31,
    'min_child_samples':     20,
    'subsample':             0.8,
    'colsample_bytree':      0.8,
    'n_estimators':          300,
    'early_stopping_rounds': 20,
    'verbose':               -1,
    'n_jobs':                -1,
    'random_state':          42,
}

# Action importance share threshold: features above this are considered
# action-responsive. 0.05 = actions account for at least 5% of total importance.
ACTION_SHARE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_transition_pairs(df, target_col, predictor_cols):
    """
    Build (X_t, y_{t+1}) pairs from bloc-level data.

    For each stay, shifts target_col by -1 within the stay to create the
    next-bloc value. Drops the last bloc per stay (no next bloc exists).

    Returns X (predictor DataFrame) and y (Series of next-bloc target values).
    Only rows where both the target and all predictors are non-NaN are kept.
    """
    df_s = df.sort_values([C_ICUSTAYID, C_BLOC]).copy()
    df_s['__next_target__'] = df_s.groupby(C_ICUSTAYID)[target_col].shift(-1)

    # Keep only rows that have a valid next-bloc target (drops last bloc)
    df_s = df_s[df_s['__next_target__'].notna()]

    present = [c for c in predictor_cols if c in df_s.columns]
    X = df_s[present].copy()
    y = df_s['__next_target__']
    return X, y


def stay_level_split(df, test_frac=0.2, seed=42):
    """
    Assign stays to train/test at stay level (not row level).
    Returns two sets of icustayid values.
    """
    stays = df[C_ICUSTAYID].unique()
    rng = np.random.default_rng(seed)
    rng.shuffle(stays)
    n_test = int(len(stays) * test_frac)
    test_stays = set(stays[:n_test])
    train_stays = set(stays[n_test:])
    return train_stays, test_stays


def compute_action_share(feature_names, importances, action_cols_present):
    """
    Compute what fraction of total LightGBM importance is attributable
    to action columns.

    action_share = sum(imp[a] for a in action_cols) / sum(all importances)
    """
    imp_series = pd.Series(importances, index=feature_names)
    action_imp  = imp_series[imp_series.index.isin(action_cols_present)].sum()
    total_imp   = imp_series.sum()
    share = float(action_imp / total_imp) if total_imp > 0 else 0.0

    # Top 3 action columns by importance
    action_imp_detail = (
        imp_series[imp_series.index.isin(action_cols_present)]
        .sort_values(ascending=False)
        .head(3)
        .to_dict()
    )
    return share, action_imp_detail


def run_one_feature(df_train, df_test, target_col, predictor_cols, is_binary=False):
    """
    Train LightGBM predicting next-bloc value of target_col.
    Returns (metric_value, action_share, action_detail_dict, best_iter).

    metric_value: R2 for continuous, AUC for binary.
    action_share: fraction of total importance from action columns.
    """
    params = LGB_PARAMS_BINARY if is_binary else LGB_PARAMS_REGRESSION

    X_train, y_train = make_transition_pairs(df_train, target_col, predictor_cols)
    X_test,  y_test  = make_transition_pairs(df_test,  target_col, predictor_cols)

    if len(y_train) < 100 or y_train.nunique() < 2:
        return None  # Not enough data or no variance

    # Encode object columns as category
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        X_train[c] = X_train[c].astype('category')
        X_test[c]  = X_test[c].astype('category')

    model = lgb.LGBMClassifier(**params) if is_binary else lgb.LGBMRegressor(**params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        callbacks=[
            lgb.early_stopping(params['early_stopping_rounds'], verbose=False),
            lgb.log_evaluation(period=-1),
        ],
    )

    preds = model.predict_proba(X_test)[:, 1] if is_binary else model.predict(X_test)

    if is_binary:
        if y_test.nunique() < 2:
            metric_val = float('nan')
        else:
            metric_val = roc_auc_score(y_test, preds)
    else:
        metric_val = r2_score(y_test, preds)

    action_cols_present = [c for c in ACTION_COLS if c in X_train.columns]
    action_share, action_detail = compute_action_share(
        X_train.columns.tolist(), model.feature_importances_, action_cols_present)

    return metric_val, action_share, action_detail, model.best_iteration_


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(
        REPO_DIR, 'data', 'processed', 'icu_readmit', 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=os.path.join(
        REPO_DIR, 'reports', 'icu_readmit', 'step_09b_causal_actions', 'step_02_transition_selection'))
    parser.add_argument('--action-share-threshold', type=float,
                        default=ACTION_SHARE_THRESHOLD,
                        help='Min action importance share to flag a feature as responsive')
    parser.add_argument('--features', nargs='+', default=None,
                        help='Run only these target features (for smoke testing)')
    parser.add_argument('--max-rows', type=int, default=None,
                        help='Subsample this many rows for smoke testing')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    log_file = args.log or os.path.join(REPO_DIR, 'logs', 'step_09b_step_02_transition_selection.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 09b.2 started. input=%s", args.input)
    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    if args.max_rows:
        stays_sub = df[C_ICUSTAYID].unique()[:args.max_rows // 25]  # ~25 blocs per stay
        df = df[df[C_ICUSTAYID].isin(stays_sub)]
        logging.info("Subsampled to %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    # Stay-level train/test split
    train_stays, test_stays = stay_level_split(df, test_frac=0.2)
    df_train = df[df[C_ICUSTAYID].isin(train_stays)]
    df_test  = df[df[C_ICUSTAYID].isin(test_stays)]
    logging.info("Split: %d train stays / %d test stays", len(train_stays), len(test_stays))

    # Predictor columns = state + static context + actions
    all_state_cols = (
        [c for c in CONTINUOUS_TARGETS + BINARY_TARGETS if c in df.columns]
        + [c for c in STATIC_CONTEXT if c in df.columns]
        + [c for c in ACTION_COLS if c in df.columns]
    )

    # Target features to model
    continuous_targets = [c for c in CONTINUOUS_TARGETS if c in df.columns]
    binary_targets     = [c for c in BINARY_TARGETS     if c in df.columns]

    if args.features:
        continuous_targets = [c for c in continuous_targets if c in args.features]
        binary_targets     = [c for c in binary_targets     if c in args.features]

    all_targets = continuous_targets + binary_targets
    logging.info("Modelling %d features (%d continuous + %d binary)",
                 len(all_targets), len(continuous_targets), len(binary_targets))

    # ---------------------------------------------------------------------------
    # Main loop: one model per target feature
    # ---------------------------------------------------------------------------
    rows = []
    action_detail_rows = []

    for i, feat in enumerate(all_targets):
        is_binary = feat in binary_targets
        metric_label = 'auc' if is_binary else 'r2'

        result = run_one_feature(
            df_train, df_test, feat, all_state_cols, is_binary=is_binary)

        if result is None:
            logging.info("  [%d/%d] %-30s  SKIPPED (insufficient data)",
                         i + 1, len(all_targets), feat)
            continue

        metric_val, action_share, action_detail, best_iter = result
        responsive = action_share >= args.action_share_threshold

        logging.info("  [%d/%d] %-30s  %s=%.4f  action_share=%.3f  iter=%d  %s",
                     i + 1, len(all_targets), feat,
                     metric_label, metric_val, action_share, best_iter,
                     "[RESPONSIVE]" if responsive else "")

        rows.append({
            'feature':      feat,
            'metric_type':  metric_label,
            'metric_value': round(metric_val, 4),
            'action_share': round(action_share, 4),
            'best_iter':    best_iter,
            'responsive':   responsive,
        })

        for act, imp_val in action_detail.items():
            action_detail_rows.append({
                'target_feature': feat,
                'action_col':     act,
                'importance':     imp_val,
            })

    # ---------------------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------------------
    responsiveness = (
        pd.DataFrame(rows)
        .sort_values('action_share', ascending=False)
        .reset_index(drop=True)
    )
    responsiveness['rank'] = responsiveness.index + 1
    responsiveness.to_csv(
        os.path.join(args.out_dir, 'feature_responsiveness.csv'), index=False)

    action_detail_df = pd.DataFrame(action_detail_rows)
    action_detail_df.to_csv(
        os.path.join(args.out_dir, 'action_importance_detail.csv'), index=False)

    responsive_features = responsiveness.loc[
        responsiveness['responsive'], 'feature'].tolist()

    out = {
        'action_share_threshold': args.action_share_threshold,
        'n_features_modelled':    len(rows),
        'n_responsive':           len(responsive_features),
        'responsive_features':    responsive_features,
        'top10_by_action_share':  responsiveness.head(10)[
            ['feature', 'metric_type', 'metric_value', 'action_share']
        ].to_dict(orient='records'),
    }
    with open(os.path.join(args.out_dir, 'variable_selection_09b.json'), 'w') as f:
        json.dump(out, f, indent=2)

    logging.info("=== RESULTS SUMMARY ===")
    logging.info("  Responsive features (action_share >= %.2f): %d / %d",
                 args.action_share_threshold, len(responsive_features), len(rows))
    logging.info("  Top 5 by action_share:")
    for _, r in responsiveness.head(5).iterrows():
        logging.info("    %-28s  action_share=%.3f  %s=%.4f",
                     r['feature'], r['action_share'], r['metric_type'], r['metric_value'])
    logging.info("Outputs written to %s", args.out_dir)
    logging.info("Step 09b.2 complete.")
