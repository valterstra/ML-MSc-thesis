"""
Step 09b.1 -- Variable selection: outcome-relevant features.

PURPOSE
-------
Identify which features are predictive of 30-day readmission (readmit_30d).
This is the first half of a two-stage variable selection:

  Step 09b.1 (this script): outcome relevance -- which features predict readmission?
  Step 09b.2 (next script): transition relevance -- which features respond to actions?

The final RL state space will be the INTERSECTION of both.

MODELS
------
Four LightGBM classifiers, all predicting readmit_30d at stay level (~61k stays).
Each model sees a different view of the data to prevent one feature group from
drowning out another. The combined model (Model 4) then reveals which features
dominate when all groups compete head-to-head.

  Model 1 -- time-varying labs + vitals (last bloc per stay, ~90 features)
    Captures the patient's physiological state at discharge.
    Uses last bloc because that is closest to the readmission event.

  Model 2 -- static demographics + comorbidities (first bloc, ~31 features)
    Captures who the patient is: age, Charlson score, 18 comorbidity flags,
    discharge disposition, insurance, etc.
    Uses first bloc because static features are constant across the stay.

  Model 3 -- action patterns (stay-level aggregates, ~22 features)
    Captures how the patient was treated: frac_<drug> (fraction of blocs where
    drug was active), ever_<drug> (any exposure), mean/max dose.
    Aggregated across all blocs to represent the full treatment trajectory.

  Model 4 -- COMBINED (all features from Models 1-3, ~143 features)
    All three feature groups joined at stay level. This reveals:
      (a) absolute importance of each feature across all groups
      (b) whether static factors still dominate when competing with labs/actions
      (c) which features are redundant (drop in importance vs isolated models)
    This is the primary model for deciding the final RL state variables.

AGGREGATION STRATEGY (Model 4)
-------------------------------
Each group uses a different aggregation before the join:
  - Time-varying: last bloc value (closest to outcome)
  - Static: first bloc value (same every row, first is fine)
  - Actions: frac_/ever_/mean_/max_ aggregates across all blocs
All joined on icustayid, then a single LightGBM is trained on the merged table.

Train/test split: 80/20 stratified by readmit_30d, random_state=42.

Outputs:
  reports/icu_readmit/step_09b_causal_actions/step_01_variable_selection/
    model1_importance.csv    -- time-varying features ranked by importance
    model2_importance.csv    -- static features ranked
    model3_importance.csv    -- action features ranked
    model4_importance.csv    -- combined features ranked (primary output)
    all_models_summary.csv   -- stacked importance table across all 4 models
    variable_selection.json  -- AUC + top features per model + metadata

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_01_variable_selection.py
    python scripts/icu_readmit/step_01_variable_selection.py --top-n 20
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_BLOC, C_READMIT_30D,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES, CHARLSON_FLAG_COLS,
)

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------
TIME_VARYING_FEATURES = (
    CHART_FIELD_NAMES
    + LAB_FIELD_NAMES
    + ['SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'mechvent', 'extubated']
)

STATIC_FEATURES = (
    ['age', 'gender', 'charlson_score', 're_admission', 'prior_ed_visits_6m',
     'drg_severity', 'drg_mortality', 'discharge_disposition',
     'race', 'insurance', 'marital_status', 'admission_type', 'admission_location']
    + CHARLSON_FLAG_COLS
)

ACTION_COLS = [
    'vasopressor_dose', 'ivfluid_dose',
    'antibiotic_active', 'anticoagulant_active', 'diuretic_active',
    'steroid_active', 'insulin_active', 'opioid_active',
    'sedation_active', 'transfusion_active', 'electrolyte_active',
]

LGB_PARAMS = {
    'objective':        'binary',
    'metric':           'auc',
    'learning_rate':    0.05,
    'num_leaves':       31,
    'min_child_samples': 20,
    'subsample':        0.8,
    'colsample_bytree': 0.8,
    'n_estimators':     500,
    'early_stopping_rounds': 30,
    'verbose':          -1,
    'n_jobs':           -1,
    'random_state':     42,
}


def build_stay_level(df, feature_cols, agg='last'):
    """
    Reduce bloc-level data to one row per stay.
    agg='last'  -> last bloc (time-varying features)
    agg='any'   -> any bloc active (binary action flags)
    agg='mean'  -> mean across blocs (dose/rate features)
    agg='first' -> first bloc (static features -- same value every bloc)
    """
    present = [c for c in feature_cols if c in df.columns]
    if agg == 'last':
        stay_df = (df.sort_values(C_BLOC)
                     .groupby(C_ICUSTAYID)[present + [C_READMIT_30D]]
                     .last()
                     .reset_index())
    elif agg == 'first':
        stay_df = (df.groupby(C_ICUSTAYID)[present + [C_READMIT_30D]]
                     .first()
                     .reset_index())
    elif agg == 'mean':
        stay_df = (df.groupby(C_ICUSTAYID)[present + [C_READMIT_30D]]
                     .mean()
                     .reset_index())
    else:
        raise ValueError(f"Unknown agg: {agg}")
    return stay_df, present


def build_action_stay_level(df, action_cols):
    """
    Build stay-level action features:
      - frac_<action>: fraction of blocs where action was active
      - ever_<action>: 1 if ever active during stay (binary)
    Dose columns (vasopressor_dose, ivfluid_dose) get mean and max.
    """
    present = [c for c in action_cols if c in df.columns]
    binary_actions = [c for c in present if c not in ('vasopressor_dose', 'ivfluid_dose')]
    dose_actions   = [c for c in present if c in ('vasopressor_dose', 'ivfluid_dose')]

    grp = df.groupby(C_ICUSTAYID)
    parts = [grp[C_READMIT_30D].first()]

    if binary_actions:
        frac = grp[binary_actions].mean().add_prefix('frac_')
        ever = (grp[binary_actions].max() > 0).astype(int).add_prefix('ever_')
        parts += [frac, ever]

    if dose_actions:
        parts.append(grp[dose_actions].mean().add_prefix('mean_'))
        parts.append(grp[dose_actions].max().add_prefix('max_'))

    stay_df = pd.concat(parts, axis=1).reset_index()
    feat_cols = [c for c in stay_df.columns if c not in (C_ICUSTAYID, C_READMIT_30D)]
    return stay_df, feat_cols


def run_lgb_model(stay_df, feature_cols, label, model_name):
    """Train LightGBM, return AUC + feature importance DataFrame."""
    X = stay_df[feature_cols].copy()
    y = stay_df[C_READMIT_30D].astype(int)

    # Encode categoricals
    cat_cols = X.select_dtypes(include='object').columns.tolist()
    for c in cat_cols:
        X[c] = X[c].astype('category')

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(LGB_PARAMS['early_stopping_rounds'], verbose=False),
                   lgb.log_evaluation(period=-1)],
    )

    val_pred = model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, val_pred)
    best_iter = model.best_iteration_
    logging.info("  %s: AUC=%.4f  best_iter=%d  features=%d",
                 model_name, auc, best_iter, len(feature_cols))

    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_,
        'model': model_name,
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    imp['rank'] = imp.index + 1

    return imp, auc, model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=os.path.join(
        REPO_DIR, 'data', 'processed', 'icu_readmit', 'ICUdataset.csv'))
    parser.add_argument('--out-dir', default=os.path.join(
        REPO_DIR, 'reports', 'icu_readmit', 'step_09b_causal_actions', 'step_01_variable_selection'))
    parser.add_argument('--top-n', type=int, default=20,
                        help='Top N features to include per group in summary JSON')
    parser.add_argument('--log', default=None)
    args = parser.parse_args()

    log_file = args.log or os.path.join(REPO_DIR, 'logs', 'step_09b_step_01_variable_selection.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 09b.1 started. input=%s", args.input)
    df = pd.read_csv(args.input)
    logging.info("Loaded %d rows, %d stays", len(df), df[C_ICUSTAYID].nunique())

    results = {}

    # ------------------------------------------------------------------
    # Model 1: time-varying labs + vitals (last bloc per stay)
    # ------------------------------------------------------------------
    logging.info("Model 1: time-varying labs + vitals (last bloc per stay)")
    stay1, feats1 = build_stay_level(df, TIME_VARYING_FEATURES, agg='last')
    logging.info("  %d stays, %d features", len(stay1), len(feats1))
    imp1, auc1, _ = run_lgb_model(stay1, feats1, C_READMIT_30D, 'time_varying')
    imp1.to_csv(os.path.join(args.out_dir, 'model1_importance.csv'), index=False)
    results['time_varying'] = {'auc': auc1, 'n_features': len(feats1),
                                'top_features': imp1.head(args.top_n)['feature'].tolist()}

    # ------------------------------------------------------------------
    # Model 2: static demographics + comorbidities (first bloc per stay)
    # ------------------------------------------------------------------
    logging.info("Model 2: static demographics + comorbidities")
    stay2, feats2 = build_stay_level(df, STATIC_FEATURES, agg='first')
    logging.info("  %d stays, %d features", len(stay2), len(feats2))
    imp2, auc2, _ = run_lgb_model(stay2, feats2, C_READMIT_30D, 'static')
    imp2.to_csv(os.path.join(args.out_dir, 'model2_importance.csv'), index=False)
    results['static'] = {'auc': auc2, 'n_features': len(feats2),
                          'top_features': imp2.head(args.top_n)['feature'].tolist()}

    # ------------------------------------------------------------------
    # Model 3: action patterns (stay-level aggregates)
    # ------------------------------------------------------------------
    logging.info("Model 3: action patterns (stay-level aggregates)")
    stay3, feats3 = build_action_stay_level(df, ACTION_COLS)
    logging.info("  %d stays, %d features", len(stay3), len(feats3))
    imp3, auc3, _ = run_lgb_model(stay3, feats3, C_READMIT_30D, 'actions')
    imp3.to_csv(os.path.join(args.out_dir, 'model3_importance.csv'), index=False)
    results['actions'] = {'auc': auc3, 'n_features': len(feats3),
                           'top_features': imp3.head(args.top_n)['feature'].tolist()}

    # ------------------------------------------------------------------
    # Model 4: combined -- all three groups joined at stay level
    #
    # Join strategy:
    #   - stay1: last bloc per stay (time-varying)
    #   - stay2: first bloc per stay (static)
    #   - stay3: stay-level action aggregates
    # All joined on icustayid; readmit_30d taken from stay1.
    # Features are renamed/suffixed where necessary to avoid collisions.
    # ------------------------------------------------------------------
    logging.info("Model 4: combined (time-varying + static + action patterns)")

    # Drop readmit_30d from stay2 and stay3 before merging to avoid duplicates
    stay2_no_label = stay2.drop(columns=[C_READMIT_30D], errors='ignore')
    stay3_no_label = stay3.drop(columns=[C_READMIT_30D], errors='ignore')

    stay4 = (stay1
             .merge(stay2_no_label, on=C_ICUSTAYID, how='inner')
             .merge(stay3_no_label, on=C_ICUSTAYID, how='inner'))

    # Feature columns: everything except the identifier and label
    feats4 = [c for c in stay4.columns if c not in (C_ICUSTAYID, C_READMIT_30D)]

    logging.info("  %d stays, %d features (time-varying=%d + static=%d + actions=%d)",
                 len(stay4), len(feats4), len(feats1), len(feats2), len(feats3))

    imp4, auc4, _ = run_lgb_model(stay4, feats4, C_READMIT_30D, 'combined')
    imp4.to_csv(os.path.join(args.out_dir, 'model4_importance.csv'), index=False)
    results['combined'] = {'auc': auc4, 'n_features': len(feats4),
                           'top_features': imp4.head(args.top_n)['feature'].tolist()}

    # ------------------------------------------------------------------
    # Summary outputs
    # ------------------------------------------------------------------
    # Stack all four importance tables into one file with a 'model' column
    # for easy filtering/plotting.
    summary = pd.concat([imp1, imp2, imp3, imp4], ignore_index=True)
    summary.to_csv(os.path.join(args.out_dir, 'all_models_summary.csv'), index=False)

    out_json = os.path.join(args.out_dir, 'variable_selection.json')
    with open(out_json, 'w') as f:
        json.dump(results, f, indent=2)

    logging.info("=== RESULTS SUMMARY ===")
    for grp, res in results.items():
        logging.info("  %-15s AUC=%.4f  top5=%s",
                     grp, res['auc'], res['top_features'][:5])
    logging.info("Outputs written to %s", args.out_dir)
    logging.info("Step 09b.1 complete.")
