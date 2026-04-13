"""
Step 07 -- Final imputation (linear interpolation + KNN + computed features).

Adapted from scripts/sepsis/step_07_impute_final.py.

Key differences from sepsis step_07:
  - Replaces elixhauser NaN fill with charlson_score NaN fill
  - Fills NaN binary drug flags (9 action columns) with 0
  - Same generic imputation and computed features (SOFA, SIRS, P/F, shock index)
  - KNN operates on CHART_FIELD_NAMES + LAB_FIELD_NAMES (extended list for icu_readmit)

Note: KNN is the most time-consuming step (30-60 min on full dataset).
No checkpointing needed — KNN operates on the full matrix at once.

Inputs:  data/interim/icu_readmit/intermediates/states_actions.csv
Outputs: data/interim/icu_readmit/intermediates/states_actions_final.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_07_impute_final.py \\
        data/interim/icu_readmit/intermediates/states_actions.csv \\
        data/interim/icu_readmit/intermediates/states_actions_final.csv \\
        2>&1 | tee logs/step_07_icu_readmit.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_TIMESTEP, C_BLOC,
    C_GENDER, C_AGE, C_CHARLSON, C_MECHVENT, C_CAM_ICU,
    C_MEDIAN_DOSE_VASO, C_MAX_DOSE_VASO,
    C_PAO2_FIO2, C_SHOCK_INDEX, C_SOFA, C_SIRS,
    C_RE_ADMISSION, C_DIED_WITHIN_48H_OF_OUT_TIME, C_DIED_IN_HOSP,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES, BINARY_ACTION_COLS,
)
from careai.icu_readmit.utils import load_csv
# Generic imputation and derived feature functions reused from sepsis
from careai.sepsis.imputation import fixgaps, knn_impute
from careai.sepsis.derived_features import (
    compute_pao2_fio2, compute_shock_index, compute_sofa, compute_sirs,
)

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def correct_features(df):
    """
    Clean up and standardise feature types.

    Differences vs sepsis:
    - charlson_score NaN -> median (not elixhauser)
    - Binary drug flags NaN -> 0 (9 new action columns)
    """
    # Sepsis pipeline stored gender as 1/2 numeric; ICU readmit stores 'M'/'F' strings
    if df[C_GENDER].dtype == object:
        df[C_GENDER] = df[C_GENDER].map({'M': 1, 'F': 0}).fillna(0).astype(int)
    else:
        df[C_GENDER] = df[C_GENDER] - 1
    ii = df[C_AGE] > 150
    logging.info("  Clamping %d ages > 150 -> 91", ii.sum())
    df.loc[ii, C_AGE] = 91

    n_nan_mv = pd.isna(df[C_MECHVENT]).sum()
    logging.info("  Filling %d NaN mechvent with 0", n_nan_mv)
    df.loc[pd.isna(df[C_MECHVENT]), C_MECHVENT] = 0
    df.loc[df[C_MECHVENT] > 0, C_MECHVENT] = 1

    if C_CAM_ICU in df.columns:
        n_nan_cam = pd.isna(df[C_CAM_ICU]).sum()
        logging.info("  Filling %d NaN cam_icu with 0 (not assessed = no delirium flag)", n_nan_cam)
        df.loc[pd.isna(df[C_CAM_ICU]), C_CAM_ICU] = 0
        df[C_CAM_ICU] = df[C_CAM_ICU].astype(int)

    n_nan_charlson = pd.isna(df[C_CHARLSON]).sum()
    logging.info("  Filling %d NaN charlson_score with median", n_nan_charlson)
    df.loc[pd.isna(df[C_CHARLSON]), C_CHARLSON] = np.nanmedian(df[C_CHARLSON])

    df.loc[pd.isna(df[C_MEDIAN_DOSE_VASO]), C_MEDIAN_DOSE_VASO] = 0
    df.loc[pd.isna(df[C_MAX_DOSE_VASO]),    C_MAX_DOSE_VASO]    = 0

    # Binary drug flags: NaN -> 0 (drug not given in that bloc)
    for col in BINARY_ACTION_COLS:
        if col in df.columns:
            n_nan = pd.isna(df[col]).sum()
            if n_nan > 0:
                logging.info("  Filling %d NaN %s with 0", n_nan, col)
            df.loc[pd.isna(df[col]), col] = 0
            df[col] = df[col].astype(int)

    # Numeric conversion for boolean/flag columns
    for col in [C_RE_ADMISSION, C_DIED_WITHIN_48H_OF_OUT_TIME, C_DIED_IN_HOSP]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Final imputation: linear interpolation + KNN + computed features.'))
    parser.add_argument('input',  type=str, help='Path to states_actions.csv')
    parser.add_argument('output', type=str, help='Path for output CSV')
    parser.add_argument('--resolution', dest='resolution', type=float, default=4.0)
    parser.add_argument('--no-correct-features', dest='correct_features_flag',
                        default=True, action='store_false')
    parser.add_argument('--no-interpolation', dest='interpolation', default=True,
                        action='store_false')
    parser.add_argument('--no-knn', dest='knn', default=True, action='store_false')
    parser.add_argument('--no-computed-features', dest='computed_features',
                        default=True, action='store_false')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_07_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 07 started. input=%s", args.input)
    df = load_csv(args.input, null_icustayid=True)
    n_before = len(df)
    df = df.dropna(subset=[C_ICUSTAYID])
    if n_before != len(df):
        logging.warning("Dropped %d rows with NA icustayid (checkpoint artifact)", n_before - len(df))
    logging.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if args.correct_features_flag:
        logging.info("Correcting features")
        df = correct_features(df)

    if args.interpolation:
        miss = pd.isna(df).sum() / len(df)
        impute_columns = (miss > 0) & (miss < 0.05)
        cols_to_interp = [c for c in df.loc[:, impute_columns].columns
                          if pd.api.types.is_numeric_dtype(df[c])]
        logging.info("Linear interpolation on %d columns (<5%% missing)", len(cols_to_interp))
        for col in cols_to_interp:
            logging.info("  Interpolating %s", col)
            df[col] = fixgaps(df[col])

    if args.knn:
        knn_cols = [c for c in CHART_FIELD_NAMES + LAB_FIELD_NAMES if c in df.columns]
        logging.info("KNN imputation on %d columns (this is the slow step)", len(knn_cols))
        df[knn_cols] = knn_impute(df[knn_cols], na_threshold=0.9)
        logging.info("KNN imputation complete")

    if args.computed_features:
        logging.info("Computing P/F ratio")
        df[C_PAO2_FIO2] = compute_pao2_fio2(df)
        logging.info("Computing shock index")
        df[C_SHOCK_INDEX] = compute_shock_index(df)
        logging.info("Computing SOFA")
        df[C_SOFA] = compute_sofa(df, timestep_resolution=args.resolution)
        logging.info("Computing SIRS")
        df[C_SIRS] = compute_sirs(df)

    logging.info("Writing output -> %s", args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False, float_format='%g')
    logging.info("Step 07 complete. %d rows written.", len(df))
