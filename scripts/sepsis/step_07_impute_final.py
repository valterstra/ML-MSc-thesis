"""
Step 07 — Final imputation of states+actions (interpolation + KNN + computed features).

Faithful adaptation of ai_clinician/preprocessing/06_impute_states_actions.py.
Changes: import paths only (ai_clinician.* -> careai.sepsis.*).
Provenance tracking removed (optional in original; omitted here for simplicity).

Tasks:
  - Correct gender, clamp age > 150 -> 91, fill NaN mechvent with 0, fix elixhauser NaN
  - Fill NaN vasopressor doses with 0
  - Linear interpolation on columns with < 5% missingness
  - KNN imputation on all chart + lab columns
  - Compute PaO2/FiO2, Shock Index, SOFA, SIRS

Note: KNN is the most time-consuming step (30-60 min on full dataset).
No checkpointing needed here — KNN operates on the full matrix at once.

Inputs:
  states_actions.csv         (from step 06)
Outputs:
  states_actions_final.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/sepsis/step_07_impute_final.py \\
        data/interim/sepsis/intermediates/states_actions.csv \\
        data/interim/sepsis/intermediates/states_actions_final.csv \\
        2>&1 | tee logs/step_07_impute_final.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis.columns import (
    C_ICUSTAYID, C_TIMESTEP, C_BLOC,
    C_GENDER, C_AGE, C_ELIXHAUSER, C_MECHVENT,
    C_MEDIAN_DOSE_VASO, C_MAX_DOSE_VASO,
    C_PAO2_FIO2, C_SHOCK_INDEX, C_SOFA, C_SIRS,
    C_RE_ADMISSION, C_DIED_WITHIN_48H_OF_OUT_TIME, C_DIED_IN_HOSP,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES,
)
from careai.sepsis.utils import load_csv
from careai.sepsis.imputation import fixgaps, knn_impute
from careai.sepsis.derived_features import (
    compute_pao2_fio2, compute_shock_index, compute_sofa, compute_sirs,
)

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def correct_features(df):
    df[C_GENDER] = df[C_GENDER] - 1
    ii = df[C_AGE] > 150
    logging.info("  Clamping %d ages > 150 -> 91", ii.sum())
    df.loc[ii, C_AGE] = 91
    n_nan_mv = pd.isna(df[C_MECHVENT]).sum()
    logging.info("  Filling %d NaN mechvent with 0", n_nan_mv)
    df.loc[pd.isna(df[C_MECHVENT]), C_MECHVENT] = 0
    df.loc[df[C_MECHVENT] > 0, C_MECHVENT] = 1
    n_nan_elix = pd.isna(df[C_ELIXHAUSER]).sum()
    logging.info("  Filling %d NaN elixhauser with median", n_nan_elix)
    df.loc[pd.isna(df[C_ELIXHAUSER]), C_ELIXHAUSER] = np.nanmedian(df[C_ELIXHAUSER])
    df.loc[pd.isna(df[C_MEDIAN_DOSE_VASO]), C_MEDIAN_DOSE_VASO] = 0
    df.loc[pd.isna(df[C_MAX_DOSE_VASO]),    C_MAX_DOSE_VASO]    = 0

    # Fix boolean/string columns that need to be numeric for downstream comparisons
    for col in [C_RE_ADMISSION, C_DIED_WITHIN_48H_OF_OUT_TIME, C_DIED_IN_HOSP]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0).astype(int)
            logging.info("  Converted %s to int", col)
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Final imputation: linear interpolation + KNN + computed features.'))
    parser.add_argument('input',  type=str, help='Path to states_actions.csv')
    parser.add_argument('output', type=str, help='Path for output CSV')
    parser.add_argument('--resolution', dest='resolution', type=float, default=4.0)
    parser.add_argument('--no-correct-features', dest='correct_features', default=True,
                        action='store_false')
    parser.add_argument('--no-interpolation', dest='interpolation', default=True,
                        action='store_false')
    parser.add_argument('--no-knn', dest='knn', default=True, action='store_false')
    parser.add_argument('--no-computed-features', dest='computed_features', default=True,
                        action='store_false')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_07_impute_final.log')
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

    if args.correct_features:
        logging.info("Correcting features")
        df = correct_features(df)

    if args.interpolation:
        miss = pd.isna(df).sum() / len(df)
        impute_columns = (miss > 0) & (miss < 0.05)
        cols_to_interp = df.loc[:, impute_columns].columns.tolist()
        logging.info("Linear interpolation on %d columns (<5%% missing)", len(cols_to_interp))
        for col in cols_to_interp:
            logging.info("  Interpolating %s", col)
            df[col] = fixgaps(df[col])

    if args.knn:
        knn_cols = CHART_FIELD_NAMES + LAB_FIELD_NAMES
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
