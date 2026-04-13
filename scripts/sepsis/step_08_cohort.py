"""
Step 08 — Build final sepsis cohort.

Faithful adaptation of ai_clinician/preprocessing/07_build_sepsis_cohort.py.
Changes: import paths only (ai_clinician.* -> careai.sepsis.*).

Filters the state/action dataframe to a Sepsis-3 cohort:
  - Remove patients with extreme UO, bilirubin, or fluid intake (outliers)
  - Remove patients where treatment was stopped before death (suggests withdrawal)
  - Remove patients who died in ICU during the data collection window
  - Keep only patients with max SOFA >= 2

Outputs:
  data/processed/sepsis/sepsis_cohort.csv   — summary per ICU stay
  data/processed/sepsis/MKdataset.csv       — full filtered dataset (sepsisRL input)
    Column renames for sepsisRL compat: output_step -> output_4hourly,
                                        input_step  -> input_4hourly_tev

Usage:
    source ../.venv/Scripts/activate
    python scripts/sepsis/step_08_cohort.py \\
        data/interim/sepsis/intermediates/states_actions_final.csv \\
        data/interim/sepsis/intermediates/patient_states/qstime.csv \\
        data/processed/sepsis/ \\
        2>&1 | tee logs/step_08_cohort.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis.columns import (
    C_ICUSTAYID, C_BLOC, C_MORTA_90, C_MAX_DOSE_VASO, C_SOFA, C_SIRS,
    C_OUTPUT_STEP, C_TOTAL_BILI, C_INPUT_STEP,
    C_DIED_WITHIN_48H_OF_OUT_TIME, C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH,
    C_LAST_VASO, C_LAST_SOFA, C_NUM_BLOCS,
    C_MAX_SOFA, C_MAX_SIRS, C_ONSET_TIME,
)
from careai.sepsis.utils import load_csv

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def outlier_stay_ids(df):
    outliers = set()
    outliers |= set(df[df[C_OUTPUT_STEP] > 12000][C_ICUSTAYID].unique())
    outliers |= set(df[df[C_TOTAL_BILI]  > 10000][C_ICUSTAYID].unique())
    outliers |= set(df[df[C_INPUT_STEP]  > 10000][C_ICUSTAYID].unique())
    return outliers


def treatment_stopped_stay_ids(df):
    a = df[[C_BLOC, C_ICUSTAYID, C_MORTA_90, C_MAX_DOSE_VASO, C_SOFA]]
    grouped = a.groupby(C_ICUSTAYID)
    d = pd.merge(
        grouped.agg('max'),
        grouped.size().rename(C_NUM_BLOCS),
        how='left', left_index=True, right_index=True,
    ).drop(C_BLOC, axis=1)
    last_bloc = (a.sort_values(C_BLOC, ascending=False)
                  .drop_duplicates(C_ICUSTAYID)
                  .rename({C_MAX_DOSE_VASO: C_LAST_VASO, C_SOFA: C_LAST_SOFA}, axis=1)
                  .drop(C_MORTA_90, axis=1))
    d = pd.merge(d, last_bloc, how='left', left_index=True,
                 right_on=C_ICUSTAYID).set_index(C_ICUSTAYID, drop=True)
    return d[
        (d[C_MORTA_90] == 1) &
        (pd.isna(d[C_LAST_VASO]) | (d[C_LAST_VASO] < 0.01)) &
        (d[C_MAX_DOSE_VASO] > 0.3) &
        (d[C_LAST_SOFA] >= d[C_SOFA] / 2) &
        (d[C_NUM_BLOCS] < 20)
    ].index


def died_in_icu_stay_ids(df):
    return df[
        (df[C_DIED_WITHIN_48H_OF_OUT_TIME] == 1) &
        (df[C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH] < 24)
    ][C_ICUSTAYID].unique()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter states to Sepsis-3 cohort.')
    parser.add_argument('input',   type=str, help='states_actions_final.csv')
    parser.add_argument('qstime',  type=str, help='qstime.csv')
    parser.add_argument('output',  type=str, help='Output directory')
    parser.add_argument('--no-outlier-exclusion', dest='outlier_exclusion',
                        default=True, action='store_false')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_08_cohort.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 08 started. input=%s", args.input)
    df = load_csv(args.input)
    qstime = load_csv(args.qstime).set_index(C_ICUSTAYID, drop=True)
    logging.info("Loaded %d rows, %d ICU stays", len(df), len(set(df[C_ICUSTAYID])))

    logging.info("Before filtering: %d ICU stays", len(set(df[C_ICUSTAYID])))

    if args.outlier_exclusion:
        outliers = outlier_stay_ids(df)
        logging.info("Removing %d outlier stays (extreme UO/bili/intake)", len(outliers))
        df = df[~df[C_ICUSTAYID].isin(outliers)]

    stopped_treatment = treatment_stopped_stay_ids(df)
    logging.info("Removing %d stays where treatment was stopped before death", len(stopped_treatment))
    df = df[~df[C_ICUSTAYID].isin(stopped_treatment)]

    died_in_icu = died_in_icu_stay_ids(df)
    logging.info("Removing %d stays where patient died in ICU during data collection", len(died_in_icu))
    df = df[~df[C_ICUSTAYID].isin(died_in_icu)]

    logging.info("After filtering: %d ICU stays", len(set(df[C_ICUSTAYID])))

    sepsis = df.groupby(C_ICUSTAYID).agg({
        C_MORTA_90: 'first',
        C_SOFA:     'max',
        C_SIRS:     'max',
    }).rename({C_SOFA: C_MAX_SOFA, C_SIRS: C_MAX_SIRS}, axis=1)
    sepsis = sepsis[sepsis[C_MAX_SOFA] >= 2]
    logging.info("%d patients with max SOFA >= 2 (Sepsis-3 cohort)", len(sepsis))

    sepsis = pd.merge(sepsis, qstime[C_ONSET_TIME], how='left', left_index=True, right_index=True)
    cohort_path = os.path.join(out_dir, "sepsis_cohort.csv")
    sepsis.to_csv(cohort_path)
    logging.info("Cohort summary written: %s", cohort_path)

    # Filter full df to cohort stays -> MKdataset.csv (sepsisRL input)
    cohort_ids = set(sepsis.index)
    df_cohort  = df[df[C_ICUSTAYID].isin(cohort_ids)].copy()
    logging.info("MKdataset: %d rows from %d stays", len(df_cohort), len(cohort_ids))

    # Two column renames for sepsisRL compatibility
    df_cohort = df_cohort.rename(columns={
        'output_step': 'output_4hourly',
        'input_step':  'input_4hourly_tev',
    })

    out_path = os.path.join(out_dir, "MKdataset.csv")
    df_cohort.to_csv(out_path, index=False, float_format='%g')
    logging.info("MKdataset.csv written: %s (%d rows)", out_path, len(df_cohort))
    logging.info("Step 08 complete.")
