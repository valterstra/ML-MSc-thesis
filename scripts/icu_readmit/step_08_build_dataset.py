"""
Step 08 -- Build final ICU readmission dataset (ICUdataset.csv).

Adapted from scripts/sepsis/step_08_cohort.py.

Key differences from sepsis step_08:
  - No SOFA >= 2 filter (not a sepsis pipeline)
  - No "treatment stopped before death" filter (mortality not our outcome)
  - ICU deaths already excluded in step_03 (hospital death exclusion)
  - Adds vasopressor_dose (0-4) and ivfluid_dose (0-4) quartile discretization
    from max_dose_vaso and input_step respectively (per cohort quartiles)
  - Outcome: readmit_30d (already in df, broadcast from step_06)
  - Output: data/processed/icu_readmit/ICUdataset.csv

Column renames (for RL compatibility, same as sepsis):
  output_step -> output_4hourly
  input_step  -> input_4hourly_tev

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_08_build_dataset.py \\
        data/interim/icu_readmit/intermediates/states_actions_final.csv \\
        data/processed/icu_readmit/ \\
        2>&1 | tee logs/step_08_icu_readmit.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_BLOC, C_MORTA_90, C_MAX_DOSE_VASO,
    C_SOFA, C_SIRS, C_NUM_BLOCS,
    C_OUTPUT_STEP, C_TOTAL_BILI, C_INPUT_STEP,
    C_VASOPRESSOR_DOSE, C_IVFLUID_DOSE,
    C_READMIT_30D,
)
from careai.icu_readmit.utils import load_csv

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Outlier thresholds (same as sepsis pipeline)
UO_STEP_MAX    = 12000   # ml per 4-hour bloc
BILI_MAX       = 10000   # mg/dL total bilirubin — extreme outlier
INPUT_STEP_MAX = 10000   # ml per 4-hour bloc


def outlier_stay_ids(df):
    """Remove stays with physiologically impossible I/O or bilirubin values."""
    outliers = set()
    if C_OUTPUT_STEP in df.columns:
        outliers |= set(df[df[C_OUTPUT_STEP] > UO_STEP_MAX][C_ICUSTAYID].unique())
    if C_TOTAL_BILI in df.columns:
        outliers |= set(df[df[C_TOTAL_BILI]  > BILI_MAX][C_ICUSTAYID].unique())
    if C_INPUT_STEP in df.columns:
        outliers |= set(df[df[C_INPUT_STEP]  > INPUT_STEP_MAX][C_ICUSTAYID].unique())
    return outliers


def discretize_doses(df):
    """
    Discretize vasopressor and IV fluid doses into 5 bins (0-4).
    Bin 0 = none; bins 1-4 = quartiles of non-zero doses.
    Uses population quartiles computed from the full cohort.

    This mirrors the AI-Clinician discretization approach used for the
    sepsis pipeline (applied at RL preprocessing stage).
    """
    # --- Vasopressor dose ---
    vaso_col = C_MAX_DOSE_VASO
    if vaso_col in df.columns:
        vaso_nonzero = df.loc[df[vaso_col] > 0, vaso_col]
        if len(vaso_nonzero) > 0:
            q1, q2, q3 = vaso_nonzero.quantile([0.25, 0.50, 0.75]).values
            df[C_VASOPRESSOR_DOSE] = 0
            df.loc[df[vaso_col] > 0,  C_VASOPRESSOR_DOSE] = 1
            df.loc[df[vaso_col] >= q1, C_VASOPRESSOR_DOSE] = 2
            df.loc[df[vaso_col] >= q2, C_VASOPRESSOR_DOSE] = 3
            df.loc[df[vaso_col] >= q3, C_VASOPRESSOR_DOSE] = 4
            logging.info("  Vasopressor quartiles: Q1=%.4f Q2=%.4f Q3=%.4f", q1, q2, q3)
        else:
            df[C_VASOPRESSOR_DOSE] = 0
        logging.info("  vasopressor_dose distribution:\n%s", df[C_VASOPRESSOR_DOSE].value_counts().to_string())
    else:
        logging.warning("  %s not found — skipping vasopressor discretization", vaso_col)

    # --- IV fluid dose ---
    fluid_col = C_INPUT_STEP
    if fluid_col in df.columns:
        fluid_nonzero = df.loc[df[fluid_col] > 0, fluid_col]
        if len(fluid_nonzero) > 0:
            q1, q2, q3 = fluid_nonzero.quantile([0.25, 0.50, 0.75]).values
            df[C_IVFLUID_DOSE] = 0
            df.loc[df[fluid_col] > 0,  C_IVFLUID_DOSE] = 1
            df.loc[df[fluid_col] >= q1, C_IVFLUID_DOSE] = 2
            df.loc[df[fluid_col] >= q2, C_IVFLUID_DOSE] = 3
            df.loc[df[fluid_col] >= q3, C_IVFLUID_DOSE] = 4
            logging.info("  IV fluid quartiles: Q1=%.1f Q2=%.1f Q3=%.1f ml", q1, q2, q3)
        else:
            df[C_IVFLUID_DOSE] = 0
        logging.info("  ivfluid_dose distribution:\n%s", df[C_IVFLUID_DOSE].value_counts().to_string())
    else:
        logging.warning("  %s not found — skipping IV fluid discretization", fluid_col)

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build final ICU readmission dataset.')
    parser.add_argument('input',  type=str, help='states_actions_final.csv')
    parser.add_argument('output', type=str, help='Output directory (e.g. data/processed/icu_readmit/)')
    parser.add_argument('--no-outlier-exclusion', dest='outlier_exclusion',
                        default=True, action='store_false')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()
    out_dir = args.output
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_08_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 08 started. input=%s", args.input)
    df = load_csv(args.input)
    logging.info("Loaded %d rows, %d ICU stays", len(df), df[C_ICUSTAYID].nunique())

    # ---- Fix discharge_disposition (naming mismatch from step_06) ----------
    # demog.csv stores this as 'discharge_location'; step_06 looked for
    # 'discharge_disposition' and produced all-NaN. Merge the real values here.
    if df['discharge_disposition'].isna().all():
        demog_path = os.path.join(REPO_DIR, 'data', 'interim', 'icu_readmit', 'intermediates', 'demog.csv')
        demog = pd.read_csv(demog_path, usecols=[C_ICUSTAYID, 'discharge_location'])
        df = df.drop(columns=['discharge_disposition'])
        df = df.merge(demog.rename(columns={'discharge_location': 'discharge_disposition'}),
                      on=C_ICUSTAYID, how='left')
        n_filled = df['discharge_disposition'].notna().sum()
        logging.info("discharge_disposition filled from demog.csv: %d / %d rows non-null",
                     n_filled, len(df))

    # ---- Outlier exclusion ------------------------------------------------
    if args.outlier_exclusion:
        outliers = outlier_stay_ids(df)
        logging.info("Removing %d outlier stays (extreme UO/bili/intake)", len(outliers))
        df = df[~df[C_ICUSTAYID].isin(outliers)]

    # ---- NOTE: No SOFA >= 2 filter (general ICU, not sepsis) --------------
    # ---- NOTE: No treatment-stopped filter (mortality not our outcome) -----

    logging.info("After filtering: %d ICU stays (%d rows)", df[C_ICUSTAYID].nunique(), len(df))

    # ---- Outcome summary --------------------------------------------------
    if C_READMIT_30D in df.columns:
        n_readmit = df.drop_duplicates(C_ICUSTAYID)[C_READMIT_30D].sum()
        n_total   = df[C_ICUSTAYID].nunique()
        logging.info("30-day readmission: %d / %d stays (%.1f%%)",
                     int(n_readmit), n_total, 100.0 * n_readmit / n_total)

    # ---- Discretize dose-level actions ------------------------------------
    logging.info("Discretizing vasopressor and IV fluid doses to 0-4 bins")
    df = discretize_doses(df)

    # ---- Per-stay summary -------------------------------------------------
    agg_spec = {C_READMIT_30D: 'first', C_SOFA: 'max', C_SIRS: 'max', C_BLOC: 'max'}
    if C_MORTA_90 in df.columns:
        agg_spec[C_MORTA_90] = 'first'
    agg = df.groupby(C_ICUSTAYID).agg(agg_spec).rename(
        {C_BLOC: C_NUM_BLOCS, C_SOFA: 'max_SOFA', C_SIRS: 'max_SIRS'}, axis=1)

    cohort_path = os.path.join(out_dir, "icu_cohort_summary.csv")
    agg.to_csv(cohort_path)
    logging.info("Cohort summary written: %s (%d stays)", cohort_path, len(agg))

    # ---- Column renames (RL compatibility, same as sepsis) ----------------
    df = df.rename(columns={
        'output_step':  'output_4hourly',
        'input_step':   'input_4hourly_tev',
    })

    # ---- Save final dataset -----------------------------------------------
    out_path = os.path.join(out_dir, "ICUdataset.csv")
    df.to_csv(out_path, index=False, float_format='%g')
    logging.info("ICUdataset.csv written: %s (%d rows, %d stays)",
                 out_path, len(df), df[C_ICUSTAYID].nunique())
    logging.info("Step 08 complete.")
