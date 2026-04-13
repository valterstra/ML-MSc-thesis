"""
Step 03 -- Define ICU cohort.

Replaces step_03_sepsis_onset.py in the sepsis pipeline.
Instead of detecting sepsis onset, this step applies inclusion/exclusion
criteria to produce the general ICU cohort.

Inclusion / exclusion criteria (confirmed with user):
  INCLUDE: All ICU stays with LOS >= 24 hours
  EXCLUDE: Hospital death (covers ICU death + post-ICU in-hospital death)
  EXCLUDE: Died within 30 days of hospital discharge (competing risk)
  EXCLUDE: Obstetric patients (admission_location == 'LABOR & DELIVERY')
  EXCLUDE: Neonatal patients (admission_type == 'NEWBORN')
  NO age minimum (user confirmed: include all ages)

Input:   data/interim/icu_readmit/intermediates/demog.csv
Output:  data/interim/icu_readmit/intermediates/icu_cohort.csv
         data/interim/icu_readmit/intermediates/cohort_exclusion_counts.csv

Key difference from sepsis: anchors observation window to ICU admission
(intime), not a derived onset time.

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_03_cohort_filter.py 2>&1 | tee logs/step_03_icu_readmit.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.utils import load_csv
from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_HADM_ID, C_SUBJECT_ID,
    C_LOS, C_INTIME, C_OUTTIME, C_DISCHTIME,
    C_DOD, C_MORTA_HOSP, C_ADM_ORDER, C_RE_ADMISSION,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_READMIT_30D,
)

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cohort parameters
MIN_ICU_LOS_DAYS = 1.0          # 24 hours
OBS_ADMISSION_LOCATIONS = {'LABOR & DELIVERY'}
NEONATAL_ADMISSION_TYPES = {'NEWBORN'}
MAX_DAYS_POST_DISCHARGE_DEATH = 30


def main():
    parser = argparse.ArgumentParser(
        description="Apply ICU cohort inclusion/exclusion criteria."
    )
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Data parent directory (default: data/interim/icu_readmit/)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Output directory (default: data/interim/icu_readmit/intermediates/)')
    parser.add_argument('--log', dest='log_file', type=str, default=None,
                        help='Log file (default: logs/step_03_icu_readmit.log)')
    args = parser.parse_args()

    data_dir = args.data_dir or os.path.join(REPO_DIR, 'data', 'interim', 'icu_readmit')
    out_dir  = args.output_dir or os.path.join(data_dir, 'intermediates')
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_03_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.info("Step 03 started.")

    # ---- Load demog -------------------------------------------------------
    demog_path = os.path.join(out_dir, "demog.csv")
    demog = load_csv(demog_path, null_icustayid=True)
    n_total = len(demog)
    logging.info("Loaded demog.csv: %d ICU stays", n_total)

    exclusion_counts = []

    def log_exclusion(label, mask):
        n = mask.sum()
        exclusion_counts.append({"criterion": label, "excluded": int(n)})
        logging.info("  Exclude %-45s: %d stays", label, n)
        return mask

    # ---- Exclusion criteria -----------------------------------------------
    # 1. Short LOS (< 24h) — los is in days in icustays
    mask_short_los = log_exclusion(
        "LOS < 24h",
        demog[C_LOS] < MIN_ICU_LOS_DAYS
    )

    # 2. Hospital death (covers ICU death + post-ICU-in-hospital death)
    mask_hosp_death = log_exclusion(
        "Hospital death (morta_hosp == 1)",
        demog[C_MORTA_HOSP] == 1
    )

    # 3. Post-discharge death within 30 days (competing risk)
    #    dod is a date string; dischtime is epoch integer (seconds)
    #    Convert dischtime to pandas datetime for comparison
    if C_DOD in demog.columns and demog[C_DOD].notna().any():
        dod_dt = pd.to_datetime(demog[C_DOD], errors='coerce')
        disch_dt = pd.to_datetime(demog[C_DISCHTIME], unit='s', errors='coerce')
        days_to_death = (dod_dt - disch_dt).dt.total_seconds() / 86400.0
        mask_postdisch_death = log_exclusion(
            f"Post-discharge death within {MAX_DAYS_POST_DISCHARGE_DEATH}d",
            (days_to_death >= 0) & (days_to_death <= MAX_DAYS_POST_DISCHARGE_DEATH)
        )
    else:
        logging.warning("  dod column missing or all null — skipping post-discharge death filter")
        mask_postdisch_death = pd.Series(False, index=demog.index)

    # 4. Obstetric stays
    if C_ADMISSION_LOC in demog.columns:
        mask_obs = log_exclusion(
            "Obstetric (admission_location == LABOR & DELIVERY)",
            demog[C_ADMISSION_LOC].isin(OBS_ADMISSION_LOCATIONS)
        )
    else:
        mask_obs = pd.Series(False, index=demog.index)

    # 5. Neonatal stays
    if C_ADMISSION_TYPE in demog.columns:
        mask_neo = log_exclusion(
            "Neonatal (admission_type == NEWBORN)",
            demog[C_ADMISSION_TYPE].isin(NEONATAL_ADMISSION_TYPES)
        )
    else:
        mask_neo = pd.Series(False, index=demog.index)

    # ---- Apply combined exclusion mask ------------------------------------
    exclude = (
        mask_short_los | mask_hosp_death | mask_postdisch_death | mask_obs | mask_neo
    )
    cohort = demog[~exclude].copy()
    n_cohort = len(cohort)
    logging.info(
        "Cohort after exclusions: %d / %d stays (%.1f%%)",
        n_cohort, n_total, 100.0 * n_cohort / n_total
    )

    # ---- Derived flag: re_admission (adm_order > 1) ----------------------
    if C_ADM_ORDER in cohort.columns:
        cohort[C_RE_ADMISSION] = (cohort[C_ADM_ORDER] > 1).astype(int)
        logging.info(
            "  Re-admissions: %d (%.1f%%)",
            cohort[C_RE_ADMISSION].sum(),
            100.0 * cohort[C_RE_ADMISSION].mean()
        )

    # ---- Outcome summary --------------------------------------------------
    if C_READMIT_30D in cohort.columns:
        n_readmit = cohort[C_READMIT_30D].sum()
        logging.info(
            "  30-day readmission: %d / %d (%.1f%%)",
            int(n_readmit), n_cohort, 100.0 * n_readmit / n_cohort
        )

    # ---- Save outputs -----------------------------------------------------
    cohort_path = os.path.join(out_dir, "icu_cohort.csv")
    cohort.to_csv(cohort_path, index=False)
    logging.info("icu_cohort.csv written: %d rows -> %s", n_cohort, cohort_path)

    excl_df = pd.DataFrame(exclusion_counts)
    excl_df.to_csv(os.path.join(out_dir, "cohort_exclusion_counts.csv"), index=False)
    logging.info("cohort_exclusion_counts.csv written")

    logging.info("Step 03 complete.")


if __name__ == "__main__":
    main()
