"""
Step 05 -- Impute patient states (outlier removal + vitals + sample-and-hold).

Adapted from scripts/sepsis/step_05_impute_states.py.

Key differences from sepsis step_05:
  - GCS: compute GCS_Total = GCS_Eye + GCS_Verbal + GCS_Motor (NO RASS estimation).
    The GCS components are extracted directly in step_04 using MetaVision itemids
    (220739/223900/223901). RASS estimation was only needed in sepsis because
    CareVue GCS itemid 198 returned 100% NaN in MIMIC-IV.
  - Additional outlier bounds for: NIBP_Diastolic, Arterial_BP_Sys/Dia, RR_Spont/Total,
    TidalVolume_Observed, Pain_Level, Phosphate, Anion_Gap, Alk_Phos, LDH, Fibrinogen,
    and all 5 WBC differential percentages.
  - Imports from careai.icu_readmit.columns (extended CHART_FIELD_NAMES, LAB_FIELD_NAMES,
    SAH_FIELD_NAMES, SAH_HOLD_DURATION).
  - Generic functions (fill_outliers, fill_stepwise, sample_and_hold) are reused from
    careai.sepsis.imputation (no sepsis-specific logic in those functions).

Inputs:
  data/interim/icu_readmit/intermediates/patient_states/patient_states.csv
Outputs:
  data/interim/icu_readmit/intermediates/patient_states/patient_states_imputed.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_05_impute_states.py \\
        data/interim/icu_readmit/intermediates/patient_states/patient_states.csv \\
        data/interim/icu_readmit/intermediates/patient_states/patient_states_imputed.csv \\
        2>&1 | tee logs/step_05_icu_readmit.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_TIMESTEP, C_BLOC,
    C_TEMP_C, C_TEMP_F, C_FIO2_100, C_FIO2_1, C_O2FLOW, C_INTERFACE,
    C_SYSBP, C_MEANBP, C_DIABP, C_HR, C_RR, C_SPO2, C_WEIGHT, C_HEIGHT,
    C_NIBP_DIA, C_ART_SYS, C_ART_DIA,
    C_RR_SPONT, C_RR_TOTAL,
    C_PEEP, C_TIDALVOLUME, C_TIDAL_OBS, C_MINUTEVENTIL,
    C_CVP, C_PAPSYS, C_PAPMEAN, C_PAPDIA, C_CI, C_SVR,
    C_PAWMEAN, C_PAWPEAK, C_PAWPLATEAU, C_PAIN_LEVEL,
    C_GCS, C_GCS_EYE, C_GCS_VERBAL, C_GCS_MOTOR, C_RASS,
    C_POTASSIUM, C_SODIUM, C_CHLORIDE, C_GLUCOSE, C_BUN, C_CREATININE,
    C_MAGNESIUM, C_CALCIUM, C_IONISED_CA, C_CO2_MEQL,
    C_SGPT, C_SGOT, C_TOTAL_BILI, C_DIRECT_BILI,
    C_TOTAL_PROTEIN, C_ALBUMIN, C_TROPONIN, C_CRP,
    C_HB, C_HT, C_RBC_COUNT, C_WBC_COUNT, C_PLATELETS_COUNT,
    C_PTT, C_PT, C_ACT,
    C_INR, C_ARTERIAL_PH, C_PAO2, C_PACO2, C_ARTERIAL_BE, C_ARTERIAL_LACTATE,
    C_HCO3, C_ETCO2, C_SVO2,
    C_PHOSPHATE, C_ANION_GAP, C_ALK_PHOS, C_LDH, C_FIBRINOGEN,
    C_NEUTS_PCT, C_LYMPHS_PCT, C_MONOS_PCT, C_EOS_PCT, C_BASOS_PCT,
    SAH_FIELD_NAMES, SAH_HOLD_DURATION,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES,
)
from careai.icu_readmit.utils import load_csv
# Generic imputation functions reused from sepsis module (no sepsis-specific logic)
from careai.sepsis.imputation import fill_outliers, fill_stepwise, sample_and_hold

tqdm.pandas()

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def remove_outliers(df):
    # --- Sentinel filter (MIMIC-IV 999,999 placeholders) ---
    sentinel_cols = [c for c in CHART_FIELD_NAMES + LAB_FIELD_NAMES
                     if c in df.columns and c != C_INTERFACE]
    n_sentinel = 0
    for col in sentinel_cols:
        mask = pd.to_numeric(df[col], errors='coerce') > 10_000
        if mask.any():
            cnt = mask.sum()
            n_sentinel += cnt
            logging.info("  Sentinel filter: %s - %d values > 10,000 -> NaN", col, cnt)
            df.loc[mask, col] = pd.NA
    logging.info("  Sentinel filter total: %d values removed", n_sentinel)

    # --- Unit corrections ---
    wrong_unit_temps = (df[C_TEMP_C] > 45) & pd.isna(df[C_TEMP_F])
    df.loc[wrong_unit_temps, C_TEMP_F] = df.loc[wrong_unit_temps, C_TEMP_C]
    df.loc[wrong_unit_temps, C_TEMP_C] = pd.NA
    df.loc[df[C_FIO2_100] < 1, C_FIO2_100] = df.loc[df[C_FIO2_100] < 1, C_FIO2_100] * 100

    # --- Physiological bounds ---
    bounds = {
        # Vitals
        C_HEIGHT:       (50, 250),
        C_WEIGHT:       (0.5, 300),
        C_GCS_EYE:      (1, 4),      # GCS Eye: 1 (no response) to 4 (spontaneous)
        C_GCS_VERBAL:   (1, 5),      # GCS Verbal: 1 (none) to 5 (oriented)
        C_GCS_MOTOR:    (1, 6),      # GCS Motor: 1 (none) to 6 (obeys commands)
        C_GCS:          (3, 15),     # summed GCS
        C_RASS:         (-5, 4),
        C_HR:           (0, 250),
        C_SYSBP:        (0, 300),
        C_MEANBP:       (0, 200),
        C_DIABP:        (0, 200),
        C_NIBP_DIA:     (0, 200),    # non-invasive diastolic
        C_ART_SYS:      (0, 300),    # invasive arterial systolic
        C_ART_DIA:      (0, 200),    # invasive arterial diastolic
        C_RR:           (0, 80),
        C_RR_SPONT:     (0, 80),     # spontaneous respiratory rate
        C_RR_TOTAL:     (0, 80),     # total respiratory rate
        C_SPO2:         (0, 100),
        C_TEMP_C:       (25, 45),
        C_TEMP_F:       (77, 113),
        C_PAIN_LEVEL:   (0, 10),     # NRS 0-10 pain scale
        # Hemodynamics
        C_CVP:          (-20, 40),
        C_PAPSYS:       (5, 120),
        C_PAPMEAN:      (5, 80),
        C_PAPDIA:       (0, 60),
        C_CI:           (0.5, 10),
        C_SVR:          (100, 4000),
        # Respiratory
        C_FIO2_100:     (20, 100),
        C_FIO2_1:       (0.2, 1.0),
        C_O2FLOW:       (0, 70),
        C_PEEP:         (0, 40),
        C_TIDALVOLUME:  (0, 1800),
        C_TIDAL_OBS:    (0, 1800),   # observed tidal volume
        C_MINUTEVENTIL: (0, 50),
        C_PAWMEAN:      (0, 50),
        C_PAWPEAK:      (0, 80),
        C_PAWPLATEAU:   (0, 60),
        # Electrolytes / Chemistry
        C_POTASSIUM:    (1, 12),
        C_SODIUM:       (95, 178),
        C_CHLORIDE:     (70, 150),
        C_GLUCOSE:      (1, 1000),
        C_BUN:          (1, 300),
        C_CREATININE:   (0.1, 50),
        C_MAGNESIUM:    (0.5, 10),
        C_CALCIUM:      (4, 20),
        C_IONISED_CA:   (0.2, 5),
        C_CO2_MEQL:     (5, 60),
        C_PHOSPHATE:    (0.5, 15),   # mg/dL (extreme: rhabdo, renal failure)
        C_ANION_GAP:    (0, 40),     # mEq/L
        # Liver / Enzymes
        C_SGOT:         (0, 10000),
        C_SGPT:         (0, 10000),
        C_TOTAL_BILI:   (0, 100),
        C_DIRECT_BILI:  (0, 70),
        C_TOTAL_PROTEIN:(1, 15),
        C_ALBUMIN:      (0.5, 7),
        C_TROPONIN:     (0, 100),
        C_CRP:          (0, 500),
        C_ALK_PHOS:     (0, 5000),   # IU/L; severe cholestasis
        C_LDH:          (0, 10000),  # IU/L; massive hemolysis/rhabdo
        C_FIBRINOGEN:   (50, 1000),  # mg/dL; 0 is DIC terminal
        # Hematology
        C_HB:           (1, 25),
        C_HT:           (5, 70),
        C_RBC_COUNT:    (0.5, 10),
        C_WBC_COUNT:    (0, 500),
        C_PLATELETS_COUNT: (1, 2000),
        # WBC differential percentages (0-100%)
        C_NEUTS_PCT:    (0, 100),
        C_LYMPHS_PCT:   (0, 100),
        C_MONOS_PCT:    (0, 100),
        C_EOS_PCT:      (0, 100),
        C_BASOS_PCT:    (0, 100),
        # Coagulation
        C_PTT:          (10, 200),
        C_PT:           (7, 200),
        C_ACT:          (50, 600),
        C_INR:          (0.5, 25),
        # Blood gas
        C_ARTERIAL_PH:       (6.7, 7.8),
        C_PAO2:              (10, 700),
        C_PACO2:             (5, 200),
        C_ARTERIAL_BE:       (-50, 40),
        C_ARTERIAL_LACTATE:  (0.1, 30),
        C_HCO3:              (2, 55),
        C_ETCO2:             (0, 100),
        C_SVO2:              (0, 100),
    }
    # Only apply bounds for columns that exist in df
    bounds_present = {k: v for k, v in bounds.items() if k in df.columns}
    df = fill_outliers(df, bounds_present)
    return df


def compute_gcs_total(df):
    """
    Compute GCS_Total = GCS_Eye + GCS_Verbal + GCS_Motor.

    This replaces the sepsis pipeline's RASS-based GCS estimation.
    The three components come directly from MetaVision itemids
    (220739 / 223900 / 223901) with 99.4% coverage.
    """
    for comp in [C_GCS_EYE, C_GCS_VERBAL, C_GCS_MOTOR]:
        if comp not in df.columns:
            logging.warning("GCS component '%s' missing from patient_states — cannot compute GCS", comp)
            return df

    # Compute where all three components are present
    have_all = ~pd.isna(df[C_GCS_EYE]) & ~pd.isna(df[C_GCS_VERBAL]) & ~pd.isna(df[C_GCS_MOTOR])
    if C_GCS not in df.columns:
        df[C_GCS] = pd.NA
    df.loc[have_all, C_GCS] = (
        df.loc[have_all, C_GCS_EYE] +
        df.loc[have_all, C_GCS_VERBAL] +
        df.loc[have_all, C_GCS_MOTOR]
    )
    n_computed = have_all.sum()
    logging.info("  GCS computed from components: %d rows (%.1f%%)", n_computed,
                 100.0 * n_computed / max(1, len(df)))
    return df


def convert_fio2_units(df):
    missing_fio2_set = pd.isna(df[C_FIO2_1]) & ~pd.isna(df[C_FIO2_100])
    df.loc[missing_fio2_set, C_FIO2_1] = df.loc[missing_fio2_set, C_FIO2_100] / 100
    missing_fio2 = ~pd.isna(df[C_FIO2_1]) & pd.isna(df[C_FIO2_100])
    df.loc[missing_fio2, C_FIO2_100] = df.loc[missing_fio2, C_FIO2_1] * 100
    return df


def estimate_fio2(df):
    """Identical to sepsis pipeline FiO2 estimation."""
    # FiO2_1 (itemid 190) is CareVue-only and absent from MetaVision MIMIC-IV.
    # Add as all-NA so convert_fio2_units can derive it from FiO2_100 / 100.
    if C_FIO2_1 not in df.columns:
        df[C_FIO2_1] = pd.NA
        logging.info("  FiO2_1 absent from data — column created, will derive from FiO2_100")
    df = convert_fio2_units(df)
    sah_fio2 = {}
    for col in [C_INTERFACE, C_FIO2_100, C_O2FLOW]:
        logging.info("  SAH on %s for FiO2", col)
        sah_fio2[col] = sample_and_hold(df[C_ICUSTAYID], df[C_TIMESTEP], df[col],
                                         SAH_HOLD_DURATION[col])

    def apply(mask_expr, fill):
        mask = mask_expr
        logging.info("  FiO2 rule: %d rows", mask.sum())
        df.loc[mask, C_FIO2_100] = fill

    apply(pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) &
          ((sah_fio2[C_INTERFACE] == 0) | (sah_fio2[C_INTERFACE] == 2)),
          fill_stepwise(sah_fio2[C_O2FLOW].loc[
              pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) &
              ((sah_fio2[C_INTERFACE] == 0) | (sah_fio2[C_INTERFACE] == 2))],
              zip(*([15, 12, 10, 8, 6, 5, 4, 3, 2, 1], [70, 62, 55, 50, 44, 40, 36, 32, 28, 24]))))

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) &
            ((sah_fio2[C_INTERFACE] == 0) | (sah_fio2[C_INTERFACE] == 2)))
    df.loc[mask, C_FIO2_100] = 21

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) &
            (pd.isna(sah_fio2[C_INTERFACE]) | sah_fio2[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))))
    df.loc[mask, C_FIO2_100] = fill_stepwise(
        sah_fio2[C_O2FLOW].loc[mask],
        zip(*([15, 12, 10, 8, 6, 4], [75, 69, 66, 58, 40, 36])))

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) &
            (pd.isna(sah_fio2[C_INTERFACE]) | sah_fio2[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))))
    df.loc[mask, C_FIO2_100] = pd.NA

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) &
            (sah_fio2[C_INTERFACE] == 7))
    df.loc[mask, C_FIO2_100] = fill_stepwise(
        sah_fio2[C_O2FLOW].loc[mask],
        zip(*([9.99, 8, 6], [80, 70, 60])),
        zip(*([10, 15], [90, 100])))

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) &
            (sah_fio2[C_INTERFACE] == 7))
    df.loc[mask, C_FIO2_100] = pd.NA

    df = convert_fio2_units(df)
    return df


def estimate_vitals(df):
    """BP, Temp, Hb/Ht, bilirubin cross-imputation — identical to sepsis pipeline."""
    logging.info("  BP imputation")
    ii = ~pd.isna(df[C_SYSBP]) & ~pd.isna(df[C_MEANBP]) & pd.isna(df[C_DIABP])
    df.loc[ii, C_DIABP] = (3 * df.loc[ii, C_MEANBP] - df.loc[ii, C_SYSBP]) / 2
    ii = ~pd.isna(df[C_SYSBP]) & ~pd.isna(df[C_DIABP]) & pd.isna(df[C_MEANBP])
    df.loc[ii, C_MEANBP] = (df.loc[ii, C_SYSBP] + 2 * df.loc[ii, C_DIABP]) / 3
    ii = ~pd.isna(df[C_MEANBP]) & ~pd.isna(df[C_DIABP]) & pd.isna(df[C_SYSBP])
    df.loc[ii, C_SYSBP] = 3 * df.loc[ii, C_MEANBP] - 2 * df.loc[ii, C_DIABP]

    logging.info("  Temp imputation")
    ii = (df[C_TEMP_F] > 25) & (df[C_TEMP_F] < 45)
    df.loc[ii, C_TEMP_C] = df.loc[ii, C_TEMP_F]
    df.loc[ii, C_TEMP_F] = np.NaN
    ii = df[C_TEMP_C] > 70
    df.loc[ii, C_TEMP_F] = df.loc[ii, C_TEMP_C]
    df.loc[ii, C_TEMP_C] = np.NaN
    ii = ~pd.isna(df[C_TEMP_C]) & pd.isna(df[C_TEMP_F])
    df.loc[ii, C_TEMP_F] = df.loc[ii, C_TEMP_C] * 1.8 + 32
    ii = ~pd.isna(df[C_TEMP_F]) & pd.isna(df[C_TEMP_C])
    df.loc[ii, C_TEMP_C] = (df.loc[ii, C_TEMP_F] - 32) / 1.8

    logging.info("  Hb/Ht imputation")
    ii = ~pd.isna(df[C_HB]) & pd.isna(df[C_HT])
    df.loc[ii, C_HT] = (df.loc[ii, C_HB] * 2.862) + 1.216
    ii = ~pd.isna(df[C_HT]) & pd.isna(df[C_HB])
    df.loc[ii, C_HB] = (df.loc[ii, C_HT] - 1.216) / 2.862

    logging.info("  Bilirubin imputation")
    ii = ~pd.isna(df[C_TOTAL_BILI]) & pd.isna(df[C_DIRECT_BILI])
    df.loc[ii, C_DIRECT_BILI] = (df.loc[ii, C_TOTAL_BILI] * 0.6934) - 0.1752
    ii = ~pd.isna(df[C_DIRECT_BILI]) & pd.isna(df[C_TOTAL_BILI])
    df.loc[ii, C_TOTAL_BILI] = (df.loc[ii, C_DIRECT_BILI] + 0.1752) / 0.6934
    neg_dbili = df[C_DIRECT_BILI] < 0
    if neg_dbili.any():
        df.loc[neg_dbili, C_DIRECT_BILI] = 0

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Imputes outliers, computes GCS from components, estimates FiO2, '
        'estimates derived vitals, and applies sample-and-hold to ICU patient states.'))
    parser.add_argument('input',  type=str, help='Path to patient_states.csv')
    parser.add_argument('output', type=str, help='Path for output patient_states_imputed.csv')
    parser.add_argument('--no-outliers',    dest='outliers',        default=True, action='store_false')
    parser.add_argument('--no-gcs',         dest='gcs',             default=True, action='store_false')
    parser.add_argument('--no-fio2',        dest='fio2',            default=True, action='store_false')
    parser.add_argument('--no-vitals',      dest='vitals',          default=True, action='store_false')
    parser.add_argument('--no-sample-hold', dest='sample_and_hold', default=True, action='store_false')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_05_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.info("Step 05 started. input=%s output=%s", args.input, args.output)
    df = load_csv(args.input)
    logging.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    if args.outliers:
        logging.info("Removing outliers")
        df = remove_outliers(df)

    if args.gcs:
        logging.info("Computing GCS from components (Eye + Verbal + Motor)")
        df = compute_gcs_total(df)

    if args.fio2:
        logging.info("Estimating FiO2")
        df = estimate_fio2(df)

    if args.vitals:
        logging.info("Estimating vitals")
        df = estimate_vitals(df)

    if args.sample_and_hold:
        logging.info("Sample and hold (%d columns)", len(SAH_FIELD_NAMES))
        sah_series = {C_BLOC: df[C_BLOC], C_ICUSTAYID: df[C_ICUSTAYID], C_TIMESTEP: df[C_TIMESTEP]}
        # Include GCS in SAH (computed above; not in SAH_FIELD_NAMES as it's derived)
        if C_GCS in df.columns:
            sah_series[C_GCS] = sample_and_hold(
                df[C_ICUSTAYID], df[C_TIMESTEP], df[C_GCS], 6)  # 6h hold for GCS
        for col in SAH_FIELD_NAMES:
            if col not in df.columns:
                logging.warning("  SAH: column '%s' not in data, skipping", col)
                continue
            logging.info("  SAH on %s", col)
            sah_series[col] = sample_and_hold(
                df[C_ICUSTAYID], df[C_TIMESTEP], df[col], SAH_HOLD_DURATION[col])
            n_before = pd.isna(df[col]).sum()
            n_after  = pd.isna(sah_series[col]).sum()
            pct = (1 - n_after / max(1, n_before)) * 100
            logging.info("    Eliminated %.1f%% of NA values", pct)
        df = pd.DataFrame(sah_series)

    logging.info("Writing output -> %s", args.output)
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df.to_csv(args.output, index=False, float_format='%g')
    logging.info("Step 05 complete. %d rows written.", len(df))
