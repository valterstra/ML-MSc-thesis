"""
Step 05 -Impute patient states (outlier removal + FiO2 + vitals + sample-and-hold).

Faithful adaptation of ai_clinician/preprocessing/04_impute_states.py.
Changes: import paths only (ai_clinician.* -> careai.sepsis.*).
Provenance tracking removed (optional in original; omitted here for simplicity).

Tasks:
  - Remove physiological outliers (clamp to plausible ranges -> NaN)
  - Estimate GCS from RASS when GCS is missing
  - Estimate FiO2 from O2 flow / delivery device interface
  - Estimate BP, temperature, Hb/Ht, bilirubin cross-columns
  - Sample-and-hold forward-fill within each ICU stay

Inputs:
  data/interim/sepsis/intermediates/patient_states/patient_states.csv
Outputs:
  data/interim/sepsis/intermediates/patient_states/patient_states_imputed.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/sepsis/step_05_impute_states.py \\
        data/interim/sepsis/intermediates/patient_states/patient_states.csv \\
        data/interim/sepsis/intermediates/patient_states/patient_states_imputed.csv \\
        2>&1 | tee logs/step_05_impute_states.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis.columns import (
    C_ICUSTAYID, C_TIMESTEP, C_BLOC,
    C_TEMP_C, C_TEMP_F, C_FIO2_100, C_FIO2_1, C_O2FLOW, C_INTERFACE,
    C_SYSBP, C_MEANBP, C_DIABP, C_HR, C_RR, C_SPO2, C_WEIGHT, C_HEIGHT,
    C_PEEP, C_TIDALVOLUME, C_MINUTEVENTIL,
    C_CVP, C_PAPSYS, C_PAPMEAN, C_PAPDIA, C_CI, C_SVR,
    C_PAWMEAN, C_PAWPEAK, C_PAWPLATEAU,
    C_POTASSIUM, C_SODIUM, C_CHLORIDE, C_GLUCOSE, C_BUN, C_CREATININE,
    C_MAGNESIUM, C_CALCIUM, C_IONISED_CA, C_CO2_MEQL,
    C_SGPT, C_SGOT, C_TOTAL_BILI, C_DIRECT_BILI,
    C_TOTAL_PROTEIN, C_ALBUMIN, C_TROPONIN, C_CRP,
    C_HB, C_HT, C_RBC_COUNT, C_WBC_COUNT, C_PLATELETS_COUNT,
    C_PTT, C_PT, C_ACT,
    C_INR, C_ARTERIAL_PH, C_PAO2, C_PACO2, C_ARTERIAL_BE, C_ARTERIAL_LACTATE,
    C_HCO3, C_ETCO2, C_SVO2,
    C_GCS, C_RASS,
    SAH_FIELD_NAMES, SAH_HOLD_DURATION,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES,
)
from careai.sepsis.utils import load_csv
from careai.sepsis.imputation import fill_outliers, fill_stepwise, sample_and_hold

tqdm.pandas()

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def remove_outliers(df):
    # --- Blanket sentinel filter (MIMIC-IV 999,999 placeholders) ---
    # Any value > 10,000 in lab/clinical columns is a sentinel, not real data.
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

    # --- Unit corrections (before outlier removal) ---
    # Temp_C values > 45 with missing Temp_F: likely recorded in wrong unit
    wrong_unit_temps = (df[C_TEMP_C] > 45) & pd.isna(df[C_TEMP_F])
    df.loc[wrong_unit_temps, C_TEMP_F] = df.loc[wrong_unit_temps, C_TEMP_C]
    df.loc[wrong_unit_temps, C_TEMP_C] = pd.NA
    # FiO2 recorded as fraction instead of percentage
    df.loc[df[C_FIO2_100] < 1, C_FIO2_100] = df.loc[df[C_FIO2_100] < 1, C_FIO2_100] * 100

    # --- Complete physiological bounds for ALL chart + lab columns ---
    # Covers every column in CHART_FIELD_NAMES + LAB_FIELD_NAMES.
    # Bounds are conservative (err on keeping extreme-but-possible values).
    df = fill_outliers(df, {
        # -- Vitals --
        C_HEIGHT:           (50, 250),      # tallest human ~251 cm
        C_WEIGHT:           (0.5, 300),     # bariatric upper bound
        C_GCS:              (3, 15),        # GCS scale
        C_RASS:             (-5, 4),        # RASS scale
        C_HR:               (0, 250),       # asystole=0 charted; tachy max ~250
        C_SYSBP:            (0, 300),
        C_MEANBP:           (0, 200),
        C_DIABP:            (0, 200),
        C_RR:               (0, 80),
        C_SPO2:             (0, 100),       # percentage, cannot exceed 100
        C_TEMP_C:           (25, 45),       # hypothermia 25C to hyperthermia 45C
        C_TEMP_F:           (77, 113),      # = 25C to 45C in Fahrenheit
        # -- Hemodynamics --
        C_CVP:              (-20, 40),      # central venous pressure cmH2O
        C_PAPSYS:           (5, 120),       # pulmonary artery systolic
        C_PAPMEAN:          (5, 80),        # pulmonary artery mean
        C_PAPDIA:           (0, 60),        # pulmonary artery diastolic
        C_CI:               (0.5, 10),      # cardiac index L/min/m2
        C_SVR:              (100, 4000),    # systemic vascular resistance
        # -- Respiratory --
        C_FIO2_100:         (20, 100),      # percentage, 21% room air to 100%
        C_FIO2_1:           (0.2, 1.0),     # fraction
        C_O2FLOW:           (0, 70),        # L/min
        C_PEEP:             (0, 40),        # cmH2O
        C_TIDALVOLUME:      (0, 1800),      # mL
        C_MINUTEVENTIL:     (0, 50),        # L/min
        C_PAWMEAN:          (0, 50),        # mean airway pressure cmH2O
        C_PAWPEAK:          (0, 80),        # peak inspiratory pressure
        C_PAWPLATEAU:       (0, 60),        # plateau pressure
        # -- Electrolytes / Chemistry --
        C_POTASSIUM:        (1, 12),        # mEq/L
        C_SODIUM:           (95, 178),      # mEq/L
        C_CHLORIDE:         (70, 150),      # mEq/L
        C_GLUCOSE:          (1, 1000),      # mg/dL
        C_BUN:              (1, 300),       # mg/dL; 0 is lab error
        C_CREATININE:       (0.1, 50),      # mg/dL; >30 extreme rhabdo
        C_MAGNESIUM:        (0.5, 10),      # mg/dL
        C_CALCIUM:          (4, 20),        # mg/dL; <4 incompatible with life
        C_IONISED_CA:       (0.2, 5),       # mmol/L
        C_CO2_MEQL:         (5, 60),        # mEq/L
        # -- Liver / Enzymes --
        C_SGOT:             (0, 10000),     # massive hepatic necrosis possible
        C_SGPT:             (0, 10000),
        C_TOTAL_BILI:       (0, 100),       # mg/dL; fulminant liver failure
        C_DIRECT_BILI:      (0, 70),        # mg/dL; negative is impossible
        C_TOTAL_PROTEIN:    (1, 15),        # g/dL
        C_ALBUMIN:          (0.5, 7),       # g/dL
        C_TROPONIN:         (0, 100),       # ng/mL; assay-dependent
        C_CRP:              (0, 500),       # mg/L
        # -- Hematology --
        C_HB:               (1, 25),        # g/dL; polycythemia can reach ~22
        C_HT:               (5, 70),        # %; polycythemia
        C_RBC_COUNT:        (0.5, 10),      # M/uL
        C_WBC_COUNT:        (0, 500),       # K/uL; leukemoid reactions
        C_PLATELETS_COUNT:  (1, 2000),      # K/uL
        # -- Coagulation --
        C_PTT:              (10, 200),      # sec; 0 is lab error
        C_PT:               (7, 200),       # sec
        C_ACT:              (50, 600),      # sec
        C_INR:              (0.5, 25),      # ratio
        # -- Blood gas --
        C_ARTERIAL_PH:      (6.7, 7.8),    # pH > 7.8 essentially fatal
        C_PAO2:             (10, 700),      # mmHg; 0 is impossible in living patient
        C_PACO2:            (5, 200),       # mmHg; 0 is impossible
        C_ARTERIAL_BE:      (-50, 40),      # mEq/L; >40 is erroneous
        C_ARTERIAL_LACTATE: (0.1, 30),      # mmol/L
        C_HCO3:             (2, 55),        # mEq/L; >55 is erroneous
        C_ETCO2:            (0, 100),       # mmHg; cannot exceed ~80 physiologically
        C_SVO2:             (0, 100),       # percentage
    })
    return df


def convert_fio2_units(df):
    missing_fio2_set = pd.isna(df[C_FIO2_1]) & ~pd.isna(df[C_FIO2_100])
    df.loc[missing_fio2_set, C_FIO2_1] = df.loc[missing_fio2_set, C_FIO2_100] / 100
    missing_fio2 = ~pd.isna(df[C_FIO2_1]) & pd.isna(df[C_FIO2_100])
    df.loc[missing_fio2, C_FIO2_100] = df.loc[missing_fio2, C_FIO2_1] * 100
    return df


def estimate_fio2(df):
    df = convert_fio2_units(df)
    sah_fio2 = {}
    for col in [C_INTERFACE, C_FIO2_100, C_O2FLOW]:
        logging.info("  SAH on %s", col)
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

    # Room air -no O2 flow, no interface or cannula
    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) &
            ((sah_fio2[C_INTERFACE] == 0) | (sah_fio2[C_INTERFACE] == 2)))
    logging.info("  FiO2 room air: %d rows", mask.sum())
    df.loc[mask, C_FIO2_100] = 21

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) &
            (pd.isna(sah_fio2[C_INTERFACE]) | sah_fio2[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))))
    logging.info("  FiO2 face mask/vent with O2 flow: %d rows", mask.sum())
    df.loc[mask, C_FIO2_100] = fill_stepwise(
        sah_fio2[C_O2FLOW].loc[mask],
        zip(*([15, 12, 10, 8, 6, 4], [75, 69, 66, 58, 40, 36])))

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) &
            (pd.isna(sah_fio2[C_INTERFACE]) | sah_fio2[C_INTERFACE].isin((1, 3, 4, 5, 6, 9, 10))))
    logging.info("  FiO2 face mask/vent no O2 flow: %d rows -> NA", mask.sum())
    df.loc[mask, C_FIO2_100] = pd.NA

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & ~pd.isna(sah_fio2[C_O2FLOW]) &
            (sah_fio2[C_INTERFACE] == 7))
    logging.info("  FiO2 non-rebreather with O2 flow: %d rows", mask.sum())
    df.loc[mask, C_FIO2_100] = fill_stepwise(
        sah_fio2[C_O2FLOW].loc[mask],
        zip(*([9.99, 8, 6], [80, 70, 60])),
        zip(*([10, 15], [90, 100])))

    mask = (pd.isna(sah_fio2[C_FIO2_100]) & pd.isna(sah_fio2[C_O2FLOW]) &
            (sah_fio2[C_INTERFACE] == 7))
    logging.info("  FiO2 non-rebreather no O2 flow: %d rows -> NA", mask.sum())
    df.loc[mask, C_FIO2_100] = pd.NA

    df = convert_fio2_units(df)
    return df


def estimate_gcs(rass):
    if rass >= 0:    return 15
    elif rass == -1: return 14
    elif rass == -2: return 12
    elif rass == -3: return 11
    elif rass == -4: return 6
    elif rass == -5: return 3
    return pd.NA


def estimate_vitals(df):
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
    # Clamp negative Direct_bili from formula artifact (Total_bili < 0.25 -> negative)
    neg_dbili = df[C_DIRECT_BILI] < 0
    if neg_dbili.any():
        logging.info("  Clamping %d negative Direct_bili -> 0", neg_dbili.sum())
        df.loc[neg_dbili, C_DIRECT_BILI] = 0

    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Imputes outliers, estimates FiO2, estimates derived vitals, and '
        'applies sample-and-hold to patient states.'))
    parser.add_argument('input',  type=str, help='Path to patient_states.csv')
    parser.add_argument('output', type=str, help='Path for output CSV')
    parser.add_argument('--no-outliers',    dest='outliers',       default=True, action='store_false')
    parser.add_argument('--no-fio2',        dest='fio2',           default=True, action='store_false')
    parser.add_argument('--no-gcs',         dest='gcs',            default=True, action='store_false')
    parser.add_argument('--no-vitals',      dest='vitals',         default=True, action='store_false')
    parser.add_argument('--no-sample-hold', dest='sample_and_hold',default=True, action='store_false')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_05_impute_states.log')
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

    if args.fio2:
        logging.info("Estimating GCS from RASS")
        ii = pd.isna(df[C_GCS])
        df.loc[ii, C_GCS] = df.loc[ii, C_RASS].apply(estimate_gcs)
        logging.info("  Filled %d GCS values from RASS", ii.sum())

    if args.gcs:
        logging.info("Estimating FiO2")
        df = estimate_fio2(df)

    if args.vitals:
        logging.info("Estimating vitals")
        df = estimate_vitals(df)

    if args.sample_and_hold:
        logging.info("Sample and hold (%d columns)", len(SAH_FIELD_NAMES))
        sah_series = {C_BLOC: df[C_BLOC], C_ICUSTAYID: df[C_ICUSTAYID], C_TIMESTEP: df[C_TIMESTEP]}
        for col in SAH_FIELD_NAMES:
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
