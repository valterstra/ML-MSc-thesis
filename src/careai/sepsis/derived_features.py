"""
Derived feature computation: sepsis onset, SOFA, SIRS, Shock Index, PaO2/FiO2.
Adapted from ai_clinician/preprocessing/derived_features.py — only import path changed.
All formulas and thresholds identical to the original.
"""
import numpy as np
import pandas as pd
from careai.sepsis.columns import *


def calculate_onset(abx, bacterio, stay_id):
    """
    Compute presumed sepsis onset time for a single ICU stay.
    Logic: antibiotic first + culture <= 24h later  -> onset = antibiotic time
           culture first + antibiotic <= 72h later  -> onset = culture time
    Identical to AI-Clinician-MIMICIV derived_features.calculate_onset().
    """
    matching_abs = abx.loc[abx[C_ICUSTAYID] == stay_id, C_STARTDATE].reset_index(drop=True).sort_values()
    matching_bacts = bacterio[bacterio[C_ICUSTAYID] == stay_id].reset_index(drop=True).sort_values(C_CHARTTIME)
    if matching_abs.empty or matching_bacts.empty:
        return None
    for _, ab_time in matching_abs.items():
        dists = [abs(bact_row[C_CHARTTIME] - ab_time) / 3600 for _, bact_row in matching_bacts.iterrows()]
        min_index = int(np.argmin(dists))
        bact = matching_bacts.iloc[min_index]
        if dists[min_index] <= 24 and ab_time <= bact[C_CHARTTIME]:
            return {C_SUBJECT_ID: bact[C_SUBJECT_ID], C_ICUSTAYID: stay_id, C_ONSET_TIME: ab_time}
        elif dists[min_index] <= 72 and ab_time >= bact[C_CHARTTIME]:
            return {C_SUBJECT_ID: bact[C_SUBJECT_ID], C_ICUSTAYID: stay_id, C_ONSET_TIME: bact[C_CHARTTIME]}
    return None


def compute_pao2_fio2(df):
    if C_PAO2 not in df.columns or C_FIO2_1 not in df.columns:
        return pd.NA
    return df[C_PAO2] / df[C_FIO2_1]


def compute_shock_index(df):
    """Recompute Shock Index without NaN and Inf; replace remaining NaN with column mean (~0.8)."""
    result = df[C_HR] / df[C_SYSBP]
    result[np.isinf(result)] = pd.NA
    d = np.nanmean(result)
    print("Replacing shock index NaN with average value", d)
    result[pd.isna(result)] = d
    return result


def compute_sofa(df, timestep_resolution=4.0):
    """
    Compute SOFA score (0-24) from six organ-system components.
    Identical to AI-Clinician-MIMICIV derived_features.compute_sofa().
    """
    s = df[[C_PAO2_FIO2, C_PLATELETS_COUNT, C_TOTAL_BILI,
            C_MEANBP, C_MAX_DOSE_VASO, C_GCS, C_CREATININE, C_OUTPUT_STEP]]

    s1 = pd.DataFrame([
        s[C_PAO2_FIO2] > 400,
        (s[C_PAO2_FIO2] >= 300) & (s[C_PAO2_FIO2] < 400),
        (s[C_PAO2_FIO2] >= 200) & (s[C_PAO2_FIO2] < 300),
        (s[C_PAO2_FIO2] >= 100) & (s[C_PAO2_FIO2] < 200),
        s[C_PAO2_FIO2] < 100,
    ], index=range(5))
    s2 = pd.DataFrame([
        s[C_PLATELETS_COUNT] > 150,
        (s[C_PLATELETS_COUNT] >= 100) & (s[C_PLATELETS_COUNT] < 150),
        (s[C_PLATELETS_COUNT] >= 50)  & (s[C_PLATELETS_COUNT] < 100),
        (s[C_PLATELETS_COUNT] >= 20)  & (s[C_PLATELETS_COUNT] < 50),
        s[C_PLATELETS_COUNT] < 20,
    ], index=range(5))
    s3 = pd.DataFrame([
        s[C_TOTAL_BILI] < 1.2,
        (s[C_TOTAL_BILI] >= 1.2) & (s[C_TOTAL_BILI] < 2),
        (s[C_TOTAL_BILI] >= 2)   & (s[C_TOTAL_BILI] < 6),
        (s[C_TOTAL_BILI] >= 6)   & (s[C_TOTAL_BILI] < 12),
        s[C_TOTAL_BILI] > 12,
    ], index=range(5))
    s4 = pd.DataFrame([
        s[C_MEANBP] >= 70,
        (s[C_MEANBP] < 70) & (s[C_MEANBP] >= 65),
        s[C_MEANBP] < 65,
        (s[C_MAX_DOSE_VASO] > 0) & (s[C_MAX_DOSE_VASO] <= 0.1),
        s[C_MAX_DOSE_VASO] > 0.1,
    ], index=range(5))
    s5 = pd.DataFrame([
        s[C_GCS] > 14,
        (s[C_GCS] > 12) & (s[C_GCS] <= 14),
        (s[C_GCS] > 9)  & (s[C_GCS] <= 12),
        (s[C_GCS] > 5)  & (s[C_GCS] <= 9),
        s[C_GCS] <= 5,
    ], index=range(5))
    s6 = pd.DataFrame([
        s[C_CREATININE] < 1.2,
        (s[C_CREATININE] >= 1.2) & (s[C_CREATININE] < 2),
        (s[C_CREATININE] >= 2)   & (s[C_CREATININE] < 3.5),
        ((s[C_CREATININE] >= 3.5) & (s[C_CREATININE] < 5)) | (s[C_OUTPUT_STEP] < 500 * timestep_resolution / 24),
        (s[C_CREATININE] > 5) | (s[C_OUTPUT_STEP] < 200 * timestep_resolution / 24),
    ], index=range(5))

    return s1.idxmax(axis=0) + s2.idxmax(axis=0) + s3.idxmax(axis=0) + s4.idxmax(axis=0) + s5.idxmax(axis=0) + s6.idxmax(axis=0)


def compute_sirs(df):
    """
    Compute SIRS score (0-4) — count of positive criteria.
    Identical to AI-Clinician-MIMICIV derived_features.compute_sirs().
    """
    s = df[[C_TEMP_C, C_HR, C_RR, C_PACO2, C_WBC_COUNT]]
    s1 = (s[C_TEMP_C] >= 38) | (s[C_TEMP_C] <= 36)
    s2 = s[C_HR] > 90
    s3 = (s[C_RR] >= 20) | (s[C_PACO2] <= 32)
    s4 = (s[C_WBC_COUNT] >= 12) | (s[C_WBC_COUNT] < 4)
    return s1.astype(int) + s2.astype(int) + s3.astype(int) + s4.astype(int)
