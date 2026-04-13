"""
Column name constants and itemid reference lists for the ICU readmission pipeline.

Adapted from src/careai/sepsis/columns.py (AI-Clinician-MIMICIV structure).

Key differences from sepsis pipeline:
  - REF_LABS extended with phosphate (lab 36) + ICU-audit additions (labs 37-45)
  - REF_VITALS extended with GCS components, NIBP_Dia, pain, WBC diff, etc.
  - DEMOGRAPHICS extended: charlson (not elixhauser), race, insurance, marital_status,
    admission_type, prior_ed_visits_6m, 18 Charlson component flags
  - ACTIONS: 9 binary drug classes + 2 dose-level (vasopressor, ivfluid) + mechvent
  - Outcome: readmit_30d (from hospital discharge) + discharge_disposition (terminal action)
"""
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Core identifiers and timestamps
# ---------------------------------------------------------------------------
C_BLOC       = 'bloc'
C_ICUSTAYID  = 'icustayid'        # = stay_id in MIMIC-IV
C_HADM_ID    = 'hadm_id'
C_SUBJECT_ID = 'subject_id'
C_CHARTTIME  = 'charttime'
C_STARTTIME  = 'starttime'
C_ENDTIME    = 'endtime'
C_STARTDATE  = 'startdate'
C_ITEMID     = 'itemid'
C_VALUENUM   = 'valuenum'
C_VALUE      = 'value'
C_TIMESTEP   = 'timestep'
C_BIN_INDEX  = 'bin_index'
C_ADMITTIME  = 'admittime'
C_DISCHTIME  = 'dischtime'
C_INTIME     = 'intime'
C_OUTTIME    = 'outtime'
C_LOS        = 'los'
C_DOD        = 'dod'
C_AMOUNT     = 'amount'
C_RATE       = 'rate'
C_TEV        = 'tev'
C_RATESTD    = 'ratestd'
C_NORM_INFUSION_RATE = 'norm_infusion_rate'
C_INPUT_PREADM = 'input_preadm'
C_DATEDIFF_MINUTES = 'datediff_minutes'

# Onset/cohort timing
C_FIRST_TIMESTEP = 'first_timestep'
C_LAST_TIMESTEP  = 'last_timestep'

# ---------------------------------------------------------------------------
# Static demographic / confounder columns
# ---------------------------------------------------------------------------
C_GENDER            = 'gender'
C_AGE               = 'age'
C_WEIGHT            = 'Weight_kg'
C_RACE              = 'race'
C_INSURANCE         = 'insurance'
C_MARITAL_STATUS    = 'marital_status'
C_ADMISSION_TYPE    = 'admission_type'       # emergency vs elective vs urgent
C_ADMISSION_LOC     = 'admission_location'   # where patient came from
C_CHARLSON          = 'charlson_score'
C_RE_ADMISSION      = 're_admission'
C_PRIOR_ED_VISITS   = 'prior_ed_visits_6m'  # LACE E component
C_DRG_SEVERITY      = 'drg_severity'
C_DRG_MORTALITY     = 'drg_mortality'

# Outcome and terminal action
C_READMIT_30D          = 'readmit_30d'
C_DISCHARGE_DISPOSITION = 'discharge_disposition'

# Derived from sepsis pipeline — kept for compatibility
C_DIED_IN_HOSP                          = 'died_in_hosp'
C_DIED_WITHIN_48H_OF_OUT_TIME           = 'died_within_48h_of_out_time'
C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH = 'delay_end_of_record_and_discharge_or_death'
C_ADM_ORDER  = 'adm_order'
C_MORTA_HOSP = 'morta_hosp'
C_MORTA_90   = 'morta_90'

# ---------------------------------------------------------------------------
# Chart event columns (vitals/respiratory — same as sepsis)
# ---------------------------------------------------------------------------
C_HEIGHT      = 'Height_cm'
C_GCS         = 'GCS'           # GCS_Total (sum of Eye+Verbal+Motor)
C_GCS_EYE     = 'GCS_Eye'
C_GCS_VERBAL  = 'GCS_Verbal'
C_GCS_MOTOR   = 'GCS_Motor'
C_CAM_ICU     = 'cam_icu'       # CAM-ICU delirium screen: 1=positive, 0=negative, NaN=not assessed
C_RASS        = 'RASS'
C_HR          = 'HR'
C_SYSBP       = 'SysBP'
C_MEANBP      = 'MeanBP'
C_DIABP       = 'DiaBP'
C_NIBP_DIA    = 'NIBP_Diastolic'   # non-invasive diastolic (new)
C_ART_SYS     = 'Arterial_BP_Sys'  # invasive arterial systolic (new)
C_ART_DIA     = 'Arterial_BP_Dia'  # invasive arterial diastolic (new)
C_RR          = 'RR'
C_RR_SPONT    = 'RR_Spontaneous'   # spontaneous RR (new)
C_RR_TOTAL    = 'RR_Total'         # total RR (new)
C_SPO2        = 'SpO2'
C_TEMP_C      = 'Temp_C'
C_TEMP_F      = 'Temp_F'
C_CVP         = 'CVP'
C_PAPSYS      = 'PAPsys'
C_PAPMEAN     = 'PAPmean'
C_PAPDIA      = 'PAPdia'
C_CI          = 'CI'
C_SVR         = 'SVR'
C_INTERFACE   = 'Interface'
C_FIO2_100    = 'FiO2_100'
C_FIO2_1      = 'FiO2_1'
C_O2FLOW      = 'O2flow'
C_PEEP        = 'PEEP'
C_TIDALVOLUME = 'TidalVolume'
C_TIDAL_OBS   = 'TidalVolume_Observed'  # observed TV (new)
C_MINUTEVENTIL = 'MinuteVentil'
C_PAWMEAN     = 'PAWmean'
C_PAWPEAK     = 'PAWpeak'
C_PAWPLATEAU  = 'PAWplateau'
C_PAIN_LEVEL  = 'Pain_Level'            # numeric pain score (new)

# ---------------------------------------------------------------------------
# Lab columns — 36 locked labs (same names as sepsis; phosphate added)
# ---------------------------------------------------------------------------
C_POTASSIUM        = 'Potassium'
C_SODIUM           = 'Sodium'
C_CHLORIDE         = 'Chloride'
C_GLUCOSE          = 'Glucose'
C_BUN              = 'BUN'
C_CREATININE       = 'Creatinine'
C_MAGNESIUM        = 'Magnesium'
C_CALCIUM          = 'Calcium'
C_IONISED_CA       = 'Ionised_Ca'
C_CO2_MEQL         = 'CO2_mEqL'
C_SGOT             = 'SGOT'
C_SGPT             = 'SGPT'
C_TOTAL_BILI       = 'Total_bili'
C_DIRECT_BILI      = 'Direct_bili'
C_TOTAL_PROTEIN    = 'Total_protein'
C_ALBUMIN          = 'Albumin'
C_TROPONIN         = 'Troponin'
C_CRP              = 'CRP'
C_HB               = 'Hb'
C_HT               = 'Ht'
C_RBC_COUNT        = 'RBC_count'
C_WBC_COUNT        = 'WBC_count'
C_PLATELETS_COUNT  = 'Platelets_count'
C_PTT              = 'PTT'
C_PT               = 'PT'
C_ACT              = 'ACT'
C_INR              = 'INR'
C_ARTERIAL_PH      = 'Arterial_pH'
C_PAO2             = 'paO2'
C_PACO2            = 'paCO2'
C_ARTERIAL_BE      = 'Arterial_BE'
C_ARTERIAL_LACTATE = 'Arterial_lactate'
C_HCO3             = 'HCO3'
C_ETCO2            = 'ETCO2'
C_SVO2             = 'SvO2'
C_PHOSPHATE        = 'Phosphate'        # lab 36 — new vs sepsis

# ICU audit additional lab columns (beyond the 36 locked labs)
C_ANION_GAP     = 'Anion_Gap'           # directly measured (96.5%)
C_ALK_PHOS      = 'Alkaline_Phosphatase' # (51.0%)
C_LDH           = 'LDH'                 # (37.0%)
C_FIBRINOGEN    = 'Fibrinogen'           # (25.8%)
C_NEUTS_PCT     = 'Neuts_pct'           # WBC differential
C_LYMPHS_PCT    = 'Lymphs_pct'
C_MONOS_PCT     = 'Monos_pct'
C_EOS_PCT       = 'Eos_pct'
C_BASOS_PCT     = 'Basos_pct'

# Ventilation
C_MECHVENT  = 'mechvent'
C_EXTUBATED = 'extubated'

# Derived severity / computed
C_SHOCK_INDEX = 'Shock_Index'
C_PAO2_FIO2   = 'PaO2_FiO2'
C_SOFA        = 'SOFA'
C_SIRS        = 'SIRS'
C_LAST_SOFA   = 'last_SOFA'
C_NUM_BLOCS   = 'num_blocs'

# Vasopressor / fluid I/O (same as sepsis)
C_MEDIAN_DOSE_VASO  = 'median_dose_vaso'
C_MAX_DOSE_VASO     = 'max_dose_vaso'
C_VASOPRESSOR_DOSE  = 'vasopressor_dose'   # discretized 0-4
C_IVFLUID_DOSE      = 'ivfluid_dose'        # discretized 0-4
C_INPUT_TOTAL       = 'input_total'
C_INPUT_STEP        = 'input_step'
C_OUTPUT_TOTAL      = 'output_total'
C_OUTPUT_STEP       = 'output_step'
C_CUMULATED_BALANCE = 'cumulated_balance'

# ---------------------------------------------------------------------------
# Binary action columns (per 4-hour bloc)
# ---------------------------------------------------------------------------
C_ANTIBIOTIC_ACTIVE    = 'antibiotic_active'
C_ANTICOAGULANT_ACTIVE = 'anticoagulant_active'
C_DIURETIC_ACTIVE      = 'diuretic_active'
C_STEROID_ACTIVE       = 'steroid_active'
C_INSULIN_ACTIVE       = 'insulin_active'
C_OPIOID_ACTIVE        = 'opioid_active'
C_SEDATION_ACTIVE      = 'sedation_active'
C_TRANSFUSION_ACTIVE   = 'transfusion_active'
C_ELECTROLYTE_ACTIVE   = 'electrolyte_active'

BINARY_ACTION_COLS = [
    C_ANTIBIOTIC_ACTIVE,
    C_ANTICOAGULANT_ACTIVE,
    C_DIURETIC_ACTIVE,
    C_STEROID_ACTIVE,
    C_INSULIN_ACTIVE,
    C_OPIOID_ACTIVE,
    C_SEDATION_ACTIVE,
    C_TRANSFUSION_ACTIVE,
    C_ELECTROLYTE_ACTIVE,
    C_MECHVENT,     # ventilation is both a state (ongoing) and a clinical action decision
]

# ---------------------------------------------------------------------------
# 18 Charlson comorbidity component flags
# ---------------------------------------------------------------------------
C_CC_MI          = 'cc_mi'
C_CC_CHF         = 'cc_chf'
C_CC_PVD         = 'cc_pvd'
C_CC_CVD         = 'cc_cvd'
C_CC_DEMENTIA    = 'cc_dementia'
C_CC_COPD        = 'cc_copd'
C_CC_RHEUM       = 'cc_rheum'
C_CC_PUD         = 'cc_pud'
C_CC_MILD_LIVER  = 'cc_mild_liver'
C_CC_DM_NO_CC    = 'cc_dm_no_cc'
C_CC_DM_CC       = 'cc_dm_cc'
C_CC_PARALYSIS   = 'cc_paralysis'
C_CC_RENAL       = 'cc_renal'
C_CC_MALIGN      = 'cc_malign'
C_CC_SEV_LIVER   = 'cc_sev_liver'
C_CC_METASTATIC  = 'cc_metastatic'
C_CC_HIV         = 'cc_hiv'
C_CC_HEMIPLEGIA  = 'cc_hemiplegia'

CHARLSON_FLAG_COLS = [
    C_CC_MI, C_CC_CHF, C_CC_PVD, C_CC_CVD, C_CC_DEMENTIA, C_CC_COPD,
    C_CC_RHEUM, C_CC_PUD, C_CC_MILD_LIVER, C_CC_DM_NO_CC, C_CC_DM_CC,
    C_CC_PARALYSIS, C_CC_RENAL, C_CC_MALIGN, C_CC_SEV_LIVER,
    C_CC_METASTATIC, C_CC_HIV, C_CC_HEMIPLEGIA,
]

STATIC_CONFOUNDER_COLS = [
    C_GENDER, C_AGE, C_WEIGHT,
    C_RACE, C_INSURANCE, C_MARITAL_STATUS,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_CHARLSON, C_RE_ADMISSION,
    C_PRIOR_ED_VISITS,
    C_DRG_SEVERITY, C_DRG_MORTALITY,
    C_DISCHARGE_DISPOSITION,    # known at discharge; used as training confounder, not real-time action
] + CHARLSON_FLAG_COLS

# ---------------------------------------------------------------------------
# Dtype specs for loading CSVs
# ---------------------------------------------------------------------------
STAY_ID_OPTIONAL_DTYPE_SPEC = {
    C_HADM_ID:   pd.Int64Dtype(),
    C_SUBJECT_ID: pd.Int64Dtype(),
    C_TIMESTEP:  pd.Int64Dtype(),
    C_ICUSTAYID: pd.Int64Dtype(),
    C_ITEMID:    pd.Int64Dtype(),
}

# ---------------------------------------------------------------------------
# REF_LABS — itemid groups for lab extraction
# Group index i+1 maps to LAB_FIELD_NAMES[i]
#
# Labs 1-35: identical to sepsis pipeline (AI-Clinician-MIMICIV)
# Lab 36: Phosphate — new vs sepsis
# Labs 37-45: ICU audit additions (Anion_Gap, Alk_Phos, LDH, Fibrinogen, WBC diff)
# ---------------------------------------------------------------------------
REF_LABS = [
    [829, 1535, 227442, 227464, 3792, 50971, 50822],                                    # 1  Potassium
    [837, 220645, 4194, 3725, 3803, 226534, 1536, 4195, 3726, 50983, 50824],           # 2  Sodium
    [788, 220602, 1523, 4193, 3724, 226536, 3747, 50902, 50806],                       # 3  Chloride
    [225664, 807, 811, 1529, 220621, 226537, 3744, 50809, 50931],                      # 4  Glucose
    [781, 1162, 225624, 3737, 51006, 52647],                                            # 5  BUN
    [791, 1525, 220615, 3750, 50912, 51081],                                            # 6  Creatinine
    [821, 1532, 220635, 50960],                                                         # 7  Magnesium
    [786, 225625, 1522, 3746, 50893, 51624],                                            # 8  Calcium
    [816, 225667, 3766, 50808],                                                         # 9  Ionised_Ca
    [777, 787, 50804],                                                                  # 10 CO2_mEqL
    [770, 3801, 50878, 220587],                                                         # 11 SGOT
    [769, 3802, 50861, 220644],                                                         # 12 SGPT (+ alt itemid 220644)
    [1538, 848, 225690, 51464, 50885],                                                  # 13 Total_bili
    [803, 1527, 225651, 50883],                                                         # 14 Direct_bili
    [3807, 1539, 849, 50976],                                                           # 15 Total_protein
    [772, 1521, 227456, 3727, 50862],                                                   # 16 Albumin
    [227429, 851, 51002, 51003],                                                        # 17 Troponin
    [227444, 50889],                                                                    # 18 CRP
    [814, 220228, 50811, 51222],                                                        # 19 Hb
    [813, 220545, 3761, 226540, 51221, 50810],                                         # 20 Ht
    [4197, 3799, 51279],                                                                # 21 RBC_count
    [1127, 1542, 220546, 4200, 3834, 51300, 51301],                                    # 22 WBC_count
    [828, 227457, 3789, 51265],                                                         # 23 Platelets_count
    [825, 1533, 227466, 3796, 51275, 52923, 52165, 52166, 52167],                      # 24 PTT
    [824, 1286, 51274, 227465],                                                         # 25 PT
    [1671, 1520, 768, 220507],                                                          # 26 ACT
    [815, 1530, 227467, 51237],                                                         # 27 INR
    [780, 1126, 3839, 4753, 50820],                                                     # 28 Arterial_pH
    [779, 490, 3785, 3838, 3837, 50821, 220224, 226063, 226770, 227039],               # 29 paO2
    [778, 3784, 3836, 3835, 50818, 220235, 226062, 227036],                            # 30 paCO2
    [776, 224828, 3736, 4196, 3740, 74, 50802],                                        # 31 Arterial_BE
    [225668, 1531, 50813],                                                              # 32 Arterial_lactate
    [227443, 50882, 50803],                                                             # 33 HCO3
    [1817, 228640],                                                                     # 34 ETCO2
    [823, 227686, 223772],                                                              # 35 SvO2
    [225677, 50970],                                                                    # 36 Phosphate (NEW)
    [227073],                                                                           # 37 Anion_Gap (ICU audit)
    [225612],                                                                           # 38 Alkaline_Phosphatase
    [220632],                                                                           # 39 LDH
    [227468],                                                                           # 40 Fibrinogen
    [225641],                                                                           # 41 Neuts_pct
    [225642],                                                                           # 42 Lymphs_pct
    [225643],                                                                           # 43 Monos_pct
    [225645],                                                                           # 44 Eos_pct
    [225644],                                                                           # 45 Basos_pct
]

# ---------------------------------------------------------------------------
# REF_VITALS — itemid groups for chart event extraction
# Group index i+1 maps to CHART_FIELD_NAMES[i]
#
# Items 1-28: identical to sepsis pipeline
# Items 29-37: new ICU-specific chartevents
# ---------------------------------------------------------------------------
REF_VITALS = [
    [226707, 226730],                                                   # 1  Height_cm
    [581, 580, 224639, 226512],                                         # 2  Weight_kg
    [220739],                                                           # 3  GCS_Eye  (replaces broken CareVue 198)
    [223900],                                                           # 4  GCS_Verbal
    [223901],                                                           # 5  GCS_Motor
    [228096],                                                           # 6  RASS
    [211, 220045],                                                      # 7  HR
    [220179, 225309, 6701, 6, 227243, 224167, 51, 455],                # 8  SysBP
    [220181, 220052, 225312, 224322, 6702, 443, 52, 456],              # 9  MeanBP
    [8368, 8441, 225310, 8555, 8440],                                   # 10 DiaBP
    [220180],                                                           # 11 NIBP_Diastolic (new)
    [220050],                                                           # 12 Arterial_BP_Sys (new)
    [220051],                                                           # 13 Arterial_BP_Dia (new)
    [220210, 3337, 224422, 618, 3603, 615],                            # 14 RR
    [224689],                                                           # 15 RR_Spontaneous (new)
    [224690],                                                           # 16 RR_Total (new)
    [220277, 646, 834],                                                 # 17 SpO2
    [3655, 223762],                                                     # 18 Temp_C
    [223761, 678],                                                      # 19 Temp_F
    [220074, 113],                                                      # 20 CVP
    [492, 220059],                                                      # 21 PAPsys
    [491, 220061],                                                      # 22 PAPmean
    [8448, 220060],                                                     # 23 PAPdia
    [116, 1372, 1366, 228368, 228177],                                  # 24 CI
    [626],                                                              # 25 SVR
    [467, 226732],                                                      # 26 Interface
    [223835, 3420, 160, 727],                                           # 27 FiO2_100
    [190],                                                              # 28 FiO2_1
    [470, 471, 223834, 227287, 194, 224691],                           # 29 O2flow
    [220339, 506, 505, 224700],                                         # 30 PEEP
    [224686, 224684, 684, 224421, 3083, 2566, 654, 3050, 681, 2311],   # 31 TidalVolume (set)
    [224685],                                                           # 32 TidalVolume_Observed (new)
    [224687, 450, 448, 445],                                            # 33 MinuteVentil
    [224697, 444],                                                      # 34 PAWmean
    [224695, 535],                                                      # 35 PAWpeak
    [224696, 543],                                                      # 36 PAWplateau
    [223791],                                                           # 37 Pain_Level (new, 85.3%)
    [229326],                                                           # 38 CAM-ICU (delirium screen, text->0/1 in ce())
]

# ---------------------------------------------------------------------------
# Field name lists (CHART_FIELD_NAMES[i] corresponds to REF_VITALS[i] for i < 37)
# NOTE: GCS components (indices 2-4) are extracted from chartevents in step_04;
#       GCS_Total is summed from them in step_05 and appended at position 37.
#       C_GCS (index 37) has NO REF_VITALS entry — step_04 never reads it from
#       chartevents. It only enters the dataset via step_05's compute_gcs_total().
# ---------------------------------------------------------------------------
CHART_FIELD_NAMES = [
    C_HEIGHT, C_WEIGHT,
    C_GCS_EYE, C_GCS_VERBAL, C_GCS_MOTOR,           # components — summed to GCS in step_05
    C_RASS, C_HR, C_SYSBP, C_MEANBP, C_DIABP,
    C_NIBP_DIA, C_ART_SYS, C_ART_DIA,
    C_RR, C_RR_SPONT, C_RR_TOTAL,
    C_SPO2, C_TEMP_C, C_TEMP_F,
    C_CVP, C_PAPSYS, C_PAPMEAN, C_PAPDIA,
    C_CI, C_SVR, C_INTERFACE,
    C_FIO2_100, C_FIO2_1, C_O2FLOW, C_PEEP,
    C_TIDALVOLUME, C_TIDAL_OBS,
    C_MINUTEVENTIL, C_PAWMEAN, C_PAWPEAK, C_PAWPLATEAU,
    C_PAIN_LEVEL,
    C_CAM_ICU,  # delirium screen — REF_VITALS group 38 (text->0/1 via ce() CASE)
    C_GCS,      # GCS_Total — computed in step_05; no REF_VITALS group (index 38, beyond extraction range)
]

LAB_FIELD_NAMES = [
    C_POTASSIUM, C_SODIUM, C_CHLORIDE, C_GLUCOSE, C_BUN,
    C_CREATININE, C_MAGNESIUM, C_CALCIUM, C_IONISED_CA, C_CO2_MEQL,
    C_SGOT, C_SGPT, C_TOTAL_BILI, C_DIRECT_BILI, C_TOTAL_PROTEIN,
    C_ALBUMIN, C_TROPONIN, C_CRP,
    C_HB, C_HT, C_RBC_COUNT, C_WBC_COUNT, C_PLATELETS_COUNT,
    C_PTT, C_PT, C_ACT, C_INR,
    C_ARTERIAL_PH, C_PAO2, C_PACO2, C_ARTERIAL_BE, C_ARTERIAL_LACTATE,
    C_HCO3, C_ETCO2, C_SVO2,
    C_PHOSPHATE,                                    # lab 36
    C_ANION_GAP, C_ALK_PHOS, C_LDH, C_FIBRINOGEN,  # ICU audit additions
    C_NEUTS_PCT, C_LYMPHS_PCT, C_MONOS_PCT, C_EOS_PCT, C_BASOS_PCT,
]

VENT_FIELD_NAMES = [C_MECHVENT, C_EXTUBATED]

# All fields that get sample-and-hold forward-fill
SAH_FIELD_NAMES = CHART_FIELD_NAMES + LAB_FIELD_NAMES + VENT_FIELD_NAMES

COMPUTED_FIELD_NAMES = [C_SHOCK_INDEX, C_PAO2_FIO2, C_SOFA, C_SIRS]

IO_FIELD_NAMES = [
    C_MEDIAN_DOSE_VASO, C_MAX_DOSE_VASO,
    C_INPUT_TOTAL, C_INPUT_STEP,
    C_OUTPUT_TOTAL, C_OUTPUT_STEP,
    C_CUMULATED_BALANCE,
]

DEMOGRAPHICS_FIELD_NAMES = (
    [C_GENDER, C_AGE, C_WEIGHT,
     C_RACE, C_INSURANCE, C_MARITAL_STATUS,
     C_ADMISSION_TYPE, C_ADMISSION_LOC,
     C_CHARLSON, C_RE_ADMISSION,
     C_PRIOR_ED_VISITS, C_DRG_SEVERITY, C_DRG_MORTALITY,
     C_DISCHARGE_DISPOSITION,
     C_DIED_IN_HOSP, C_DIED_WITHIN_48H_OF_OUT_TIME,
     C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH]
    + CHARLSON_FLAG_COLS
)

# ---------------------------------------------------------------------------
# Sample-and-hold durations in hours
# Values for labs 1-35 identical to sepsis; new fields use clinical judgment.
# ---------------------------------------------------------------------------
_CHART_SAH = [
    168, 72,                      # Height, Weight
    6, 6, 6,                      # GCS_Eye, GCS_Verbal, GCS_Motor
    6, 2, 2, 2, 2,                # RASS, HR, SysBP, MeanBP, DiaBP
    2, 2, 2,                      # NIBP_Dia, Art_Sys, Art_Dia
    2, 2, 2,                      # RR, RR_Spont, RR_Total
    2, 6, 6,                      # SpO2, Temp_C, Temp_F
    2, 2, 2, 2,                   # CVP, PAPsys, PAPmean, PAPdia
    2, 2, 24,                     # CI, SVR, Interface
    12, 12, 12, 6,                # FiO2_100, FiO2_1, O2flow, PEEP
    6, 6,                         # TidalVolume, TidalVolume_Observed
    6, 6, 6, 6,                   # MinuteVentil, PAWmean, PAWpeak, PAWplateau
    8,                            # Pain_Level (reassessed ~q4-8h)
    4,                            # CAM-ICU (assessed q4-8h in ICU; short hold — reassessed frequently)
    6,                            # GCS_Total (same hold as components)
]

_LAB_SAH = [
    28, 28, 28, 28, 28,   # K, Na, Cl, Glucose, BUN
    28, 28, 28, 28, 28,   # Cr, Mg, Ca, Ionised_Ca, CO2
    28, 28, 28, 28, 28,   # SGOT, SGPT, Total_bili, Direct_bili, Total_protein
    28, 28, 28,           # Albumin, Troponin, CRP
    28, 28, 28, 28, 28,   # Hb, Ht, RBC, WBC, Platelets
    28, 28, 28, 28,       # PTT, PT, ACT, INR
    8, 8, 8, 8, 8, 8, 8, # Arterial gas x7 (pH,paO2,paCO2,BE,lactate,HCO3,ETCO2)
    28,                   # SvO2
    28,                   # Phosphate
    28, 28, 28, 28,       # Anion_Gap, Alk_Phos, LDH, Fibrinogen
    28, 28, 28, 28, 28,   # WBC differential x5
]

_VENT_SAH = [6, 6]  # mechvent, extubated

SAH_HOLD_DURATION = {
    f: v for f, v in zip(SAH_FIELD_NAMES, _CHART_SAH + _LAB_SAH + _VENT_SAH)
}

assert len(SAH_FIELD_NAMES) == len(_CHART_SAH) + len(_LAB_SAH) + len(_VENT_SAH), (
    f"SAH mismatch: {len(SAH_FIELD_NAMES)} fields vs "
    f"{len(_CHART_SAH)+len(_LAB_SAH)+len(_VENT_SAH)} durations"
)
