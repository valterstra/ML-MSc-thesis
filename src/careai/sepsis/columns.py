"""
Column name constants and itemid reference lists for the sepsis dataset pipeline.
Adapted from ai_clinician/preprocessing/columns.py (AI-Clinician-MIMICIV).
Only change: removed ai_clinician import paths.
"""
import pandas as pd
import numpy as np

# Demographics and timestamps
C_BLOC = 'bloc'
C_ICUSTAYID = 'icustayid'
C_CHARTTIME = 'charttime'
C_GENDER = 'gender'
C_AGE = 'age'
C_ELIXHAUSER = 'elixhauser'
C_RE_ADMISSION = 're_admission'
C_DIED_IN_HOSP = 'died_in_hosp'
C_DIED_WITHIN_48H_OF_OUT_TIME = 'died_within_48h_of_out_time'
C_MORTA_90 = 'morta_90'
C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH = 'delay_end_of_record_and_discharge_or_death'
C_READMIT_30D = 'readmit_30d'

# Chart events
C_HEIGHT = 'Height_cm'
C_WEIGHT = 'Weight_kg'
C_GCS = 'GCS'
C_RASS = 'RASS'
C_HR = 'HR'
C_SYSBP = 'SysBP'
C_MEANBP = 'MeanBP'
C_DIABP = 'DiaBP'
C_RR = 'RR'
C_SPO2 = 'SpO2'
C_TEMP_C = 'Temp_C'
C_TEMP_F = 'Temp_F'
C_CVP = 'CVP'
C_PAPSYS = 'PAPsys'
C_PAPMEAN = 'PAPmean'
C_PAPDIA = 'PAPdia'
C_CI = 'CI'
C_SVR = 'SVR'
C_INTERFACE = 'Interface'
C_FIO2_100 = 'FiO2_100'
C_FIO2_1 = 'FiO2_1'
C_O2FLOW = 'O2flow'
C_PEEP = 'PEEP'
C_TIDALVOLUME = 'TidalVolume'
C_MINUTEVENTIL = 'MinuteVentil'
C_PAWMEAN = 'PAWmean'
C_PAWPEAK = 'PAWpeak'
C_PAWPLATEAU = 'PAWplateau'

# Labs
C_POTASSIUM = 'Potassium'
C_SODIUM = 'Sodium'
C_CHLORIDE = 'Chloride'
C_GLUCOSE = 'Glucose'
C_BUN = 'BUN'
C_CREATININE = 'Creatinine'
C_MAGNESIUM = 'Magnesium'
C_CALCIUM = 'Calcium'
C_IONISED_CA = 'Ionised_Ca'
C_CO2_MEQL = 'CO2_mEqL'
C_SGOT = 'SGOT'
C_SGPT = 'SGPT'
C_TOTAL_BILI = 'Total_bili'
C_DIRECT_BILI = 'Direct_bili'
C_TOTAL_PROTEIN = 'Total_protein'
C_ALBUMIN = 'Albumin'
C_TROPONIN = 'Troponin'
C_CRP = 'CRP'
C_HB = 'Hb'
C_HT = 'Ht'
C_RBC_COUNT = 'RBC_count'
C_WBC_COUNT = 'WBC_count'
C_PLATELETS_COUNT = 'Platelets_count'
C_PTT = 'PTT'
C_PT = 'PT'
C_ACT = 'ACT'
C_INR = 'INR'
C_ARTERIAL_PH = 'Arterial_pH'
C_PAO2 = 'paO2'
C_PACO2 = 'paCO2'
C_ARTERIAL_BE = 'Arterial_BE'
C_ARTERIAL_LACTATE = 'Arterial_lactate'
C_HCO3 = 'HCO3'
C_ETCO2 = 'ETCO2'
C_SVO2 = 'SvO2'

# Ventilation
C_MECHVENT = 'mechvent'
C_EXTUBATED = 'extubated'

# Computed
C_SHOCK_INDEX = 'Shock_Index'
C_PAO2_FIO2 = 'PaO2_FiO2'

# Vasopressors, input/output
C_MEDIAN_DOSE_VASO = 'median_dose_vaso'
C_MAX_DOSE_VASO = 'max_dose_vaso'
C_INPUT_TOTAL = 'input_total'
C_INPUT_STEP = 'input_step'
C_OUTPUT_TOTAL = 'output_total'
C_OUTPUT_STEP = 'output_step'
C_CUMULATED_BALANCE = 'cumulated_balance'

# Onset data
C_ONSET_TIME = "onset_time"
C_FIRST_TIMESTEP = "first_timestep"
C_LAST_TIMESTEP = "last_timestep"

# Derived dataframe columns
C_TIMESTEP = "timestep"
C_BIN_INDEX = "bin_index"
C_SOFA = "SOFA"
C_SIRS = "SIRS"
C_LAST_VASO = "last_vaso"
C_LAST_SOFA = "last_SOFA"
C_NUM_BLOCS = "num_blocs"
C_MAX_SOFA = "max_SOFA"
C_MAX_SIRS = "max_SIRS"

# Raw data columns
C_HADM_ID = "hadm_id"
C_SUBJECT_ID = "subject_id"
C_STARTDATE = "startdate"
C_ENDDATE = "enddate"
C_STARTTIME = "starttime"
C_ENDTIME = "endtime"
C_CHARTDATE = "chartdate"
C_ITEMID = "itemid"
C_ADMITTIME = "admittime"
C_DISCHTIME = "dischtime"
C_ADM_ORDER = "adm_order"
C_UNIT = "unit"
C_INTIME = "intime"
C_OUTTIME = "outtime"
C_LOS = "los"
C_DOB = "dob"
C_DOD = "dod"
C_EXPIRE_FLAG = "expire_flag"
C_MORTA_HOSP = "morta_hosp"
C_VALUE = "value"
C_VALUENUM = "valuenum"
C_SELFEXTUBATED = "selfextubated"
C_INPUT_PREADM = "input_preadm"
C_AMOUNT = "amount"
C_RATE = "rate"
C_TEV = "tev"
C_RATESTD = "ratestd"
C_DATEDIFF_MINUTES = "datediff_minutes"
C_GSN = "gsn"
C_NDC = "ndc"
C_DOSE_VAL = "dose_val"
C_DOSE_UNIT = "dose_unit"
C_ROUTE = "route"
C_ORG_ITEMID = "org_itemid"
C_SPEC_ITEMID = "spec_itemid"
C_AB_ITEMID = "ab_itemid"
C_INTERPRETATION = "interpretation"
C_NORM_INFUSION_RATE = "norm_infusion_rate"

# Comorbidities
C_CONGESTIVE_HEART_FAILURE = "congestive_heart_failure"
C_CARDIAC_ARRHYTHMIAS = "cardiac_arrhythmias"
C_VALVULAR_DISEASE = "valvular_disease"
C_PULMONARY_CIRCULATION = "pulmonary_circulation"
C_PERIPHERAL_VASCULAR = "peripheral_vascular"
C_HYPERTENSION = "hypertension"
C_PARALYSIS = "paralysis"
C_OTHER_NEUROLOGICAL = "other_neurological"
C_CHRONIC_PULMONARY = "chronic_pulmonary"
C_DIABETES_UNCOMPLICATED = "diabetes_uncomplicated"
C_DIABETES_COMPLICATED = "diabetes_complicated"
C_HYPOTHYROIDISM = "hypothyroidism"
C_RENAL_FAILURE = "renal_failure"
C_LIVER_DISEASE = "liver_disease"
C_PEPTIC_ULCER = "peptic_ulcer"
C_AIDS = "aids"
C_LYMPHOMA = "lymphoma"
C_METASTATIC_CANCER = "metastatic_cancer"
C_SOLID_TUMOR = "solid_tumor"
C_RHEUMATOID_ARTHRITIS = "rheumatoid_arthritis"
C_COAGULOPATHY = "coagulopathy"
C_OBESITY = "obesity"
C_WEIGHT_LOSS = "weight_loss"
C_FLUID_ELECTROLYTE = "fluid_electrolyte"
C_BLOOD_LOSS_ANEMIA = "blood_loss_anemia"
C_DEFICIENCY_ANEMIAS = "deficiency_anemias"
C_ALCOHOL_ABUSE = "alcohol_abuse"
C_DRUG_ABUSE = "drug_abuse"
C_PSYCHOSES = "psychoses"
C_DEPRESSION = "depression"

COMORBIDITY_FIELDS = [
    C_CONGESTIVE_HEART_FAILURE, C_CARDIAC_ARRHYTHMIAS, C_VALVULAR_DISEASE,
    C_PULMONARY_CIRCULATION, C_PERIPHERAL_VASCULAR, C_HYPERTENSION, C_PARALYSIS,
    C_OTHER_NEUROLOGICAL, C_CHRONIC_PULMONARY, C_DIABETES_UNCOMPLICATED,
    C_DIABETES_COMPLICATED, C_HYPOTHYROIDISM, C_RENAL_FAILURE, C_LIVER_DISEASE,
    C_PEPTIC_ULCER, C_AIDS, C_LYMPHOMA, C_METASTATIC_CANCER, C_SOLID_TUMOR,
    C_RHEUMATOID_ARTHRITIS, C_COAGULOPATHY, C_OBESITY, C_WEIGHT_LOSS,
    C_FLUID_ELECTROLYTE, C_BLOOD_LOSS_ANEMIA, C_DEFICIENCY_ANEMIAS, C_ALCOHOL_ABUSE,
    C_DRUG_ABUSE, C_PSYCHOSES, C_DEPRESSION,
]

DTYPE_SPEC = {
    C_HADM_ID: pd.Int64Dtype(),
    C_SUBJECT_ID: pd.Int64Dtype(),
    C_TIMESTEP: pd.Int64Dtype(),
    C_ICUSTAYID: np.int64,
    C_ITEMID: pd.Int64Dtype(),
}

STAY_ID_OPTIONAL_DTYPE_SPEC = {
    C_HADM_ID: pd.Int64Dtype(),
    C_SUBJECT_ID: pd.Int64Dtype(),
    C_TIMESTEP: pd.Int64Dtype(),
    C_ICUSTAYID: pd.Int64Dtype(),
    C_ITEMID: pd.Int64Dtype(),
}

# ---------------------------------------------------------------------------
# Itemid reference lists — identical to AI-Clinician-MIMICIV columns.py
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
    [769, 3802, 50861],                                                                 # 12 SGPT
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
]

REF_VITALS = [
    [226707, 226730],                                                                   # 1  Height_cm
    [581, 580, 224639, 226512],                                                         # 2  Weight_kg
    [198],                                                                              # 3  GCS
    [228096],                                                                           # 4  RASS
    [211, 220045],                                                                      # 5  HR
    [220179, 225309, 6701, 6, 227243, 224167, 51, 455],                               # 6  SysBP
    [220181, 220052, 225312, 224322, 6702, 443, 52, 456],                             # 7  MeanBP (note: MeanBP is index 7 but DIABP label; see CHART_FIELD_NAMES order)
    [8368, 8441, 225310, 8555, 8440],                                                  # 8  DiaBP
    [220210, 3337, 224422, 618, 3603, 615],                                            # 9  RR
    [220277, 646, 834],                                                                 # 10 SpO2
    [3655, 223762],                                                                     # 11 Temp_C
    [223761, 678],                                                                      # 12 Temp_F
    [220074, 113],                                                                      # 13 CVP
    [492, 220059],                                                                      # 14 PAPsys
    [491, 220061],                                                                      # 15 PAPmean
    [8448, 220060],                                                                     # 16 PAPdia
    [116, 1372, 1366, 228368, 228177],                                                  # 17 CI
    [626],                                                                              # 18 SVR
    [467, 226732],                                                                      # 19 Interface (O2 delivery device)
    [223835, 3420, 160, 727],                                                           # 20 FiO2_100
    [190],                                                                              # 21 FiO2_1
    [470, 471, 223834, 227287, 194, 224691],                                           # 22 O2flow
    [220339, 506, 505, 224700],                                                         # 23 PEEP
    [224686, 224684, 684, 224421, 3083, 2566, 654, 3050, 681, 2311],                  # 24 TidalVolume
    [224687, 450, 448, 445],                                                            # 25 MinuteVentil
    [224697, 444],                                                                      # 26 PAWmean
    [224695, 535],                                                                      # 27 PAWpeak
    [224696, 543],                                                                      # 28 PAWplateau
]

CHART_FIELD_NAMES = [
    C_HEIGHT, C_WEIGHT, C_GCS,
    C_RASS, C_HR, C_SYSBP,
    C_MEANBP, C_DIABP, C_RR,
    C_SPO2, C_TEMP_C, C_TEMP_F,
    C_CVP, C_PAPSYS, C_PAPMEAN,
    C_PAPDIA, C_CI, C_SVR,
    C_INTERFACE, C_FIO2_100, C_FIO2_1,
    C_O2FLOW, C_PEEP, C_TIDALVOLUME,
    C_MINUTEVENTIL, C_PAWMEAN, C_PAWPEAK,
    C_PAWPLATEAU,
]

LAB_FIELD_NAMES = [
    C_POTASSIUM, C_SODIUM,
    C_CHLORIDE, C_GLUCOSE, C_BUN,
    C_CREATININE, C_MAGNESIUM, C_CALCIUM,
    C_IONISED_CA, C_CO2_MEQL, C_SGOT,
    C_SGPT, C_TOTAL_BILI, C_DIRECT_BILI,
    C_TOTAL_PROTEIN, C_ALBUMIN, C_TROPONIN,
    C_CRP, C_HB, C_HT,
    C_RBC_COUNT, C_WBC_COUNT, C_PLATELETS_COUNT,
    C_PTT, C_PT, C_ACT,
    C_INR, C_ARTERIAL_PH, C_PAO2,
    C_PACO2, C_ARTERIAL_BE, C_ARTERIAL_LACTATE,
    C_HCO3, C_ETCO2, C_SVO2,
]

VENT_FIELD_NAMES = [C_MECHVENT, C_EXTUBATED]

COMPUTED_FIELD_NAMES = [C_SHOCK_INDEX, C_PAO2_FIO2]

IO_FIELD_NAMES = [
    C_MEDIAN_DOSE_VASO, C_MAX_DOSE_VASO,
    C_INPUT_TOTAL, C_INPUT_STEP,
    C_OUTPUT_TOTAL, C_OUTPUT_STEP,
    C_CUMULATED_BALANCE,
]

DEMOGRAPHICS_FIELD_NAMES = [
    C_GENDER, C_AGE, C_ELIXHAUSER, C_RE_ADMISSION,
    C_DIED_IN_HOSP, C_DIED_WITHIN_48H_OF_OUT_TIME,
    C_MORTA_90, C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH,
]

SAH_FIELD_NAMES = CHART_FIELD_NAMES + LAB_FIELD_NAMES + VENT_FIELD_NAMES

# Sample-and-hold durations in hours — identical to AI-Clinician
SAH_HOLD_DURATION = {f: v for f, v in zip(SAH_FIELD_NAMES, [
    168, 72, 6,
    6, 2, 2,
    2, 2, 2,
    2, 6, 6,
    2, 2, 2,
    2, 2, 2,
    24, 12, 12,
    12, 6, 6,
    6, 6, 6,
    6, 144, 14,
    14, 14, 28,
    28, 28, 28,
    28, 28, 28,
    28, 28, 28,
    28, 28, 28,
    28, 14, 28,
    28, 28, 28,
    28, 28, 28,
    28, 8, 8,
    8, 8, 8,
    8, 8, 8,
    6, 6,
])}
