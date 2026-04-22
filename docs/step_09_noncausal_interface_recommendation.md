# Step 09 Non-Causal Interface Recommendation

Purpose:
- define a broad predictive state/action interface as a side track to the active
  selected-causal Step 09 branch
- preserve the current causal workflow unchanged
- create a much wider modeling interface for a non-causal transformer branch

Context:
- the active Step 09 recommendation narrows the problem through causal
  relevance and controllability
- this side track intentionally removes those structural restrictions
- the goal is not "include everything blindly", but "keep the interface broad
  while removing obvious leakage, duplicates, zero-variance columns, and
  extremely sparse defaults"

This document is paired with:
- `scripts/icu_readmit/step_09_state_action_selection/step_06_build_noncausal_interface.py`

Outputs:
- `data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_dataset.parquet`
- `data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_spec.json`
- `data/processed/icu_readmit/step_09_noncausal_interface/noncausal_interface_missingness.csv`

## Main design

The non-causal branch keeps:
- broad dynamic physiology
- broad static context
- broad binary actions
- `SOFA` as an auxiliary support / reward column

The non-causal branch does not use:
- identifiers as model features
- terminal/outcome leakage variables as model features
- duplicated representations where one cleaner version exists
- zero-variance columns
- extremely sparse specialty variables as default inputs

## Keep

### Dynamic state

Vitals / monitoring / respiratory:
- `HR`
- `SysBP`
- `MeanBP`
- `DiaBP`
- `NIBP_Diastolic`
- `Arterial_BP_Sys`
- `Arterial_BP_Dia`
- `RR`
- `RR_Spontaneous`
- `RR_Total`
- `SpO2`
- `Temp_C`
- `Interface`
- `FiO2_1`
- `O2flow`
- `PEEP`
- `TidalVolume`
- `TidalVolume_Observed`
- `MinuteVentil`
- `PAWmean`
- `PAWpeak`
- `PAWplateau`
- `Pain_Level`
- `GCS`
- `mechvent`

Labs:
- `Potassium`
- `Sodium`
- `Chloride`
- `Glucose`
- `BUN`
- `Creatinine`
- `Magnesium`
- `Calcium`
- `Ionised_Ca`
- `CO2_mEqL`
- `SGOT`
- `SGPT`
- `Total_bili`
- `Direct_bili`
- `Albumin`
- `Hb`
- `Ht`
- `RBC_count`
- `WBC_count`
- `Platelets_count`
- `PTT`
- `PT`
- `INR`
- `Arterial_pH`
- `paO2`
- `paCO2`
- `Arterial_BE`
- `Arterial_lactate`
- `HCO3`
- `Phosphate`
- `Anion_Gap`
- `Alkaline_Phosphatase`
- `Fibrinogen`
- `Neuts_pct`
- `Lymphs_pct`
- `Monos_pct`

### Static context
- `gender`
- `age`
- `Weight_kg`
- `race`
- `insurance`
- `marital_status`
- `admission_type`
- `admission_location`
- `charlson_score`
- `re_admission`
- `prior_ed_visits_6m`

### Actions

Binary actions used in the side-track interface:
- `vasopressor_active`
- `ivfluid_active`
- `antibiotic_active`
- `anticoagulant_active`
- `diuretic_active`
- `insulin_active`
- `opioid_active`
- `sedation_active`
- `transfusion_active`
- `electrolyte_active`
- `mechvent_active`

Derived binary encodings:
- `vasopressor_active = 1` if `vasopressor_dose > 0` or `max_dose_vaso > 0`
- `ivfluid_active = 1` if `ivfluid_dose > 0` or `input_4hourly_tev > 0`
- `mechvent_active = mechvent`

### Auxiliary support columns
- `SOFA`
- `readmit_30d`

Interpretation:
- `SOFA` is kept available for reward construction and monitoring
- `readmit_30d` is kept as an outcome label, not an input feature

## Exclude

### Identifiers / ordering only
- `bloc`
- `icustayid`
- `timestep`

These are retained structurally in the materialized dataset for sequence
construction, but they are not part of the model feature lists.

### Outcome / terminal leakage
- `readmit_30d`
- `discharge_disposition`
- `died_in_hosp`
- `died_within_48h_of_out_time`
- `delay_end_of_record_and_discharge_or_death`

### Duplicated representations
- `Temp_F`
- `FiO2_100`
- `GCS_Eye`
- `GCS_Verbal`
- `GCS_Motor`

Reason:
- keep `Temp_C` instead of `Temp_F`
- keep `FiO2_1` instead of `FiO2_100`
- keep total `GCS` instead of the components

### Zero-variance columns in the current processed dataset
- `extubated`
- `cam_icu`
- `drg_severity`
- `drg_mortality`
- `steroid_active`

These were confirmed to have only a single observed value in the current
`ICUdataset.csv` and therefore carry no predictive information.

### Extremely sparse default exclusions
- `Basos_pct`
- `Eos_pct`
- `SVR`
- `CI`
- `Total_protein`
- `ACT`
- `CRP`
- `PAPmean`
- `PAPdia`
- `PAPsys`
- `SvO2`
- `ETCO2`

Reason:
- they are too sparse to justify default inclusion in the first broad branch

## Optional but not in the default side-track dataset

- `CVP`
- `Troponin`
- `LDH`
- `PaO2_FiO2`
- `Shock_Index`
- `SIRS`
- `input_total`
- `output_total`
- `output_4hourly`
- `cumulated_balance`
- `median_dose_vaso`

These are not unusable. They are left out of the first side-track materialization
to keep the branch broad but still disciplined.

## SOFA note

The broader dataset contains enough raw information to support a cleaner SOFA-based
reward design than the selected-causal branch.

SOFA-relevant raw inputs retained here include:
- `paO2`
- `FiO2_1`
- `Platelets_count`
- `Total_bili`
- `MeanBP`
- `GCS`
- `Creatinine`
- vasopressor activity

Therefore:
- `PaO2_FiO2` does not need to be a default input feature
- `SOFA` can be kept as an auxiliary support column rather than a primary
  transformer input

## Intended role of this side track

This side track is the non-causal counterpart to the active selected-causal
Step 09 branch.

The intended next steps are:
- Step 10 side-track preprocessing from the broad non-causal interface
- Step 11 side-track transformer simulator training without structural causal masks

The current selected-causal Step 09 outputs remain unchanged and should continue
to be treated as the canonical causal branch.
