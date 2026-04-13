# ICU Readmission Pipeline -- Data Preprocessing (Steps 01-08)

## Purpose

Builds **ICUdataset.csv** -- a general ICU cohort with 4-hourly state-action sequences
for training and evaluating a treatment policy aimed at reducing 30-day readmission.

This pipeline is structurally based on the AI-Clinician-MIMICIV / sepsis pipeline but
generalizes from Sepsis-3 patients to all ICU admissions, and replaces in-hospital mortality
with 30-day post-discharge readmission as the primary outcome.

---

## Data Source

**MIMIC-IV** (local PostgreSQL, db=`mimic`). Same schema as the sepsis pipeline.

---

## Final Output

**`data/processed/icu_readmit/ICUdataset.csv`**
- All ICU stays meeting cohort criteria (not Sepsis-3 filtered)
- 4-hour time resolution
- 39 chart features + 45 lab features + 2 ventilation flags + 4 derived scores
- 2 dose-level actions + 10 binary actions (9 drugs + mechvent) + readmission label
- Static: 14 demographic/comorbidity fields + 18 Charlson component flags

---

## Cohort Criteria (Step 03)

| Criterion | Decision |
|-----------|----------|
| Minimum ICU LOS | >= 24 hours |
| Hospital death | EXCLUDED (covers ICU death + post-ICU in-hospital death) |
| Died within 30 days of discharge | EXCLUDED (competing risk for readmission) |
| Obstetric patients | EXCLUDED (admission_location = 'LABOR & DELIVERY') |
| Neonatal patients | EXCLUDED (admission_type = 'NEWBORN') |
| Age minimum | NONE -- all ages included |

Observation window anchored to **ICU intime** (not a derived onset time like sepsis).

---

## Lab Set (36 locked labs + 9 ICU audit additions)

### Locked labs 1-36

| Group | Labs |
|-------|------|
| Electrolytes/Chemistry (10) | Potassium, Sodium, Chloride, Glucose, BUN, Creatinine, Magnesium, Calcium, Ionised_Ca, CO2_mEqL |
| Liver/Enzymes (5) | SGOT, SGPT, Total_bili, Direct_bili, Total_protein |
| Other chemistry (3) | Albumin, Troponin, CRP |
| Hematology (5) | Hb, Ht, RBC_count, WBC_count, Platelets_count |
| Coagulation (4) | PTT, PT, ACT, INR |
| Arterial blood gas (7) | Arterial_pH, paO2, paCO2, Arterial_BE, Arterial_lactate, HCO3, ETCO2 |
| Other (2) | SvO2, Phosphate |

### ICU audit additions (labs 37-45)

| Lab | Coverage | Note |
|-----|----------|------|
| Anion_Gap | 96.5% | Directly measured (itemid 227073) -- no derivation needed |
| Alkaline_Phosphatase | 51.0% | itemid 225612 |
| LDH | 37.0% | itemid 220632 |
| Fibrinogen | 25.8% | itemid 227468 |
| Neuts_pct | 41.9% | WBC differential (itemid 225641) |
| Lymphs_pct | 41.9% | itemid 225642 |
| Monos_pct | 41.9% | itemid 225643 |
| Eos_pct | 41.9% | itemid 225645 |
| Basos_pct | 41.9% | itemid 225644 |

---

## Chart Feature Set (39 vitals/respiratory + 2 ventilation flags)

Positions 1-37 have a 1-to-1 mapping to REF_VITALS itemid groups (used in step_02/step_04).
CAM-ICU (position 38) has a REF_VITALS entry but requires text-to-binary conversion in SQL.
GCS_Total (position 39) is computed in step_05; it has no REF_VITALS entry.

### New vs sepsis pipeline
- GCS components (Eye/Verbal/Motor) replace broken CareVue itemid 198 (100% NaN in MIMIC-IV)
- NIBP_Diastolic (220180, 98.6%) -- completes non-invasive BP triad
- Arterial_BP_Sys/Dia (220050/51, 38%) -- invasive arterial line
- RR_Spontaneous (224689) + RR_Total (224690) -- ventilator RR breakdown
- TidalVolume_Observed (224685) -- actual vs set tidal volume
- Pain_Level (223791, 85.3%) -- numeric pain score
- CAM-ICU (229326, 49.7%) -- delirium screen: text "Positive"->1, "Negative"->0, other->NULL
- GCS_Total -- summed from components in step_05

---

## Action Space (locked)

Time resolution: **4-hour blocs** (same as sepsis pipeline).

### Dose-level (0-4, quartile-binned per cohort)
| Action | Rationale |
|--------|-----------|
| vasopressor_dose | Dose matters -- small vs large norepinephrine is a different decision |
| ivfluid_dose | Dose matters -- 250ml vs 2000ml is a different decision |

### Binary (given/not given in this 4-hour bloc)
| Action | Top drugs |
|--------|-----------|
| antibiotic_active | Vancomycin, Cefazolin, Cefepime, Pip/Tazo, Ceftriaxone |
| anticoagulant_active | Heparin (prophylactic + therapeutic) |
| diuretic_active | Furosemide, Torsemide |
| steroid_active | From prescriptions |
| insulin_active | Regular, Humalog, Glargine |
| opioid_active | Fentanyl, Morphine, Hydromorphone |
| sedation_active | Propofol, Midazolam, Dexmedetomidine |
| transfusion_active | PRBCs, FFP, Platelets, Albumin |
| electrolyte_active | Magnesium Sulfate, KCl, Calcium Gluconate |
| mechvent | Mechanical ventilation (also a state feature) |

---

## Key Differences from Sepsis Pipeline

| Aspect | Sepsis | ICU Readmit |
|--------|--------|-------------|
| Cohort | Sepsis-3 (abx + culture, SOFA >= 2) | All ICU admissions (LOS >= 24h) |
| Time anchor | Sepsis onset time | ICU admission time (intime) |
| Step 03 | Sepsis onset detection | General cohort filter (LOS/death/obstetric/newborn) |
| Outcome | In-hospital death (morta_hosp) | 30-day readmission post-discharge (readmit_30d) |
| Comorbidities | Elixhauser score (31 flags) | Charlson score (18 component flags) |
| Actions | IV fluid + vasopressor (dose 0-4) | 2 dose-level + 10 binary (9 drugs + mechvent) |
| GCS | Estimated from RASS (CareVue issue) | Summed from Eye+Verbal+Motor components |
| Drug extraction | Not in sepsis preprocessing | drugs_mv: 9 binary drug classes from inputevents |

---

## Design Decisions

### Charlson vs Elixhauser
Used Charlson Comorbidity Index (not Elixhauser) because:
- Charlson is the standard for readmission prediction (LACE index uses it directly)
- Better predictive validity for 30-day outcomes vs Elixhauser's mortality focus
- 18 component flags captured individually as static features for RL state

### Mechvent as dual state + action
`mechvent` appears in both `VENT_FIELD_NAMES` (state, via sample-and-hold) and
`BINARY_ACTION_COLS` (action, per-bloc binary). Duplicates are removed in step_06 via
`dict.fromkeys()`. Rationale: initiating or continuing ventilation is a clinical decision
that plausibly affects patient trajectory; it also describes the current state.

### CAM-ICU encoding
MIMIC-IV chartevents stores CAM-ICU as text ("Positive", "Negative", "Unable to Assess").
Converted in the `ce()` SQL query via CASE statement: Positive->1, Negative->0, other->NULL.
NULL forward-filled in step_07 correct_features() with 0 (not assessed = no delirium flag).
Short sample-and-hold duration (4h) since it is reassessed frequently in the ICU.

### discharge_disposition as static feature
Discharge location (home, SNF, rehab, AMA, etc.) is known only at the end of the stay.
It is recorded as a static demographic field (not updated per bloc), carried through all
steps as a confounder, and available in the final dataset for analysis.
It is NOT used as a real-time RL action; however, it may inform the reward model
(discharge to SNF/rehab correlates with readmission risk).

### No age minimum
User confirmed: include all ages. No pediatric filter applied in step_03.
Ages > 150 are clamped to 91 in step_07 (MIMIC anonymization artifact).

### Prior ED visits (LACE E component)
`prior_ed_visits_6m` extracted in demog() -- counts ED visits in the 6 months before
admission. This is the E component of the LACE readmission risk score.

---

## 8-Step Pipeline

### Step 01 -- Extract raw data from PostgreSQL
**Script:** `scripts/icu_readmit/step_01_extract.py`
**Output:** `data/interim/icu_readmit/raw_data/`

Extracts 12 tables from MIMIC-IV. Run order:

| Table | File | Note |
|-------|------|------|
| chartevents | `ce<min><max>.csv` | Batched by stay_id (30M-40M, 1M steps) |
| lab chartevents | `labs_ce.csv` | Lab itemids via chartevents |
| lab events | `labs_le.csv` | `mimiciv_hosp_typed.labevents` (integer hadm_id schema) |
| mechvent | `mechvent.csv` | Per-timestep ventilation flags |
| mechvent_pe | `mechvent_pe.csv` | Ventilation procedure events (start/end times) |
| preadm_fluid | `preadm_fluid.csv` | Pre-admission fluid total |
| fluid_mv | `fluid_mv.csv` | MetaVision inputevents (IV fluids) |
| vaso_mv | `vaso_mv.csv` | Vasopressor inputevents |
| preadm_uo | `preadm_uo.csv` | Pre-admission urine output |
| uo | `uo.csv` | ICU urine output |
| drugs_mv | `drugs_mv.csv` | 9 binary drug classes (new vs sepsis) |
| demog | `demog.csv` | Requires charlson_flags temp table (materialized first) |

`charlson_flags` is materialized as a session-local temp table before any queries run.
Age uses MIMIC-IV's `anchor_age`/`anchor_year` (no `dob` field -- removed for privacy).

**Flags:**
- `--skip-existing`: skip files that already exist (safe to resume after crash)
- `--smoke-test`: limit every query to 5000 rows for quick end-to-end test
- `--only <name>`: run a single table (e.g. `--only demog`)

---

### Step 02 -- Preprocess raw data
**Script:** `scripts/icu_readmit/step_02_preprocess.py`
**Input:** `data/interim/icu_readmit/raw_data/`
**Output:** `data/interim/icu_readmit/intermediates/`

Tasks:
- Remap chartevents itemids: replace raw MIMIC itemids with column index (1..37) using REF_VITALS converter
- Remap lab itemids: replace raw MIMIC itemids with column index (1..45) using REF_LABS converter
- Copy `demog.csv` to intermediates, filling NaN in `morta_hosp`, `morta_90`, `charlson_score`, and all 18 `cc_*` flags with 0
- Compute normalized IV fluid infusion rate (`norm_infusion_rate` = amount / (rate * duration))
- Filter null `icustayid` rows from mechvent, vaso_mv, drugs_mv

---

### Step 03 -- Define ICU cohort
**Script:** `scripts/icu_readmit/step_03_cohort_filter.py`
**Input:** `data/interim/icu_readmit/intermediates/demog.csv`
**Output:**
- `data/interim/icu_readmit/intermediates/icu_cohort.csv`
- `data/interim/icu_readmit/intermediates/cohort_exclusion_counts.csv`

Applies inclusion/exclusion criteria (see Cohort Criteria section above).
Outputs a counts table showing how many stays were dropped at each filter step.
Observation window anchored to `intime` (ICU admission), not a derived onset time.

---

### Step 04 -- Build patient states (unbinned)
**Script:** `scripts/icu_readmit/step_04_patient_states.py`
**Input:**
- `data/interim/icu_readmit/raw_data/ce*.csv` (batched chartevents)
- `data/interim/icu_readmit/intermediates/labs_ce.csv`, `labs_le.csv`
- `data/interim/icu_readmit/intermediates/mechvent.csv`, `mechvent_pe.csv`
- `data/interim/icu_readmit/intermediates/icu_cohort.csv`
**Output:**
- `data/interim/icu_readmit/intermediates/patient_states/patient_states.csv`
- `data/interim/icu_readmit/intermediates/patient_states/icu_stay_bounds.csv`

Collects chart and lab events within each ICU stay window (intime to outtime).
Bins events into 4-hour blocs. One row per (icustayid, timestep).
`icu_stay_bounds.csv`: per-stay first/last timestep + dischtime (used in step_06).

**Checkpointing:** flushes to disk every `--checkpoint-every` patients. Use `--resume`
to skip already-processed stays after a crash.

---

### Step 05 -- Impute patient states
**Script:** `scripts/icu_readmit/step_05_impute_states.py`
**Input:** `patient_states.csv`
**Output:** `patient_states_imputed.csv`

Tasks:
1. **Outlier removal**: clip physiologically impossible values per feature (extended bounds for 45 labs + new vitals)
2. **Compute GCS_Total**: sum GCS_Eye + GCS_Verbal + GCS_Motor (extracted directly from MetaVision itemids 220739/223900/223901 -- no RASS estimation needed unlike sepsis pipeline)
3. **Sample-and-hold forward-fill**: each feature holds its last observed value for its clinical hold duration (e.g. 28h for most labs, 2h for HR/BP, 4h for CAM-ICU)

All SAH durations defined in `columns.py:SAH_HOLD_DURATION`.

---

### Step 06 -- Combine states with actions
**Script:** `scripts/icu_readmit/step_06_states_actions.py`
**Input:**
- `patient_states_imputed.csv`
- `fluid_mv.csv`, `vaso_mv.csv`, `drugs_mv.csv` (from intermediates)
- `icu_stay_bounds.csv`, `demog.csv`, `icu_cohort.csv`
**Output:** `data/interim/icu_readmit/intermediates/states_actions.csv`

Joins patient states with per-bloc action summaries:
- `vasopressor_dose` / `ivfluid_dose`: raw dose aggregates (discretized to 0-4 in step_08)
- 9 binary drug flags (antibiotic/anticoagulant/diuretic/steroid/insulin/opioid/sedation/transfusion/electrolyte): 1 if any dose given in bloc
- `mechvent`: binary, both state and action
- Static demographic fields (gender, age, race, insurance, Charlson, discharge_disposition, etc.) broadcast to every bloc
- `readmit_30d` label broadcast to every bloc from cohort

`_OUTPUT_COLUMNS` uses `dict.fromkeys()` to deduplicate (mechvent appears in both SAH_FIELD_NAMES and BINARY_ACTION_COLS).

---

### Step 07 -- Final imputation
**Script:** `scripts/icu_readmit/step_07_impute_final.py`
**Input:** `states_actions.csv`
**Output:** `states_actions_final.csv`

Tasks:
1. **correct_features()**: clamp ages > 150 -> 91, fill NaN mechvent/CAM-ICU -> 0, fill NaN charlson -> median, fill NaN drug flags -> 0
2. **Linear interpolation**: columns with < 5% missing (fixes small gaps without KNN cost)
3. **KNN imputation**: on CHART_FIELD_NAMES + LAB_FIELD_NAMES (extended 84-column list). Slow step: 30-60 min on full dataset.
4. **Computed features**: P/F ratio, shock index, SOFA, SIRS

Note: KNN operates on the full matrix at once -- no checkpointing needed for this step.

---

### Step 08 -- Build final dataset
**Script:** `scripts/icu_readmit/step_08_build_dataset.py`
**Input:** `states_actions_final.csv`
**Output:** `data/processed/icu_readmit/ICUdataset.csv`

Tasks:
- Quartile-discretize `vasopressor_dose` (0-4) from `max_dose_vaso` per cohort quartiles
- Quartile-discretize `ivfluid_dose` (0-4) from `input_step` per cohort quartiles
- Column renames for RL compatibility: `output_step` -> `output_4hourly`, `input_step` -> `input_4hourly_tev`
- No SOFA filter (not a sepsis pipeline)
- ICU deaths already excluded in step_03

---

## Source Code Map

```
src/careai/icu_readmit/
  columns.py          -- itemids (REF_LABS 45 groups, REF_VITALS 38 groups),
                         column constants, SAH_HOLD_DURATION, BINARY_ACTION_COLS,
                         CHARLSON_FLAG_COLS, DEMOGRAPHICS_FIELD_NAMES
  queries.py          -- SQL queries (PostgreSQL dialect, mimiciv_hosp_typed schema)
                         charlson(), demog(), ce(), labs_ce(), labs_le(),
                         mechvent(), mechvent_pe(), preadm_fluid(), fluid_mv(),
                         vaso_mv(), preadm_uo(), uo(), drugs_mv()
  imputation.py       -- fill_outliers, sample_and_hold (extended bounds for 45 labs)
  utils.py            -- load_csv() with optional null icustayid handling
  rl/                 -- see icu_readmit_rl_pipeline.md

scripts/icu_readmit/
  step_01_extract.py        -- PostgreSQL extraction (12 tables)
  step_02_preprocess.py     -- itemid remapping, NaN fill, infusion rate
  step_03_cohort_filter.py  -- inclusion/exclusion, cohort CSV
  step_04_patient_states.py -- binned states from raw events (checkpointed)
  step_05_impute_states.py  -- outlier removal, GCS sum, sample-and-hold
  step_06_states_actions.py -- join states + actions + demographics + label
  step_07_impute_final.py   -- linear interp + KNN + derived features
  step_08_build_dataset.py  -- discretize doses, rename columns, output CSV
```

---

## Run Commands

```bash
source ../.venv/Scripts/activate

# Step 01 (smoke test first)
PGUSER=postgres PGPASSWORD="Liverpool1892*" \
  python scripts/icu_readmit/step_01_extract.py --smoke-test

# Step 01 (full run, resume-safe)
PGUSER=postgres PGPASSWORD="Liverpool1892*" \
  python scripts/icu_readmit/step_01_extract.py --skip-existing \
  2>&1 | tee logs/step_01_icu_readmit_full.log

# Steps 02-08 (run after step_01 completes)
python scripts/icu_readmit/step_02_preprocess.py \
  2>&1 | tee logs/step_02_icu_readmit.log

python scripts/icu_readmit/step_03_cohort_filter.py \
  2>&1 | tee logs/step_03_icu_readmit.log

python scripts/icu_readmit/step_04_patient_states.py \
    data/interim/icu_readmit/intermediates/patient_states \
    2>&1 | tee logs/step_04_icu_readmit.log

python scripts/icu_readmit/step_05_impute_states.py \
    data/interim/icu_readmit/intermediates/patient_states/patient_states.csv \
    data/interim/icu_readmit/intermediates/patient_states/patient_states_imputed.csv \
    2>&1 | tee logs/step_05_icu_readmit.log

python scripts/icu_readmit/step_06_states_actions.py \
    2>&1 | tee logs/step_06_icu_readmit.log

python scripts/icu_readmit/step_07_impute_final.py \
    data/interim/icu_readmit/intermediates/states_actions.csv \
    data/interim/icu_readmit/intermediates/states_actions_final.csv \
    2>&1 | tee logs/step_07_icu_readmit.log

python scripts/icu_readmit/step_08_build_dataset.py \
    data/interim/icu_readmit/intermediates/states_actions_final.csv \
    data/processed/icu_readmit/ \
    2>&1 | tee logs/step_08_icu_readmit.log
```
