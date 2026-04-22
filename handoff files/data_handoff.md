# Data Chapter Handoff

## 1. Section objective

This section should explain the full data foundation of the active ICU-readmission thesis pipeline.

It should allow the thesis-writing agent to answer, with evidence from the codebase:
- what raw data were used,
- how the cohort was constructed,
- how time was represented,
- how the outcome was defined,
- which state and action variables were available before modeling,
- how missing data were handled,
- what the final sample size was,
- and how the active selected datasets used by the simulator and RL branches were derived.

The scope of this handoff is the active branch from ICU extraction through the selected replay dataset and its directly related derived data objects.

---

## 2. Facts from the codebase

### 2.1 Data source and access conditions

- The source data are **MIMIC-IV** tables stored in a **local PostgreSQL database**.
- The active config file sets:
  - host: `localhost`
  - port: `5432`
  - database: `mimic`
- The pipeline uses both:
  - `mimiciv_icu`
  - `mimiciv_hosp`
  - and `mimiciv_hosp_typed`
- `mimiciv_hosp_typed` is used for at least `admissions` and `labevents` because the typed schema avoids `hadm_id` text/integer mismatch problems.
- Age is computed from MIMIC-IV anonymized fields:
  - `anchor_age`
  - `anchor_year`
  - ICU `intime`
- There is no direct `dob` field in the pipeline logic because MIMIC-IV removes it for privacy.

### 2.2 High-level data workflow

The active data workflow has three layers:

1. **Raw extraction and preprocessing**
- build a general ICU cohort
- extract chart events, labs, ventilation, input/output, and demographic context
- construct 4-hour trajectories

2. **Final longitudinal cohort dataset**
- output file: `ICUdataset.csv`
- this is the broad, pre-selection ICU trajectory dataset

3. **Selected active modeling datasets**
- selected replay dataset for simulator/RL work
- selected static context table
- selected scaler metadata
- selected support-model training data embedded in the selected replay dataset

### 2.3 Raw extracted tables

Step 01 extracts **12 raw/intermediate tables** from MIMIC-IV:

- `ce*.csv`
  - batched chart events by ICU stay range
- `labs_ce.csv`
  - lab-like measurements from `chartevents`
- `labs_le.csv`
  - lab events from `labevents`
- `mechvent.csv`
  - ventilation-related chart events
- `mechvent_pe.csv`
  - ventilation procedure events
- `preadm_fluid.csv`
  - pre-admission fluid
- `fluid_mv.csv`
  - MetaVision fluid input events
- `vaso_mv.csv`
  - vasopressor input events
- `preadm_uo.csv`
  - pre-admission urine output
- `uo.csv`
  - ICU urine output
- `drugs_mv.csv`
  - binary drug-action input events
- `demog.csv`
  - stay-level demographics, comorbidity, admission context, mortality, and readmission labels

The `demog` extraction depends on a session-local temp table:
- `charlson_flags`

### 2.4 Cohort definition

The active ICU cohort is **all ICU admissions** meeting the following criteria:

Inclusion:
- ICU length of stay `>= 24 hours`

Exclusion:
- hospital death
- post-discharge death within 30 days
- obstetric admissions:
  - `admission_location == 'LABOR & DELIVERY'`
- neonatal admissions:
  - `admission_type == 'NEWBORN'`

No age minimum is applied.

The observation window is anchored to:
- **ICU admission time (`intime`)**

not:
- sepsis onset
- discharge time
- or another derived anchor.

### 2.5 Cohort counts

From `demog.csv`:
- initial ICU stays before cohort filtering: `94,458`

From `cohort_exclusion_counts.csv`:
- excluded for LOS `< 24h`: `19,615`
- excluded for hospital death: `11,350`
- excluded for post-discharge death within 30 days: `5,542`
- excluded for obstetric admission location: `0`
- excluded for neonatal admission type: `0`

From the final cohort outputs:
- final ICU stays after filtering: `61,771`
- final cohort retention: about `65.4%` of the original stay-level `demog` table

Readmission prevalence in the final cohort:
- stay-level `readmit_30d` prevalence: `20.69%`

### 2.6 Time granularity and longitudinal structure

The active ICU dataset is organized into **4-hour blocs**.

This is consistent across:
- the broad ICU longitudinal dataset
- the selected replay dataset
- the simulator/control branches

In the final broad ICU dataset:
- total rows: `1,500,857`
- total ICU stays: `61,771`

Stay-length summary from `icu_cohort_summary.csv`:
- mean blocs per stay: `24.34`
- median blocs per stay: `14`
- 25th percentile: `9`
- 75th percentile: `25`
- 90th percentile: `49`
- 95th percentile: `78`
- max: `1,359`

Important interpretation:
- the dataset is strongly longitudinal
- trajectories vary substantially in length
- the RL/simulator branch later truncates/controls horizon, but the underlying cohort data are not fixed-length

### 2.7 Outcome definition

The main outcome is:
- `readmit_30d`

Definition:
- whether the patient has a new hospital admission within 30 days of discharge

This is constructed in the SQL logic from admissions-level timing.

Important exclusions tied to the outcome:
- patients who die in hospital are excluded
- patients who die within 30 days of discharge are excluded as a competing risk

So the active target is:
- **post-discharge 30-day readmission among patients discharged alive and not dying within the 30-day readmission window**

### 2.8 Broad state space before variable selection

The broad final ICU dataset contains:

- 39 chart / bedside / respiratory variables
- 45 lab variables
- 2 ventilation flags
- 4 derived severity or physiology variables
- static demographic and comorbidity fields
- dose-level and binary actions

The broad final CSV columns include:
- identifiers and time:
  - `bloc`
  - `icustayid`
  - `timestep`
- demographics and admission context:
  - `gender`
  - `age`
  - `race`
  - `insurance`
  - `marital_status`
  - `admission_type`
  - `admission_location`
  - `charlson_score`
  - `re_admission`
  - `prior_ed_visits_6m`
  - `drg_severity`
  - `drg_mortality`
  - `discharge_disposition`
- 18 Charlson component flags
- chart and respiratory features
- lab features
- mechanical ventilation and extubation flags
- intake/output variables
- binary drug-action flags
- the final outcome label `readmit_30d`
- derived scores:
  - `PaO2_FiO2`
  - `Shock_Index`
  - `SOFA`
  - `SIRS`
- discretized action variables:
  - `vasopressor_dose`
  - `ivfluid_dose`

### 2.9 Broad action space before variable selection

The broad active action space includes:

Dose-level actions:
- `vasopressor_dose`
- `ivfluid_dose`

Binary actions:
- `antibiotic_active`
- `anticoagulant_active`
- `diuretic_active`
- `steroid_active`
- `insulin_active`
- `opioid_active`
- `sedation_active`
- `transfusion_active`
- `electrolyte_active`
- `mechvent`

Important design choice:
- `mechvent` is treated as both a state-like ongoing support variable and an intervention/action variable

### 2.10 Feature construction and missing-data handling

The active preprocessing pipeline handles missingness in several stages:

1. **Item remapping and cleanup**
- raw MIMIC itemids are mapped into stable feature indices
- null `icustayid` rows are filtered from selected tables

2. **Outlier handling**
- physiologically impossible values are clipped or removed before final imputation

3. **Sample-and-hold forward fill**
- applied to chart, lab, and ventilation features
- hold durations vary by feature family
- examples:
  - many labs use `28h`
  - HR/BP-type variables use shorter holds such as `2h`
  - CAM-ICU uses a short hold because it is reassessed frequently

4. **Derived feature construction**
- GCS total is reconstructed from:
  - `GCS_Eye`
  - `GCS_Verbal`
  - `GCS_Motor`
- additional derived features include:
  - `Shock_Index`
  - `PaO2_FiO2`
  - `SOFA`
  - `SIRS`

5. **Final imputation**
- small-gap linear interpolation
- KNN imputation over the main chart/lab matrix
- explicit fills for:
  - `mechvent`
  - `cam_icu`
  - missing drug flags
  - selected static fields such as Charlson score

### 2.11 Residual missingness after preprocessing

The broad final `ICUdataset.csv` is **not fully dense** across all variables.

Examples of variables with very high remaining missingness:
- `Basos_pct`: `1,500,857` missing rows
- `SVR`: `1,500,857`
- `Eos_pct`: `1,500,857`
- `CI`: `1,495,037`
- `Total_protein`: `1,486,137`
- `ACT`: `1,485,524`

Some fields still have partial missingness:
- `marital_status`: `142,577` missing
- `insurance`: `20,920` missing
- `discharge_disposition`: `2,552` missing

Important contrast:
- the **selected variables used in the active simulator/RL branch are fully observed** in the final broad dataset

Verified zero missingness in the selected active variables:
- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`
- `age`
- `charlson_score`
- `prior_ed_visits_6m`
- `SOFA`
- `readmit_30d`
- `vasopressor_dose`
- `ivfluid_dose`
- `antibiotic_active`
- `diuretic_active`
- `mechvent`

### 2.12 Causal selection outputs relevant for data

The broad ICU dataset is narrowed downstream using causal screening and responsiveness analyses.

Evidence from state-side variable selection:
- time-varying model AUC: `0.6182`
- static model AUC: `0.6961`
- action-summary model AUC: `0.5928`

Top outcome-relevant time-varying variables include:
- `Platelets_count`
- `WBC_count`
- `Hb`
- `HR`
- `PT`
- `PTT`
- `BUN`
- `Glucose`
- `Creatinine`
- `Shock_Index`

Evidence from action-responsiveness screening:
- modelled features: `85`
- responsive features above threshold: `24`

The current selected active set is:

Dynamic states:
- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

Static confounders carried with them:
- `age`
- `charlson_score`
- `prior_ed_visits_6m`

Selected actions:
- `ivfluid`
- `diuretic`
- `vasopressor`
- `mechvent`
- `antibiotic`

### 2.13 Selected replay dataset used by the active simulator/RL branch

The broad ICU dataset is transformed into a selected replay dataset:
- `data/processed/icu_readmit/rl_dataset_selected.parquet`

This is the main derived data object consumed by:
- simulator training
- simulator evaluation
- simulator-side RL
- offline RL comparison

The selected state space has dimension `9`:
- 6 dynamic states
- 3 static confounders

The selected action space has dimension `5`:
- 5 binary actions
- `32` possible combinations

Columns in the selected replay dataset:
- identifiers and split:
  - `icustayid`
  - `bloc`
  - `split`
- outcome and termination:
  - `readmit_30d`
  - `done`
- current state:
  - `s_Hb`
  - `s_BUN`
  - `s_Creatinine`
  - `s_Phosphate`
  - `s_HR`
  - `s_Chloride`
  - `s_age`
  - `s_charlson_score`
  - `s_prior_ed_visits_6m`
- action encoding:
  - `a`
  - `vasopressor_b`
  - `ivfluid_b`
  - `antibiotic_b`
  - `diuretic_b`
  - `mechvent_b`
- reward:
  - `r`
- next state:
  - `s_next_*` for all 9 selected state dimensions

### 2.14 Selected dataset sizes and splits

For `rl_dataset_selected.parquet`:

Total rows:
- `1,500,857`

Total stays:
- `61,771`

Row counts by split:
- train: `1,057,632`
- val: `222,408`
- test: `220,817`

Stay counts by split:
- train: `43,239`
- val: `9,265`
- test: `9,267`

Terminal rows:
- exactly `61,771`
- one terminal row per stay

Stay-level readmission prevalence by split:
- train: `20.65%`
- val: `20.42%`
- test: `21.18%`

### 2.15 Normalization and transformation in the selected dataset

The selected replay dataset is not just a column subset.
It applies train-only transformations:

Dynamic selected variables:
- z-scored on training stays only
- clipped before scaling
- `BUN` and `Creatinine` receive `log1p` transforms before z-scoring

Static selected variables:
- also z-scored on training stays only
- clipped before scaling

Metadata are stored in:
- `data/processed/icu_readmit/scaler_params_selected.json`

Static context used by the active branch is stored separately in:
- `data/processed/icu_readmit/static_context_selected.parquet`

That table contains one row per stay:
- rows: `61,771`
- columns:
  - `icustayid`
  - `split`
  - `age`
  - `charlson_score`
  - `prior_ed_visits_6m`
  - `gender`
  - `race`
  - `re_admission`

### 2.16 Reward-related derived data objects tied to the selected dataset

Two later active data objects are directly derived from the selected dataset:

1. **Severity surrogate training data**
- source rows: the broad ICU dataset
- features:
  - `Hb`
  - `BUN`
  - `Creatinine`
  - `Phosphate`
  - `HR`
  - `Chloride`
- target:
  - `SOFA`

2. **Terminal readmission model training data**
- source rows: terminal rows from `rl_dataset_selected.parquet`
- features:
  - `s_Hb`
  - `s_BUN`
  - `s_Creatinine`
  - `s_Phosphate`
  - `s_HR`
  - `s_Chloride`
  - `s_age`
  - `s_charlson_score`
  - `s_prior_ed_visits_6m`
- target:
  - `readmit_30d`

Terminal-model data size:
- one row per stay
- total terminal rows: `61,771`

Split-specific terminal-row counts:
- train: `43,239`
- val: `9,265`
- test: `9,267`

### 2.17 What the active branch actually uses from the data side

For the final thesis pipeline, the data objects that matter most are:

Broad cohort data:
- `data/processed/icu_readmit/ICUdataset.csv`

Selected replay data:
- `data/processed/icu_readmit/rl_dataset_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`
- `data/processed/icu_readmit/static_context_selected.parquet`

Support-model data products:
- `models/icu_readmit/severity_selected/`
- `models/icu_readmit/terminal_readmit_selected/`

Everything after that uses these derived data objects rather than going back to the broad raw ICU table design.

---

## 3. Interpretation

### 3.1 What the data pipeline achieves

The data pipeline is doing more than ordinary preprocessing.

It turns raw MIMIC-IV ICU data into:
- a broad longitudinal ICU cohort,
- a clinically interpretable intervention space,
- and a selected replay dataset that is directly usable for simulation and RL.

This is important because the thesis is not built around a single flat prediction table.
It is built around **trajectories**.

### 3.2 Why the cohort construction matters

The outcome is readmission after live discharge, not mortality.

That makes the cohort definition especially important:
- excluding in-hospital death prevents mixing discharge failure with readmission
- excluding death within 30 days of discharge removes a competing-risk problem
- anchoring to ICU `intime` makes the longitudinal design clinically interpretable

### 3.3 Why the broad dataset and the selected dataset should be clearly separated in the thesis

The thesis should distinguish between:

1. **the broad ICUdataset**
- large, rich, clinically detailed
- contains many variables not used in the final RL branch

2. **the selected replay dataset**
- purpose-built for the final active simulator/RL workflow
- much lower-dimensional
- directly supports the main comparison in the thesis

This distinction matters because the main experiments are not run on the full broad variable set.

### 3.4 Why missingness should be described carefully

The broad cohort dataset is not fully dense.

That is not necessarily a weakness if it is described correctly:
- many peripheral or low-coverage ICU variables remain sparse
- the active selected variables used for the final simulator/RL branch are fully observed after preprocessing

So the correct narrative is:
- the broad dataset remains partially sparse,
- but the final active modeling subset is fully usable.

### 3.5 What the selected dataset means scientifically

The selected replay dataset is the point where the thesis stops being a generic ICU data project and becomes the final intervention-learning setup.

It encodes:
- the selected state variables,
- the selected actions,
- the split structure,
- the transition structure,
- and the reward-related quantities needed downstream.

That is the real data interface used by the active simulators and policy-learning branches.

---

## 4. Evidence

### 4.1 Core pipeline and config

- `configs/icu_readmit.yaml`
- `docs/icu_readmit_pipeline.md`
- `scripts/icu_readmit/step_01_extract.py`
- `scripts/icu_readmit/step_02_preprocess.py`
- `scripts/icu_readmit/step_03_cohort_filter.py`
- `scripts/icu_readmit/step_04_patient_states.py`
- `scripts/icu_readmit/step_05_impute_states.py`
- `scripts/icu_readmit/step_06_states_actions.py`
- `scripts/icu_readmit/step_07_impute_final.py`
- `scripts/icu_readmit/step_08_build_dataset.py`

### 4.2 Schema, variable lists, and preprocessing constants

- `src/careai/icu_readmit/columns.py`
- `src/careai/icu_readmit/queries.py`

### 4.3 Causal selection and selected-set definition

- `docs/step_09_state_action_recommendation.md`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance/variable_selection.json`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness/step_21b_transition_responsiveness.json`

### 4.4 Selected replay dataset and support-model data objects

- `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`
- `scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py`
- `scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py`
- `data/processed/icu_readmit/rl_dataset_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`
- `data/processed/icu_readmit/static_context_selected.parquet`

### 4.5 Generated quantitative evidence used in this handoff

- `data/interim/icu_readmit/intermediates/demog.csv`
- `data/interim/icu_readmit/intermediates/cohort_exclusion_counts.csv`
- `data/interim/icu_readmit/intermediates/icu_cohort.csv`
- `data/processed/icu_readmit/ICUdataset.csv`
- `data/processed/icu_readmit/icu_cohort_summary.csv`
- `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`
- `reports/icu_readmit/terminal_readmit_selected/terminal_readmit_selected_metrics.json`

---

## 5. Caveats

- The broad `ICUdataset.csv` still contains substantial residual missingness in several peripheral variables.
- The active thesis branch does not rely on all of those variables, but the thesis should not imply that the entire broad dataset is fully dense.
- Some older documentation in the repo still refers to earlier numbering or older branch names. The active data story should be based on the current selected pipeline, not the legacy broad/tier2/discharge RL branch.
- The cohort exclusion counts are reported per criterion, but the exclusion criteria are combined; the counts should not be naively summed as if they were mutually exclusive.
- The row-level prevalence of `readmit_30d` in the broad longitudinal dataset is larger than the stay-level prevalence because the label is repeated across all blocs within a stay.
- The selected replay dataset’s reward field `r` reflects preprocessing choices used for downstream modeling. If the thesis discusses reward design, that belongs in the methods chapter rather than the data chapter.
- The severity surrogate and terminal readmission model are downstream support-model data products. They are relevant to the data story, but they are not primary cohort-construction steps.

---

## 6. Suggested thesis text ingredients

- State clearly that the source is MIMIC-IV in local PostgreSQL form, with derived typed hospital tables used where necessary for schema compatibility.
- Explain that the cohort is a general ICU cohort, not a sepsis-only cohort.
- State the inclusion and exclusion criteria explicitly.
- Emphasize that the unit of analysis is the 4-hour ICU bloc within a stay.
- Define the main outcome as 30-day readmission after live discharge.
- Distinguish between:
  - the broad final ICU trajectory dataset,
  - and the selected replay dataset used in the final modeling branch.
- Report the final broad cohort sample size:
  - `61,771` ICU stays
  - `1,500,857` 4-hour rows
- Report the final stay-level readmission prevalence:
  - about `20.7%`
- Explain that broad preprocessing includes sample-and-hold, interpolation, KNN imputation, and derived physiology scores.
- Note that residual missingness remains in several peripheral broad variables, but the selected variables used in the final active pipeline are fully observed after preprocessing.
- Explain that the active selected dataset reduces the broad feature/action space to:
  - 6 dynamic states
  - 3 static confounders
  - 5 binary actions
- Report the selected replay dataset split sizes and stay counts.
- Mention that the selected dataset is the direct data interface used by the simulator and RL experiments.
- Mention that static context and normalization metadata are stored separately and are part of the active data infrastructure.
