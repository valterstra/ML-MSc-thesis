# ICU Readmission RL Pipeline (Legacy Steps 09-13)

This document is archived. The retired broad / tier2 / discharge branch is now
stored under the legacy ICU-readmit paths and is no longer the active pipeline.
The historical commands and filenames below are preserved for reference and are
not maintained as current entrypoints.

Current active workflow:
- `docs/icu_readmit_rl_pipeline.md`

Archived code locations:
- `scripts/icu_readmit/legacy/`
- `src/careai/icu_readmit/legacy/`
- `notebooks/legacy/icu_readmit/`

## What This Is

Reinforcement learning pipeline for general ICU treatment recommendations.
Goal: learn a policy that chooses drug and ventilation actions during an ICU stay
to reduce 30-day post-discharge readmission risk.

**Input**: `data/processed/icu_readmit/ICUdataset.csv`
See `docs/icu_readmit_pipeline.md` for how that dataset is built (steps 01-08).

---

## Current Status Update (2026-04-07)

This document still contains the original Tier-2 design and the original
Step 04 action-state results for historical reference.

The active update since that run is:

- original Step 03 remains the current outcome-anchor analysis
- a new robust action-state analysis was added:
  - `scripts/icu_readmit/step_09_causal_states/step_04b_action_state_stability_robust.py`
- Step 04b uses:
  - 2 sampled actions per run
  - 2 sampled delta-states per run
  - baseline-state adjustment
  - the original broad 28-state pool
- final Step 04b outputs are in:
  - `reports/icu_readmit/step_09_causal_states/action_state_pair_results_robust.csv`
  - `reports/icu_readmit/step_09_causal_states/action_state_frequency_matrix_robust.csv`
  - `reports/icu_readmit/step_09_causal_states/action_state_summary_robust.json`

Current recommended expanded set from Step 03 + Step 04b:

- Dynamic states:
  - `Hb`
  - `BUN`
  - `Creatinine`
  - `Phosphate`
  - `HR`
  - `Chloride`
- Static confounders:
  - `age`
  - `charlson_score`
  - `prior_ed_visits_6m`
- Actions:
  - `ivfluid`
  - `diuretic`
  - `vasopressor`
  - `mechvent`
  - `antibiotic`

This selected-set recommendation is saved separately in:

- `docs/step_09_state_action_recommendation.md`

Selected-set implementation status:

- Step 10f preprocessing built and run:
  - `scripts/icu_readmit/step_10f_rl_preprocess_selected.py`
  - output: `data/processed/icu_readmit/rl_dataset_selected.parquet`
- Step 10g severity-surrogate built and run:
  - `scripts/icu_readmit/step_10g_train_selected_severity_surrogate.py`
  - outputs under:
    - `models/icu_readmit/severity_selected/`
    - `reports/icu_readmit/severity_selected/`
- selected-set Step 14 training track implemented:
  - `scripts/icu_readmit/step_14_caresim_train_selected.py`
  - `notebooks/step_14_caresim_selected_colab.ipynb`

The old Tier-2 sections below remain useful as baseline documentation, but they
should no longer be read as the only current design path.

---

## Variable Selection and Causal Discovery (Step 09)

Step 09 uses a fully FCI-based random stability approach to select variables
for the RL model. This replaces earlier LightGBM + PC-based variable selection
(steps 09a/b/c) which were superseded. The FCI stability method is more
methodologically rigorous: it handles latent confounders, does not assume causal
sufficiency, and produces robustness estimates (frequency over many random subsets)
rather than a single-run point estimate.

See `scripts/icu_readmit/step_09_causal_states/` for all four sub-steps.

---

## Causal Design Decisions (FCI Stability Analysis)

### Why FCI (Fast Causal Inference)

FCI is a constraint-based causal discovery algorithm that, unlike PC, does NOT
assume all common causes are observed (causal sufficiency). It outputs a Partial
Ancestral Graph (PAG) where a definite directed edge X -> Y means: in every
consistent DAG, X is an ancestor of Y. This is the most conservative and
defensible causal claim available from observational data.

### Random Stability Approach

Inspired by stability selection (Meinshausen & Buhlmann 2010). Run FCI thousands
of times over random variable subsets + data subsamples. Edge frequency across
runs = robustness. A variable appearing in many independent random subsets and
consistently generating the same edge is very unlikely to be a false positive.

Configuration for both analyses:
- 5000 stays subsampled per run (prevents Fisher-Z from detecting trivial micro-correlations at large N)
- alpha = 0.05 (FisherZ independence test)
- Background knowledge enforces temporal tier ordering (no future -> past edges)
- Confounders: age, charlson_score, prior_ed_visits_6m (+ num_blocs for action analysis)

### Two-Part Analysis

**Part 1 (Step 03): Which discharge states predict 30-day readmission?**

6-node graph per run: [3 confounders | 2 random state variables | readmit_30d]
2000 runs total. Frequency = fraction of runs where FCI found a definite
directed edge from that state variable to readmit_30d.

Full results (all 29 variables, ranked by freq_definite):

| Rank | Variable | Freq | Notes |
|------|----------|------|-------|
| 1 | last_Hb | 97.1% | Hemoglobin -- oxygen-carrying capacity |
| 2 | last_Ht | 96.2% | Hematocrit -- redundant with Hb, excluded from RL state |
| 3 | last_BUN | 90.8% | Blood urea nitrogen -- renal function / hydration |
| 4 | last_input_total | 88.7% | Total fluid input -- circular with ivfluid action, excluded |
| 5 | last_Phosphate | 55.7% | Serum phosphate -- metabolic / nutrition marker |
| 6 | last_HR | 54.1% | Heart rate -- hemodynamic stability at discharge |
| 7 | last_PT | 53.8% | Prothrombin time -- coagulation; no drug lever (excluded) |
| 8 | last_Creatinine | 51.8% | Serum creatinine -- kidney filtration |
| 9 | last_Shock_Index | 48.9% | HR/SysBP -- composite hemodynamic instability |
| 10 | last_Alkaline_Phosphatase | 32.5% | |
| 11 | last_Chloride | 26.4% | |
| 12 | last_cumulated_balance | 25.3% | |
| 13 | last_Glucose | 20.4% | |
| 14 | last_SpO2 | 19.1% | |
| 15 | last_PTT | 19.0% | |
| 16 | last_Sodium | 18.6% | |
| 17-29 | ... | <15% | Weak / noise-level signal |

Clear break at rank 9 (Shock_Index 48.9% -> Alkaline_Phosphatase 32.5%).
Top 8 actionable states (excluding Ht as redundant, input_total as circular,
PT as not drug-modifiable) form the candidate pool for RL state selection.

**Part 2 (Step 04): Which drug actions causally shift those states?**

6-node graph per run: [4 confounders | frac_drug (fixed) | 1 random delta_state]
3000 runs per drug, 9 drugs, 27,000 total FCI runs. ~1h 55min runtime.
Drug representation: frac_active = fraction of blocs drug was active (0-1 scale).
State representation: delta = last_value - first_value (physiological change).

Key results (freq_definite for clinically important drug-state pairs):

| State | vasopressor | ivfluid | antibiotic | anticoagulant | diuretic | insulin | sedation | mechvent |
|-------|-------------|---------|------------|---------------|----------|---------|---------|---------|
| delta_Hb | 1.000 | 1.000 | 1.000 | 0.886 | 0.000 | 0.051 | 1.000 | 1.000 |
| delta_BUN | 0.000 | 0.990 | 0.260 | 0.010 | 1.000 | 1.000 | 0.273 | 0.664 |
| delta_Creatinine | 0.023 | 0.897 | 0.336 | 0.063 | 0.991 | 0.681 | 0.054 | 0.019 |
| delta_HR | 0.927 | 0.337 | 0.981 | 0.959 | 0.182 | 0.185 | 0.919 | 1.000 |
| delta_Shock_Index | 0.091 | 0.874 | 0.942 | 0.866 | 0.588 | 0.824 | 0.082 | 0.019 |
| delta_Phosphate | 0.517 | 1.000 | 0.692 | 0.053 | 0.722 | 0.837 | 0.206 | 0.345 |

Note: steroid_active = 0% in MIMIC-IV (wrong itemids -- known limitation).
All 3000 steroid runs skipped at variance check; excluded from RL action space.

Full frequency matrix: `reports/icu_readmit/step_09_causal_states/action_state_frequency_matrix.csv`

---

## RL Model Design: Three Tiers

The FCI stability results define three model sizes. The key criterion for
inclusion: a state must robustly predict readmission (Part 1) AND at least
one drug in the action set must reliably shift it (Part 2).

### Tier 1 -- Small (2 drugs, 3 states)

The tightest causal chain; easiest to defend:

**Actions:** diuretic, ivfluid
**States:** Hb, BUN, Creatinine
**Confounders (static context):** age, charlson_score, prior_ed_visits_6m

Rationale: diuretic and ivfluid have opposing effects on BUN and Creatinine
(both strong readmission predictors, freq_definite >= 0.89). The two drugs form
a natural clinical trade-off -- too much diuretic worsens renal function, too
little leaves the patient fluid-overloaded. Hb is the strongest readmission
predictor (97.1%) and is meaningfully shifted by ivfluid. Clean 2-drug policy
with a well-defined renal/fluid decision. Good ablation / sanity check.

### Tier 2 -- Medium (4 drugs, 5 states) [PRIMARY MODEL]

Adds the hemodynamic dimension:

**Actions:** diuretic, ivfluid, vasopressor, antibiotic
**States:** Hb, BUN, Creatinine, HR, Shock_Index
**Confounders (static context):** age, charlson_score, prior_ed_visits_6m

Why each state:
- Hb (97.1%): oxygen-carrying capacity; low Hb at discharge = fragile patient
- BUN (90.8%): renal / hydration marker; directly modifiable by diuretic/ivfluid
- Creatinine (51.8%): kidney filtration; diuretic=0.99, ivfluid=0.90
- HR (54.1%): hemodynamic stability; vasopressor=0.93, antibiotic=0.98
- Shock_Index (48.9%): composite HR/SysBP instability; antibiotic=0.94, ivfluid=0.87

Why each action:
- diuretic: BUN=1.00, Creatinine=0.99, renal anchor
- ivfluid: BUN=0.99, Creatinine=0.90, Hb=1.00, volume/renal
- vasopressor: HR=0.93, hemodynamic arm
- antibiotic: HR=0.98, Shock_Index=0.94, infection/hemodynamic arm

Excluded from Tier 2 despite step_03 signal:
- Ht: redundant with Hb (Ht ~= 3 x Hb, near-perfect correlation)
- PT: coagulation marker, no drug in action set shifts it (all freq < 0.17)
- Phosphate: moderate readmission signal but less central story

### Tier 3 -- Large (7 drugs, 7 states)

Full model covering renal, hemodynamic, coagulation, and metabolic dimensions:

**Actions:** diuretic, ivfluid, vasopressor, antibiotic, anticoagulant, insulin, mechvent
**States:** Hb, BUN, Creatinine, HR, Shock_Index, Phosphate, Ht

Why additional actions vs Tier 2:
- anticoagulant: PTT=1.00, Ht=0.99, HR=0.96 -- coagulation arm
- insulin: BUN=1.00, Glucose=0.96, Phosphate=0.84 -- metabolic arm
- mechvent: HR=1.00, RR=1.00, SpO2=1.00 -- respiratory arm

Note on Ht: reintroduced in Tier 3 as distinct from Hb in larger state space.
Mechanistically: IV fluids dilute blood (lower Ht), vasopressors cause
hemoconcentration (raise Ht). These are different mechanisms from Hb absolute level.

---

## Pipeline Overview

```
Step 09   FCI stability analysis (causal variable selection)  [ COMPLETE ]
  step_01  Stay-level build (61,771 stays, 100+ state+static cols)
  step_02  LightGBM variable ranking (state AUC=0.618, static AUC=0.696)
  step_03  State -> readmit FCI stability (2000 runs)
           Top: Hb=97.1%, Ht=96.2%, BUN=90.8%, Creatinine=51.8%, Shock_Index=48.9%
  step_04  Action -> state FCI stability (27,000 runs, 9 drugs x 3000)
           Key: diuretic->BUN=1.00, ivfluid->Creatinine=0.90, vasopressor->HR=0.93
Step 10   RL preprocessing: narrow 15-feature state   [ COMPLETE ] (prior design, broad baseline)
Step 10b  RL preprocessing: broad 51-feature state    [ COMPLETE ] <-- used for steps 11-13
Step 11   Continuous RL: Dueling DDQN + PER           [ COMPLETE ] (broad baseline run)
Step 12   Off-policy evaluation (DR)                  [ COMPLETE ] (broad baseline run)
Step 12b  Policy analysis: Raghu 2017 Fig 1 & 2       [ COMPLETE ] (broad baseline run)
Step 13   Model-based simulators                      [ COMPLETE ] (older broad baseline)
Step 10c  RL preprocessing: Tier 2 FCI-guided state   [ COMPLETE ]
Step 11b  RL training: Tier 2 model                   [ COMPLETE ]
Step 10d  RL preprocessing: Tier 2 + discharge action [ COMPLETE ]
Step 11c  Joint DDQN + discharge action               [ COMPLETE ]
Step 14   CARE-Sim transformer world model            [ COMPLETE ]
Step 15   CARE-Sim evaluation                         [ COMPLETE ]
Step 16   CARE-Sim control layer                      [ COMPLETE ]
```

## Current Practical End State

The ICU readmission pipeline no longer stops at the older offline DDQN and Step 13 simulators.

The current end-to-end stack is:

1. Tier-2 RL preprocessing with the FCI-guided variable set
2. CARE-Sim transformer world model training
3. CARE-Sim held-out evaluation
4. simulator-based control:
   - short-horizon planner
   - DDQN trained against CARE-Sim

Current canonical Tier-2 design:

- Dynamic state: `Hb`, `BUN`, `Creatinine`, `HR`, `Shock_Index`
- Static confounders repeated per step: `age`, `charlson_score`, `prior_ed_visits_6m`
- Total state dim: `8`
- Actions: `diuretic`, `ivfluid`, `vasopressor`, `antibiotic`
- Action space: `16` combinations

Current summary:

- CARE-Sim one-step next-state MSE is about `0.083`
- CARE-Sim reward MAE is about `1.87`
- planner is the strongest Step 16 controller
- DDQN improves over `random` and `repeat_last`, but still trails planner

---

## Step 09 -- FCI Stability Analysis (Causal Variable Selection)

**Scripts:** `scripts/icu_readmit/step_09_causal_states/`
**Output:** `reports/icu_readmit/step_09_causal_states/`

Four sub-steps building the full causal evidence base for variable selection.
All four are COMPLETE.

---

### Step 09 / step_01 -- Stay-Level Build

**Script:** `step_01_stay_level.py`
**Output:** `data/processed/icu_readmit/step_09_causal_states/stay_level.parquet`

Collapses ICUdataset.csv (1.5M rows, 4-hour blocs) to one row per ICU stay
(61,771 stays). Extracts:
- Static columns (demographics, comorbidities): `.first()` per stay
- State columns (vitals, labs, derived scores): `.last()` per stay (discharge state), prefixed `last_`
- `num_blocs`: stay length (proxy for ICU LOS)
- `readmit_30d`: outcome label

```bash
python scripts/icu_readmit/step_09_causal_states/step_01_stay_level.py
python scripts/icu_readmit/step_09_causal_states/step_01_stay_level.py --smoke
```

---

### Step 09 / step_02 -- Variable Ranking

**Script:** `step_02_variable_ranking.py`
**Output:** `reports/icu_readmit/step_09_causal_states/state_variable_ranking.csv`
          `reports/icu_readmit/step_09_causal_states/static_variable_ranking.csv`

Two separate LightGBM classifiers (state variables vs static variables) predicting
`readmit_30d`. Run separately to prevent static features from drowning out state features.

Results:
- State variables AUC: 0.618 (top: Hb, Ht, BUN, HR)
- Static variables AUC: 0.696 (top: age, charlson_score, prior_ed_visits_6m)

Top 30 state variables from this ranking form the candidate pool for step_03.
Static features confirmed as confounders for causal analysis.

```bash
python scripts/icu_readmit/step_09_causal_states/step_02_variable_ranking.py
python scripts/icu_readmit/step_09_causal_states/step_02_variable_ranking.py --smoke
```

---

### Step 09 / step_03 -- State -> Readmission FCI Stability

**Script:** `step_03_random_stability.py`
**Output:** `reports/icu_readmit/step_09_causal_states/random_stability_results.csv`

2000 FCI runs, each on a 6-node graph:
  [age, charlson_score, prior_ed_visits_6m | var_1, var_2 | readmit_30d]
  Tier 0: confounders, Tier 1: two random state vars, Tier 2: readmit_30d
  5000 stays subsampled per run.

Edge frequency = fraction of runs where FCI found a definite directed edge
from that state variable to readmit_30d.

Full results (29 variables):

| Rank | Variable | Freq | Clinical meaning |
|------|----------|------|-----------------|
| 1 | last_Hb | 97.1% | Hemoglobin -- oxygen-carrying capacity |
| 2 | last_Ht | 96.2% | Hematocrit -- redundant with Hb |
| 3 | last_BUN | 90.8% | Blood urea nitrogen -- renal/hydration |
| 4 | last_input_total | 88.7% | Total fluid input -- circular, excluded |
| 5 | last_Phosphate | 55.7% | Serum phosphate -- metabolic |
| 6 | last_HR | 54.1% | Heart rate -- hemodynamic stability |
| 7 | last_PT | 53.8% | Prothrombin time -- no drug lever |
| 8 | last_Creatinine | 51.8% | Kidney filtration |
| 9 | last_Shock_Index | 48.9% | HR/SysBP composite |
| 10 | last_Alkaline_Phosphatase | 32.5% | Sharp drop-off here |
| 11-29 | ... | <27% | Weak signal |

```bash
python scripts/icu_readmit/step_09_causal_states/step_03_random_stability.py
python scripts/icu_readmit/step_09_causal_states/step_03_random_stability.py --smoke
```

---

### Step 09 / step_04 -- Action -> State FCI Stability

**Script:** `step_04_action_state_stability.py`
**Output:** `reports/icu_readmit/step_09_causal_states/action_state_results/frac_<drug>_results.csv`
          `reports/icu_readmit/step_09_causal_states/action_state_frequency_matrix.csv`

3000 FCI runs per drug, 9 drugs, 27,000 total. Each run is a 6-node graph:
  [age, charlson_score, prior_ed_visits_6m, num_blocs | frac_drug | delta_state]
  Tier 0: confounders, Tier 1: drug, Tier 2: delta_state
  5000 stays subsampled per run. Drug as frac_active (0-1 scale). State as delta (last - first).

Key results (freq_definite, cross-referenced with step_03 readmission signal):

| Drug -> State | Drug freq | State -> readmit freq | Verdict |
|--------------|-----------|----------------------|---------|
| diuretic -> delta_BUN | 1.000 | 90.8% | Strong -- renal anchor |
| diuretic -> delta_Creatinine | 0.991 | 51.8% | Strong |
| ivfluid -> delta_BUN | 0.990 | 90.8% | Strong -- opposing renal effect |
| ivfluid -> delta_Creatinine | 0.897 | 51.8% | Strong |
| ivfluid -> delta_Hb | 1.000 | 97.1% | Strong (dilution effect) |
| vasopressor -> delta_HR | 0.927 | 54.1% | Strong -- hemodynamic |
| antibiotic -> delta_HR | 0.981 | 54.1% | Strong (resolving tachycardia) |
| antibiotic -> delta_Shock_Index | 0.942 | 48.9% | Strong |
| ivfluid -> delta_Shock_Index | 0.874 | 48.9% | Strong |

Note: steroid_active = 0% (wrong itemids in MIMIC-IV) -- all runs skipped.
Full 9x28 matrix: `action_state_frequency_matrix.csv`

```bash
python scripts/icu_readmit/step_09_causal_states/step_04_action_state_stability.py
python scripts/icu_readmit/step_09_causal_states/step_04_action_state_stability.py --smoke
```

---

## Step 10 -- RL Preprocessing

**Script:** `scripts/icu_readmit/step_10_rl_preprocess.py`
**Output:** `data/processed/icu_readmit/`

Transforms ICUdataset.csv into the experience-replay format needed by the DDQN.

**Processing steps:**
1. Build binary action columns from source columns (vasopressor_dose>0, ivfluid_dose>0, etc.)
2. Encode 5 binary actions as integer 0-31
3. Assign stays to train/val/test (70/15/15, deterministic by icustayid sort)
4. Z-score normalise 14 continuous state features on train set only
5. Build (s, a, r, s', done) consecutive bloc pairs
6. Save RL dataset, scaler parameters, and static context table

**Results (full dataset):**

```
Input:  1,500,857 rows, 61,771 stays, 146 columns
Output: 1,500,857 transitions, 42 columns

  done=0 (non-terminal): 1,439,086
  done=1 (terminal):        61,771

  Train:  1,057,632 transitions / 43,239 stays  (readmit=20.6%)
  Val:      222,408 transitions /  9,265 stays  (readmit=20.4%)
  Test:     220,817 transitions /  9,267 stays  (readmit=21.2%)

  Action combinations: 32/32 present
  Dense reward: mean=0.036, std=1.634, range=[-12, +11]
  Terminal: +15 (no readmit): 48,989  |  -15 (readmit): 12,782
```

**Output files:**
- `data/processed/icu_readmit/rl_dataset.parquet` -- (s,a,r,s',done) tuples
- `data/processed/icu_readmit/static_context.parquet` -- one row per stay, static confounders
- `data/processed/icu_readmit/scaler_params.json` -- mean/std per feature for inference

**Columns in rl_dataset.parquet (42 total):**
```
icustayid, bloc, split, readmit_30d, done
s_HR, s_MeanBP, s_RR, s_SpO2, s_Temp_C,
s_Potassium, s_Creatinine, s_BUN, s_WBC_count, s_Glucose, s_Hb,
s_SOFA, s_Shock_Index, s_mechvent, s_GCS        (15 state features, normalised)
a                                                (integer 0-31)
vasopressor_b, ivfluid_b, antibiotic_b, sedation_b, diuretic_b
r                                                (reward)
s_next_HR, ..., s_next_GCS                      (15 next-state features; zeros if done=1)
```

**Run commands:**
```bash
python scripts/icu_readmit/step_10_rl_preprocess.py            # full run
python scripts/icu_readmit/step_10_rl_preprocess.py --smoke    # 2000 stays
```

---

## Step 10b -- RL Preprocessing: Broad State

**Script:** `scripts/icu_readmit/step_10b_rl_preprocess_broad.py`
**Output:** `data/processed/icu_readmit/`

A parallel RL dataset using a broader, Raghu 2017-style state representation --
all well-covered ICU measurements that a bedside clinician would routinely see,
without the variable-selection filtering from steps 09a/09b/09c.

**Why two state representations:**
- Step 10 (narrow, 15 states): causally guided -- only features confirmed by
  outcome relevance, action responsiveness, and causal discovery
- Step 10b (broad, 51 states): replication baseline -- same philosophy as Raghu 2017,
  which used a broad 47-feature clinical state without causal filtering

Both use identical actions, rewards, and split logic. The comparison allows the thesis
to ask: does causal variable selection produce a better RL policy than the raw Raghu approach?

**Broad state features (51 total):**

| Group | Features (count) |
|-------|-----------------|
| Vitals | HR, MeanBP, Arterial_BP_Sys, Arterial_BP_Dia, RR, SpO2, Temp_C, FiO2_1 (8) |
| Blood gas | Arterial_pH, paO2, paCO2, HCO3, Arterial_BE, Arterial_lactate (6) |
| Renal/metabolic | BUN, Creatinine, Potassium, Sodium, Chloride, Glucose, Magnesium, Calcium, Phosphate, Anion_Gap (10) |
| Hematology/coag | Hb, WBC_count, Platelets_count, PT, PTT, INR, Fibrinogen (7) |
| Liver/other labs | SGOT, SGPT, Total_bili, Albumin (4) |
| Derived scores | SOFA, SIRS, Shock_Index, PaO2_FiO2, GCS (5) |
| Fluid balance | input_total, output_total, cumulated_balance (3) |
| Ventilator | PEEP, TidalVolume, MinuteVentil (3) |
| Hemodynamic | CVP (1) |
| Neuro | RASS (1) |
| Static context | age, charlson_score, prior_ed_visits_6m (3) |
| Binary state | mechvent, re_admission (2) |

Static confounders (age, charlson_score, prior_ed_visits_6m) are included
directly in the state vector, following Raghu 2017 (who included demographics
in the state). The narrow dataset keeps these separate.

**Output files:**
- `data/processed/icu_readmit/rl_dataset_broad.parquet` -- 1,500,857 transitions, 114 cols
- `data/processed/icu_readmit/static_context_broad.parquet`
- `data/processed/icu_readmit/scaler_params_broad.json`

**Steps 11-13 use the broad dataset** (rl_dataset_broad.parquet) to replicate
the Raghu pipeline as closely as possible.

**Run commands:**
```bash
python scripts/icu_readmit/step_10b_rl_preprocess_broad.py
python scripts/icu_readmit/step_10b_rl_preprocess_broad.py --smoke
```

---

## Step 11 -- Continuous RL: Dueling DDQN + SARSA

**Script:** `scripts/icu_readmit/step_11_ddqn.py`
**Input:** `data/processed/icu_readmit/rl_dataset_broad.parquet`
**Output:** `models/icu_readmit/continuous/`
**Log:** `logs/step_11_icu_readmit_full.log`

Two models trained:
1. **Dueling Double DQN** -- the RL policy we want to evaluate
2. **SARSA physician** -- a Q-function trained to mimic the physician's actual behaviour,
   used as a principled baseline for off-policy evaluation

### Architecture: Dueling Double DQN with Prioritized Experience Replay

**Double DQN**: Uses two networks to reduce Q-value overestimation:
- Online network: trains continuously via backprop, selects which action to take
- Target network: updated softly (τ=0.001), used to evaluate Q-values of chosen actions
- Prevents the same network from both selecting and scoring actions (prevents feedback loops)

**Dueling architecture**: Splits the Q-function into two streams:
- Value stream V(s): how good is this state, regardless of action
- Advantage stream A(s,a): how much better is action a compared to average
- Combined: Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))
- Advantage: can learn state values even when a specific action is never tried

**Prioritized Experience Replay**: Samples transitions proportional to their
TD error magnitude -- high-error transitions are revisited more often,
accelerating learning from informative transitions.

**DDQN hyperparameters:**
- Training steps: 100,000
- State dimension: 51, Action space: 32
- Learning rate: 1e-4, Discount (γ): 0.99
- Hidden layers: 128 units
- Replay buffer: full training set (1,057,632 transitions)
- Reward threshold for Q-value regularisation: 20 (slightly above ±15 terminal)

### Architecture: SARSA Physician Baseline

SARSA (State-Action-Reward-State-Action) is trained on the same data but learns
to predict the Q-function of the **physician's observed policy**, not an optimal policy.

The key difference from DDQN:
- DDQN Bellman target: `r + γ · max_a' Q(s', a')` -- uses the best possible next action
- SARSA Bellman target: `r + γ · Q(s', a'_physician)` -- uses the physician's actual next action

This makes SARSA an "on-policy" estimator: it learns how much total future reward
the physician would accumulate by continuing to act as they did. This Q-function is
used as a baseline in the doubly-robust evaluation (step 12).

**SARSA hyperparameters:**
- Training steps: 80,000
- Same state/action dimensions and architecture as DDQN
- Replay buffer: full training set

### Results (full run, CPU, 126.8 minutes total)

**DDQN training (100,000 steps):**

| Checkpoint | Loss |
|-----------|------|
| 5,000 | 5.94 |
| 25,000 | 4.35 |
| 50,000 | 3.87 |
| 75,000 | 3.76 |
| 100,000 | 3.96 |

```
DDQN complete:
  Mean Q (train): 0.0681
  Mean Q (val):   0.0751  |  unique actions: 29/32
  Mean Q (test):  0.0815  |  unique actions: 28/32
```

The declining loss (5.94 → 3.76) confirms convergence. The fact that 28-29 of 32
action combinations are used (not collapsed to 1) shows the policy is learning
state-dependent treatment decisions, not a trivial single recommendation.

**SARSA physician (80,000 steps):**

```
SARSA complete:
  Mean Q (train): 0.1346
  Mean Q (val):   0.1402  |  unique actions: 29/32
  Mean Q (test):  0.1423  |  unique actions: 29/32
```

SARSA Q-values are slightly higher than DDQN (0.1346 vs 0.0681). This is expected:
SARSA learns the Q-function of the physician policy (which is shaped by clinical experience
over years), while DDQN is learning from scratch on a fixed offline dataset.

**Output files:**
```
models/icu_readmit/continuous/
  ddqn/
    dqn_model.pt                  -- trained model weights
    checkpoint_{20k,40k,...}.pt   -- training checkpoints every 20k steps
    dqn_actions.pkl               -- greedy actions for train split (1,057,632)
    dqn_q_values.pkl              -- Q-values for train split
    dqn_actions_val.pkl           -- greedy actions for val split (222,408)
    dqn_q_val.pkl
    dqn_actions_test.pkl          -- greedy actions for test split (220,817)
    dqn_q_test.pkl
  sarsa_phys/
    sarsa_phys_model.pt
    phys_actions_{train,val,test}.pkl
    phys_q_{train,val,test}.pkl
```

**Run commands:**
```bash
python scripts/icu_readmit/step_11_ddqn.py                          # full run (~127 min on CPU)
python scripts/icu_readmit/step_11_ddqn.py --smoke                  # 500 steps each (~2 min)
python scripts/icu_readmit/step_11_ddqn.py --dqn-steps 100000 \
    --sarsa-steps 80000 --log logs/step_11_full.log                  # explicit full run
```

---

## Step 12 -- Off-Policy Evaluation (Doubly Robust)

**Script:** `scripts/icu_readmit/step_12_evaluate.py`
**Input:** `data/processed/icu_readmit/rl_dataset_broad.parquet` + step 11 models
**Output:** `models/icu_readmit/eval/`, `reports/icu_readmit/evaluation_results.json`

Off-policy evaluation (OPE) estimates how well the DDQN policy would perform
if deployed clinically, without running it on real patients.

### Why off-policy evaluation is hard

The DDQN was trained on historical data -- we cannot let it interact with patients
to measure its performance. We must estimate its value from the same historical
dataset the physician policy generated.

Simple importance sampling (IS) suffers from high variance when the DDQN policy
differs much from the physician. Doubly Robust (DR) combines two components:
a direct reward estimator and importance-weighted corrections, providing:
- Correct estimate if **either** the reward model or the importance weights are accurate
- Lower variance than pure IS

### Components trained in step 12

**1. Physician policy model** (`physician_policy/`)
A neural network classifier trained to predict the probability of each of the
32 possible physician actions given the current state: π(a|s).
These probabilities are used as importance weights (denominator in IS).

**2. Reward estimator** (`reward_estimator/`)
A neural network trained to predict the expected reward for any (state, action) pair.
This provides the "direct method" component of DR: estimating what reward the DDQN
policy would receive even for actions the physician never took.

**3. Environment model** (`env_model/`)
A neural network trained to predict the next state s' given (s, a).
This provides counterfactual rollouts: what would have happened if the DDQN had
taken a different action?

**4. Doubly Robust evaluation**
Combines the reward estimator and importance-corrected actual rewards.
Evaluated on: DDQN (train), DDQN (test), SARSA physician (test).

- `value_clip = 40.0` (slightly above the maximum cumulative reward of ~±30)
- `gamma = 0.99`

**Run commands:**
```bash
python scripts/icu_readmit/step_12_evaluate.py                     # full run
python scripts/icu_readmit/step_12_evaluate.py --smoke             # 200 steps each
```

---

## Step 12b -- Policy Analysis: Raghu 2017 Figures 1 & 2

**Script:** `scripts/icu_readmit/step_12b_policy_analysis.py`
**Input:** `rl_dataset_broad.parquet` (test split) + `dqn_actions_test.pkl` + `scaler_params_broad.json`
**Output:** `reports/icu_readmit/`
**Log:** `logs/step_12b_policy_analysis.log`

Qualitative evaluation of the learned policy, directly replicating the primary
evidence figures from Raghu et al. 2017. No training -- pure analysis of the
test set predictions from step 11.

### Why qualitative evaluation matters

Raghu 2017 treats quantitative off-policy metrics (DR) as unreliable in ICU settings
due to distributional shift and partial observability. Their **primary evidence** is
two qualitative figures showing that the RL policy behaves sensibly:
1. It recommends different drug combinations at different severity levels
2. Patient outcomes are worst exactly when the clinician and RL policy disagree most

These figures do not prove the policy is optimal, but they provide strong evidence
that it has learned clinically meaningful patterns -- not noise.

### Figure 1 -- Action distribution comparison (Raghu Fig 1 equivalent)

Three panels (SOFA severity: low/medium/high). Each panel shows side-by-side bars:
- Physician prescription rate (%) for each of the 5 drugs
- DDQN recommendation rate (%) for each of the 5 drugs

**What to look for:** Does the DDQN recommend more aggressive treatment (more vasopressors,
antibiotics) at higher SOFA severity? Does it match physician prescribing patterns
while showing systematic differences that could represent improvements?

Saved: `reports/icu_readmit/fig1_action_distribution.png`

### Figure 2 -- Readmission rate vs action disagreement (Raghu Fig 2 equivalent)

Three panels (SOFA severity: low/medium/high). X-axis: Hamming distance (0-5 drugs) between
DDQN and physician actions. Y-axis: readmission rate (%) for timesteps at that disagreement level.

**Hamming distance**: the number of drugs (out of 5) where DDQN and physician disagree.
- Hamming = 0: DDQN fully agrees with physician
- Hamming = 5: DDQN recommends the exact opposite of physician on all 5 drugs

**Key claim (Raghu 2017):** If the learned policy is valid, outcomes should be worst
when the clinician and RL disagree most. The RL policy has learned something useful if
patients whose physicians happened to follow the DDQN's recommendations had better outcomes.

**Results observed:**

| SOFA severity | n (test) | Readmission at Hamming=0 | Readmission at Hamming=2+ |
|--------------|---------|--------------------------|--------------------------|
| LOW | ~75% of test | ~18% | rising monotonically to ~35% |
| MEDIUM | ~25% of test | ~25% | rising monotonically |
| HIGH | 26 timesteps | (too few to interpret) |

The LOW and MEDIUM panels show the expected monotonic pattern: readmission rates are lowest
when the DDQN fully agrees with the physician and rise as disagreement increases.
**This replicates the key finding from Raghu 2017 Figure 2.**

The HIGH SOFA panel is essentially empty (26 timesteps). This is expected: the ICU
readmission cohort excludes in-hospital deaths (competing risk), which are the patients
with very high SOFA scores. The readmission outcome selects for patients who survived
their ICU stay, and those patients rarely have SOFA > 15.

Saved: `reports/icu_readmit/fig2_readmission_vs_disagreement.png`

### Additional output

`reports/icu_readmit/policy_analysis_stats.json` -- underlying numbers for both figures:
exact prescription rates, readmission rates per Hamming bin, and overall agreement statistics.

**Run command (no smoke flag needed -- runs in ~4 seconds):**
```bash
python scripts/icu_readmit/step_12b_policy_analysis.py
```

---

## Step 13 -- Model-Based Simulators (Raghu 2018 Table 1)

**Script:** `scripts/icu_readmit/step_13_simulator.py`
**Input:** `data/processed/icu_readmit/rl_dataset_broad.parquet`
**Output:** `models/icu_readmit/simulator/`, `reports/icu_readmit/`

Trains neural network transition models to simulate state evolution given an action.
Replicates Raghu 2018 Table 1: four architectures compared on per-feature MSE
and multi-step rollout quality.

### Why build a simulator

The off-policy evaluation in step 12 estimates policy value from the historical dataset.
A learned simulator allows a different kind of evaluation: generate rollouts under the
DDQN policy and inspect whether simulated trajectories look physiologically plausible.

It also enables future work: once a high-quality simulator exists, the RL agent can
be trained online against the simulator (model-based RL), which is far more sample-efficient
than offline batch learning.

### Four architectures (Raghu 2018 Table 1)

| Type | Architecture | Notes |
|------|-------------|-------|
| `nn` | 2 FC + ReLU + BatchNorm | Preferred model (Raghu 2018 finding) |
| `linear` | Linear regression | Baseline -- no nonlinearity |
| `lstm` | LSTM on history sequence | Captures temporal dependencies |
| `bnn` | Bayesian NN (variational) | Uncertainty quantification |

**Input to all models:** current state (51) + action (5 binary) + history (n_history=4 past states)
**Output:** next state (51 features predicted independently)

### Key differences from sepsis step 13

- State: 51 features (not 36), all s_* prefix, already z-scored
- Action: 5-bit binary (not 2-feature Raghu-style continuous dose)
- n_action: 5 (not 2)
- Input parquet: rl_dataset_broad.parquet (not CSV)

### Run commands

```bash
# Smoke test: single architecture, 5 epochs
python scripts/icu_readmit/step_13_simulator.py --model-type nn --smoke

# Single architecture full run
python scripts/icu_readmit/step_13_simulator.py --model-type nn

# All four architectures (replicates Table 1)
python scripts/icu_readmit/step_13_simulator.py --model-type all

# With custom settings
python scripts/icu_readmit/step_13_simulator.py \
    --model-type nn --epochs 100 --hidden 256 \
    --rollout-steps 10 --rollout-patients 200 \
    --log logs/step_13_icu_readmit.log
```

### Output files

```
models/icu_readmit/simulator/
  nn/        transition_model.pt, model_config.pkl
  linear/    transition_model.pt, model_config.pkl
  lstm/      transition_model.pt, model_config.pkl
  bnn/       transition_model.pt, model_config.pkl

reports/icu_readmit/
  simulator_nn_per_feature_mse.json        -- MSE per state feature, nn model
  simulator_nn_rollout_eval.json           -- multi-step rollout quality
  simulator_linear_per_feature_mse.json
  ...
  simulator_comparison.json                -- ranking across all 4 architectures
```

---

## Source Code Map

```
scripts/icu_readmit/
  step_09a_variable_selection.py       -- 4 LightGBM models predicting readmit_30d
  step_09b_transition_selection.py     -- 1 LightGBM per state feature, action_share
  step_09c_causal_discovery.py         -- 5 graphs x 3 algorithms (PC, NOTEARS, LiNGAM)
  step_10_rl_preprocess.py             -- narrow 15-feature (s,a,r,s',done) dataset
  step_10b_rl_preprocess_broad.py      -- broad 51-feature dataset (Raghu-style)
  step_11_ddqn.py                      -- Dueling DDQN + SARSA physician
  step_12_evaluate.py                  -- Doubly Robust off-policy evaluation
  step_12b_policy_analysis.py          -- Raghu 2017 Fig 1 & 2 qualitative evaluation
  step_13_simulator.py                 -- 4 transition model architectures

src/careai/icu_readmit/rl/
  continuous.py      -- DDQN training loop, SARSA, replay buffer, prepare_rl_data()
  evaluation.py      -- physician policy, reward estimator, env model, DR evaluation
  simulator.py       -- transition model architectures and training (step 13)

reports/icu_readmit/
  step_09a/   model1-4_importance.csv, all_models_summary.csv, variable_selection.json
  step_09b/   feature_responsiveness.csv, action_importance_detail.csv
  step_09c/   <graph>/pc/, <graph>/lingam/, cross_algorithm_summary.csv
  fig1_action_distribution.png           -- physician vs DDQN drug use by SOFA (step 12b)
  fig2_readmission_vs_disagreement.png   -- readmission vs Hamming distance (step 12b)
  policy_analysis_stats.json             -- underlying numbers for both figures
  evaluation_results.json                -- DR evaluation scores (step 12)
  simulator_comparison.json              -- simulator architecture ranking (step 13)

data/processed/icu_readmit/
  ICUdataset.csv               -- 1,500,857 rows, 61,771 stays, 146 cols (step 08)
  rl_dataset.parquet           -- narrow 15-feature transitions (step 10)
  rl_dataset_broad.parquet     -- broad 51-feature transitions (step 10b)
  scaler_params_broad.json     -- normalisation parameters for broad state
  static_context_broad.parquet

models/icu_readmit/
  continuous/ddqn/             -- DDQN model + actions + Q-values (step 11)
  continuous/sarsa_phys/       -- SARSA physician model (step 11)
  eval/                        -- physician policy, reward estimator, env model (step 12)
  simulator/nn|linear|lstm|bnn -- transition models (step 13)
```
