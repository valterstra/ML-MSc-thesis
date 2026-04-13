# CARE-AI Project Guide

## What This Project Is
A causal reinforcement learning pipeline for hospital treatment recommendations.
Built on MIMIC-IV (prototype). Final target: Region Stockholm VAL Data Warehouse.
Goal: move from prediction → causal inference → RL policy that recommends drug interventions.
See STATUS.md for current pipeline state and what needs to be run next.

## Three Independent Pipelines

This repo contains three separate RL pipelines:

1. **V1/V3 Causal RL Pipeline** (daily drug recommendations for general hospital patients)
   - 6-step pipeline documented below
   - Source: `src/careai/sim_daily/`, `src/careai/causal_daily/`, `src/careai/rl_daily/`, `src/careai/causal_v3/`
   - Scripts: `scripts/hosp_daily/`, `scripts/sim/`, `scripts/causal/`, `scripts/rl/`, `scripts/causal_v3/`

2. **Sepsis RL Pipeline** (IV fluid + vasopressor dosing for sepsis ICU patients)
   - 13-step pipeline replicating Raghu et al. 2017/2018
   - **Full documentation: `docs/sepsis_pipeline.md`** (steps 01-08, data preprocessing)
   - **Full documentation: `docs/sepsis_rl_pipeline.md`** (steps 09-13, RL + simulators)
   - Source: `src/careai/sepsis/` (queries, columns, imputation, derived features, rl/)
   - Scripts: `scripts/sepsis/step_01_extract.py` through `step_13_simulator.py`
   - Data: `data/processed/sepsis/MKdataset.csv` (325k rows, 37k ICU stays)
   - Models: `models/sepsis_rl/` (discrete/, continuous/, eval/, simulator/)
   - Reports: `reports/sepsis_rl/`

3. **ICU Readmission Pipeline** (general ICU treatment policy targeting 30-day readmission)
   - 13-step pipeline, general ICU cohort (not sepsis-specific), built on sepsis pipeline structure
   - **Full documentation: `docs/icu_readmit_pipeline.md`** (steps 01-08, data preprocessing)
   - **Full documentation: `docs/icu_readmit_rl_pipeline.md`** (steps 09-13, RL + simulators)
   - Source: `src/careai/icu_readmit/` (columns, queries, imputation, derived features, rl/)
   - Scripts: `scripts/icu_readmit/step_01_extract.py` through `step_13_simulator.py`
   - Config: `configs/icu_readmit.yaml`
   - Data: `data/processed/icu_readmit/ICUdataset.csv` (target output)
   - Models: `models/icu_readmit/` (discrete/, continuous/, eval/, simulator/)
   - Reports: `reports/icu_readmit/`

### ICU Readmission Pipeline -- Key Design Decisions

```
Cohort:      All ICU stays, LOS >= 24h. No age min. Exclude: in-hosp death,
             died within 30 days (competing risk), obstetric, newborn.
Outcome:     readmit_30d -- 30-day readmission post-discharge (not mortality)
Comorbidity: Charlson score + 18 component flags (not Elixhauser)
             -- Charlson is the standard for readmission; used in LACE index
Time:        4-hour blocs anchored to ICU intime
GCS:         Summed from Eye+Verbal+Motor (MetaVision itemids 220739/223900/223901)
             -- CareVue itemid 198 returns 100% NaN in MIMIC-IV
CAM-ICU:     Text->binary in SQL (Positive=1, Negative=0, other=NULL)
             NULL filled with 0 in step_07 (not assessed = no delirium flag)
mechvent:    Both state (SAH forward-fill) AND action (binary per bloc)
             -- initiating ventilation is a clinical decision
discharge_disposition: Static confounder (known only at discharge)
             -- not a real-time action; kept for reward model training
Drug extraction: drugs_mv table -- 9 binary drug classes from inputevents
             (new vs sepsis pipeline which had no drug extraction)
```

### ICU Readmission Pipeline -- Run Status

```
Step 01 -- Extract raw data               [ COMPLETE ]
Step 02 -- Preprocess                     [ COMPLETE ]
Step 03 -- Cohort filter                  [ COMPLETE ]
Step 04 -- Patient states                 [ COMPLETE ]
Step 05 -- Impute states                  [ COMPLETE ]
Step 06 -- States + actions               [ COMPLETE ] 1,503,312 rows, 61,822 stays, 140 cols
Step 07 -- Final imputation               [ COMPLETE ] 144 cols (+ SOFA, SIRS, Shock_Index, PaO2_FiO2)
Step 08 -- Build ICUdataset.csv           [ COMPLETE ] 1,500,857 rows, 61,771 stays, 146 cols, readmit=20.7%
Step 09 -- FCI stability analysis         [ COMPLETE ] scripts/icu_readmit/step_09_causal_states/
  step_01: stay-level build (61,771 stays)
  step_02: LightGBM variable ranking -- state AUC=0.618, static AUC=0.696
  step_03: state->readmit FCI stability (2000 runs) -- Hb=97%, Ht=96%, BUN=91%
  step_04: action->state FCI stability (27,000 runs, 9 drugs x 3000)
           diuretic->BUN=1.00, ivfluid->Creatinine=0.90, vasopressor->HR=0.93
  Tier 2 (PRIMARY): drugs={diuretic,ivfluid,vasopressor,antibiotic}, states={Hb,BUN,Creatinine,HR,Shock_Index}
Step 10  -- RL preprocessing (narrow)     [ COMPLETE ] 1.5M transitions, 32/32 action combos, rl_dataset.parquet
Step 10b -- RL preprocessing (broad)      [ COMPLETE ] 1.5M transitions, 51 state features, rl_dataset_broad.parquet
Step 11  -- Dueling DDQN + SARSA          [ COMPLETE ] DDQN 100k steps (loss 5.9->3.8), SARSA 80k; 126.8 min CPU
Step 12  -- Off-policy evaluation (DR)    [ COMPLETE ] Physician policy + reward estimator + env model + DR scores
Step 12b -- Policy analysis (Raghu figs)  [ COMPLETE ] Fig1 drug distribution, Fig2 readmit vs disagreement confirmed
Step 13  -- Model-based simulators        [ COMPLETE ] 4 architectures (nn/linear/lstm/bnn), per-feature MSE + rollout
Step 10c -- RL preprocessing (Tier 2 FCI) [ COMPLETE ] 1,500,857 rows, 21 cols, 16/16 action combos, rl_dataset_tier2.parquet
Step 11b -- RL training: Tier 2 model     [ COMPLETE ] DDQN 100k steps + SARSA 80k; Colab T4 GPU
             Output: models/icu_readmit/tier2/ddqn/ and models/icu_readmit/tier2/sarsa_phys/
             Architecture: Dueling DDQN + SARSA, n_state=5, n_actions=16, hidden=128, lr=1e-4, gamma=0.99
Step 10d -- RL preprocessing (Tier 2 + discharge) [ COMPLETE ] 1,560,338 rows
             (1,500,857 in-stay phase-0 + 59,481 discharge phase-1)
             Discharge categories: 0=Home(25%), 1=Home+Services(32%), 2=Institutional(42%), ~4% excluded
             Output: rl_dataset_tier2_discharge.parquet
Step 11c -- Joint DDQN + discharge action  [ COMPLETE ] DDQN 100k steps + SARSA 80k; Colab T4 GPU
             Two-phase MDP: Phase 0=drugs(16 actions, reward=SOFA delta),
             Phase 1=discharge(3 actions, reward=+-15 readmit terminal)
             Cross-phase Bellman: last in-stay bloc bootstraps from Q_discharge
             KEY FINDING: 76.9% drug policy shift vs step 11b (adding discharge
             terminal fundamentally reshaped drug recommendations; vasopressor+ivfluid
             now prominent where step 11b showed <2%)
             Discharge Q ~7.7 (theoretical ~8.6, conservative -- healthy)
             DDQN vs SARSA drug disagreement: 76.5% (was 40.7% in step 11b)
             Output: models/icu_readmit/tier2_discharge/ddqn/ and sarsa_phys/
             Both models: drug .pt + discharge .pt, losses.pkl, val/test splits
Step 14  -- CARE-Sim transformer world model [ COMPLETE ] 5-model ensemble trained on Tier-2
             Trained on 8-state Tier-2 design:
             dynamic={Hb,BUN,Creatinine,HR,Shock_Index}
             static={age,charlson_score,prior_ed_visits_6m}
             Output: models/icu_readmit/caresim/
Step 15  -- CARE-Sim evaluation           [ COMPLETE ]
             One-step next-state MSE ~0.083, reward MAE ~1.87,
             terminal accuracy ~0.956, rollout step-1 MSE ~0.11
             Output: reports/icu_readmit/caresim/
Step 16  -- CARE-Sim control layer        [ COMPLETE ]
             Planner + simulator-based DDQN on CARE-Sim
             Latest results (100-episode eval):
               val:  planner=7.65, ddqn=4.16, repeat_last=2.44, random=2.24
               test: planner=8.02, ddqn=4.77, repeat_last=2.87, random=2.56
             DDQN improves over weak baselines but still trails planner
             and remains partially action-collapsed on actions {0,2}
             Output: models/icu_readmit/caresim_control/ and reports/icu_readmit/caresim_control/
```

Known limitations:
- steroid_active = 0% (wrong itemids in MIMIC-IV inputevents -- accepted)
- 15 columns remain NaN after KNN imputation (SVR, Eos_pct, Basos_pct: near-zero coverage)
- Causal discovery: LiNGAM finds reversed edges (confounding by indication -- documented as finding)
- CARE-Sim reward encodes readmission implicitly; no explicit readmission-probability head yet
- Step 16 rollouts are fixed-horizon and planner remains stronger than DDQN

### ICU Readmission Pipeline -- MIMIC-IV Schema Notes

```
- mimiciv_hosp.admissions.hadm_id is TEXT; use mimiciv_hosp_typed.admissions (INTEGER)
- mimiciv_hosp.labevents.hadm_id is TEXT; use mimiciv_hosp_typed.labevents (INTEGER)
- patients table: use anchor_age + anchor_year (no dob field -- removed for privacy)
  age = anchor_age + EXTRACT(YEAR FROM intime)::int - anchor_year
- morta_90: h.dod is already a bigint epoch; subtract intime directly (no ::timestamp cast)
- charlson_flags temp table must be materialized before demog() query runs
```

## The 6-Step Pipeline

```
Step 1: Build Dataset
  scripts/hosp_daily/build_hosp_daily.py
  → data/processed/hosp_daily_transitions.csv   ✅ 3.06M rows, 498k admissions

Step 2: Train Transition Simulator
  scripts/sim/train_daily_transition.py
  → models/sim_daily_full/                       ✅ 23 LightGBM models, R²=0.37–0.94
  scripts/sim/evaluate_daily_sim.py
  → reports/sim_daily_full/, reports/sim_daily_5day/

Step 2b: Estimate Causal Treatment Effects (AIPW)
  scripts/causal/estimate_treatment_effects.py
  → reports/causal_daily_full/treatment_effects.json  ✅ 9 drug→outcome ATEs
  scripts/causal/check_overlap.py                     (propensity overlap diagnostics)

Step 2c: Train CATE Models (heterogeneous effects)
  scripts/causal/train_cate_models.py
  → models/cate_daily_full/                      ✅ 27 models (3 per pair: outcome + propensity + CausalForestDML)
  → reports/cate_daily_full/
  (models/cate_daily/ = 5k sample reference, kept for comparison)

Step 3: Train Readmission Reward Model
  scripts/rl/train_readmission_reward.py
  → models/rl_daily_full/                        ✅ AUC=0.647, 2.14M train rows

Step 3b: Run RL Policy Evaluation
  scripts/rl/run_policy.py --policy-type ate    → reports/rl_daily_full/ate/  ✅ done
  scripts/rl/run_policy.py --policy-type cate   → reports/rl_daily_full/cate/ ✅ done

Step 4: Fitted Q-Iteration (FQI) — Multi-Step RL
  scripts/rl/train_fqi_agent.py         → models/fqi_daily/,        reports/rl_daily_full/fqi/         ✅ antibiotic-only baseline
  scripts/rl/train_fqi_multi_agent.py   → models/fqi_multi/,        reports/rl_daily_full/fqi_multi/   ✅ 5-drug, 26 features (3k patients, 64 seqs)
  scripts/rl/train_fqi_multi_agent.py   → models/fqi_multi_large/,  reports/rl_daily_full/fqi_multi_large/ ✅ scaled up (5k patients, 128 seqs)
```

## Policy Branches (all completed)

All branches share the same transition model (sim_daily_full) and reward model (rl_daily_full).

```
ATE branch (Option C):                                                 ✅ done
  1-step greedy. Scalar AIPW effect per (drug, outcome) — same delta for every patient.
  Source: reports/causal_daily_full/treatment_effects.json
  Result: mean risk 0.4228, beats do-nothing 98.6% of patients

CATE branch (Option D):                                                ✅ done
  1-step greedy. CausalForestDML — per-patient heterogeneous effect based on current state.
  Source: models/cate_daily_full/
  Result: mean risk 0.4219, beats do-nothing 98.8% of patients

FQI baseline (Option E):                                               ✅ done
  3-step planning. Antibiotic only. 7 RL state features. 5k train patients, 8 seqs.
  Source: models/fqi_daily/
  Result: mean risk 0.4902, beats do-nothing 63.6% of patients

FQI-multi (Option F):                                                  ✅ done
  3-step planning. 5 drugs jointly. Full 26-feature state. 3k patients x 64 seqs.
  Source: models/fqi_multi/
  Result: mean risk 0.4879, beats do-nothing 73.4% of patients

FQI-multi-large (Option G):                                            ✅ done
  3-step planning. 5 drugs jointly. Full 26-feature state. 5k patients x 128 seqs.
  Source: models/fqi_multi_large/
  Result: mean risk 0.4874, beats do-nothing 75.8% of patients
```

ATE/CATE: exhaustive search over 2^5=32 drug combinations, 1 simulator step, score with readmission model.
FQI variants: 3 decision points (day 0/2/4), 2 sim days per step, sparse terminal reward, 10 FQI iterations.

## Source Code Map (src/careai/)

```
sim_daily/
  features.py      — STATE_CONTINUOUS (15 labs), STATE_BINARY (4), ACTION_COLS (6),
                     STATIC_FEATURES (7), MEASURED_FLAGS (15)
  data.py          — prepare_daily_data() → PreparedData(raw, one_step_train, initial_states)
  transition.py    — TransitionModel, load_model(), predict_next(model, state_dict, action_dict)
  evaluate.py      — single-step R²/AUC, rollout KS tests
  env.py           — step-based environment for multi-step rollouts

causal_daily/
  features.py      — ALL_CONFOUNDERS, TREATMENT_OUTCOME_PAIRS (9 pairs), EXPECTED_DIRECTION
  propensity.py    — P(drug=1|confounders) logistic model — nuisance model for AIPW
  estimators.py    — naive_ate(), ipw_ate(), aipw_ate(), bootstrap_ci()
  balance.py       — SMD before/after IPW weighting (covariate balance diagnostics)
  evaluate.py      — run_causal_analysis(), print_results_table()
  cate.py          — CATEModel, CATERegistry, fit_cate_registry(), predict_cate(),
                     save_cate_registry(), load_cate_registry()

rl_daily/
  readmission.py   — ReadmissionModel (LightGBM), fit_readmission_model(),
                     predict_readmission_risk(), save/load_readmission_model()
                     NOTE: this is the RL reward model, not an episode readmission predictor
  policy.py        — ATE_DRUGS (5), load_ate_table(), causal_exhaustive_policy()
                     apply_ate_corrections() — additive scalar ATE shift to base next-state
  policy_cate.py   — cate_exhaustive_policy() — same interface as policy.py but per-patient
                     CATEs precomputed once per patient (9 calls), reused across 32 combos
  evaluate.py      — evaluate_policy(ate_table=... OR cate_registry=...) — pass exactly one
                     Compares: policy vs do-nothing vs real clinical actions
  fqi.py           — FittedQIteration: antibiotic-only, 7 RL state features, 3 steps x 2 days
                     collect_trajectories() — batched rollout (2^3=8 seqs enumerated)
                     FittedQIteration.fit() — backward induction, 1 LightGBM Q-model per step
  fqi_multi.py     — FittedQIterationMulti: 5 drugs jointly, 26-feature state, 3 steps x 2 days
                     collect_trajectories_multi() — random sampling (N_SEQS=64 default)
                     FittedQIterationMulti.fit() — same backward induction, 31-feature Q-function
                     FittedQIterationMulti.best_combo() — argmax over all 32 drug combos

hosp_daily/        — dataset builder (Step 1, already run)
  build.py         — 8-step pipeline: spine → static → location → service →
                     labs → infection → actions → label/split
  drug_lists.py    — regex patterns for 6 drug classes

transitions/       — used internally by hosp_daily/build.py
  sampling.py      — subject_level_sample() — deterministic hash-based sampling
  split.py         — assign_subject_splits() — 70/15/15 by subject_id

io/                — used internally by hosp_daily/build.py
  load_inputs.py   — load_yaml(), resolve_from_config()
  write_outputs.py — write_csv(), write_json()
```

## 9 Treatment-Outcome Pairs
```python
("insulin_active",       "glucose")                    # confounded — shows positive (up not down)
("antibiotic_active",    "wbc")                        # down ✓
("antibiotic_active",    "positive_culture_cumulative") # confounded
("diuretic_active",      "bun")                        # up ✓
("diuretic_active",      "potassium")                  # down ✓
("diuretic_active",      "sodium")                     # confounded
("steroid_active",       "glucose")                    # up ✓
("steroid_active",       "wbc")                        # up ✓
("anticoagulant_active", "inr")                        # up ✓
```

## Full-Data run_policy.py Commands
```bash
# ATE policy
python scripts/rl/run_policy.py \
  --policy-type ate \
  --csv data/processed/hosp_daily_transitions.csv \
  --transition-model-dir models/sim_daily_full \
  --readmission-model-dir models/rl_daily_full \
  --report-dir reports/rl_daily_full

# CATE policy (requires models/cate_daily_full/ to exist first)
python scripts/rl/run_policy.py \
  --policy-type cate \
  --csv data/processed/hosp_daily_transitions.csv \
  --transition-model-dir models/sim_daily_full \
  --readmission-model-dir models/rl_daily_full \
  --cate-model-dir models/cate_daily_full \
  --report-dir reports/rl_daily_full
```

## Environment Notes
- Python venv: `.venv/` — activate before running
- DB: PostgreSQL localhost:5432, db=mimic (pass credentials inline on Windows bash)
- econml requires scikit-learn <1.7 — InconsistentVersionWarning on model load is harmless
- Unicode characters in print() break Windows cp1252 terminal — use ASCII only
