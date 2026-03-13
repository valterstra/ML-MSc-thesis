# CARE-AI Project Guide

## What This Project Is
A causal reinforcement learning pipeline for hospital treatment recommendations.
Built on MIMIC-IV (prototype). Final target: Region Stockholm VAL Data Warehouse.
Goal: move from prediction → causal inference → RL policy that recommends drug interventions.
See STATUS.md for current pipeline state and what needs to be run next.

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
