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
  → models/cate_daily_full/                      ⏳ NOT YET RUN on full data
  → reports/cate_daily_full/
  (models/cate_daily/ = 5k sample reference, temporary)

Step 3: Train Readmission Reward Model
  scripts/rl/train_readmission_reward.py
  → models/rl_daily_full/                        ✅ AUC=0.647, 2.14M train rows

Step 3b: Run RL Policy Evaluation
  scripts/rl/run_policy.py --policy-type ate    → reports/rl_daily_full/ate/  ⏳ pending
  scripts/rl/run_policy.py --policy-type cate   → reports/rl_daily_full/cate/ ⏳ pending
```

## Two Parallel Policy Branches

Both branches share the same transition model (sim_daily_full) and reward model (rl_daily_full).
They differ only in how drug effects are estimated and applied:

```
ATE branch (Option C):
  Scalar AIPW effect per (drug, outcome) pair — same delta for every patient
  Source: reports/causal_daily_full/treatment_effects.json
  Run:    scripts/rl/run_policy.py --policy-type ate

CATE branch (Option D):
  CausalForestDML — per-patient heterogeneous effect based on current state
  Source: models/cate_daily_full/  (pending full-data training)
  Run:    scripts/rl/run_policy.py --policy-type cate
```

Both do exhaustive search over 2^5=32 combinations of 5 drugs
(antibiotic, anticoagulant, diuretic, insulin, steroid — opioid excluded).
Pick the combo that minimises P(readmit_30d=1) from the readmission model.

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
