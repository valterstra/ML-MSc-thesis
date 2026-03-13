# CareAI Pipeline Status

Last updated: 2026-03-13

## Current State

| Step | Description | Status | Output |
|------|-------------|--------|--------|
| 1 | Build hosp_daily dataset | ✅ Done | `data/processed/hosp_daily_transitions.csv` |
| 2 | Train transition simulator | ✅ Done | `models/sim_daily_full/` |
| 2b | Estimate AIPW ATEs (causal) | ✅ Done | `reports/causal_daily_full/treatment_effects.json` |
| 2c | Train CATE models (full data) | ✅ Done | `models/cate_daily_full/` |
| 3 | Train readmission reward model | ✅ Done | `models/rl_daily_full/` (AUC 0.647) |
| 3b-ATE | Run ATE policy on full data | ✅ Done | `reports/rl_daily_full/ate/` |
| 3b-CATE | Run CATE policy on full data | ✅ Done | `reports/rl_daily_full/cate/` |
| 4a | FQI — antibiotic-only baseline | ✅ Done | `models/fqi_daily/`, `reports/rl_daily_full/fqi/` |
| 4b | FQI-multi — 5 drugs, 26 features (3k/64) | ✅ Done | `models/fqi_multi/`, `reports/rl_daily_full/fqi_multi/` |
| 4c | FQI-multi-large — scaled up (5k/128) | ✅ Done | `models/fqi_multi_large/`, `reports/rl_daily_full/fqi_multi_large/` |

## What To Run Next

All core pipeline steps are complete. Candidate next improvements:

1. **Dense intermediate rewards** — reward at every RL step, not just terminal. Addresses sparse reward problem in FQI.
2. **CATE corrections inside FQI** — replace scalar ATE with personalized CATE deltas at each simulator step.
3. **Result analysis** — per-patient subgroup analysis, policy agreement/disagreement between branches.

## Key Results

### Readmission model (full data)
- AUC: **0.647** | Brier: 0.231 | Prevalence: 22.7% | Features: 43
- Trained on 2.14M rows, tested on 458k rows

### Policy comparison (500 test patients, 3-way evaluation)

| Policy | Mean readmission risk | Beats do-nothing |
|--------|----------------------|------------------|
| CATE (1-step greedy, 5 drugs, personalized) | 0.4219 | 98.8% |
| ATE (1-step greedy, 5 drugs, scalar) | 0.4228 | 98.6% |
| FQI-multi-large (3-step, 5 drugs, 26 feat, 5k/128) | 0.4874 | 75.8% |
| FQI-multi (3-step, 5 drugs, 26 feat, 3k/64) | 0.4879 | 73.4% |
| FQI baseline (3-step, antibiotic only, 7 feat) | 0.4902 | 63.6% |
| Do-nothing | 0.4959 | — |
| Real clinical (day-0 drugs held constant) | 0.4955 | — |

### AIPW ATEs (full data)
| Treatment | Outcome | ATE | Expected direction |
|-----------|---------|-----|--------------------|
| insulin_active | glucose | +25.19 | down (confounded) |
| antibiotic_active | wbc | -0.035 | down ✓ |
| antibiotic_active | positive_culture_cumulative | +0.007 | down ✗ |
| diuretic_active | bun | +1.58 | up ✓ |
| diuretic_active | potassium | -0.036 | down ✓ |
| diuretic_active | sodium | -0.093 | up ✗ |
| steroid_active | glucose | +3.37 | up ✓ |
| steroid_active | wbc | +0.232 | up ✓ |
| anticoagulant_active | inr | +0.025 | up ✓ |

### CATE models (full data)
- 27 models total (3 per treatment-outcome pair: outcome nuisance, propensity nuisance, CausalForestDML)
- Trained on full data in `models/cate_daily_full/`
- Population-level ATEs consistent with AIPW

## Notes
- `models/cate_daily/` = 5k sample CATE models, kept as reference
- `models/cate_daily_full/` = full-data CATE models (production)
- econml requires scikit-learn <1.7 — InconsistentVersionWarning on model load is harmless
- On Windows/bash: DB credentials must be passed inline (not inherited from system env vars)
- FQI scripts: `scripts/rl/train_fqi_agent.py` (antibiotic-only) and `scripts/rl/train_fqi_multi_agent.py` (5-drug)
