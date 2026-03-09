# CareAI Pipeline Status

Last updated: 2026-03-06

## Current State

| Step | Description | Status | Output |
|------|-------------|--------|--------|
| 1 | Build hosp_daily dataset | ✅ Done | `data/processed/hosp_daily_transitions.csv` |
| 2 | Train transition simulator | ✅ Done | `models/sim_daily_full/` |
| 2b | Estimate AIPW ATEs (causal) | ✅ Done | `reports/causal_daily_full/treatment_effects.json` |
| 2c | Train CATE models (full data) | ⏳ Pending | `models/cate_daily_full/` — not yet run |
| 3 | Train readmission reward model | ✅ Done | `models/rl_daily_full/` (AUC 0.647) |
| 3b-ATE | Run ATE policy on full data | ⏳ Pending | `reports/rl_daily_full/ate/` — not yet run |
| 3b-CATE | Run CATE policy on full data | ⏳ Pending | `reports/rl_daily_full/cate/` — blocked on 2c |

## What To Run Next

### Step 2c — Train CATE models on full data (~1-2 hours)
```bash
cd CareAI
python scripts/causal/train_cate_models.py \
  --csv data/processed/hosp_daily_transitions.csv \
  --model-dir models/cate_daily_full \
  --report-dir reports/cate_daily_full
```

### Step 3b — Run both policies (after 2c completes)
```bash
# ATE policy (fast, ~5 min)
python scripts/rl/run_policy.py \
  --policy-type ate \
  --csv data/processed/hosp_daily_transitions.csv \
  --transition-model-dir models/sim_daily_full \
  --readmission-model-dir models/rl_daily_full \
  --report-dir reports/rl_daily_full

# CATE policy (~20-30 min)
python scripts/rl/run_policy.py \
  --policy-type cate \
  --csv data/processed/hosp_daily_transitions.csv \
  --transition-model-dir models/sim_daily_full \
  --readmission-model-dir models/rl_daily_full \
  --cate-model-dir models/cate_daily_full \
  --report-dir reports/rl_daily_full
```

## Key Results So Far

### Readmission model (full data)
- AUC: **0.647** | Brier: 0.231 | Prevalence: 22.7% | Features: 43
- Trained on 2.14M rows, tested on 458k rows

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

### CATE models (5k sample reference — full data pending)
Trained on 43k rows. Population ATEs broadly consistent with AIPW.
Full-data models will replace `models/cate_daily/` with `models/cate_daily_full/`.

## Notes
- `models/cate_daily/` = 5k sample CATE models, kept as reference until full-data run completes
- `models/cate_daily_full/` = does not exist yet
- econml requires scikit-learn <1.7 — InconsistentVersionWarning on model load is harmless
- On Windows/bash: DB credentials must be passed inline (not inherited from system env vars)
