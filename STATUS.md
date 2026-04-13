# CareAI Status

Last updated: 2026-04-03

## Current Thesis State

The active thesis track is the **ICU readmission pipeline with CARE-Sim**.

The important transition over the last work cycle is:

- the older ICU readmission RL/simulator stack (steps 09-13) is no longer the end state
- the current pipeline now extends through:
  - Step 14: CARE-Sim training
  - Step 15: CARE-Sim evaluation
  - Step 16: CARE-Sim control layer (planner + DDQN)

## ICU Readmission Pipeline

| Step | Description | Status | Primary Output |
|------|-------------|--------|----------------|
| 01-08 | ICU cohort build and preprocessing | Complete | `data/processed/icu_readmit/ICUdataset.csv` |
| 09 | FCI stability causal discovery | Complete | `reports/icu_readmit/step_09_causal_states/` |
| 10c | Tier-2 RL preprocessing | Complete | `data/processed/icu_readmit/rl_dataset_tier2.parquet` |
| 11b | Tier-2 offline DDQN + SARSA | Complete | `models/icu_readmit/tier2/` |
| 11c | Tier-2 + discharge action DDQN | Complete | `models/icu_readmit/tier2_discharge/` |
| 13 | Older model-based simulators | Complete | `models/icu_readmit/simulator/` |
| 14 | CARE-Sim transformer world model | Complete | `models/icu_readmit/caresim/` |
| 15 | CARE-Sim held-out evaluation | Complete | `reports/icu_readmit/caresim/` |
| 16 | CARE-Sim control layer | Complete | `models/icu_readmit/caresim_control/`, `reports/icu_readmit/caresim_control/` |

## Current Primary Model Design

Tier-2 CARE-Sim uses:

- Dynamic state: `Hb`, `BUN`, `Creatinine`, `HR`, `Shock_Index`
- Static confounders: `age`, `charlson_score`, `prior_ed_visits_6m`
- Total state dimension: `8`
- Actions: `diuretic`, `ivfluid`, `vasopressor`, `antibiotic`
- Action space: `16` binary combinations

Readmission is **not** modeled as an explicit simulator output head. It is currently encoded implicitly through the terminal reward target used during RL preprocessing.

## Step 15 -- CARE-Sim Evaluation

Held-out CARE-Sim summary from `reports/icu_readmit/caresim/caresim_summary.json`:

- One-step val next-state MSE: `0.0830`
- One-step test next-state MSE: `0.0834`
- Reward MAE: about `1.87` on both val/test
- Terminal accuracy: about `0.956` on both val/test
- Mean uncertainty: about `0.0706`

Closed-loop rollout behavior:

- Val step-1 state MSE: `0.1169`
- Val step-5 state MSE: `0.2199`
- Test step-1 state MSE: `0.1080`
- Test step-5 state MSE: `0.1764`

Interpretation:

- the simulator is stable enough for short-horizon control experiments
- rollout error grows with horizon, as expected
- uncertainty remains low enough to support planner/DDQN experiments

## Step 16 -- CARE-Sim Control Layer

Current outputs:

- Model folder: `models/icu_readmit/caresim_control/`
- Reports: `reports/icu_readmit/caresim_control/`
- Main summary: `step_16_summary.json`
- Diagnostics: `step_16_diagnostics_val.json`, `step_16_diagnostics_test.json`

Latest 100-episode evaluation:

| Split | Planner | DDQN | Repeat-last | Random |
|------|---------:|-----:|------------:|-------:|
| Val  | 7.65 | 4.16 | 2.44 | 2.24 |
| Test | 8.02 | 4.77 | 2.87 | 2.56 |

Diagnostics:

- planner is still the strongest policy
- DDQN now clearly beats `random` and is moderately ahead of `repeat_last`
- DDQN improved after increasing training budget and slowing epsilon decay
- DDQN still uses only two actions (`0` and `2`) and remains partially action-collapsed

Val pairwise diagnostics:

- `ddqn - planner`: mean diff `-3.49`, win rate `0.04`
- `ddqn - random`: mean diff `+1.92`, win rate `0.84`
- `ddqn - repeat_last`: mean diff `+1.72`, win rate `0.58`

## What To Do Next

The repo is now in a state where two paths are reasonable:

1. Treat planner as the strongest current control result and write it up as the main Step 16 result.
2. Run one more structural DDQN improvement cycle if closing the planner gap is important.

If continuing DDQN work, the next changes should be structural rather than just increasing training time again.

## Important Current Files

- `scripts/icu_readmit/step_14_caresim_train.py`
- `scripts/icu_readmit/step_15_caresim_evaluate.py`
- `scripts/icu_readmit/step_16_caresim_control.py`
- `notebooks/step_14_caresim_colab.ipynb`
- `notebooks/step_16_caresim_control_colab.ipynb`
- `docs/caresim_playbook.md`
