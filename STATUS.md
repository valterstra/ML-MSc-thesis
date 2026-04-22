# CareAI Status

Last updated: 2026-04-15

## Branch Scope

This `main` branch tracks the active ICU-readmission pipeline only.

Active numbering:
- `01-08`: cohort and dataset pipeline
- `09`: state/action selection
- `10a-10c`: selected RL preprocessing and reward models
- `11a-11c`: simulator training
- `12a-12c`: simulator evaluation
- `13a-13c`: control
- `14`: offline RL comparison

## Current State

| Step | Description | Status | Primary Output |
|------|-------------|--------|----------------|
| `01-08` | ICU readmission preprocessing pipeline | Complete | `ICUdataset.csv` |
| `09` | state/action selection | Complete | selected state/action set |
| `21` | legacy action-selection diagnostics | Archived | supporting diagnostic reports |
| `10a` | selected RL dataset | Complete | `rl_dataset_selected.parquet` |
| `10b` | selected severity surrogate | Complete | `models/icu_readmit/severity_selected/` |
| `10c` | selected terminal readmission model | Complete | `models/icu_readmit/terminal_readmit_selected/` |
| `11a` | CARE-Sim selected causal training | Complete | `models/icu_readmit/caresim_selected_causal/` |
| `11b` | MarkovSim selected causal training | Complete | `models/icu_readmit/markovsim_selected_causal/` |
| `11c` | DAG-aware temporal world model training | Complete | `models/icu_readmit/dagaware_selected_causal/` |
| `12a` | CARE-Sim evaluation | Complete | `reports/icu_readmit/caresim_selected_causal/` |
| `12b` | MarkovSim evaluation | Complete | `reports/icu_readmit/markovsim_selected_causal/` |
| `12c` | DAG-aware temporal world model evaluation | Complete | `reports/icu_readmit/dagaware_selected_causal/` |
| `13a` | CARE-Sim planner + DDQN control | Complete | `models/icu_readmit/caresim_control_selected_causal/` |
| `13b` | MarkovSim planner + DDQN control | Complete | `models/icu_readmit/markovsim_control_selected_causal/` |
| `13c` | DAG-aware planner + DDQN control | Planned | `models/icu_readmit/dagaware_control_selected_causal/` |
| `14` | offline DDQN + OPE comparison | Complete | `reports/icu_readmit/offline_selected/` |

## Current Comparison Structure

Two simulator families:
- `11a/12a/13a`: CARE-Sim transformer world model
- `11b/12b/13b`: MarkovSim causal Markov baseline
- `11c/12c/13c`: DAG-aware temporal transformer world model

One real-data comparison branch:
- `14`: offline DDQN trained on logged ICU data and evaluated with OPE

Step `14` is the policy comparison branch for:
- `offline_ddqn`
- `caresim_ddqn`
- `markovsim_ddqn`

## Main Files

- `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`
- `scripts/icu_readmit/step_11b_markovsim_train.py`
- `scripts/icu_readmit/step_13a_caresim_control.py`
- `scripts/icu_readmit/step_13b_markovsim_control.py`
- `scripts/icu_readmit/step_13c_dagaware_control.py`
- `scripts/icu_readmit/step_14_offline_selected.py`
- `scripts/prepare_colab_upload.py`

## Notes

- Generated data, models, reports, logs, and temporary archives are intentionally not versioned on `main`.
- Broader experimental code has been separated from this branch so the active pipeline remains readable.
- The old `09b` action-analysis branch is preserved as legacy `step 21` under `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/`.
