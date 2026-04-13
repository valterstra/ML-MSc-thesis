# ICU Readmit Active Pipeline

This is the active ICU-readmit workflow currently used in the project.

## Current Steps

- Steps `01-08`: cohort construction and ICU dataset building
- Step `09a`: causal state analysis
- Step `09b`: causal action analysis
- Step `10a`: selected replay dataset build
- Step `10b`: selected severity surrogate
- Step `10c`: selected terminal readmission model
- Step `11a`: selected causal CARE-Sim training
- Step `11b`: selected causal MarkovSim training
- Step `12a`: CARE-Sim evaluation
  - one-step prediction
  - rollout sanity
  - counterfactual checks
- Step `12b`: MarkovSim evaluation
  - one-step prediction
  - rollout sanity
  - counterfactual checks
- Step `13a`: CARE-Sim control
  - planner
  - DDQN
  - random / repeat-last baselines
- Step `13b`: MarkovSim control
  - planner
  - DDQN
  - random / repeat-last baselines
- Step `14`: offline RL comparison on held-out logged data
  - `offline_ddqn`
  - `caresim_ddqn`
  - `markovsim_ddqn`

## Active Core Files

- Selected preprocessing:
  - `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`
  - `scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py`
  - `scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py`
- Active simulators:
  - `src/careai/icu_readmit/caresim/`
  - `src/careai/icu_readmit/markovsim/`
- Shared active RL helpers:
  - `src/careai/icu_readmit/rl/continuous.py`
  - `src/careai/icu_readmit/rl/evaluation.py`
  - `src/careai/icu_readmit/rl/networks.py`

## Notes

- The old broad / tier2 / discharge RL branch is no longer part of the active pipeline.
- That retired branch is preserved under:
  - `scripts/icu_readmit/legacy/`
  - `src/careai/icu_readmit/legacy/`
  - `notebooks/legacy/icu_readmit/`
  - `docs/legacy/icu_readmit/`
- Historical details for the retired branch are in:
  - `docs/legacy/icu_readmit/icu_readmit_rl_pipeline_legacy.md`
