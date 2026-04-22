# CareAI

This `main` branch is the active ICU-readmission thesis pipeline.

The branch is intentionally curated around the current `01-14` workflow:

1. `01-08`: ICU cohort extraction, preprocessing, state/action construction, and final dataset build
2. `09`: state/action selection and selected-set recommendation
3. `10a-10c`: selected RL preprocessing, severity surrogate, and terminal readmission model
4. `11a-11b`: simulator training
   - `11a` CARE-Sim selected causal transformer
   - `11b` MarkovSim selected causal baseline
   - `11c` DAG-aware temporal world model
5. `12a-12b`: simulator evaluation
6. `13a-13c`: planner/DDQN control on top of each simulator
7. `14`: offline RL comparison on held-out logged data

## Active Entry Points

Preprocessing and causal selection:
- `scripts/icu_readmit/step_01_extract.py`
- `scripts/icu_readmit/step_08_build_dataset.py`
- `scripts/icu_readmit/step_09_state_action_selection/`
- `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`
- `scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py`
- `scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py`

Simulators:
- `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`
- `scripts/icu_readmit/step_11b_markovsim_train.py`
- `scripts/icu_readmit/step_12a_caresim_evaluate.py`
- `scripts/icu_readmit/step_12b_markovsim_evaluate.py`

Control and comparison:
- `scripts/icu_readmit/step_13a_caresim_control.py`
- `scripts/icu_readmit/step_13b_markovsim_control.py`
- `scripts/icu_readmit/step_13c_dagaware_control.py`
- `scripts/icu_readmit/step_14_offline_selected.py`

## Active Notebooks

- `notebooks/step_11a_caresim_selected_causal_colab.ipynb`
- `notebooks/step_13a_caresim_selected_colab.ipynb`
- `notebooks/step_13b_markovsim_selected_colab.ipynb`
- `notebooks/step_13c_dagaware_selected_colab.ipynb`
- `notebooks/step_14_offline_selected_colab.ipynb`

## Core Docs

- `docs/icu_readmit_pipeline.md`
- `docs/icu_readmit_rl_pipeline.md`
- `docs/step_09_state_action_recommendation.md`
- `docs/icu_readmit_ddqn_explainer.md`
- `docs/icu_readmit_offline_ddqn_explainer.md`
- `docs/caresim_playbook.md`

## Branch Policy

`main` contains only the active ICU-readmission pipeline and the code it directly depends on.

Broader experimental work and archived branches are preserved outside `main`.

Legacy Step 21 action-selection diagnostics are preserved under:
- `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/`
