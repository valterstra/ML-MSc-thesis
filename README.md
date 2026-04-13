# CareAI

CareAI is an MSc thesis codebase for causal modeling, simulation, and reinforcement learning on MIMIC-IV. The repository contains multiple experimental tracks, but the current active ICU readmission track is now centered on a transformer world model (`CARE-Sim`) and a simulator-based control layer.

## Current Active Track

The most up-to-date pipeline in this repo is the **ICU readmission pipeline**:

1. Build a general ICU cohort with 4-hour state-action trajectories
2. Run FCI-based causal discovery to identify actionable Tier-2 states and drugs
3. Train a transformer world model (`CARE-Sim`) on the Tier-2 RL dataset
4. Evaluate the simulator on held-out one-step and rollout fidelity
5. Use the trained simulator as the environment for:
   - a short-horizon planner baseline
   - a simulator-based DDQN policy

The current Tier-2 primary setup is:

- Dynamic state: `Hb`, `BUN`, `Creatinine`, `HR`, `Shock_Index`
- Static confounders repeated at each step: `age`, `charlson_score`, `prior_ed_visits_6m`
- Actions: `diuretic`, `ivfluid`, `vasopressor`, `antibiotic`
- Action space size: `16` binary combinations

## Current Status

### ICU readmission track

- Steps `01-13`: completed, including FCI variable selection, Tier-2 RL preprocessing, offline DDQN baselines, and older MLP/LSTM/BNN simulators
- Step `14`: completed -- CARE-Sim transformer ensemble trained and saved in `models/icu_readmit/caresim/`
- Step `15`: completed -- CARE-Sim evaluation and simulator sanity checks
- Step `16`: completed -- planner baseline and simulator-based DDQN control layer

### Current best Step 16 result

On the latest `100`-episode `val/test` evaluation:

- `planner` is the strongest controller
- `ddqn` beats `random` and `repeat_last`, but still trails planner
- DDQN performance improved after increasing training budget and slowing epsilon decay, but the learned policy is still concentrated on two actions

Latest Step 16 mean discounted returns:

| Split | Planner | DDQN | Repeat-last | Random |
|------|---------:|-----:|------------:|-------:|
| Val  | 7.65 | 4.16 | 2.44 | 2.24 |
| Test | 8.02 | 4.77 | 2.87 | 2.56 |

Interpretation:

- CARE-Sim is usable as a control environment
- short-horizon planning is currently the best-performing decision method
- DDQN learns a nontrivial policy, but still shows action collapse

## Key Artifacts

### CARE-Sim world model

- Model: `models/icu_readmit/caresim/`
- Training script: `scripts/icu_readmit/step_14_caresim_train.py`
- Evaluation script: `scripts/icu_readmit/step_15_caresim_evaluate.py`
- Colab notebook: `notebooks/step_14_caresim_colab.ipynb`

### CARE-Sim control layer

- Script: `scripts/icu_readmit/step_16_caresim_control.py`
- Control code: `src/careai/icu_readmit/caresim/control/`
- Model outputs: `models/icu_readmit/caresim_control/`
- Reports: `reports/icu_readmit/caresim_control/`
- Colab notebook: `notebooks/step_16_caresim_control_colab.ipynb`

### Current evaluation summaries

- CARE-Sim simulator summary: `reports/icu_readmit/caresim/caresim_summary.json`
- Step 16 control summary: `reports/icu_readmit/caresim_control/step_16_summary.json`
- Step 16 diagnostics: `reports/icu_readmit/caresim_control/step_16_diagnostics_val.json`

## Repository Layout

```text
CareAI/
  configs/
  data/
  docs/
  models/
  notebooks/
  reports/
  scripts/
    icu_readmit/
  src/careai/
    icu_readmit/
      caresim/
      rl/
```

Relevant ICU readmission entry points:

- `scripts/icu_readmit/step_10e_rl_preprocess_tier2.py`
- `scripts/icu_readmit/step_14_caresim_train.py`
- `scripts/icu_readmit/step_15_caresim_evaluate.py`
- `scripts/icu_readmit/step_16_caresim_control.py`

## Other Tracks

This repo also contains:

- a daily hospital pipeline
- a sepsis RL pipeline
- older ICU readmission baselines before CARE-Sim

Those tracks are still useful as references, but they are not the current center of the thesis workflow.

## Where To Read Next

- `STATUS.md` -- concise current repo state
- `CLAUDE.md` -- internal project guide / working memory style summary
- `docs/icu_readmit_pipeline.md` -- ICU readmission preprocessing steps 01-08
- `docs/icu_readmit_rl_pipeline.md` -- ICU readmission RL + simulator steps 09-16
- `docs/caresim_playbook.md` -- CARE-Sim implementation and current design notes
