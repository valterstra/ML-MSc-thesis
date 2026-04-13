# CARE-Sim Playbook
# Transformer World Model and Control Stack for ICU Readmission RL

Written as an internal implementation guide. This file should reflect the
current state of the CARE-Sim stack, not the aspirational state.

---

## 1. Current Status

There are now two relevant CARE-Sim tracks:

1. baseline Tier-2 track
   - old 8-state / 4-action setup
   - fully trained and evaluated
2. selected-set track
   - new 9-state / 5-action setup
   - Step 10 preprocessing complete
   - severity surrogate complete
   - Step 14 training ready
   - Step 12 / Step 13 rollout wiring still pending

So the current stack is no longer just a proposed simulator. The trained
baseline ensemble, evaluation, and control layer all exist and have been run.

---

## 2. Where CARE-Sim Lives in the Pipeline

```text
Step 01-08  ICU preprocessing + cohort build        data/processed/icu_readmit/ICUdataset.csv
Step 09     FCI stability analysis                  reports/icu_readmit/step_09a_causal_states/
Step 10e    Tier-2 RL preprocessing                 data/processed/icu_readmit/rl_dataset_tier2.parquet
Step 10f    Selected-set preprocessing              data/processed/icu_readmit/rl_dataset_selected.parquet
Step 10g    Selected-state severity surrogate       models/icu_readmit/severity_selected/
Step 11b/c  Offline DDQN + discharge baselines      models/icu_readmit/tier2/, tier2_discharge/
Step 13     Older MLP/LSTM/BNN simulators           models/icu_readmit/simulator/
Step 14     CARE-Sim training                       models/icu_readmit/caresim*/ 
Step 15     CARE-Sim evaluation                     reports/icu_readmit/caresim*/
Step 13     CARE-Sim control layer                  models/icu_readmit/caresim_control*/
                                                    reports/icu_readmit/caresim_control*/
```

---

## 3. Input Data

### Baseline Tier-2 input

- `data/processed/icu_readmit/rl_dataset_tier2.parquet`

This canonical baseline dataset contains:

- Dynamic state (5):
  - `s_Hb`
  - `s_BUN`
  - `s_Creatinine`
  - `s_HR`
  - `s_Shock_Index`
- Static confounders repeated at each step (3):
  - `s_age`
  - `s_charlson_score`
  - `s_prior_ed_visits_6m`
- Actions (4):
  - `vasopressor_b`
  - `ivfluid_b`
  - `antibiotic_b`
  - `diuretic_b`

Interpretation:

- state dimension = `8`
- action dimension = `4`
- action space for control = `16` binary combinations

### Selected-set input

- `data/processed/icu_readmit/rl_dataset_selected.parquet`

This selected-set dataset contains:

- Dynamic state (6):
  - `s_Hb`
  - `s_BUN`
  - `s_Creatinine`
  - `s_Phosphate`
  - `s_HR`
  - `s_Chloride`
- Static confounders repeated at each step (3):
  - `s_age`
  - `s_charlson_score`
  - `s_prior_ed_visits_6m`
- Actions (5):
  - `vasopressor_b`
  - `ivfluid_b`
  - `antibiotic_b`
  - `diuretic_b`
  - `mechvent_b`

Interpretation:

- state dimension = `9`
- action dimension = `5`
- action space for control = `32` binary combinations

---

## 4. Reward Design

### Baseline Tier-2 track

The baseline simulator uses:

- dense reward based on SOFA delta
- terminal reward encoding readmission implicitly

In that baseline path, the reward is part of the replay parquet and the model
still includes a reward head.

### Selected-set track

The selected-set direction is cleaner:

- Step 14 selected training disables the reward head
- a learned selected-state severity surrogate is trained separately against real SOFA
- future rollout reward should come from:
  - `severity(state_t) - severity(state_{t+1})`

Current severity-surrogate artifacts:

- `models/icu_readmit/severity_selected/ridge_sofa_surrogate.joblib`
- `models/icu_readmit/severity_selected/severity_surrogate_config.json`
- `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`
- `reports/icu_readmit/severity_selected/severity_surrogate_coefficients.csv`

Current full-data surrogate performance:

- validation MAE: about `1.86`
- validation R^2: about `0.279`
- validation Spearman: about `0.517`
- test MAE: about `1.84`
- test R^2: about `0.273`
- test Spearman: about `0.501`

---

## 5. Termination Design

The simulator's `terminal_prob` means:

- probability the ICU trajectory ends at that step

It does **not** mean 30-day readmission probability.

In rollouts, `done` occurs when:

1. `terminal_prob > 0.5`, or
2. the rollout hits the configured `max_steps`

The newer structured / selected Step 14 tracks now use:

- random training windows
- explicit time feature
- static context freezing

to reduce the old sequence-boundary artifact.

---

## 6. File Map

### Core CARE-Sim source

```text
src/careai/icu_readmit/caresim/
    model.py          -- CareSimGPT transformer
    dataset.py        -- replay dataset loader, now supports inferred parquet schema
    train.py          -- loss, training loop, EnsembleTrainer
    ensemble.py       -- multi-model inference wrapper
    simulator.py      -- CareSimEnvironment
```

### Control layer

```text
src/careai/icu_readmit/caresim/control/
    actions.py        -- action encoding/decoding
    observation.py    -- recent state/action window builder
    planner.py        -- short-horizon action planner
    ddqn.py           -- simulator-based DDQN training
    evaluation.py     -- seed episode loading + policy evaluation
```

### Relevant scripts

```text
scripts/icu_readmit/
    step_10a_rl_preprocess_selected.py
    step_10b_train_selected_severity_surrogate.py
    step_14_caresim_train.py
    step_14_caresim_train_structured.py
    step_14_caresim_train_selected.py
    step_12a_caresim_evaluate.py
    step_13a_caresim_control.py
```

### Colab entry points

```text
notebooks/
    step_14_caresim_colab.ipynb
    step_14_caresim_structured_colab.ipynb
    step_14_caresim_selected_colab.ipynb
    step_13a_caresim_selected_colab.ipynb
    step_13b_markovsim_selected_colab.ipynb
```

---

## 7. Baseline Empirical Results

### Step 15 baseline CARE-Sim quality

From `reports/icu_readmit/caresim/caresim_summary.json`:

- one-step val next-state MSE: about `0.0830`
- one-step test next-state MSE: about `0.0834`
- reward MAE: about `1.87`
- terminal accuracy: about `0.956`
- mean uncertainty: about `0.0706`

Closed-loop rollout:

- val step-1 state MSE: about `0.1169`
- val step-5 state MSE: about `0.2199`
- test step-1 state MSE: about `0.1080`
- test step-5 state MSE: about `0.1764`

Interpretation:

- baseline CARE-Sim is stable enough for short-horizon control
- error grows with horizon, so policy comparisons should stay modest-horizon

### Step 13 baseline planner vs DDQN

Latest 100-episode evaluation:

| Split | Planner | DDQN | Repeat-last | Random |
|------|---------:|-----:|------------:|-------:|
| Val  | 7.65 | 4.16 | 2.44 | 2.24 |
| Test | 8.02 | 4.77 | 2.87 | 2.56 |

Interpretation:

- planner is the strongest control policy
- DDQN clearly beats random
- DDQN moderately beats repeat-last
- DDQN still trails planner and remains partially action-collapsed

---

## 8. Current Working Conclusion

The baseline Tier-2 CARE-Sim stack remains the current fully executed reference.

The selected-set track is now the main forward path for new experiments:

- selected replay dataset: ready
- selected severity surrogate: ready
- selected Step 14 training: ready
- selected Step 12 / Step 13 reward wiring: next implementation task
