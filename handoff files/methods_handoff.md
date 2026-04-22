# Methods Handoff

This document is a code-grounded methods handoff for thesis writing. It explains how the active ICU-readmission pipeline moves from the processed trajectory dataset to the final causal selection, simulator construction, reinforcement learning setup, and evaluation design.

This handoff is based on the active repository code under:

- `scripts/icu_readmit/`
- `src/careai/icu_readmit/`
- `docs/step_09_state_action_recommendation.md`

It also uses the currently saved artifacts and report files under:

- `data/processed/icu_readmit/`
- `models/icu_readmit/`
- `reports/icu_readmit/`

Where the code and saved artifacts are not perfectly aligned, that is stated explicitly.

## 1. Modeling objective

### Final empirical objective

The final active pipeline is not only a prediction pipeline. The final objective is **policy learning for ICU intervention recommendation under a 30-day readmission objective**, using two simulator-based control branches and one offline RL comparison branch.

The final question implemented in code is:

- given a reduced ICU state representation and a reduced intervention/action space,
- can we learn a policy that chooses actions expected to improve short-horizon patient trajectories and reduce terminal readmission risk,
- and how do simulator-trained policies compare with a policy learned directly from logged real trajectories?

In the active codebase, this final objective has three components:

1. **Causally informed selection of the modeling interface**
- choose a small set of state variables and actions that are both clinically meaningful and plausibly modifiable

2. **Simulator-based optimization**
- train a patient transition model
- use that simulator to train and evaluate control policies

3. **Off-policy evaluation of learned policies on held-out logged data**
- train an offline DDQN directly on logged ICU trajectories
- evaluate `offline_ddqn`, `caresim_ddqn`, and `markovsim_ddqn` on the same held-out logged-data benchmark

### What the final thesis objective is not

The final thesis objective is **not**:

- pure treatment-effect estimation in the causal inference sense
- only risk prediction
- only a simulator paper
- only off-policy evaluation

The active thesis pipeline combines:

- causal variable/action selection,
- simulator learning,
- simulator-based policy optimization,
- and offline RL comparison.

### Final versus exploratory branches

**Final methods kept in the active thesis pipeline**

- selected causal state/action selection
- selected replay preprocessing
- severity surrogate model
- terminal readmission model
- CARE-Sim selected causal simulator
- MarkovSim selected causal simulator
- simulator-side planner baseline
- simulator-side DDQN
- offline DDQN with held-out OPE

**Exploratory or non-final elements still present in code**

- generic CARE-Sim reward-head training path (`predict_reward=True`) is available in generic CARE-Sim code but is not used in the selected final pipeline
- support models used for OPE in `src/careai/icu_readmit/rl/evaluation.py` are not final policies
- `train_sarsa_physician()` exists in `src/careai/icu_readmit/rl/continuous.py` but is not called by any active step script

## 2. Selection from broad data to final modeling interface

### Starting point

The broad processed ICU trajectory dataset used by the active selection/preprocessing branch is:

- `data/processed/icu_readmit/ICUdataset.csv`

This file is the output of the upstream data-construction pipeline (steps 01-08). The final modeling pipeline does not start from raw MIMIC tables. It starts from this bloc-level processed ICU stay dataset.

### Broad candidate state variables considered

There are two broad candidate-state pools in the active selection code.

#### A. Outcome-relevance state pool used in `legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance.py`

This script defines:

- `TIME_VARYING_FEATURES = CHART_FIELD_NAMES + LAB_FIELD_NAMES + ['SOFA', 'SIRS', 'Shock_Index', 'PaO2_FiO2', 'mechvent', 'extubated']`

`CHART_FIELD_NAMES` from `src/careai/icu_readmit/columns.py` are:

- `Height_cm`
- `Weight_kg`
- `GCS_Eye`
- `GCS_Verbal`
- `GCS_Motor`
- `RASS`
- `HR`
- `SysBP`
- `MeanBP`
- `DiaBP`
- `NIBP_Diastolic`
- `Arterial_BP_Sys`
- `Arterial_BP_Dia`
- `RR`
- `RR_Spontaneous`
- `RR_Total`
- `SpO2`
- `Temp_C`
- `Temp_F`
- `CVP`
- `PAPsys`
- `PAPmean`
- `PAPdia`
- `CI`
- `SVR`
- `Interface`
- `FiO2_100`
- `FiO2_1`
- `O2flow`
- `PEEP`
- `TidalVolume`
- `TidalVolume_Observed`
- `MinuteVentil`
- `PAWmean`
- `PAWpeak`
- `PAWplateau`
- `Pain_Level`
- `cam_icu`
- `GCS`

`LAB_FIELD_NAMES` from `src/careai/icu_readmit/columns.py` are:

- `Potassium`
- `Sodium`
- `Chloride`
- `Glucose`
- `BUN`
- `Creatinine`
- `Magnesium`
- `Calcium`
- `Ionised_Ca`
- `CO2_mEqL`
- `SGOT`
- `SGPT`
- `Total_bili`
- `Direct_bili`
- `Total_protein`
- `Albumin`
- `Troponin`
- `CRP`
- `Hb`
- `Ht`
- `RBC_count`
- `WBC_count`
- `Platelets_count`
- `PTT`
- `PT`
- `ACT`
- `INR`
- `Arterial_pH`
- `paO2`
- `paCO2`
- `Arterial_BE`
- `Arterial_lactate`
- `HCO3`
- `ETCO2`
- `SvO2`
- `Phosphate`
- `Anion_Gap`
- `Alkaline_Phosphatase`
- `LDH`
- `Fibrinogen`
- `Neuts_pct`
- `Lymphs_pct`
- `Monos_pct`
- `Eos_pct`
- `Basos_pct`

Derived state-like variables explicitly added on top of those lists are:

- `SOFA`
- `SIRS`
- `Shock_Index`
- `PaO2_FiO2`
- `mechvent`
- `extubated`

#### B. Robust action-state controllability pool used in `step_09_state_action_selection/step_04b_action_state_stability_robust.py`

The robust multivariate action-state analysis was not run on the full 80+ variable pool. It was run on a narrower state pool:

- `BUN`
- `Hb`
- `Platelets_count`
- `WBC_count`
- `cumulated_balance`
- `Creatinine`
- `PT`
- `input_4hourly_tev`
- `PTT`
- `Glucose`
- `output_total`
- `HR`
- `RR`
- `Alkaline_Phosphatase`
- `Ht`
- `Temp_C`
- `SpO2`
- `Phosphate`
- `Shock_Index`
- `Chloride`
- `CO2_mEqL`
- `Fibrinogen`
- `SGOT`
- `Pain_Level`
- `Lymphs_pct`
- `Sodium`
- `paCO2`
- `TidalVolume_Observed`

### Final selected state variables

The final selected state set is documented in both:

- `docs/step_09_state_action_recommendation.md`
- `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`

The final selected dynamic states are:

- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

The final selected static confounders carried into the state are:

- `age`
- `charlson_score`
- `prior_ed_visits_6m`

Therefore the final selected state dimension is `9`.

### Broad candidate actions considered

Again there are two broad candidate-action pools.

#### A. Outcome-relevance / stay-level treatment summary pool in `legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance.py`

This action pool contains:

- `vasopressor_dose`
- `ivfluid_dose`
- `antibiotic_active`
- `anticoagulant_active`
- `diuretic_active`
- `steroid_active`
- `insulin_active`
- `opioid_active`
- `sedation_active`
- `transfusion_active`
- `electrolyte_active`

This script uses stay-level aggregates such as `frac_*`, `ever_*`, `mean_*`, and `max_*`.

#### B. Robust action-state controllability pool in `step_09_state_action_selection/step_04b_action_state_stability_robust.py`

The robust multivariate controllability run uses:

- `vasopressor`
- `ivfluid`
- `antibiotic`
- `anticoagulant`
- `diuretic`
- `steroid`
- `insulin`
- `sedation`
- `mechvent`

### Final selected actions

The final selected action set, documented in `docs/step_09_state_action_recommendation.md` and implemented in `step_10a_rl_preprocess_selected.py`, is:

- `vasopressor`
- `ivfluid`
- `antibiotic`
- `diuretic`
- `mechvent`

These are encoded as five binary columns in the replay dataset:

- `vasopressor_b`
- `ivfluid_b`
- `antibiotic_b`
- `diuretic_b`
- `mechvent_b`

The integer action code `a` is built as:

- `vasopressor_b * 1`
- `ivfluid_b * 2`
- `antibiotic_b * 4`
- `diuretic_b * 8`
- `mechvent_b * 16`

This yields a discrete action space of `32` possible binary combinations.

### Exact criteria used for keeping or dropping variables

The final selected interface is **not** the output of one single automatic selector. It is a combination of predictive screening, causal/action-responsiveness screening, and manual pruning.

#### Criterion 1: predictive relevance to `readmit_30d`

Implemented in:

- `scripts/icu_readmit/step_09_state_action_selection/step_02_variable_ranking.py`
- `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance.py`

Relevant code behavior:

- LightGBM models rank features by importance for predicting `readmit_30d`
- step 09 also flags high missingness (`missing_frac > 0.20`)
- legacy step 21a builds separate models for:
  - time-varying physiological features
  - static features
  - action summaries
  - combined feature sets

#### Criterion 2: action responsiveness / controllability

Implemented in:

- `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness.py`

For each candidate feature `f`, the code predicts `f_{t+1}` from:

- current time-varying states
- static context (`age`, `charlson_score`, `gender`)
- current actions

The explicit screening statistic is:

- `action_share = action_importance / total_importance`

The implemented threshold is:

- `ACTION_SHARE_THRESHOLD = 0.05`

Features above this threshold are treated as action-responsive.

#### Criterion 3: robust causal/action-state stability

Implemented in:

- `scripts/icu_readmit/step_09_state_action_selection/step_03b_random_stability_robust.py`
- `scripts/icu_readmit/step_09_state_action_selection/step_04b_action_state_stability_robust.py`

These scripts produce stability summaries stored under:

- `reports/icu_readmit/step_09_state_action_selection/random_stability_summary.json`
- `reports/icu_readmit/step_09_state_action_selection/action_state_summary_robust.json`

These summaries are used as evidence for whether a variable is:

- strongly associated with readmission,
- robustly modifiable by actions,
- and non-redundant enough to keep in a reduced modeling interface.

#### Criterion 4: manual redundancy and clinical plausibility pruning

This final step is documented in:

- `docs/step_09_state_action_recommendation.md`

Examples explicitly stated there:

- `Ht` is not kept in the final main set because it is treated as redundant with `Hb`
- `Shock_Index` is marked as borderline
- `insulin` is marked as borderline
- `anticoagulant` is dropped from the main set because its strongest support is for `PTT`, which is not kept in the main final state set
- `sedation` is dropped because it is described as mixed and more procedural/confounded
- `steroid` is dropped because it was unusable in the robust action-state run

### Was the final selection causal, predictive, heuristic, or a combination?

It is a **combination**:

- predictive screening
- action-responsiveness screening
- causal stability evidence
- manual domain pruning

The correct description is:

- **causally informed variable and action selection**

It is not correct to describe the final state/action interface as purely causal discovery output or purely predictive feature selection output.

### Scripts and documents that implement this step

Main scripts:

- `scripts/icu_readmit/step_09_state_action_selection/step_01_stay_level.py`
- `scripts/icu_readmit/step_09_state_action_selection/step_02_variable_ranking.py`
- `scripts/icu_readmit/step_09_state_action_selection/step_03_random_stability.py`
- `scripts/icu_readmit/step_09_state_action_selection/step_03b_random_stability_robust.py`
- `scripts/icu_readmit/step_09_state_action_selection/step_04_action_state_stability.py`
- `scripts/icu_readmit/step_09_state_action_selection/step_04b_action_state_stability_robust.py`
- `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21a_outcome_relevance.py`
- `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness.py`
- `scripts/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21c_focused_causal_graphs.py`

Final decision document:

- `docs/step_09_state_action_recommendation.md`

Final selected preprocessing implementation:

- `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`

### Final output files created by this step

Selection evidence / reports:

- `reports/icu_readmit/step_09_state_action_selection/random_stability_summary.json`
- `reports/icu_readmit/step_09_state_action_selection/action_state_summary_robust.json`
- `reports/icu_readmit/legacy/step_21_action_selection_diagnostics/step_21b_transition_responsiveness/step_21b_transition_responsiveness.json`
- `docs/step_09_state_action_recommendation.md`

Final selected modeling files:

- `data/processed/icu_readmit/rl_dataset_selected.parquet`
- `data/processed/icu_readmit/scaler_params_selected.json`
- `data/processed/icu_readmit/static_context_selected.parquet`

## 3. Formal decision process definition

### One observation unit

The basic row-level unit in the selected replay dataset is one ICU stay bloc at time `t`, stored in:

- `data/processed/icu_readmit/rl_dataset_selected.parquet`

The transition unit is one within-stay consecutive pair:

- current bloc `t`
- next bloc `t+1`

The code constructs this explicitly in:

- `build_transitions()` in `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`

### State

The final selected state is a 9-dimensional vector composed of:

Dynamic state variables:

- `s_Hb`
- `s_BUN`
- `s_Creatinine`
- `s_Phosphate`
- `s_HR`
- `s_Chloride`

Static confounders:

- `s_age`
- `s_charlson_score`
- `s_prior_ed_visits_6m`

These are standardized using training-split statistics in `step_10a_rl_preprocess_selected.py`, with clipping and some `log1p` transforms before z-scoring.

### Action

The final action is a 5-dimensional binary vector:

- `vasopressor_b`
- `ivfluid_b`
- `antibiotic_b`
- `diuretic_b`
- `mechvent_b`

The action is also stored as a single integer code:

- `a` in `{0, ..., 31}`

### Next state

The next state is stored as:

- `s_next_Hb`
- `s_next_BUN`
- `s_next_Creatinine`
- `s_next_Phosphate`
- `s_next_HR`
- `s_next_Chloride`
- `s_next_age`
- `s_next_charlson_score`
- `s_next_prior_ed_visits_6m`

For static variables, `step_10a` sets:

- `s_next_age = s_age`
- `s_next_charlson_score = s_charlson_score`
- `s_next_prior_ed_visits_6m = s_prior_ed_visits_6m`

### Reward

There are **two reward layers** in the active codebase, and this distinction matters.

#### Stored replay reward in `rl_dataset_selected.parquet`

Constructed in `step_10a_rl_preprocess_selected.py`:

- nonterminal: `r = SOFA_t - SOFA_{t+1}`
- terminal: `r = +15` if `readmit_30d == 0`, else `r = -15`

This reward is stored in the parquet file as column `r`.

#### Final active reward used by simulator control and offline RL comparison

The final active modeling/evaluation pipeline does **not** rely only on the stored `r`.

In:

- `src/careai/icu_readmit/caresim/simulator.py`
- `src/careai/icu_readmit/markovsim/simulator.py`
- `scripts/icu_readmit/step_14_offline_selected.py`

the effective reward is recomputed as:

1. **dense severity improvement reward**
- default current mode: `handcrafted`
- reward = `severity(current_state) - severity(next_state)`

2. **terminal readmission reward**
- via `LightGBMReadmitModel.terminal_reward(next_state)`
- implemented as:
  - `reward_scale - 2 * reward_scale * p_readmit`
- default `reward_scale = 15.0`

So the final control/offline RL branch uses a reward function that is aligned across:

- CARE-Sim control
- MarkovSim control
- offline DDQN comparison

but not identical to the raw `r` stored in step 10a.

### Terminal condition

There are two terminal notions in the active pipeline.

#### Logged-data terminal condition

In `step_10a`, terminal means:

- last bloc of a stay

This is stored as:

- `done = 1` if current row is the final bloc for that stay

#### Simulator control terminal condition

In both simulator environments, terminal occurs if:

- the simulator's terminal probability exceeds `0.5`, or
- the rollout hits the configured max step cap

This logic is implemented in:

- `src/careai/icu_readmit/caresim/simulator.py`
- `src/careai/icu_readmit/markovsim/simulator.py`

### Horizon / episode length

For logged trajectories:

- horizon is variable and equals the observed ICU stay length in blocs

For simulator control:

- seed history length is `5` rows
- rollout cap is `5` decision steps
- planner lookahead horizon is `3`

These values are used in the active control code.

### How readmission enters the setup

Readmission enters the pipeline in three distinct places.

1. **Selection target**
- Step 09 uses `readmit_30d` to identify outcome-relevant variables

2. **Replay-data terminal label**
- step 10a terminal reward uses `readmit_30d`

3. **Final terminal reward model**
- `scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py`
- trains a LightGBM classifier on terminal rows only
- predicts readmission probability from the selected state
- this model is then used to assign terminal reward during simulator control and in step 14 reward recomputation

### Is this explicitly framed as an MDP?

The active pipeline treats the problem as an **approximate finite-horizon MDP/control problem**, but not in a perfectly strict textbook Markov sense.

The code makes two approximations:

1. the selected 9-dimensional state is treated as the current patient state for transition modeling
2. the policy does not observe only the current state; it observes a **5-step history window**

The policy observation builder in:

- `src/careai/icu_readmit/caresim/control/observation.py`
- `src/careai/icu_readmit/markovsim/control/observation.py`

constructs a flattened `70`-dimensional observation:

- `5 * (9 state + 5 action) = 70`

So the practical framing is:

- simulator transition model: approximate state-action transition model
- policy input: short-history observation window

## 4. Simulator / transition model

There are **two final simulators** in the active thesis pipeline:

- CARE-Sim
- MarkovSim

Both are part of the final thesis pipeline. MarkovSim is not only exploratory. It is the explicit simpler baseline simulator.

### 4.1 CARE-Sim

#### Model class

Main classes:

- `careai.icu_readmit.caresim.model.CareSimGPT`
- `careai.icu_readmit.caresim.ensemble.CareSimEnsemble`

User-facing training entrypoint:

- `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`

Internal delegated training code:

- `scripts/icu_readmit/step_14_caresim_train_selected.py`
- `scripts/icu_readmit/step_14_caresim_train.py`

#### Inputs

CARE-Sim takes a sequence of:

- state vectors of dimension `9`
- action vectors of dimension `5`

The active selected-causal path uses:

- `--freeze-static-context`
- `--use-time-feature`
- `--train-window-mode random`
- `--val-window-mode last`
- `--no-predict-reward`
- `--causal-constraints`

So the active final model sees the full selected state/action history, with a causal action mask applied inside the model.

#### Outputs

Generic CARE-Sim can output:

- next state
- optional reward head
- terminal logit

The final active selected-causal path disables the reward head:

- `predict_reward = false`

So the final active CARE-Sim predicts:

- next state
- terminal probability

#### Joint vs separate next-state prediction

CARE-Sim predicts next state jointly through one transformer model per ensemble member.

It is not a bank of separate per-variable regressors.

#### Does CARE-Sim predict reward directly?

In the active selected-causal path: **no**.

Dense reward is computed after transition prediction from the predicted next state, using either:

- handcrafted severity
- or the selected severity surrogate

The final active runs use handcrafted severity by default.

#### Uncertainty / stochasticity

CARE-Sim uses an ensemble:

- `n_models = 5` in the current saved selected-causal run metadata

Uncertainty is taken from ensemble dispersion:

- standard deviation across ensemble member predictions

The simulator exposes:

- next-state mean prediction
- uncertainty score derived from state prediction spread

Transitions in the environment are effectively deterministic at the ensemble mean. The ensemble provides epistemic uncertainty rather than explicitly sampling stochastic next states.

#### Causal constraints

The selected causal mask is implemented in:

- `FCI_SELECTED_CAUSAL_MASK` in `src/careai/icu_readmit/caresim/model.py`

The intended direct action-state links are:

- `Hb <- vasopressor, ivfluid, antibiotic, mechvent`
- `BUN <- ivfluid, diuretic`
- `Creatinine <- ivfluid, diuretic`
- `Phosphate <- ivfluid, antibiotic, diuretic`
- `HR <- vasopressor, mechvent`
- `Chloride <- ivfluid, diuretic`

#### Important implementation caveat: CARE-Sim dynamic/static index mismatch

This is the single most important simulator caveat in the active code.

In `src/careai/icu_readmit/caresim/model.py`, the hard-coded indices are:

- `DYNAMIC_STATE_IDX = (0, 1, 2, 3, 4)`
- `STATIC_STATE_IDX = (5, 6, 7)`

But the selected 9-state interface intended by step 10a is:

Dynamic:

- index 0 `Hb`
- index 1 `BUN`
- index 2 `Creatinine`
- index 3 `Phosphate`
- index 4 `HR`
- index 5 `Chloride`

Static:

- index 6 `age`
- index 7 `charlson_score`
- index 8 `prior_ed_visits_6m`

Because the selected CARE-Sim run uses `freeze_static_context=True`, the model only learns state loss on the hard-coded dynamic indices and copies the others forward.

Therefore, in the current active CARE-Sim implementation:

- `Chloride` is intended to be dynamic
- but it is effectively copied forward and not learned dynamically

This means the final saved selected CARE-Sim artifacts do **not** fully implement the intended 6-dynamic-state design.

This should be stated explicitly in any thesis methods chapter that describes the final CARE-Sim implementation.

#### Training

CARE-Sim is trained on:

- `data/processed/icu_readmit/rl_dataset_selected.parquet`

through the selected-causal training entrypoint:

- `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`

Current saved selected-causal run metadata in:

- `models/icu_readmit/caresim_selected_causal/run_meta.json`

show:

- `n_models = 5`
- `state_dim = 9`
- `action_dim = 5`
- `d_model = 256`
- `n_heads = 8`
- `n_layers = 4`
- `dropout = 0.1`
- `max_seq_len = 80`
- `predict_reward = false`
- `freeze_static_context = true`
- `use_time_feature = true`
- `use_causal_constraints = true`
- `n_epochs = 30`
- `lr = 0.001`
- `batch_size = 64`
- `w_state = 1.0`
- `w_reward = 0.5`
- `w_term = 0.5`

#### Trained artifact files

Current selected-causal simulator artifacts are stored in:

- `models/icu_readmit/caresim_selected_causal/`

Key files:

- `ensemble_config.json`
- `run_meta.json`
- `member_0/best_model.pt`
- `member_1/best_model.pt`
- `member_2/best_model.pt`
- `member_3/best_model.pt`
- `member_4/best_model.pt`

#### Diagnostics / simulator quality metrics

Evaluation script:

- `scripts/icu_readmit/step_12a_caresim_evaluate.py`

Outputs:

- `caresim_one_step_val.json`
- `caresim_one_step_test.json`
- `caresim_rollout_val.json`
- `caresim_rollout_test.json`
- `caresim_counterfactual_val.csv`
- `caresim_summary.json`

Currently saved local summary:

- `reports/icu_readmit/caresim_selected_causal/caresim_summary.json`

Current saved one-step metrics:

- val `next_state_mse = 0.06783302429281862`
- test `next_state_mse = 0.0680623747749937`
- val `terminal_accuracy = 0.9536279597306005`
- test `terminal_accuracy = 0.9535925055720187`
- val `mean_uncertainty = 0.020686472240833146`

Current saved reward MAE:

- val `reward_mae = 1.1143156878210705`

Important caveat:

- the currently saved CARE-Sim step 12 summary was generated with `use_terminal_readmit_reward = false`
- rollout/counterfactual sample counts were small (`rollout_patients = 20`, `counterfactual_patients = 3`)
- so the current saved CARE-Sim reward diagnostics are not directly aligned with the saved MarkovSim summary

### 4.2 MarkovSim

#### Model class

Main classes:

- `careai.icu_readmit.markovsim.model.MarkovSimEnsemble`
- `careai.icu_readmit.markovsim.simulator.MarkovSimEnvironment`

Training entrypoint:

- `scripts/icu_readmit/step_11b_markovsim_train.py`

#### Inputs

MarkovSim takes only:

- current selected state (`9` dims)
- current selected action (`5` dims)

It does not use full temporal history internally.

#### Outputs

MarkovSim predicts:

- next values for the dynamic state variables
- terminal probability

Static confounders are copied through unchanged.

#### Joint vs separate next-state prediction

MarkovSim predicts next-state variables **separately**:

- one `Ridge` regression per dynamic next-state target

and:

- one `LogisticRegression` for terminal probability

This is implemented in:

- `src/careai/icu_readmit/markovsim/model.py`

#### Does MarkovSim predict reward directly?

No.

It predicts transitions and terminal probability only. Dense reward is computed from state change through the severity function. Terminal reward comes from the separate terminal readmission model.

#### Uncertainty / stochasticity

MarkovSim is deterministic at the mean prediction. Uncertainty is represented by residual standard deviations estimated during training.

In practice:

- no noise is sampled during environment transitions
- uncertainty is carried as a residual-scale diagnostic quantity

#### Causal constraints

MarkovSim uses:

- `SELECTED_CAUSAL_ACTION_MASK` in `src/careai/icu_readmit/markovsim/model.py`

The design is:

- all current state features can enter each next-state regression
- only causally allowed action features are included per target

This means the causal restriction is on direct action effects, not on state-state dependencies.

#### Training

Training function:

- `fit_markovsim_from_dataframe()` in `src/careai/icu_readmit/markovsim/train.py`

Training data:

- train split of `rl_dataset_selected.parquet`

Current saved run metadata in:

- `models/icu_readmit/markovsim_selected_causal/run_meta.json`

show:

- `ridge_alpha = 1.0`
- `terminal_c = 1.0`
- `max_iter = 1000`

The model uses:

- `StandardScaler` for predictors
- `class_weight = "balanced"` in the terminal logistic regression

#### Trained artifact files

Stored in:

- `models/icu_readmit/markovsim_selected_causal/`

Key files:

- `markovsim_bundle.pkl`
- `markovsim_config.json`
- `run_meta.json`

#### Diagnostics / simulator quality metrics

Evaluation script:

- `scripts/icu_readmit/step_12b_markovsim_evaluate.py`

Outputs:

- `markovsim_one_step_val.json`
- `markovsim_one_step_test.json`
- `markovsim_rollout_val.json`
- `markovsim_rollout_test.json`
- `markovsim_counterfactual_val.csv`
- `markovsim_summary.json`

Current saved local summary:

- `reports/icu_readmit/markovsim_selected_causal/markovsim_summary.json`

Current saved one-step metrics:

- val `next_state_mse = 0.07206787914037704`
- test `next_state_mse = 0.07180938124656677`
- val `reward_mae = 0.05209772661328316`
- test `reward_mae = 0.05172107741236687`
- val `terminal_accuracy = 0.551841660371929`
- test `terminal_accuracy = 0.5428793978724464`

Current saved training metrics from `run_meta.json` / training output:

- `n_rows = 1057632`
- `feature_dim = 14`
- `transition_train_mse = 0.10653316229581833`
- `terminal_train_accuracy = 0.5478881123112765`

### Is the simulator part of the final thesis pipeline?

Yes.

Both:

- CARE-Sim
- MarkovSim

are part of the final active thesis pipeline. The point of the active repository is explicitly to compare a richer transformer world model with a simpler causal Markov simulator.

## 5. Reinforcement learning methods

This section separates:

- final methods used in the thesis
- exploratory or support methods present in code but not kept as final thesis methods

### A. Final methods used in the thesis

#### 1. CARE-Sim planner baseline

Algorithm:

- simulator-based planner
- not gradient-trained
- implemented in `src/careai/icu_readmit/caresim/control/planner.py`

Why included:

- provides a non-neural control baseline that directly searches actions inside the simulator
- serves as a strong "what-if search" comparator to DDQN

Training data:

- none; planner is not trained

Reward:

- simulator reward minus uncertainty penalty
- dense severity improvement reward
- optional terminal readmission reward

Action space:

- 32 discrete actions from the 5 binary selected actions

Important hyperparameters:

- `horizon = 3`
- `uncertainty_penalty = 0.25`

How it works:

- from the current state/history, evaluate all 32 candidate first actions
- for each candidate, repeat that same action for `horizon=3` simulated steps in a cloned simulator
- score the rollout
- choose the best current action
- execute only one real step
- replan from the new state

Script:

- `scripts/icu_readmit/step_13a_caresim_control.py`

Outputs:

Current active code writes `step_13a_*` report names.
Current saved local artifact names still use older `step_16_*` names:

- `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`
- `reports/icu_readmit/caresim_control_selected_causal/step_16_policy_traces_val.csv`
- `reports/icu_readmit/caresim_control_selected_causal/step_16_policy_traces_test.csv`

#### 2. CARE-Sim DDQN

Algorithm:

- Dueling Double DQN

Implementation:

- `src/careai/icu_readmit/caresim/control/ddqn.py`
- network class in `src/careai/icu_readmit/rl/networks.py`

Why included:

- final simulator-trained RL policy on top of CARE-Sim

Training data:

- experience generated online inside CARE-Sim

Reward:

- simulator reward minus uncertainty penalty
- simulator reward itself is dense severity improvement plus optional terminal readmission reward

Action space:

- 32 discrete actions

Observation space:

- 5-step history window
- `70` dimensions

Most important hyperparameters in current code:

- `observation_window = 5`
- `rollout_steps = 5`
- `gamma = 0.99`
- `lr = 1e-4`
- `batch_size = 64`
- `replay_capacity = 20000`
- `warmup_steps = 500`
- `train_steps = 20000`
- `target_sync_every = 250`
- `epsilon_start = 1.0`
- `epsilon_end = 0.10`
- `epsilon_decay_steps = 20000`
- `uncertainty_penalty = 0.25`

Important bookkeeping caveat:

- currently saved DDQN config files on disk use slightly different run-time values than the current code defaults
- for methods writing, the current code should be treated as the canonical algorithm definition

Script:

- `scripts/icu_readmit/step_13a_caresim_control.py`

Outputs:

Model artifacts:

- `models/icu_readmit/caresim_control_selected_causal/ddqn_model.pt`
- `models/icu_readmit/caresim_control_selected_causal/ddqn_train_config.json`
- `models/icu_readmit/caresim_control_selected_causal/ddqn_train_metrics.json`

Current saved local evaluation summary:

- `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`

Current saved discounted returns:

- val `ddqn = 8.692149344347015`
- test `ddqn = 8.836255628219314`

#### 3. MarkovSim planner baseline

Algorithm:

- same planner logic as CARE-Sim, but running inside MarkovSim

Implementation:

- `src/careai/icu_readmit/markovsim/control/planner.py`

Why included:

- search baseline for the simpler simulator branch

Training data:

- none

Reward:

- same final reward logic as CARE-Sim control

Action space:

- 32 discrete actions

Key hyperparameters:

- `horizon = 3`
- `uncertainty_penalty = 0.25`

Script:

- `scripts/icu_readmit/step_13b_markovsim_control.py`

Outputs:

Current local report naming still uses older filenames:

- `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`

Current saved discounted returns:

- val `planner = 8.973150204867125`
- test `planner = 9.131546391099691`

#### 4. MarkovSim DDQN

Algorithm:

- Dueling Double DQN

Implementation:

- `src/careai/icu_readmit/markovsim/control/ddqn.py`

Why included:

- final simulator-trained RL policy on top of MarkovSim

Training data:

- online interaction generated inside MarkovSim

Reward:

- dense severity improvement
- optional terminal readmission reward
- uncertainty penalty during policy optimization

Action space:

- 32 discrete actions

Observation space:

- same 70-dimensional 5-step `(state, action)` history as CARE-Sim control

Key hyperparameters:

- same DQN control configuration family as CARE-Sim

Script:

- `scripts/icu_readmit/step_13b_markovsim_control.py`

Outputs:

Model artifacts:

- `models/icu_readmit/markovsim_control_selected_causal/ddqn_model.pt`
- `models/icu_readmit/markovsim_control_selected_causal/ddqn_train_config.json`
- `models/icu_readmit/markovsim_control_selected_causal/ddqn_train_metrics.json`

Current saved local evaluation summary:

- `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`

Current saved discounted returns:

- val `ddqn = 8.976655859579582`
- test `ddqn = 9.101420106375041`

#### 5. Offline DDQN

Algorithm:

- Dueling Double DQN with prioritized replay and soft target updates

Implementation:

- training script: `scripts/icu_readmit/step_14_offline_selected.py`
- training loop: `src/careai/icu_readmit/rl/continuous.py`

Why included:

- final real-data comparison branch
- compares simulator-trained RL to RL learned directly from logged trajectories

Training data:

- fixed logged transitions from `rl_dataset_selected.parquet`
- windowed into 5-step observations (`obs_dim = 70`)

Reward:

- recomputed in `step_14_offline_selected.py`
- dense reward from selected severity function
- terminal reward from selected terminal readmission model

Action space:

- 32 discrete actions

Key hyperparameters in current active step 14 parser:

- `dqn_steps = 100000`
- `hidden = 128`
- `gamma = 0.99`
- `lr = 1e-4`
- `tau = 0.001`
- `batch_size = 32`
- `reward_threshold = 20`
- `reg_lambda = 5.0`
- `per_alpha = 0.6`
- `per_epsilon = 0.01`
- `beta_start = 0.9`

Script:

- `scripts/icu_readmit/step_14_offline_selected.py`

Outputs:

Model artifacts:

- `models/icu_readmit/offline_selected/ddqn/dqn_model.pt`
- checkpoints under `models/icu_readmit/offline_selected/ddqn/`

Auxiliary files:

- `dqn_actions.pkl`
- `dqn_q_values.pkl`
- `dqn_losses.pkl`

Current local report filenames still use older pre-renumbering names:

- `reports/icu_readmit/offline_selected/step_17_eval_results.json`
- `reports/icu_readmit/offline_selected/step_17_action_stats.json`

Current saved quick-check OPE means:

- val `offline_ddqn = 0.0625932763788066`
- test `offline_ddqn = 0.04820438391557281`

#### 6. Final non-learning baselines

These are part of the final simulator-control evaluation, even though they are not RL algorithms:

- `random`
- `repeat_last`

They are produced by the simulator control evaluation code and are part of the final step 13 comparisons.

### B. Exploratory methods or support models not kept as final thesis methods

#### 1. SARSA physician model

Implemented in:

- `train_sarsa_physician()` in `src/careai/icu_readmit/rl/continuous.py`

Status:

- present in code
- not called by any active step script
- should not be described as part of the final thesis method set

#### 2. OPE support models

Implemented in:

- `src/careai/icu_readmit/rl/evaluation.py`

These are:

- `PhysicianPolicy`
- `RewardEstimator`
- `EnvModel`

and training functions:

- `train_physician_policy()`
- `train_reward_estimator()`
- `train_env_model()`

Status:

- used to support doubly robust OPE in step 14
- not final policy-learning methods in their own right

#### 3. CARE-Sim reward-head path

Implemented generically in:

- `scripts/icu_readmit/step_14_caresim_train.py`
- `src/careai/icu_readmit/caresim/model.py`

Status:

- available in generic code
- not used in the active selected-causal thesis pipeline

## 6. Evaluation design

### Train / validation / test split definition

The final selected replay dataset is split by stay, not by row.

Split assignment is created in:

- `assign_splits()` in `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`

Mechanism:

- sort unique `icustayid`
- assign first 70% to train
- next 15% to val
- last 15% to test

Current split counts in `rl_dataset_selected.parquet`:

- train rows: `1,057,632`
- val rows: `222,408`
- test rows: `220,817`

- train stays: `43,239`
- val stays: `9,265`
- test stays: `9,267`

### Evaluation types used in the final pipeline

The final pipeline uses **three distinct evaluation modes**.

#### A. Observational simulator evaluation

Used in step 12:

- compare predicted next state to observed next state on held-out logged val/test rows

This is used for both:

- CARE-Sim
- MarkovSim

Metrics include:

- `next_state_mse`
- `reward_mae`
- `terminal_accuracy`
- `mean_uncertainty`

#### B. Simulator-based control evaluation

Used in step 13:

- run planner, DDQN, `repeat_last`, and `random` from the same held-out seed histories inside the simulator

This is not observational real-world evaluation. It is policy evaluation inside the learned simulator.

Metrics include:

- mean discounted return
- mean raw reward total
- mean uncertainty
- termination rate
- mean rollout steps
- action usage summaries

#### C. Off-policy evaluation on held-out logged data

Used in step 14:

- evaluate learned DDQN policies on logged val/test trajectories using doubly robust OPE

This is the only final comparison branch that evaluates all three DDQN policies on the same held-out logged-data benchmark:

- `offline_ddqn`
- `caresim_ddqn`
- `markovsim_ddqn`

### Baselines used

In simulator control evaluation:

- `planner`
- `random`
- `repeat_last`

In step 14 OPE:

- no planner baseline is currently evaluated
- comparison is between the three DDQN policies

The logged policy is also summarized in the OPE outputs for reference.

### Which results are final and intended for the thesis

The intended final thesis result set is:

1. **Selection / modeling interface**
- selected 9-state / 5-action interface

2. **Simulator quality**
- CARE-Sim vs MarkovSim on held-out next-state prediction and rollout diagnostics

3. **Simulator-side policy learning**
- CARE-Sim planner/DDQN vs `repeat_last` and `random`
- MarkovSim planner/DDQN vs `repeat_last` and `random`

4. **Held-out logged-data policy comparison**
- `offline_ddqn` vs `caresim_ddqn` vs `markovsim_ddqn`

### Current saved local results versus intended final results

The code is clear about the intended design, but the currently saved local report files are not all equally final-ready.

#### CARE-Sim simulator evaluation

Current saved file:

- `reports/icu_readmit/caresim_selected_causal/caresim_summary.json`

Caveat:

- generated with small rollout/counterfactual sample sizes
- generated without terminal readmission reward enabled

So it is useful for methods grounding, but not the cleanest direct comparison file against MarkovSim reward metrics.

#### MarkovSim simulator evaluation

Current saved file:

- `reports/icu_readmit/markovsim_selected_causal/markovsim_summary.json`

This run is closer to the final intended reward setup because it includes:

- handcrafted severity reward
- terminal readmission reward

#### Simulator control summaries

Current saved files:

- `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`
- `reports/icu_readmit/markovsim_control_selected_causal/step_16_markovsim_summary.json`

These are methodologically relevant final-policy comparison files, but the filenames still reflect the old pre-renumbering step names.

#### Offline OPE comparison

Current saved file:

- `reports/icu_readmit/offline_selected/step_17_eval_results.json`

Caveat:

- current saved local OPE results are a reduced quick-check run over `400` trajectories per split
- this is not yet a full held-out benchmark over all validation/test stays

### Exact current saved metrics that are already available

#### CARE-Sim control

From `step_16_summary.json`:

- val discounted return:
  - `ddqn = 8.692149344347015`
  - `planner = 8.582858349683406`
  - `repeat_last = 8.680327229551217`
  - `random = 8.592901410564357`
- test discounted return:
  - `ddqn = 8.836255628219314`
  - `planner = 8.78556112665876`

#### MarkovSim control

From `step_16_markovsim_summary.json`:

- val discounted return:
  - `ddqn = 8.976655859579582`
  - `planner = 8.973150204867125`
  - `repeat_last = 8.783070112458793`
  - `random = 8.609596448199428`
- test discounted return:
  - `ddqn = 9.101420106375041`
  - `planner = 9.131546391099691`
  - `random = 8.812488372628884`

#### Offline OPE quick-check

From `step_17_eval_results.json`:

- val:
  - `offline_ddqn = 0.0625932763788066`
  - `caresim_ddqn = -2.460073740973099`
  - `markovsim_ddqn = -0.8717816211236864`
- test:
  - `offline_ddqn = 0.04820438391557281`
  - `caresim_ddqn = -2.2666017163189287`
  - `markovsim_ddqn = -0.49651748054280365`

These are current saved numbers, not necessarily the final thesis-ready full-run OPE benchmark.

### Remaining uncertainties / weaknesses

These are the main unresolved issues that matter for thesis writing.

1. **CARE-Sim dynamic/static index mismatch**
- current selected CARE-Sim does not fully learn all intended dynamic state variables because `Chloride` is copied, not learned

2. **Selection stage partly manual**
- final state/action set is causally informed, but the final keep/drop decision is documented in a recommendation markdown file, not only derived from one automatic selector

3. **Saved report naming drift**
- current saved files still use pre-renumbering names such as `step_16_*` and `step_17_*`
- the active scripts now correspond conceptually to steps 13 and 14

4. **Saved simulator evaluation runs are not fully aligned**
- CARE-Sim and MarkovSim saved step 12 reports were produced with different reward settings and sample sizes

5. **Current saved step 14 OPE is a reduced run**
- the saved OPE comparison is useful as a quick check, but not yet a full held-out run across all trajectories

6. **Current saved DDQN configs are run-specific**
- current on-disk config files do not always match the newest code defaults exactly

## 7. THESIS CRITICAL FACTS

- Broad processed dataset used for final modeling: `data/processed/icu_readmit/ICUdataset.csv`
- Final replay dataset: `data/processed/icu_readmit/rl_dataset_selected.parquet`
- Final scaler metadata: `data/processed/icu_readmit/scaler_params_selected.json`
- Final static-context file: `data/processed/icu_readmit/static_context_selected.parquet`

- Final dynamic state variables:
  - `Hb`
  - `BUN`
  - `Creatinine`
  - `Phosphate`
  - `HR`
  - `Chloride`

- Final static confounders:
  - `age`
  - `charlson_score`
  - `prior_ed_visits_6m`

- Final action variables:
  - `vasopressor`
  - `ivfluid`
  - `antibiotic`
  - `diuretic`
  - `mechvent`

- Final action representation in code:
  - five binary columns
  - one discrete action code `a` in `{0,...,31}`

- Exact final outcome definition:
  - `readmit_30d`
  - used as the main outcome anchor in selection
  - used as the terminal label for the selected terminal readmission model

- Stored replay reward in step 10a:
  - dense `SOFA_t - SOFA_{t+1}`
  - terminal `+15 / -15`

- Final active reward used by simulators and offline RL:
  - dense severity improvement
  - plus terminal reward from a separate LightGBM readmission model
  - terminal reward formula:
    - `reward_scale - 2 * reward_scale * p_readmit`
  - default `reward_scale = 15`

- Final support models:
  - severity surrogate:
    - `scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py`
    - artifact `models/icu_readmit/severity_selected/ridge_sofa_surrogate.joblib`
  - terminal readmission model:
    - `scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py`
    - artifact `models/icu_readmit/terminal_readmit_selected/terminal_readmit_selected.joblib`

- Final simulators:
  - CARE-Sim:
    - training script `scripts/icu_readmit/step_11a_caresim_train_selected_causal.py`
    - main artifacts in `models/icu_readmit/caresim_selected_causal/`
  - MarkovSim:
    - training script `scripts/icu_readmit/step_11b_markovsim_train.py`
    - main artifacts in `models/icu_readmit/markovsim_selected_causal/`

- Final simulator evaluation scripts:
  - CARE-Sim: `scripts/icu_readmit/step_12a_caresim_evaluate.py`
  - MarkovSim: `scripts/icu_readmit/step_12b_markovsim_evaluate.py`

- Final simulator-side policy learning scripts:
  - CARE-Sim control: `scripts/icu_readmit/step_13a_caresim_control.py`
  - MarkovSim control: `scripts/icu_readmit/step_13b_markovsim_control.py`

- Final offline RL comparison script:
  - `scripts/icu_readmit/step_14_offline_selected.py`

- Final policy methods intended for the thesis:
  - CARE-Sim planner
  - CARE-Sim DDQN
  - MarkovSim planner
  - MarkovSim DDQN
  - Offline DDQN
  - simulator baselines `random` and `repeat_last`

- Final held-out comparison target:
  - `offline_ddqn`
  - `caresim_ddqn`
  - `markovsim_ddqn`
  - compared by doubly robust OPE on logged validation/test data

- Exact current unresolved uncertainties:
  - CARE-Sim currently does not correctly learn `Chloride` as a dynamic state because of hard-coded dynamic/static indices
  - selection from broad variable space to final interface is partly manual/documented, not purely automatic
  - current saved CARE-Sim and MarkovSim simulator-evaluation reports are not perfectly aligned in reward settings
  - current saved step 14 OPE results are reduced quick-check outputs, not yet a full held-out final run
  - current local artifact filenames still use old pre-renumbering step names (`step_16_*`, `step_17_*`)
