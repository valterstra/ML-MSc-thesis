# Step 10 Non-Causal Replay Preprocessing

Purpose:
- convert the broad non-causal Step 09 interface into a simulator/RL-ready replay dataset
- mirror the role of the active selected-causal `step_10a`, but for the broad predictive branch

Paired script:
- `scripts/icu_readmit/step_10a_rl_preprocess_noncausal.py`

Outputs:
- `data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet`
- `data/processed/icu_readmit/step_10_noncausal/static_context_noncausal.parquet`
- `data/processed/icu_readmit/step_10_noncausal/scaler_params_noncausal.json`

## What this step does

The non-causal replay builder:
- loads the Step 09 non-causal interface dataset
- splits stays into train / val / test
- preprocesses dynamic and static variables
- keeps actions as binary
- constructs a compact action ID
- builds transition targets
- keeps SOFA and readmission available for reward construction

## Preprocessing rules

### Dynamic states
- clipped with conservative physiological bounds
- selected skewed labs log-transformed
- z-scored using train-split statistics only

### Static continuous variables
- `age`
- `Weight_kg`
- `charlson_score`
- `prior_ed_visits_6m`

These are clipped conservatively and z-scored using train-split statistics.

### Static binary variables
- `gender`
- `re_admission`

These are kept as raw binary inputs.

### Static categorical variables
- `race`
- `insurance`
- `marital_status`
- `admission_type`
- `admission_location`

These are integer-coded in step 10 and intended to be embedded later in step 11.

### Actions
Binary actions are preserved as `0/1` columns:
- `vasopressor_active`
- `ivfluid_active`
- `antibiotic_active`
- `anticoagulant_active`
- `diuretic_active`
- `insulin_active`
- `opioid_active`
- `sedation_active`
- `transfusion_active`
- `electrolyte_active`
- `mechvent_active`

A compact action ID `a` is also materialized for RL convenience.

## Transition construction

The replay dataset follows the same broad semantics as the active branch:

- current row = state/action at time `t`
- next-state targets = dynamic and static inputs shifted to `t+1`
- terminal indicator `done = 1` on the last bloc of each ICU stay

## Reward support columns

The non-causal replay dataset does not force a single final reward design, but
it materializes the ingredients needed for the intended SOFA-plus-readmission setup.

Included columns:
- `SOFA`
- `SOFA_next`
- `readmit_30d`
- `done`
- `reward_sofa`
- `reward_terminal_readmit`
- `reward_default`

Definitions:
- `reward_sofa = SOFA_t - SOFA_{t+1}` on non-terminal rows, else `0`
- `reward_terminal_readmit = +15` if terminal and no readmission, `-15` if terminal and readmission, else `0`
- `reward_default = reward_sofa + reward_terminal_readmit`

This preserves flexibility while keeping a usable default reward already materialized.

## Why SOFA stays auxiliary

`SOFA` is kept in the replay dataset for:
- reward construction
- evaluation
- auxiliary supervision if needed later

It is not treated as a default transformer input in version 1 of the non-causal branch.

This keeps the predictive state space broad without letting the model rely too directly
on a summary score that is itself constructed from retained raw physiology.

## Relation to the causal branch

This step does not replace:
- `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`

Instead it creates a parallel track:
- causal branch: selected-causal replay build
- non-causal branch: broad predictive replay build

That separation is intentional and should be kept explicit in later Step 11 training.
