# Step 12a Non-Causal CARE-Sim Evaluation

This is the evaluation layer for the broad non-causal CARE-Sim branch.

It mirrors the role of the existing selected-causal `12a`, but it is explicit
about the parts that differ because this branch:

- has a broader state/action interface
- uses a single model rather than an ensemble
- predicts readmission directly as a model head
- does not use a hand-coded causal graph

## Main script

- `scripts/icu_readmit/step_12a_caresim_evaluate_noncausal.py`

## Inputs

- replay data:
  - `data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet`
- trained model:
  - `models/icu_readmit/caresim_noncausal/`

## Outputs

- `reports/icu_readmit/caresim_noncausal/caresim_noncausal_summary.json`
- `reports/icu_readmit/caresim_noncausal/caresim_noncausal_one_step_val.json`
- `reports/icu_readmit/caresim_noncausal/caresim_noncausal_one_step_test.json`
- `reports/icu_readmit/caresim_noncausal/caresim_noncausal_rollout_val.json`
- `reports/icu_readmit/caresim_noncausal/caresim_noncausal_rollout_test.json`
- `reports/icu_readmit/caresim_noncausal/caresim_noncausal_counterfactual_val.csv`

## What is evaluated

### 1. One-step prediction on val/test

Reported:

- next-state MSE on dynamic targets
- per-feature next-state MSE
- terminal accuracy
- terminal Brier score
- readmission metrics from the model readmission head:
  - AUC
  - AUPRC
  - Brier
  - log loss
  - accuracy

### 2. Closed-loop rollout under clinician actions

Each episode is seeded with a short real patient history and then rolled forward
for a fixed horizon using the logged clinician actions from held-out data.

Reported:

- per-step state MSE
- per-step done accuracy
- final rollout readmission metrics

### 3. Counterfactual one-step action sweep

Because this branch has 11 binary actions, exhaustive sweeps over all `2^11`
combinations are possible but unnecessary and noisy.

Instead the evaluator uses the most frequent observed action combinations from
the held-out replay data. For each seed patient it records:

- terminal probability
- readmission probability
- predicted next dynamic state
- the applied action bits

## Key difference from selected-causal `12a`

This non-causal branch does **not** report ensemble uncertainty in version 1.

That is intentional. The model is currently a single trained transformer, not
an ensemble. The evaluation should reflect that honestly instead of fabricating
pseudo-uncertainty.

## Why this structure is still comparable

The evaluation keeps the same three-layer structure as the other simulator tracks:

- one-step fidelity
- short rollout stability
- counterfactual responsiveness

That makes later comparison with:

- selected-causal CARE-Sim
- MarkovSim
- DAG-aware

much cleaner, even though the non-causal branch does not enforce causal
structure and has a broader interface.
