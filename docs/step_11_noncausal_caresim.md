# Step 11a Non-Causal CARE-Sim

This branch trains a broad predictive transformer on the non-causal replay dataset from step 10.

## Purpose

This is the non-causal parallel track to the selected-causal CARE-Sim branch.

It keeps:
- a broad state space
- a broad binary action space
- action-conditioned next-state prediction

It removes:
- hand-coded causal graph constraints
- the narrow selected state/action interface

## Inputs

Source replay dataset:
- `data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet`

## Model design

The model is a CARE-Sim-style sequence transformer over full vectors.
Its schema is inferred directly from the replay parquet:

- `s_*` columns are treated as input state features
- `s_next_*` columns define the dynamic next-state target
- `*_active` columns define the binary action space
- static categorical inputs are identified by the `_code` suffix

Per time step it consumes:
- all processed dynamic states
- repeated processed static context
- binary action vector

Static categorical variables are embedded inside the model rather than treated as raw numeric magnitudes.

The model predicts:
- next dynamic state
- terminal/discharge logit per time step
- readmission logit from the terminal context representation

`SOFA` is not part of the transformer input. It is retained in the replay dataset for later reward construction and evaluation.

## Loss

Default loss weights:
- next-state MSE: `1.0`
- terminal BCE: `0.2`
- readmission BCE: `0.2`

The next-state target covers dynamic variables only. Static context is conditioning-only in this branch.

## Main script

- `scripts/icu_readmit/step_11a_caresim_train_noncausal.py`

Default output folder:
- `models/icu_readmit/caresim_noncausal/`

Saved artifacts:
- `model.pt`
- `best_model.pt`
- `train_config.json`
- `train_metrics.json`

## Why this track exists

The causal branch asks whether structural medical prior knowledge helps.

This non-causal branch asks the complementary question:

> If we remove the causal restrictions and let a broader transformer model learn predictive ICU dynamics directly, do we get a better simulator?
