# CARE-Sim Change Proposals

Date: 2026-04-07

Purpose: capture the main simulator changes currently under consideration for the ICU readmission CARE-Sim stack, so they can be reused later when the final implementation direction is chosen.

This note is intentionally narrow. It does not try to redesign the whole simulator. It records the three most important design corrections discussed so far:

1. static confounders should not drift during rollout
2. reward should be consistent with simulated patient state, not treated as an unrelated prediction target
3. terminal prediction should not be tied to the fixed sequence-length boundary

---

## Implementation Update (2026-04-07)

Parts of these proposals are now implemented in parallel tracks.

Implemented already:

- static confounders as conditioning-only state context
- random windows from full stays
- explicit elapsed-time feature

These are now available in the structured CARE-Sim track and the new selected-set
Step 14 track.

Also implemented:

- new selected-set preprocessing:
  - `scripts/icu_readmit/step_10a_rl_preprocess_selected.py`
- selected-state severity surrogate trained against real SOFA:
  - `scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py`
  - outputs under:
    - `models/icu_readmit/severity_selected/`
    - `reports/icu_readmit/severity_selected/`
- selected-set Step 14 training track with no reward head:
  - `scripts/icu_readmit/step_14_caresim_train_selected.py`
  - `notebooks/step_14_caresim_selected_colab.ipynb`

Not implemented yet:

- selected-set Step 12 / Step 13 rollout logic using the severity surrogate as
  the dense reward function

So the current status is:

- preprocessing and severity-surrogate training are complete
- selected-set simulator training is ready
- selected-set simulator evaluation/control wiring is still pending

---

## 1. Static Confounders Should Not Drift

### Problem

The current CARE-Sim state includes:

- dynamic features:
  - `Hb`
  - `BUN`
  - `Creatinine`
  - `HR`
  - `Shock_Index`
- static confounders:
  - `age`
  - `charlson_score`
  - `prior_ed_visits_6m`

The current model predicts all 8 next-state dimensions. This means the model is being trained to predict:

- tomorrow's `age`
- tomorrow's `charlson_score`
- tomorrow's `prior_ed_visits_6m`

These are not true dynamics. As a result, even small one-step prediction errors accumulate during recursive rollout and static features drift.

### Existing evidence

This drift is already measured in current Step 15 evaluation.

One-step drift is nonzero in:

- `reports/icu_readmit/caresim/caresim_one_step_val.json`
- `reports/icu_readmit/caresim/caresim_one_step_test.json`

Rollout drift grows further in:

- `reports/icu_readmit/caresim/caresim_rollout_val.json`
- `reports/icu_readmit/caresim/caresim_rollout_test.json`

The Step 15 code explicitly computes static drift in:

- `scripts/icu_readmit/step_12a_caresim_evaluate.py`

### Proposed design change

Static confounders should be:

- used as conditioning inputs
- not predicted as dynamic outputs

That implies the preferred future design is:

- full input state still includes static confounders
- model predicts only the dynamic next-state dimensions
- static confounders are copied through unchanged during rollout

### Why this is the preferred fix

This is better than merely penalizing drift, because it structurally removes the problem.

Advantages:

- static features cannot drift
- model target becomes cleaner
- rollout semantics become more faithful
- simulator no longer wastes capacity predicting constants

### Implementation direction

Likely future change:

- split current state into:
  - input state = dynamic + static
  - predicted next state = dynamic only

Environment assembly would then reconstruct the full state as:

- predicted dynamic next state
- copied static confounders from the seeded episode context

### Fallback operational fix

If a full retraining redesign is delayed, the simulator can also lock static variables during rollout by overwriting them after each predicted step.

However, this is a fallback only. The preferred solution is still:

- do not train the model to predict static variables

---

## 2. Reward Should Be Consistent With Simulated State

### Problem

The current CARE-Sim model predicts:

- next state
- reward
- terminal

The current training setup treats reward as a supervised target alongside next-state prediction.

This is clean from an engineering perspective, but it raises a conceptual issue:

- in an RL environment, reward ideally should be a function of the patient state transition
- not an unrelated extra quantity learned in parallel

### Current reward design

The current dataset describes reward as:

- dense reward = SOFA delta
- terminal reward = `+15` / `-15`

This is documented in:

- `src/careai/icu_readmit/caresim/dataset.py`

### Core issue

The current Tier-2 simulator state does **not** contain full SOFA information.

Current state variables are:

- `Hb`
- `BUN`
- `Creatinine`
- `HR`
- `Shock_Index`
- plus static confounders

That means the simulator cannot currently reconstruct SOFA exactly from the predicted next state alone.

Therefore, the current reward head is doing real work:

- it is approximating the reward label attached to the transition
- because the current compressed state is not rich enough to derive the dense reward exactly

### Design principle agreed on

Conceptually, reward should be tied to the simulated patient state.

This is the long-run preferred design principle:

- simulate patient state evolution
- compute reward from the simulated state transition whenever possible

### Implication

If dense reward is intended to reflect SOFA change, then a cleaner simulator design would require one of the following:

#### Option A: add SOFA directly to the predicted state

Then dense reward can be computed directly from:

- `SOFA_t`
- `SOFA_{t+1}`

#### Option B: add a dedicated `next_SOFA` head

Then dense reward can be derived from that predicted SOFA quantity instead of using a generic learned reward head.

#### Option C: expand the state enough to reconstruct SOFA from component variables

This is the most physiologically grounded option, but also the largest redesign.

### Updated implementation decision

The selected-set CARE-Sim track now follows the cleaner direction:

- no reward head in Step 14 selected training
- transition model predicts:
  - next state
  - terminal
- dense reward is intended to come later from:
  - selected-state severity surrogate change

This is not wired into the selected-set Step 12 / Step 13 rollout code yet,
but the architectural decision has now been made.

### Severity-surrogate status

A learned selected-state severity surrogate has been trained against real SOFA.

Current artifact locations:

- `models/icu_readmit/severity_selected/ridge_sofa_surrogate.joblib`
- `models/icu_readmit/severity_selected/severity_surrogate_config.json`
- `reports/icu_readmit/severity_selected/severity_surrogate_metrics.json`
- `reports/icu_readmit/severity_selected/severity_surrogate_coefficients.csv`

Current full-data performance:

- validation MAE: about `1.86`
- validation R^2: about `0.279`
- validation Spearman: about `0.517`
- test MAE: about `1.84`
- test R^2: about `0.273`
- test Spearman: about `0.501`

So the selected-state surrogate is informative enough to support the next
environment-reward wiring step.

---

## 3. Terminal Prediction Should Not Be Tied To Sequence Length

### Problem

The current CARE-Sim uses a per-step terminal head to decide whether the ICU trajectory ends at a given rollout step.

Conceptually this is reasonable. The issue is that the current model appears to have learned a shortcut tied to the fixed training sequence length.

The current training setup uses:

- `max_seq_len = 80`
- absolute positional embeddings of length `80`
- truncation of long stays to the **last** `80` blocs

This is implemented in:

- `src/careai/icu_readmit/caresim/model.py`
- `src/careai/icu_readmit/caresim/dataset.py`

### Existing evidence

Direct long-rollout probing of the saved CARE-Sim checkpoint showed:

- 10 out of 10 tested episodes eventually terminated by `terminal_prob > 0.5`
- every one of them terminated at exactly step `75` when seeded with `history_len = 5`

Additional sensitivity checks showed:

- `history_len = 3` -> terminal fired at step `77`
- `history_len = 5` -> terminal fired at step `75`
- `history_len = 8` -> terminal fired at step `72`
- `history_len = 10` -> terminal fired at step `70`

This pattern matches:

- `80 - history_len`

which strongly suggests the terminal head is responding to the sequence-length boundary rather than a genuinely patient-specific ICU end process.

### Why this happens

The issue is not that shorter stays do not exist. They do.

The issue is that after preprocessing, the model never sees a training example where:

- absolute position `80` is **not** terminal

Reason:

- stays longer than `80` are truncated to the last `80` blocs
- terminal is `1` only at the final bloc of the stay
- therefore every truncated stay teaches the model that position `80` is terminal

So position `80` becomes a perfect shortcut feature for discharge.

### Proposed design change

The model should no longer be trained only on the last `80` blocs of long stays.

Preferred direction:

- keep full stays in the source data
- sample subsequences or windows from arbitrary parts of each stay
- keep a fixed maximum training window length for compute
- allow training windows that end before the ICU stay ends
- add elapsed-time context such as `bloc` or `hours_since_icu_admit`

### Why this is the preferred fix

This breaks the current shortcut.

With random windows:

- the end of a training window is no longer synonymous with the end of the ICU stay
- position within the window no longer deterministically reveals terminal status
- the terminal head must rely more on patient state and elapsed time

### Important clarification

Simply increasing or removing the `80` cap is not enough on its own.

That would remove the current exact `fires at 80` artifact, but it would not fully solve the underlying issue unless the training setup is also changed.

The more complete fix is:

- stop using only the last window of long stays
- train on windows sampled from across the full stay
- provide explicit time context

### Preferred future direction

The terminal head should be treated as a per-step hazard model for ICU end.

That means:

- `terminal_t` should answer "does the ICU trajectory end here?"
- but this answer should be learned from state plus elapsed time
- not from the hard boundary of the training sequence representation

An optional stronger version later would be to add:

- a `remaining_windows` or `time_to_end` head

but that is not required for the first fix.

---

## 4. Current Working Conclusions

### Change 1: static confounders

Status:

- implemented in parallel CARE-Sim tracks

Current preferred direction:

- static confounders should be inputs only, not prediction targets

### Change 2: reward/state consistency

Status:

- conceptually agreed and partly implemented

Current preferred direction:

- selected-set Step 14 uses no reward head
- dense reward should be derived from state transition using the trained
  selected-state severity surrogate
- terminal reward still remains a separate environment design question

### Change 3: terminal prediction / ICU end signal

Status:

- implemented in the structured / selected training setup

Current preferred direction:

- stop training only on the last `80` blocs of long stays
- move to sampled windows from full stays
- add elapsed-time context so the terminal head behaves like a hazard model rather than a sequence-boundary detector

---

## 5. Suggested Final Decision Path

When implementation decisions are finalized, these changes should likely be handled in this order:

1. Static confounders:
   - remove them from prediction targets
   - keep them as conditioning inputs

2. Terminal prediction:
   - stop using only last-80 truncation
   - move to sampled training windows from full stays
   - add elapsed-time context

3. Reward consistency:
   - decide whether to add SOFA directly, add a `next_SOFA` head, or expand state enough to reconstruct SOFA
   - only then consider replacing the generic dense reward head

This ordering is recommended because:

- the static-confounder fix is structurally clean and easy to justify
- the terminal fix removes a clear simulator artifact already observed in rollout
- the reward redesign is conceptually important, but depends more on the final state design
