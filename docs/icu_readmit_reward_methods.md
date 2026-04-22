# ICU Readmit Reward Methods

This note explains the reward components used in the ICU readmission pipeline:

- the **handcrafted severity reward**
- the **severity surrogate**
- the **terminal readmission reward**

All three turn a patient's simulated state into a scalar signal for control.

Relevant code:

- [severity.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/caresim/severity.py)
- [step_10b_train_selected_severity_surrogate.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py)
- [step_10c_train_selected_terminal_readmit.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py)
- [step_13a_caresim_control.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_13a_caresim_control.py)
- [step_13c_dagaware_control.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_13c_dagaware_control.py)
- [step_14_offline_selected.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_14_offline_selected.py)

## 1. Why there are multiple reward methods

The project needs a way to assign clinical meaning to a state transition.

The simulators predict:

- next state
- reward
- terminal probability

But for control and offline RL, we also want clinically meaningful reward signals that say whether one state is better or worse than another.

There are three complementary ways to get that signal:

1. a fixed handcrafted severity heuristic
2. a learned surrogate of SOFA
3. a terminal readmission-risk reward

They serve different roles in the pipeline.

## 2. The handcrafted severity reward

The handcrafted severity reward is a fixed, interpretable scoring rule over the selected dynamic simulator state.

It is defined directly in [severity.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/caresim/severity.py) as `HandcraftedSelectedSeverity`.

### 2.1 What inputs it uses

It uses these six dynamic state variables:

- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

These are not raw clinical values. They are the transformed simulator-space values used by the selected replay / model pipeline.

### 2.2 The fixed weights

The handcrafted score uses the following weights:

- `Hb`: `0.28`
- `BUN`: `0.24`
- `Creatinine`: `0.24`
- `Phosphate`: `0.10`
- `HR`: `0.09`
- `Chloride`: `0.05`

These weights are hard-coded, not learned from the RL control stage.

### 2.3 How the score is calculated

The score is built from simple monotonic penalty terms:

- low Hb is bad
- high BUN is bad
- high Creatinine is bad
- high Phosphate is bad
- high HR is bad
- Chloride is penalized by absolute deviation

The code applies:

- `0.28 * relu(-Hb)`
- `0.24 * relu(BUN)`
- `0.24 * relu(Creatinine)`
- `0.10 * relu(Phosphate)`
- `0.09 * relu(HR)`
- `0.05 * abs(Chloride)`

Then it sums those terms into a single scalar severity score.

### 2.4 What that means intuitively

In the transformed state space:

- values above zero for BUN, Creatinine, Phosphate, and HR push severity up
- values below zero for Hb push severity up
- Chloride is treated as a deviation-from-center feature

So the handcrafted score is a simple clinical heuristic that says:

> worse physiology should give a larger severity score

### 2.5 How it becomes reward

The reward is computed from the change in severity:

```text
reward = severity(current_state) - severity(next_state)
```

So:

- if the next state is less severe, reward is positive
- if the next state is more severe, reward is negative
- if severity does not change, reward is near zero

This makes the reward dense and transition-based.

### 2.6 Where it is used

The handcrafted severity reward is used in:

- step 13 control
- step 14 offline RL comparison

and can also be enabled in step 12 evaluation.

It is the reward mode used in the saved step 13 results for:

- CARE-Sim control
- MarkovSim control
- DAG-aware control

### 2.7 What it is good for

The handcrafted reward is useful because it is:

- simple
- stable
- interpretable
- easy to explain in a thesis

It is a clinical heuristic, not a learned model.

That makes it good for:

- baseline control experiments
- reward shaping
- easy interpretation

### 2.8 Limitation

It is not guaranteed to be the best possible clinical reward.

Because it is fixed by hand:

- it may overemphasize some lab values
- it may underemphasize some clinically relevant patterns
- it is only as good as the chosen weights and transforms

So it is best understood as a principled proxy, not ground truth.

## 3. The severity surrogate

The severity surrogate is a learned model that approximates SOFA from the selected dynamic state.

It is trained in [step_10b_train_selected_severity_surrogate.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_10b_train_selected_severity_surrogate.py).

### 3.1 What it learns

The surrogate learns to predict:

- `SOFA`

from the selected features:

- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

### 3.2 Model type

It is a **ridge regression** model trained with `RidgeCV`.

So unlike the handcrafted score, it is learned from data.

### 3.3 How the surrogate is trained

The training script:

1. reads the ICU dataset
2. keeps the selected severity-related features
3. assigns train / val / test splits by stay
4. clips and transforms the features
5. standardizes them using train-set statistics
6. fits ridge regression against the real SOFA target
7. saves the model, config, metrics, and coefficients

The training target is actual SOFA, not a custom reward label.

### 3.4 Feature preprocessing

The surrogate does not use raw values directly.

It applies feature-specific preprocessing:

- clipping to a clinically reasonable range
- optional `log1p` transform for skewed features
- z-scoring using training-set mean and standard deviation

That preprocessing is stored in:

- [severity_surrogate_config.json](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/models/icu_readmit/severity_selected/severity_surrogate_config.json)

### 3.5 How the surrogate score is computed

At inference time, the model:

1. transforms each selected feature using the stored preprocessing parameters
2. multiplies by the learned ridge coefficients
3. adds the learned intercept
4. optionally clips the target to the SOFA range

So the output is a learned severity score in SOFA-like units.

### 3.6 How it becomes reward

The reward is again based on change in severity:

```text
reward = surrogate(current_state) - surrogate(next_state)
```

So the learning signal is the same shape as the handcrafted reward, but the severity function itself is learned from data.

### 3.7 Where it is used

The severity surrogate can be enabled in step 13 or step 14 with:

- `--use-severity-reward`
- `--severity-mode surrogate`

It is also available in step 12 evaluation, although the stored selected runs in the repo mainly use the handcrafted mode for the control experiments.

### 3.8 What it is good for

The surrogate is useful because it is:

- data-driven
- still interpretable
- aligned with a known clinical severity target
- smoother than a purely hand-designed heuristic

So it can be a stronger candidate if you want a learned severity notion rather than a manually designed one.

### 3.9 Limitation

It is still only a surrogate.

That means:

- it predicts SOFA, not the full clinical state
- it inherits any bias in the SOFA target
- it depends on the quality of the preprocessing and the regression fit

It is more data-driven than the handcrafted score, but still not a direct outcome model.

### 3.10 How strong is the fit?

The surrogate fit is **moderate**, not strong, on held-out data.

On the test split it achieved:

- `R2 = 0.273`
- `Spearman = 0.501`
- `MAE = 1.837`
- `RMSE = 2.298`

That means:

- it explains about 27% of the variance in SOFA
- it preserves the ordering of severity reasonably well
- it is useful as a smooth reward proxy
- it is not accurate enough to be described as a high-fidelity SOFA predictor

So the right thesis-level reading is:

- **good enough for reward shaping**
- **not strong enough to oversell as a precise clinical estimator**

## 4. The terminal readmission reward

The terminal reward is a separate trained model that estimates the probability of 30-day readmission from the terminal state of a simulated episode.

It is trained in [step_10c_train_selected_terminal_readmit.py](C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_10c_train_selected_terminal_readmit.py).

### 4.1 What it learns

The model predicts:

- `readmit_30d`

from the terminal state features:

- `s_Hb`
- `s_BUN`
- `s_Creatinine`
- `s_Phosphate`
- `s_HR`
- `s_Chloride`
- `s_age`
- `s_charlson_score`
- `s_prior_ed_visits_6m`

### 4.2 Model type

It is a **LightGBM classifier**.

So this reward is learned from data, but it is only applied at the end of an episode.

### 4.3 How it is trained

The training script:

1. keeps only terminal rows where `done == 1`
2. uses the train / val / test splits
3. fits LightGBM to predict `readmit_30d`
4. evaluates the classifier with standard discrimination and calibration metrics

The held-out test performance is moderate:

- `AUC = 0.649`
- `AUPRC = 0.348`
- `Brier = 0.158`
- `logloss = 0.491`

So it is usable as a terminal risk proxy, but it is not a high-accuracy classifier.

### 4.4 What those metrics mean

- **AUC** measures how well the model separates readmitted from non-readmitted patients.
- **AUPRC** measures how well the model identifies readmission cases when the positive class is relatively rare.
- **Brier score** measures how well calibrated the predicted probabilities are.
- **Log loss** penalizes confident wrong predictions and summarizes overall probabilistic accuracy.

In plain terms, these scores say the terminal model has:

- moderate discrimination
- reasonable but imperfect calibration
- enough signal to support reward shaping
- not enough strength to oversell as a high-precision prognostic model

### 4.4 How it becomes reward

The classifier output `p_readmit` is transformed into reward by:

```text
reward = reward_scale - 2 * reward_scale * p_readmit
```

With the default `reward_scale = 15.0`:

- `p_readmit = 0` gives `+15`
- `p_readmit = 0.5` gives `0`
- `p_readmit = 1` gives `-15`

So low readmission risk is rewarded and high readmission risk is penalized.

### 4.5 When it is used

The terminal reward is only applied at terminal transitions, or when the simulator predicts that the episode ends now.

So it acts as a sparse end-of-episode outcome signal, complementing the dense severity reward.

### 4.6 Why it matters

The terminal reward makes the control problem more clinically meaningful because it does not only ask:

- “did the physiology improve?”

It also asks:

- “does this terminal state look likely to lead to 30-day readmission?”

## 5. How they are used in the pipeline

### In the simulator

The reward functions compute current vs next severity, and the terminal model can add an end-of-episode readmission signal when the episode terminates.

### In step 13 control

The control layer uses reward shaped by:

- severity improvement
- terminal readmission reward
- uncertainty penalty

### In step 14 offline RL

The offline comparison reuses the same reward logic so policies can be compared under a consistent definition.

## 6. Which severity method was used in the saved step 13 results

The saved control results in the repo use the **handcrafted severity reward**.

That is the mode used for:

- `13a` CARE-Sim control
- `13b` MarkovSim control
- `13c` DAG-aware control

So if you are reading the current results files, the severity signal there is the handcrafted one, not the surrogate.

## 7. Practical thesis interpretation

If you want the most concise thesis framing:

- the handcrafted severity reward is the fixed clinical heuristic
- the severity surrogate is the learned SOFA approximation
- the terminal reward is a learned readmission-risk signal at the end of an episode
- all three can convert simulated transitions into clinically meaningful scalar feedback

## 8. Short version

The handcrafted severity score is a fixed weighted function over six selected dynamic variables. It is not learned. The severity surrogate is a ridge regression model trained to predict SOFA from the same selected severity-relevant variables. The terminal reward is a LightGBM classifier on terminal states that predicts 30-day readmission and converts that probability into a signed reward. In all three cases, the control logic uses these signals to shape policy behavior in the simulator.
