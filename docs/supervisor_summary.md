# Supervisor Summary

## Step 09: state and action selection

### Goal

Reduce the ICU problem to a state space and action space that are small enough
to model, but still clinically meaningful.

The selection logic was:

- keep states that are informative for `30-day readmission`
- keep actions that can plausibly move those states
- avoid variables that are redundant, circular, or weakly controllable

### Step 09 workflow

We used three layers of evidence.

| Stage | What we did | Why it mattered |
|---|---|---|
| Step 02 | ranked discharge states and static variables with LightGBM | broad screening before causal work |
| Step 03 | FCI random-stability analysis: `discharge state -> readmission` | identify states with robust outcome relevance |
| Step 04b | robust multivariate FCI: `action -> delta state` | identify states that are not only predictive, but also modifiable |

### Step 02: broad screening

This was used as a screening step, not the final decision rule.

Highlights from the state-variable ranking:

| Rank | Variable | Importance |
|---|---|---:|
| 1 | `last_BUN` | 0.1113 |
| 2 | `last_Hb` | 0.0636 |
| 3 | `last_Platelets_count` | 0.0530 |
| 4 | `last_WBC_count` | 0.0407 |
| 5 | `last_cumulated_balance` | 0.0332 |
| 6 | `last_input_total` | 0.0330 |
| 7 | `last_Creatinine` | 0.0329 |
| 8 | `last_PT` | 0.0272 |
| 9 | `last_input_4hourly_tev` | 0.0227 |
| 10 | `last_PTT` | 0.0193 |

Highlights from the static-variable ranking:

| Rank | Variable | Importance |
|---|---|---:|
| 1 | `prior_ed_visits_6m` | 0.2499 |
| 2 | `race` | 0.1843 |
| 3 | `discharge_disposition` | 0.1360 |
| 4 | `age` | 0.0771 |
| 5 | `charlson_score` | 0.0665 |

Initial takeaway:

- renal markers were clearly important
- hemoglobin was important
- some high-ranked variables were not appropriate RL states because they were
  either redundant or too close to the action definition itself

### Step 03: robust state -> readmission analysis

This was the first main causal filter.

We ran a random-stability FCI setup where each graph included:

- fixed confounders:
  - `age`
  - `charlson_score`
  - `prior_ed_visits_6m`
- sampled discharge-state variables
- outcome:
  - `readmit_30d`

The score below is `freq_definite`: how often a variable showed a definite edge
to readmission across the random graphs.

| Rank | Discharge state | `freq_definite` |
|---|---|---:|
| 1 | `last_Hb` | 0.9712 |
| 2 | `last_Ht` | 0.9624 |
| 3 | `last_BUN` | 0.9083 |
| 4 | `last_input_total` | 0.8870 |
| 5 | `last_Phosphate` | 0.5573 |
| 6 | `last_HR` | 0.5414 |
| 7 | `last_PT` | 0.5385 |
| 8 | `last_Creatinine` | 0.5175 |
| 9 | `last_Shock_Index` | 0.4887 |
| 10 | `last_Alkaline_Phosphatase` | 0.3248 |
| 11 | `last_Chloride` | 0.2639 |

How we used this:

- `Hb`, `BUN`, `Creatinine`, `Phosphate`, `HR` were kept because they had clear
  outcome signal
- `Chloride` was weaker on outcome alone, but stayed alive because it later had
  a very strong controllability story in Step 04b
- `Ht` was not kept as a main state because it is largely redundant with `Hb`
- `input_total` was not kept because it is too close to the `ivfluid` action
- `PT` was not kept because the action story was weak
- `Shock_Index` stayed as a borderline candidate

### Step 04b: robust action -> state analysis

This was the second main causal filter.

We then asked a different question:

- which actions robustly move which states?

This was done with a stronger multivariate FCI setup:

- 2 drugs per run
- 2 delta-states per run
- baseline-state adjustment
- fixed confounders including `num_blocs`

The table below focuses only on the states that mattered most for the final
selection.

| State | Strongest action links from Step 04b |
|---|---|
| `Hb` | `vasopressor 1.00`, `ivfluid 0.97`, `antibiotic 0.96`, `mechvent 0.94` |
| `BUN` | `diuretic 1.00`, `ivfluid 0.95`, `insulin 0.83` |
| `Creatinine` | `diuretic 0.94`, `ivfluid 0.87` |
| `Phosphate` | `ivfluid 1.00`, `antibiotic 0.58`, `diuretic 0.57` |
| `HR` | `mechvent 1.00`, `vasopressor 0.77` |
| `Chloride` | `diuretic 1.00`, `ivfluid 0.97`, `insulin 0.89` |
| `Shock_Index` | `insulin 0.55`, `diuretic 0.51` |
| `Ht` | `vasopressor 1.00`, `mechvent 0.98`, `antibiotic 0.95`, `ivfluid 0.95` |
| `PT` | no convincing action story |

This is where the final set became clearer.

What Step 04b changed:

- strengthened the renal/fluid story:
  - `BUN`
  - `Creatinine`
  - `Chloride`
- kept `Hb` highly defensible
- made `mechvent` more credible than in earlier intuition
- weakened `Shock_Index` relative to the core selected states
- confirmed that `PT` did not belong in the main RL state

### Final selected set

#### Dynamic states

- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

#### Static context

- `age`
- `charlson_score`
- `prior_ed_visits_6m`

#### Actions

- `ivfluid`
- `diuretic`
- `vasopressor`
- `mechvent`
- `antibiotic`

### Why this final set

This was the main reasoning:

| Variable | Keep / drop | Reason |
|---|---|---|
| `Hb` | keep | strongest discharge-state signal, clearly modifiable |
| `BUN` | keep | strongest renal readmission state, very clear action story |
| `Creatinine` | keep | clinically central renal marker with robust controllability |
| `Phosphate` | keep | meaningful outcome signal and modifiable by several actions |
| `HR` | keep | outcome-relevant and still controllable |
| `Chloride` | keep | weaker outcome signal alone, but very strong controllability |
| `Shock_Index` | borderline / dropped from final core | less robust than the main six |
| `Ht` | drop | largely redundant with `Hb` |
| `PT` | drop | weak action story |
| `input_total` | drop | too close to action definition |

And for actions:

| Action | Keep / drop | Reason |
|---|---|---|
| `ivfluid` | keep | broadest and strongest coverage across selected states |
| `diuretic` | keep | strongest renal/fluid action |
| `vasopressor` | keep | important hemodynamic intervention |
| `mechvent` | keep | robust action-state links in Step 04b |
| `antibiotic` | keep | weaker than the top actions, but still useful and clinically important |
| `insulin` | drop from main set | some signal, but less aligned with the clearest selected-state story |
| `anticoagulant` | drop from main set | strong mainly for `PTT`, which was not in the final state set |
| `sedation` | drop from main set | mixed and less clean |
| `steroid` | drop | unusable in this run |

### End result of Step 09

Step 09 gave us the selected RL problem we use in the rest of the pipeline:

- state dimension = `9`
  - `6` dynamic states
  - `3` static confounders
- action dimension = `5`
- action space = `32` binary combinations

This is the state/action design that later feeds:

- Step 10 preprocessing
- Step 14 CARE-Sim training
- Step 13 model-based RL
- Step 14 offline RL comparison

## Step 10: preprocessing and terminal outcome model

### Preprocessing

The selected Step 09 variables were turned into the RL-ready state/action
representation used in the later pipeline.

- clipped extreme values to reduce recording artifacts
- log-transformed skewed renal markers:
  - `BUN`
  - `Creatinine`
- z-scored all state variables using the **training split only**
- kept the 3 static confounders in the state as fixed context
- encoded the 5 binary actions into a single discrete action ID:
  - `32` possible combinations

Main output:

- `data/processed/icu_readmit/rl_dataset_selected.parquet`

### Terminal readmission model

We also trained a separate terminal outcome model on the selected state space.

Purpose:

- estimate `30-day readmission risk` from the final selected ICU state
- use that estimate later as the terminal reward signal

Setup:

- one row per ICU stay
- only the terminal row (`done = 1`) is used
- model: `LightGBM classifier`
- target: `readmit_30d`
- performance:
  - validation AUC `0.642`
  - test AUC `0.649`

Model input:

- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`
- `age`
- `charlson_score`
- `prior_ed_visits_6m`

So the model is:

- `final selected state -> probability of 30-day readmission`

Why this matters:

- Step 14 no longer uses a reward head in the final setup
- Step 13 and Step 14 instead use this separate model to score the terminal
  state

### Reward used later in control

Dense reward:

```text
r_dense,t = severity(s_t) - severity(s_{t+1})
```

Handcrafted severity on the transformed selected state space:

```text
severity(s) =
0.28 * ReLU(-Hb)
+ 0.24 * ReLU(BUN)
+ 0.24 * ReLU(Creatinine)
+ 0.10 * ReLU(Phosphate)
+ 0.09 * ReLU(HR)
+ 0.05 * |Chloride|
```

Interpretation:

- lower `Hb` is worse
- higher `BUN`, `Creatinine`, `Phosphate`, `HR` are worse
- `Chloride` is penalized for deviation from the normalized cohort centre
- if severity drops from one step to the next, dense reward is positive

Terminal reward:

```text
r_terminal = 15 - 30 * p(readmit_30d | s_terminal)
```

Total reward:

```text
r_t = r_dense,t + r_terminal    (terminal step only)
```

## Step 14: selected causal CARE-Sim

### Goal

Train a world model that can answer:

- given a recent patient trajectory
- and a proposed treatment action
- what is the next state likely to be?

This is the model we later use for:

- short-horizon simulator rollouts
- planner search
- DDQN training inside the simulator

### Model structure

The final model is a causal transformer world model operating on ICU
trajectories.

Input per time step:

- current selected state
- current action
- elapsed time within the stay

Output per time step:

- predicted next state
- terminal probability

Architecture:

- one token = one 4-hour ICU bloc
- the model sees the full past trajectory, not only the last state
- causal attention masking prevents the model from using future information
- state and action are embedded jointly at each time step
- the final run used:
  - hidden dimension `256`
  - `8` attention heads
  - `4` transformer layers
  - an ensemble of `5` independently trained models

### How static confounders are handled

The 3 static confounders are included in the state input, but treated as fixed
context.

That means:

- they condition the trajectory dynamics
- but the model does not try to "evolve" them over time

This was implemented with `freeze_static_context = true`.

### Causal part of the architecture

The causal part is implemented directly in the transition model.

On top of the transformer prediction, we add a separate masked linear
`action -> state` residual branch.

Key idea:

- the transformer learns the general temporal dynamics
- the residual branch is reserved for direct action effects
- only action-state links supported by the Step 09 / Step 04b analysis are
  allowed in that direct branch

For the final selected set, the mask allows:

| State | Allowed direct action links |
|---|---|
| `Hb` | `vasopressor`, `ivfluid`, `antibiotic`, `mechvent` |
| `BUN` | `ivfluid`, `diuretic` |
| `Creatinine` | `ivfluid`, `diuretic` |
| `Phosphate` | `ivfluid`, `antibiotic`, `diuretic` |
| `HR` | `vasopressor`, `mechvent` |
| `Chloride` | `ivfluid`, `diuretic` |

The static confounders have no direct action links.

Technical point:

- disallowed entries in the action-state weight matrix are fixed to zero
- this constrains the explicit direct treatment effect component of the model

This does not make the whole system a fully identified causal model. The right
description is a `causally informed world model`: flexible sequence dynamics
plus a structurally constrained direct treatment-effect component.

### Why this version was chosen

This final version was chosen because it was the cleanest trade-off between
expressiveness and structure.

- smaller and more defensible state/action space
- static confounders treated correctly as context
- explicit elapsed-time feature
- random training windows, which reduced the old sequence-boundary artifact
- causal structure from Step 09 carried directly into the architecture

### Output of Step 14

Step 14 produces the trained selected causal CARE-Sim ensemble:

- `models/icu_readmit/caresim_selected_causal/`

This becomes the simulator used in:

- Step 15 for validation
- Step 13 for model-based RL

## Step 15: simulator validation

### Purpose

Step 15 asks a simple question:

- is the trained world model accurate enough to use for short-horizon control?

We evaluated it in two ways:

- `one-step prediction`: predict the next real ICU bloc from the real history
- `short rollout`: roll the simulator forward recursively for a few steps

### One-step results

| Metric | Validation | Test | Interpretation |
|---|---:|---:|---|
| Next-state MSE | 0.0678 | 0.0681 | very similar on val/test; no obvious generalization gap |
| Next-state RMSE | 0.260 | 0.261 | about `0.26` SD per state feature on average; reasonable for a learned clinical simulator |
| Terminal accuracy | 0.9536 | 0.9536 | high locally, but inflated by the fact that most rows are non-terminal |
| Mean uncertainty | 0.0207 | 0.0207 | low ensemble disagreement on held-out one-step predictions |

Main reading:

- one-step transition prediction is stable
- val and test are almost identical
- terminal accuracy looks strong, but should not be overinterpreted

### Short-rollout results

| Metric | Validation | Test | Interpretation |
|---|---:|---:|---|
| Step-1 rollout MSE | 0.0910 | 0.0891 | still close to the real trajectory after one recursive step |
| Final-step rollout MSE | 0.2497 | 0.2053 | error grows with recursive rollout, but stays usable over a short horizon |

Main reading:

- the simulator is good enough for short rollouts
- recursive error accumulation is real
- this supports a `short-horizon` control framing, not long free rollouts

### Main caveat

The main weakness from Step 15 was terminal behavior.

- one-step terminal classification looked good
- but free-rollout stopping behavior was not realistic
- because of this, Step 13 is framed as:
  - `5-step finite-horizon control`
  - not full natural-length ICU trajectory optimization

## Step 13: model-based RL in CARE-Sim

### Goal

Use the trained simulator as a decision environment and compare several control
policies on the same held-out patient seeds.

All policies start from:

- the same real 5-bloc seed history
- then control the simulator for up to 5 further steps

### Policies compared

| Policy | What it does |
|---|---|
| `ddqn` | learned policy network trained inside CARE-Sim |
| `planner` | short-horizon simulator search baseline |
| `random` | samples actions uniformly from the 32-action space |
| `repeat_last` | repeats the last observed clinician action from the seed history |

Short description of each:

- `ddqn`:
  - takes the recent 5-step history as input
  - outputs action values for the 32 possible action combinations
  - chooses the action with the highest predicted long-term return

- `planner`:
  - tries candidate actions inside the simulator
  - scores short simulated futures
  - picks the action with the best short-horizon score

- `random`:
  - serves as a minimal baseline
  - no learning, no planning

- `repeat_last`:
  - serves as a simple clinician-history baseline
  - asks whether doing "more of the same" is already competitive

### Step 13 evaluation setup

The comparison is done on held-out seed episodes.

For each held-out stay:

- take the first `5` real 4-hour blocs as seed history
- let each policy control the simulator for up to `5` further steps
- compare the resulting rollout return

So this is a short-horizon policy comparison:

- same starting point
- different action choices
- same simulator and reward function

### Main score

The main score is the discounted rollout return.

At each simulated step we compute:

- reward from the selected reward function
- minus an uncertainty penalty

and then discount later steps slightly by `gamma = 0.99`.

Interpretation:

- higher return = better
- lower uncertainty helps
- the score reflects both trajectory improvement and terminal outcome reward

### Metrics reported

| Metric | Meaning | How to read it |
|---|---|---|
| `mean_discounted_return` | main policy score over the rollout | higher is better |
| `mean_raw_reward_total` | reward before uncertainty penalty | higher is better |
| `mean_uncertainty` | average ensemble disagreement during rollout | lower is better |
| `std_discounted_return` | variability across held-out patients | lower = more stable policy |
| `p25 / p50 / p75 return` | distribution of performance across patients | helps show whether gains are broad or only in a subset |
| `action_counts` | how often each action was used | shows whether the policy is broad or concentrated |

Metrics we do **not** overinterpret:

- `termination_rate`
  - here it is almost always `1.0`
  - mainly because rollouts are capped at 5 steps

- `mean_last_terminal_prob`
  - useful only as a secondary simulator signal
  - not a direct clinical endpoint

### Step 13 results

#### Validation split

| Policy | Mean discounted return | Mean raw reward | Mean uncertainty |
|---|---:|---:|---:|
| `ddqn` | 8.692 | 9.069 | 0.0172 |
| `repeat_last` | 8.680 | 9.054 | 0.0170 |
| `random` | 8.593 | 8.968 | 0.0187 |
| `planner` | 8.583 | 8.939 | 0.0140 |

Pairwise against `ddqn` on validation:

- `ddqn - planner`: mean difference `+0.109`, win rate `63%`
- `ddqn - random`: mean difference `+0.099`, win rate `63%`
- `ddqn - repeat_last`: mean difference `+0.012`, win rate `45%`

#### Test split

| Policy | Mean discounted return | Mean raw reward | Mean uncertainty |
|---|---:|---:|---:|
| `ddqn` | 8.836 | 9.219 | 0.0165 |
| `random` | 8.789 | 9.173 | 0.0186 |
| `planner` | 8.786 | 9.148 | 0.0140 |
| `repeat_last` | 8.807 | 9.184 | 0.0177 |

Pairwise against `ddqn` on test:

- `ddqn - planner`: mean difference `+0.051`, win rate `58%`
- `ddqn - random`: mean difference `+0.047`, win rate `58%`
- `ddqn - repeat_last`: mean difference `+0.030`, win rate `52%`

### Main reading

- `ddqn` ranked first on both validation and test
- the ranking was consistent across splits
- the improvement over `planner` and `random` was positive, but modest
- `repeat_last` remained surprisingly competitive

So the Step 13 result is:

- the simulator-trained RL policy does improve over simple baselines
- but the gain is incremental rather than dramatic

## Step 14: offline RL comparison

### Goal

Train a second DDQN directly on the logged ICU data and compare it against the
Step 13 world-model DDQN on the same held-out real-data benchmark.

### Evaluation method

Step 14 does **not** roll the offline DDQN out in the simulator.

Instead, both DDQN policies are evaluated on held-out logged data using
`OPE (Off-Policy Evaluation)`.

Short version of OPE here:

- take held-out clinician trajectories
- ask what action each learned policy would have chosen
- estimate policy value with a **doubly robust** estimator
- for one logged trajectory, compute backward:

```text
V_t = Q_hat(s_t, pi(s_t)) + w_t * (r_t + gamma * V_{t+1} - Q_hat(s_t, a_logged,t))
```

- with terminal condition:

```text
V_T = 0
```

- final OPE score:

```text
mean over trajectories of V_1
```

- terms:
  - `s_t`: logged state / history at time `t`
  - `pi(s_t)`: action chosen by the evaluated policy
  - `a_logged,t`: action actually taken by the clinician
  - `r_t`: logged reward at that step
  - `Q_hat(s_t, pi(s_t))`: model-estimated value of the evaluated policy
  - `Q_hat(s_t, a_logged,t)`: model-estimated value of the logged action
  - `w_t`: importance weight correcting for clinician action propensity
  - `gamma`: discount factor

Interpretation:

- higher OPE value = better
- this is the main real-data comparison between the two DDQN policies
- it is an estimate, not a realized rollout return

### Step 14 results

| Policy | Validation OPE | Test OPE | Valid trajectories (val/test) |
|---|---:|---:|---:|
| `offline_ddqn` | -0.095 | -0.114 | 9253 / 9261 |
| `worldmodel_ddqn` | -2.903 | -2.780 | 8219 / 8207 |

Main reading:

- the offline DDQN clearly outperformed the world-model DDQN on held-out real
  data
- the difference was large and consistent across validation and test
- the world-model DDQN was also evaluated on fewer valid trajectories

### Action pattern summary

| Policy | Clinician agreement (val) | Clinician agreement (test) | Main action pattern |
|---|---:|---:|---|
| `offline_ddqn` | 15.99% | 16.20% | heavily concentrated on action `0` |
| `worldmodel_ddqn` | 26.67% | 26.94% | concentrated mainly on actions `2`, `18`, `23`, `3` |

Interesting point:

- the world-model DDQN matched clinician actions more often
- but its OPE value was still substantially worse than the offline DDQN

### Main takeaway

- Step 13 looked encouraging inside the simulator
- Step 14 gave the cleaner real-data comparison
- on that benchmark, the offline DDQN was stronger than the simulator-trained
  DDQN
