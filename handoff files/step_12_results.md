# Step 12 Results: Full Simulator Evaluation

This document is the detailed results companion to step 11.

If step 11 is the training layer, step 12 is the evaluation layer. This is where we ask whether the trained simulators actually behave sensibly on held-out data.

The three simulator tracks are:

- CARE-Sim transformer (`12a`)
- MarkovSim baseline (`12b`)
- DAG-aware transformer (`12c`)

The evaluation outputs are stored under `reports/icu_readmit/` for each track.

## What Step 12 Does

Step 12 evaluates each trained simulator on held-out ICU trajectories using the same selected state/action setup.

The core questions are:

- How well does the simulator predict the next state?
- How well does it predict terminal behavior?
- Does uncertainty behave sensibly?
- Do static confounders stay fixed?
- Does rollout stay stable when we simulate multiple steps forward?
- Does the model react in a structured way to action counterfactuals?

That means step 12 is not just “another benchmark.” It is the main evidence that the step 11 models are usable as simulators.

## The Three Evaluation Runs

### `12a`: CARE-Sim transformer

This is the matched evaluation for the large CARE-Sim ensemble.

Important note:

- the saved report that exists in the repo is a **reduced-budget evaluation**
- it is still useful, but it is not as heavy as the full `12b` / `12c` evaluations

The report file is:

- `reports/icu_readmit/caresim_selected_causal/caresim_summary.json`

### `12b`: MarkovSim baseline

This is the full MarkovSim evaluation.

The report file is:

- `reports/icu_readmit/markovsim_selected_causal/markovsim_summary.json`

### `12c`: DAG-aware transformer

This is the full DAG-aware evaluation using the 3-member ensemble.

The report file is:

- `reports/icu_readmit/dagaware_selected_causal/dagaware_summary.json`

## Shared Evaluation Setup

All three runs use the same core variables:

- `history_len = 5`
- `rollout_steps = 5`
- `rollout_patients = 200`
- `counterfactual_patients = 10`
- the same selected 9-state / 5-action ICU setup

The nine state variables are:

- `s_Hb`
- `s_BUN`
- `s_Creatinine`
- `s_Phosphate`
- `s_HR`
- `s_Chloride`
- `s_age`
- `s_charlson_score`
- `s_prior_ed_visits_6m`

The five actions are:

- `vasopressor_b`
- `ivfluid_b`
- `antibiotic_b`
- `diuretic_b`
- `mechvent_b`

## High-Level Summary

| Model | One-step test MSE | Test terminal accuracy | Mean uncertainty | Test rollout last-state MSE | Counterfactual rows |
|---|---:|---:|---:|---:|---:|
| CARE-Sim (`12a`) | 0.0681 | 0.9536 | 0.0207 | 0.2052 | 320 |
| MarkovSim (`12b`) | 0.0718 | 0.5429 | 0.2088 | 0.2064 | 320 |
| DAG-aware (`12c`) | 0.0723 | 0.9534 | 0.0145 | 0.2141 | 320 |

## What the Metrics Mean

### One-step test MSE

This is the average error when the simulator predicts the next state from the current observation.

Lower is better.

### Terminal accuracy

This measures how well the simulator predicts the terminal / readmission-related outcome.

Higher is better.

### Mean uncertainty

This is the ensemble uncertainty estimate.

Lower is not automatically better, but it matters that the value is finite and behaves consistently.

### Rollout last-state MSE

This tells us whether the simulator stays stable when rolled forward multiple steps instead of being used only one step at a time.

Lower is better.

### Counterfactual rows

This is the size of the action-sweep output table.

For each selected patient and each counterfactual action, the evaluator records:

- predicted reward
- terminal probability
- uncertainty
- action bits
- next-state predictions

## CARE-Sim (`12a`) in Detail

### Core numbers

From `caresim_summary.json`:

- `reward_source = zero`
- `use_terminal_readmit_reward = false`
- one-step val MSE: `0.067839`
- one-step test MSE: `0.068069`
- one-step test terminal accuracy: `0.953587`
- one-step test mean uncertainty: `0.020693`
- rollout test last-state MSE: `0.205211`
- counterfactual rows: `320`

### Rollout behavior

CARE-Sim’s rollout values are:

- val step1 state MSE: `0.090983`
- val last-state MSE: `0.249500`
- test step1 state MSE: `0.089132`
- test last-state MSE: `0.205211`

That means:

- the first rollout step is still reasonably close to the observed trajectory
- the error increases as expected over time
- the model does not immediately diverge

### Interpretation

CARE-Sim is the strongest raw predictive model in the set.

What stands out:

- lowest one-step test MSE
- very high terminal accuracy
- decent and finite uncertainty
- stable enough rollout behavior

This is the model that gives you the best pure fidelity benchmark.

One caveat:

- the stored `12a` report is reduced-budget compared with the other full evaluations
- so the rollout and counterfactual outputs should be treated as diagnostic evidence, not the final word

## MarkovSim (`12b`) in Detail

### Core numbers

From `markovsim_summary.json`:

- `use_severity_reward = true`
- `severity_mode = handcrafted`
- `use_terminal_readmit_reward = true`
- one-step test MSE: `0.071809`
- one-step test terminal accuracy: `0.542879`
- one-step test mean uncertainty: `0.208782`
- rollout test last-state MSE: `0.206359`
- counterfactual rows: `320`

### Per-feature one-step error

The MarkovSim test per-feature MSE is:

- `Hb`: `0.077567`
- `BUN`: `0.056487`
- `Creatinine`: `0.048991`
- `Phosphate`: `0.158593`
- `HR`: `0.231615`
- `Chloride`: `0.073033`
- `age`: `0.0`
- `charlson_score`: `0.0`
- `prior_ed_visits_6m`: `0.0`

The zero error on static features is expected because static features are not supposed to drift.

### Rollout behavior

MarkovSim’s rollout values are:

- val per-step state MSE:
  - `0.093351`
  - `0.132811`
  - `0.169222`
  - `0.194730`
  - `0.231342`
- test per-step state MSE:
  - `0.092901`
  - `0.132862`
  - `0.158667`
  - `0.179368`
  - `0.206359`

This is a consistent increasing-error pattern, which is normal for rollout.

### Reward and done behavior

MarkovSim’s test rollout done accuracy improves over the horizon:

- `0.53`
- `0.445`
- `0.43`
- `0.41`
- `0.41`

That said, its terminal accuracy in the one-step setting is much weaker than the transformer models.

### Interpretation

MarkovSim is the simplest and cheapest simulator, but it is the weakest on the clinically important terminal task.

What stands out:

- one-step MSE is only slightly worse than the transformers
- terminal accuracy is far lower
- uncertainty is much larger
- rollout is stable, but not especially expressive

The main lesson is that one-step MSE alone would make MarkovSim look better than it really is.

It is useful as a baseline, but it is not a strong final simulator.

## DAG-aware (`12c`) in Detail

### Core numbers

From `dagaware_summary.json`:

- `reward_source = zero`
- `use_terminal_readmit_reward = false`
- one-step test MSE: `0.072288`
- one-step test terminal accuracy: `0.953376`
- one-step test mean uncertainty: `0.014511`
- rollout test last-state MSE: `0.214085`
- counterfactual rows: `320`

### Rollout behavior

DAG-aware’s rollout values are:

- val step1 state MSE: `0.093224`
- val last-state MSE: `0.254414`
- test step1 state MSE: `0.091259`
- test last-state MSE: `0.214085`

So:

- the first step is close to the observed state
- the error grows over rollout horizon, as expected
- the trajectory remains numerically stable

### Uncertainty

This is the biggest practical improvement over the earlier 1-member version:

- `mean_uncertainty` is finite
- uncertainty is smaller than CARE-Sim and much smaller than MarkovSim
- the model now supports uncertainty-aware downstream use

This matters because the DAG-aware control and offline comparison work is much more meaningful when the simulator exposes a real ensemble spread.

### Static confounder preservation

The DAG-aware report shows static drift staying at zero:

- `age = 0.0`
- `charlson_score = 0.0`
- `prior_ed_visits_6m = 0.0`

That is exactly what we want from a simulator that is supposed to keep static confounders fixed.

### Interpretation

DAG-aware is the most scientifically interesting simulator.

What stands out:

- one-step MSE is essentially tied with CARE-Sim
- terminal accuracy is also essentially tied with CARE-Sim
- uncertainty is the smallest of the three
- static variables behave exactly as intended
- rollout is stable and usable

This is the best story model for the thesis because it combines:

- good fidelity
- causal structure
- usable uncertainty

## Side-by-Side Interpretation

### 1. CARE-Sim is the raw accuracy winner

If the only question is “which simulator predicts the next state most accurately?”, CARE-Sim wins by a small margin.

That makes sense because it is the largest and most expressive model.

### 2. DAG-aware is the best balanced model

DAG-aware is only slightly behind CARE-Sim on raw MSE, but it adds:

- explicit causal structure
- static-confounder discipline
- finite ensemble uncertainty that is actually useful

For the thesis, that makes DAG-aware a very strong candidate for the main proposed simulator.

### 3. MarkovSim is the baseline foil

MarkovSim is useful because it shows what happens if we strip the problem down to a cheap tabular / ridge-style transition model.

It is not competitive on terminal behavior, which is exactly the kind of result you want from a baseline:

- it works
- but it is clearly weaker than the transformer-based methods

## What the Counterfactual Sweeps Show

Each counterfactual CSV contains 320 rows.

For each selected patient state, the evaluator sweeps over actions and records:

- action bits
- predicted reward
- terminal probability
- uncertainty
- predicted next state

That tells us whether the simulator is sensitive to action changes, rather than treating all actions as effectively the same.

### CARE-Sim counterfactuals

The CARE-Sim sweep shows action-dependent changes in:

- terminal probability
- uncertainty
- next-state predictions

This is useful, but because the reward source is zero in that run, the reward column itself is not the main signal.

### MarkovSim counterfactuals

The MarkovSim sweep is broader in uncertainty and less structured in terminal behavior.

That is consistent with the one-step results:

- the model reacts to actions
- but not with the same clinically meaningful separation as the transformers

### DAG-aware counterfactuals

The DAG-aware sweep is the cleanest from a structural perspective:

- action changes are represented explicitly
- static features remain fixed
- uncertainty is low and finite

That makes the action sweep easier to reason about downstream.

## What Step 12 Means for the Thesis

Step 12 is where the thesis claims start to become credible.

The results support three separate statements:

1. The simulator family is working.
2. The transformer models are much better than the simple Markov baseline on terminal behavior.
3. The DAG-aware simulator is a strong middle ground: nearly as accurate as CARE-Sim, but more structurally disciplined and better suited for later causal/control discussion.

## Caveats You Should Keep in Mind

### 1. The CARE-Sim report in the repo is reduced-budget

It is still useful, but it should not be treated as directly identical to the full `12b` and `12c` evaluation budget.

### 2. One-step MSE does not tell the whole story

The Markov baseline would look less weak if you looked only at MSE.

Terminal accuracy and rollout behavior matter just as much.

### 3. Uncertainty behaves very differently across models

That is not an error; it reflects the way the models are built.

### 4. The DAG-aware model only becomes truly useful once the ensemble is larger than one

The 3-member version is what makes the uncertainty signal meaningful.

## Bottom Line

The step 12 story is:

- **CARE-Sim** gives the best raw predictive fidelity.
- **MarkovSim** is a useful but clearly weaker baseline, especially on terminal prediction.
- **DAG-aware** is almost as accurate as CARE-Sim while being more structured, more interpretable, and better suited for uncertainty-aware downstream use.

If you want the shortest thesis sentence:

> Step 12 shows that the DAG-aware simulator retains transformer-level predictive performance while adding causal structure and useful uncertainty, whereas MarkovSim remains a much simpler but weaker baseline.

If you want, I can next write the same kind of document for **step 13 control results** or **step 14 offline RL results**.
