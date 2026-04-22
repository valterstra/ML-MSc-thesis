# Step 13 Results: Control Policies on Top of the Simulators

This document summarizes the control experiments built on top of the three simulator tracks:

- CARE-Sim control
- MarkovSim control
- DAG-aware control

Step 13 asks a different question from step 12.

- Step 12 asks: â€œHow good is the simulator?â€
- Step 13 asks: â€œWhat happens when we use the simulator to choose actions?â€

That makes step 13 the policy layer of the thesis story.

## What Step 13 Does

Each step 13 run evaluates several policy types inside the simulator environment:

- `ddqn`
- `planner`
- `random`
- `repeat_last`

The reward shaping is the same across the control tracks:

- handcrafted severity reward
- terminal readmission reward
- uncertainty penalty

The important difference is the simulator underneath:

- CARE-Sim for `13a`
- MarkovSim for `13b`
- DAG-aware for `13c`

## How to Read the Control Results

The key metrics in the step 13 summaries are:

- `mean_discounted_return`
- `mean_raw_reward_total`
- `mean_uncertainty`
- `termination_rate`
- `mean_rollout_steps`
- `mean_last_terminal_prob`
- `std_discounted_return`
- interquartile range of discounted return
- `mean_step_reward`
- `mean_step_penalized_reward`
- `mean_step_terminal_prob`
- `action_counts`
- `first_action_counts`

These metrics matter because step 13 is not just about â€œwhich policy gets the highest return.â€

We also want to know:

- does the policy actually explore a meaningful action set?
- does the planner collapse to a single trivial action?
- does the DDQN learn a narrow or broad policy?
- does the policy behave differently from random or repeat-last?
- does the simulator create an environment where policy optimization is meaningful at all?

## A Note on Naming

The repository uses the historical step-16 file names for the older control tracks:

- CARE-Sim control report: `step_16_summary.json`
- MarkovSim control report: `step_13b_markovsim_summary.json`
- DAG-aware control report: `step_13c_summary.json`

For the thesis, these are the step 13 results for the three control branches.

## CARE-Sim Control (`13a`)

Report file:

- `reports/icu_readmit/caresim_control_selected_causal/step_16_summary.json`

### Meta setup

The CARE-Sim control run uses:

- `history_len = 5`
- `observation_window = 5`
- `rollout_steps = 5`
- `planner_horizon = 3`
- `uncertainty_penalty = 0.25`
- handcrafted severity reward
- terminal readmission reward
- terminal reward scale `15.0`

### Validation split

#### DDQN

- episodes: `100`
- mean discounted return: `8.6921`
- mean raw reward total: `9.0693`
- mean uncertainty: `0.01721`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.03917`
- std discounted return: `2.3206`
- median discounted return: `9.4068`
- mean step reward: `1.8139`
- mean step penalized reward: `1.8096`
- mean step terminal probability: `0.03368`

DDQN action use:

- action `2`: `152`
- action `3`: `54`
- action `4`: `3`
- action `6`: `20`
- action `8`: `15`
- action `10`: `10`
- action `14`: `2`
- action `15`: `1`
- action `16`: `5`
- action `18`: `90`
- action `19`: `7`
- action `21`: `1`
- action `22`: `15`
- action `23`: `102`
- action `25`: `2`
- action `26`: `6`
- action `30`: `12`
- action `31`: `2`

First-action counts show a broad initial policy, with the most common first actions being:

- `2`
- `18`
- `23`
- `3`
- `6`

#### Planner

- mean discounted return: `8.5829`
- mean raw reward total: `8.9394`
- mean uncertainty: `0.01404`
- mean rollout steps: `4.89`
- mean last terminal probability: `0.04941`

The planner explores many actions, not just one:

- it uses actions `0`, `2`, `3`, `5`, `6`, `7`, `10`, `14`, `16`, `18`, `20`, `22`, `23`, `26`, `27`, `30`

This is a broad policy, and importantly it does **not** collapse to a no-op.

#### Random

- mean discounted return: `8.5929`
- mean raw reward total: `8.9685`
- mean uncertainty: `0.01873`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.00674`

Random is surprisingly close to the learned policies on raw return, but its action distribution is, by construction, maximally broad.

#### Repeat-last

- mean discounted return: `8.6803`
- mean raw reward total: `9.0542`
- mean uncertainty: `0.01702`
- mean rollout steps: `4.98`
- mean last terminal probability: `0.06428`

Repeat-last is actually competitive with DDQN on this validation split, which is a sign that the control task is not extremely easy and that the reward signal is fairly smooth.

### Test split

#### DDQN

- mean discounted return: `8.8363`
- mean raw reward total: `9.2188`
- mean uncertainty: `0.01654`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.05131`
- std discounted return: `2.2339`
- median discounted return: `9.5550`
- mean step reward: `1.8438`
- mean step penalized reward: `1.8396`
- mean step terminal probability: `0.04234`

DDQN action use on test:

- action `2`: `206`
- action `3`: `25`
- action `4`: `8`
- action `6`: `6`
- action `8`: `15`
- action `10`: `14`
- action `14`: `4`
- action `15`: `4`
- action `18`: `81`
- action `19`: `10`
- action `22`: `29`
- action `23`: `80`
- action `25`: `2`
- action `26`: `6`
- action `30`: `9`

This is a broad policy, but still concentrated around a handful of actions.

#### Planner

- mean discounted return: `8.7856`
- mean raw reward total: `9.1478`
- mean uncertainty: `0.01400`
- mean rollout steps: `4.87`
- mean last terminal probability: `0.05581`

The planner explores a wide action space and is slightly weaker than DDQN on this test split.

#### Random

- mean discounted return: `8.7894`
- mean raw reward total: `9.1741`
- mean uncertainty: `0.01343`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.02961`

Random is again close to the learned policies in terms of total return, which suggests the reward surface is relatively flat.

#### Repeat-last

- mean discounted return: `8.7937`
- mean raw reward total: `9.1694`
- mean uncertainty: `0.01259`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.09208`

Repeat-last is essentially tied with the other policies on this split.

### Interpretation of CARE-Sim control

CARE-Sim control is the cleanest and most balanced control result in the three tracks.

What stands out:

- DDQN is competitive with planner, random, and repeat-last
- the policy does not collapse to a single action
- both planner and DDQN use a broad but structured action mix
- mean returns are all in a fairly tight band

The main interpretation is not that DDQN dominates by a large margin.

Instead, CARE-Sim provides a stable environment where:

- control is possible
- returns are meaningful
- the policy is not trivial
- the action space is used in a clinically plausible way

This is a strong result because it shows the simulator can support downstream RL without immediately degenerating.

## MarkovSim Control (13b)
Report file:
- reports/icu_readmit/markovsim_control_selected_causal/step_13b_markovsim_summary.json
### Meta setup
The MarkovSim control run uses the same reward structure:
- handcrafted severity reward
- terminal readmission reward
- uncertainty penalty
This refreshed run also sets:
- terminal_stop_threshold = 1.1
That disables premature terminal stopping during the control comparison, so MarkovSim now runs to the fixed 5-step horizon like the two transformer simulators.
### Validation split
#### DDQN
- mean discounted return: 8.5472
- mean raw reward total: 9.1625
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.30590
- mean step reward: 1.8325
- mean step penalized reward: 1.7803
- mean step terminal probability: 0.27949
DDQN action use:
- action 0: 42
- action 1: 8
- action 2: 136
- action 3: 39
- action 4: 14
- action 5: 1
- action 6: 20
- action 7: 44
- action 8: 9
- action 10: 3
- action 12: 3
- action 13: 1
- action 15: 3
- action 16: 3
- action 17: 1
- action 18: 2
- action 19: 41
- action 20: 4
- action 21: 32
- action 22: 3
- action 23: 49
- action 24: 7
- action 25: 2
- action 26: 2
- action 27: 3
- action 28: 15
- action 30: 5
- action 31: 8
This is no longer a one-step collapsed policy. The DDQN now uses a much broader action set and behaves like a real 5-step policy.
#### Planner
- mean discounted return: 8.5461
- mean raw reward total: 9.1613
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.71807
- mean step reward: 1.8323
- mean step penalized reward: 1.7801
Planner action use:
- action 0 only
The planner still collapses to a single action, but the episode no longer ends after one step. So the remaining problem is action degeneracy, not premature termination.
#### Random
- mean discounted return: 8.4482
- mean raw reward total: 9.0598
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.13605
Random has lower return than DDQN and planner, but not by a large margin.
#### Repeat-last
- mean discounted return: 8.5801
- mean raw reward total: 9.1967
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.33790
Repeat-last is surprisingly competitive, and in some sense the whole MarkovSim control setting appears to be close to the same reward level regardless of policy.
### Test split
#### DDQN
- mean discounted return: 8.7437
- mean raw reward total: 9.3670
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.36344
- mean step reward: 1.8734
- mean step penalized reward: 1.8212
DDQN action use on test:
- action 0: 29
- action 1: 4
- action 2: 181
- action 3: 40
- action 4: 25
- action 5: 2
- action 6: 18
- action 7: 27
- action 8: 10
- action 10: 1
- action 12: 1
- action 13: 2
- action 15: 3
- action 16: 4
- action 17: 5
- action 18: 9
- action 19: 26
- action 20: 4
- action 21: 25
- action 22: 5
- action 23: 27
- action 24: 8
- action 25: 5
- action 26: 2
- action 27: 3
- action 28: 13
- action 30: 14
- action 31: 7
#### Planner
- mean discounted return: 8.7577
- mean raw reward total: 9.3816
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.71659
- action counts: 0 only
This is still a degenerate planner result, but for a different reason than before:
- the planner always chooses action 0
- the rollout now lasts the full five steps
- the remaining weakness is action collapse, not early stopping
#### Random
- mean discounted return: 8.6858
- mean raw reward total: 9.3071
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.14658
#### Repeat-last
- mean discounted return: 8.7613
- mean raw reward total: 9.3853
- mean uncertainty: 0.20878
- mean rollout steps: 5.0
- mean last terminal probability: 0.37496
### Interpretation of MarkovSim control
MarkovSim control is the weakest of the three control environments.
What stands out:
- uncertainty is very large and constant
- planner collapses to a single action
- the control horizon problem has been fixed: all policies now run for the full 5 steps
- DDQN is competitive but not clearly better than the simple baselines
This suggests that the original early-termination problem was a terminal-stop artifact, but even after fixing it, the MarkovSim environment is still too coarse to support a nuanced control story.
It still provides a baseline, but it is not the best platform for a meaningful policy comparison.

## DAG-aware Control (13c)

Report file:

- `reports/icu_readmit/dagaware_control_selected_causal/step_13c_summary.json`

### Meta setup

The DAG-aware control run uses:

- handcrafted severity reward
- terminal readmission reward
- uncertainty penalty
- the 3-member DAG-aware simulator

This is the most thesis-relevant control setting because the simulator has meaningful uncertainty and an explicit causal structure.

### Validation split

#### DDQN

- episodes: `100`
- mean discounted return: `8.6013`
- mean raw reward total: `8.9687`
- mean uncertainty: `0.01229`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.07195`
- std discounted return: `2.3206`
- median discounted return: `9.2631`
- mean step reward: `1.7937`
- mean step penalized reward: `1.7907`
- mean step terminal probability: `0.06586`

DDQN action use:

- action `2`: `299`
- action `3`: `5`
- action `18`: `191`
- action `19`: `5`

This is a very narrow policy.

It is not trivial in the sense of using only one action, but it is highly concentrated on two actions:

- `2`
- `18`

First-action counts show the same pattern:

- action `2` dominates
- action `18` is second

#### Planner

- mean discounted return: `8.6390`
- mean raw reward total: `9.0039`
- mean uncertainty: `0.00940`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.10509`
- std discounted return: `2.3431`
- median discounted return: `9.3701`
- mean step reward: `1.8008`
- mean step penalized reward: `1.7984`
- mean step terminal probability: `0.08486`

Planner action use is much broader than DDQN.

It spans almost the full action set, including:

- `0`, `1`, `2`, `3`, `4`, `5`, `6`, `7`, `8`, `9`
- `10`, `11`, `12`, `13`, `14`, `15`, `16`, `17`, `18`, `19`
- `20`, `21`, `22`, `23`, `24`, `25`, `26`, `27`, `28`, `29`, `30`, `31`

This means the planner is exploring rather than collapsing.

#### Random

- mean discounted return: `8.5591`
- mean raw reward total: `8.9258`
- mean uncertainty: `0.01298`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.02992`
- std discounted return: `2.3547`
- mean step reward: `1.7852`
- mean step penalized reward: `1.7819`

Random is slightly worse than DDQN and planner on validation, but not by a huge margin.

#### Repeat-last

- mean discounted return: `8.6056`
- mean raw reward total: `8.9727`
- mean uncertainty: `0.01211`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.08594`
- std discounted return: `2.3273`
- mean step reward: `1.7945`
- mean step penalized reward: `1.7915`

Repeat-last is close to DDQN and planner.

### Test split

#### DDQN

- mean discounted return: `8.8179`
- mean raw reward total: `9.1946`
- mean uncertainty: `0.01259`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.08137`
- std discounted return: `2.1774`
- median discounted return: `9.4924`
- mean step reward: `1.8389`
- mean step penalized reward: `1.8358`
- mean step terminal probability: `0.06816`

DDQN action use on test:

- action `2`: `307`
- action `3`: `9`
- action `18`: `172`
- action `19`: `12`

This is again a narrow policy concentrated on a small action subset.

#### Planner

- mean discounted return: `8.7873`
- mean raw reward total: `9.1587`
- mean uncertainty: `0.00955`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.10654`
- std discounted return: `2.2106`
- median discounted return: `9.4832`
- mean step reward: `1.8317`
- mean step penalized reward: `1.8294`

Planner action use is broad and varied.

#### Random

- mean discounted return: `8.7970`
- mean raw reward total: `9.1741`
- mean uncertainty: `0.01343`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.02961`
- std discounted return: `2.1982`
- mean step reward: `1.8348`
- mean step penalized reward: `1.8315`

#### Repeat-last

- mean discounted return: `8.7937`
- mean raw reward total: `9.1694`
- mean uncertainty: `0.01259`
- termination rate: `1.0`
- mean rollout steps: `5.0`
- mean last terminal probability: `0.09208`
- std discounted return: `2.2379`
- mean step reward: `1.8339`
- mean step penalized reward: `1.8307`

### Interpretation of DAG-aware control

DAG-aware control is the most balanced of the three control branches.

What stands out:

- returns are tightly clustered across DDQN, planner, random, repeat-last
- the learned DDQN policy is narrow, but not degenerate
- the planner is broader than DDQN
- uncertainty is low and finite
- the model uses the full five-step horizon

The key takeaway is not that DDQN dramatically wins.

Instead:

- the simulator is stable enough to support control
- the control task is meaningful
- the reward shaping produces sensible returns
- the policy is not obviously pathological

This makes DAG-aware a very good thesis candidate because it is the cleanest combination of:

- predictive fidelity
- causal structure
- usable uncertainty
- downstream control viability

## Side-by-Side Comparison Across the Three Control Tracks

### 1. Return levels are broadly similar

Across all three control tracks, the return values are fairly close.

This means the task is not producing huge separations between policies.

That is important because it prevents overclaiming:

- these are not giant policy wins
- the control problem is relatively smooth
- the main differences are in policy structure and simulator behavior

### 2. CARE-Sim shows the healthiest policy diversity

CARE-Sim has:

- broad action usage
- full rollout horizon
- strong returns
- stable uncertainty

This is the best sign that the simulator can support real control behavior.

### 3. MarkovSim is still too coarse for a strong control story

MarkovSimâ€™s planner collapses to action `0`, and the mean rollout length is only about one step.

However:

- the planner still collapses to action  
- uncertainty remains high and constant 
- DDQN remains only modestly differentiated from the simple baselines

That means the main weakness now is no longer premature termination. It is the simplicity of the environment itself.

### 4. DAG-aware is stable but policy-narrow

DAG-aware keeps the rollout horizon intact and has very low uncertainty, but the learned DDQN focuses on just two actions.

That suggests:

- the environment is usable
- but the optimal policy may be conservative or highly constrained
- the broader action space is not being fully exploited by DDQN

## What the Action Distributions Tell Us

Action counts are one of the most informative parts of step 13.

### CARE-Sim

The DDQN and planner both use many actions.

This is a healthy sign:

- the policy is not collapsed
- the planner is not trivial
- the simulator allows nuanced choices

### MarkovSim

The planner collapses to action `0`.

That is a red flag if your goal is to show meaningful policy optimization.

It means the MarkovSim environment may be too simple or too reward-dominated for a nuanced planner comparison, even after fixing early stopping.

### DAG-aware

The planner uses many actions, but DDQN concentrates on just two.

This is a mixed result:

- control works
- the policy is stable
- but DDQN may be over-conservative or under-trained

## What the Return Numbers Suggest

The returns are useful, but they need careful reading.

### CARE-Sim

The returns are close across all policies, with DDQN slightly ahead on test.

That suggests the environment is well-formed and not trivial.

### MarkovSim

The returns are also close, but the planner degenerates.

That suggests the reward signal dominates and the simulator does not offer much policy differentiation.

### DAG-aware

The returns are close, but the learned DDQN is narrow.

That suggests the simulator is stable, but the learned policy is conservative and may need more training or a stronger reward design to become more expressive.

## Thesis Interpretation

If you want the thesis-level message from step 13, it is this:

1. The simulator choice matters for control behavior.
2. CARE-Sim gives the most natural and diverse control setting.
3. MarkovSim still serves as a baseline, but even after fixing termination, it remains too coarse to support a strong control narrative.
4. DAG-aware is the most principled model, but its DDQN policy is still conservative and does not yet fully exploit the action space.

That is a useful result because it avoids a simplistic â€œbest policy winsâ€ story.

Instead, it shows:

- the environment design influences policy structure
- causal masking and uncertainty change how the control loop behaves
- the quality of the simulator matters as much as the RL algorithm

## Bottom Line

The step 13 results support three distinct conclusions:

- **CARE-Sim** is the most balanced control environment and gives the healthiest policy diversity.
- **MarkovSim** is still the weakest environment for policy learning because the planner collapses, even though the control horizon issue has now been fixed.
- **DAG-aware** is the most principled environment, with low uncertainty and stable rollout, but the learned DDQN is conservative and concentrated on a small action subset.

For the thesis, the cleanest takeaway is:

> Control quality depends strongly on the simulator. CARE-Sim supports the richest policy behavior, MarkovSim remains too coarse even after fixing early termination, and DAG-aware offers the best structured simulator but still yields a conservative policy under the current DDQN setup.

If you want, I can next write the step 14 offline RL results document in the same level of detail.

## At-a-Glance Comparison Tables

The tables below give a compact view of how the four policies behave across the three simulator tracks on the **test split**.

### 1. Mean discounted return

Higher is better.

| Policy | CARE-Sim | MarkovSim | DAG-aware |
| --- | ---: | ---: | ---: |
| `ddqn` | 8.8363 | 8.7437 | 8.8179 |
| `planner` | 8.7856 | 8.7577 | 8.7873 |
| `random` | 8.7894 | 8.6858 | 8.7970 |
| `repeat_last` | 8.8067 | 8.7613 | 8.7937 |

### 2. Mean rollout length

This shows how long policies typically stay active before the simulator stops the episode. Lower values often mean the simulator is terminating quickly.

| Policy | CARE-Sim | MarkovSim | DAG-aware |
| --- | ---: | ---: | ---: |
| `ddqn` | 5.00 | 5.00 | 5.00 |
| `planner` | 4.87 | 5.00 | 5.00 |
| `random` | 5.00 | 5.00 | 5.00 |
| `repeat_last` | 5.00 | 5.00 | 5.00 |

### 3. Action diversity

This counts how many distinct action IDs were used on the test split.

| Policy | CARE-Sim | MarkovSim | DAG-aware |
| --- | ---: | ---: | ---: |
| `ddqn` | 16 | 28 | 4 |
| `planner` | 24 | 1 | 30 |
| `random` | 32 | 32 | 32 |
| `repeat_last` | 15 | 15 | 15 |

### 4. Quick interpretation of the tables

- CARE-Sim is the most balanced setting: DDQN, planner, random, and repeat-last all achieve similar returns, but the learned and search-based policies still show structured action use.
- MarkovSim is still the most degenerate control setting, but now for the right reason: the planner collapses to a single action even when the rollout is forced to the full five-step horizon.
- DAG-aware is the most structured simulator, but the learned DDQN becomes very narrow, using only four actions on the test split.

These tables are meant as a quick visual summary. The detailed policy-by-policy discussion above still contains the fuller interpretation.

### 5. What mean discounted return means

`mean_discounted_return` is the average total policy score per rollout episode after discounting later rewards by `gamma = 0.99`.

In practical terms, it means:

- each policy rollout produces a sequence of step rewards
- each later reward is multiplied by a slightly smaller weight than earlier rewards
- the discounted rewards are summed inside each episode
- the reported value is the average of those episode totals across the split

So this metric answers:

> On average, how much total short-horizon reward does this policy collect when we value earlier gains slightly more than later ones?

Because the horizon here is only five steps, the discounting effect is not huge, but it still matters. It means `mean_discounted_return` is the cleanest single summary of overall policy quality in the current step 13 setup.

### 6. DDQN action-count matrix on the test split

This table shows how often the **DDQN** policy selected each action ID across the three simulator tracks on the **test split**.

Action-code legend for the selected 5-action setting:

- `V` = vasopressor
- `F` = IV fluid
- `A` = antibiotic
- `D` = diuretic
- `M` = mechanical ventilation

When an action ID activates multiple interventions, the code is written as a `+`-joined combination such as `F+M` or `V+F+A+M`.

| Action ID | Action code | CARE-Sim DDQN | MarkovSim DDQN | DAG-aware DDQN |
| --- | --- | ---: | ---: | ---: |
| 0  | none      | 1   | 29  | 0   |
| 1  | V         | 0   | 4  | 0   |
| 2  | F         | 206 | 181 | 307 |
| 3  | V+F       | 25  | 40 | 9   |
| 4  | A         | 8   | 25 | 0   |
| 5  | V+A       | 0   | 2  | 0   |
| 6  | F+A       | 6   | 18 | 0   |
| 7  | V+F+A     | 0   | 27 | 0   |
| 8  | D         | 15  | 10 | 0   |
| 9  | V+D       | 0   | 0  | 0   |
| 10 | F+D       | 14  | 1  | 0   |
| 11 | V+F+D     | 0   | 0  | 0   |
| 12 | A+D       | 0   | 1  | 0   |
| 13 | V+A+D     | 0   | 2  | 0   |
| 14 | F+A+D     | 4   | 0  | 0   |
| 15 | V+F+A+D   | 4   | 3  | 0   |
| 16 | M         | 0   | 4  | 0   |
| 17 | V+M       | 0   | 5  | 0   |
| 18 | F+M       | 81  | 9  | 172 |
| 19 | V+F+M     | 10  | 26 | 12  |
| 20 | A+M       | 0   | 4  | 0   |
| 21 | V+A+M     | 0   | 25 | 0   |
| 22 | F+A+M     | 29  | 5  | 0   |
| 23 | V+F+A+M   | 80  | 27 | 0   |
| 24 | D+M       | 0   | 8  | 0   |
| 25 | V+D+M     | 2   | 5  | 0   |
| 26 | F+D+M     | 6   | 2  | 0   |
| 27 | V+F+D+M   | 0   | 3  | 0   |
| 28 | A+D+M     | 0   | 13 | 0   |
| 29 | V+A+D+M   | 0   | 0  | 0   |
| 30 | F+A+D+M   | 9   | 14 | 0   |
| 31 | V+F+A+D+M | 0   | 7  | 0   |

### 7. What the DDQN action-count matrix shows

- CARE-Sim DDQN is broad but structured: it uses many actions, with the strongest concentration on `2`, `18`, and `23`.
- MarkovSim DDQN is now clearly broader than in the earlier run, but it is still less clinically structured than CARE-Sim and much less concentrated than DAG-aware.
- DAG-aware DDQN is the narrowest by far: it is dominated by actions `2` and `18`, with only minor use of `3` and `19`.

This is the clearest single view of how the learned DDQN policy differs across the three simulators.





