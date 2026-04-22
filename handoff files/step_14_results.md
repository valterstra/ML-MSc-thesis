# Step 14 Results: Offline RL Comparison

This document covers the offline reinforcement learning comparison branch.

Step 14 asks a different question from step 13.

- Step 13 asks: “What happens if we train control policies inside the simulators?”
- Step 14 asks: “What happens if we train directly on logged data, and how does that compare to the simulator-trained policies?”

This is the comparison layer that sits above the simulator-control experiments.

## What Step 14 Does

Step 14 trains and evaluates an offline DDQN policy on the logged ICU trajectories, then compares it against the simulator-based DDQN policies.

The comparison set is:

- `offline_ddqn`
- `caresim_ddqn`
- `markovsim_ddqn`
- `dagaware_ddqn`

The evaluation uses the same selected ICU setup and the same reward structure:

- handcrafted severity reward
- terminal readmission reward

## Why Step 14 Matters

This is the part of the thesis where you ask:

- is the simulator useful beyond being a predictor?
- do simulator-trained policies look better or worse than a policy trained directly on historical data?
- do the different simulator families produce meaningfully different policy behavior?

That makes step 14 a very important bridge between simulation and decision-making.

## The Two Main Outputs

Step 14 produces two core result files:

- `reports/icu_readmit/offline_selected/step_14_eval_results.json`
- `reports/icu_readmit/offline_selected/step_14_action_stats.json`

The first file contains the value estimates and policy-level summaries.

The second file contains action agreement and action diversity statistics.

## Step 14 Meta Setup

The evaluation metadata says:

- `obs_dim = 70`
- `window_len = 5`
- comparison policies:
  - `caresim_ddqn`
  - `dagaware_ddqn`
  - `markovsim_ddqn`
  - `offline_ddqn`
- reward:
  - `severity_mode = handcrafted`
  - terminal model directory:
    - `models/icu_readmit/terminal_readmit_selected`
  - terminal reward scale:
    - `15.0`

So this is not a bare offline benchmark. It is a comparison between:

- a policy trained directly from logged data
- policies trained inside three different simulator environments

## Logged Policy Baseline

Before looking at the policy estimates, it helps to know the logged-data baseline.

### Validation

- mean: `7.3020`
- std: `2.6909`
- number of trajectories: `9265`

### Test

- mean: `7.2783`
- std: `2.6553`
- number of trajectories: `9267`

This baseline is useful because it tells us the score level of the historical physician behavior.

## OPE Summary

The step 14 evaluation reports the following off-policy estimates.

### Validation

- `offline_ddqn`: mean `-0.0109`, std `0.9924`, valid trajectories `9264`
- `caresim_ddqn`: mean `-2.8046`, std `8.0263`, valid trajectories `8220`
- `markovsim_ddqn`: mean `-0.7370`, std `3.7572`, valid trajectories `9150`
- `dagaware_ddqn`: mean `-1.6651`, std `6.4528`, valid trajectories `8514`

### Test

- `offline_ddqn`: mean `0.0036`, std `0.8213`, valid trajectories `9266`
- `caresim_ddqn`: mean `-2.6655`, std `7.7435`, valid trajectories `8204`
- `markovsim_ddqn`: mean `-0.6813`, std `3.5604`, valid trajectories `9158`
- `dagaware_ddqn`: mean `-1.4975`, std `6.0808`, valid trajectories `8492`

## How to Interpret the OPE Numbers

These values are not raw simulator rewards.

They are estimated policy values under the offline evaluation pipeline.

The main pattern is:

- `offline_ddqn` is closest to zero and has the tightest spread
- `markovsim_ddqn` is negative, but less extreme than CARE-Sim and DAG-aware
- `dagaware_ddqn` is more negative than MarkovSim but less extreme than CARE-Sim
- `caresim_ddqn` is the most negative and has the largest spread

That pattern is important, but it should be read with caution:

- these are OPE estimates, not ground-truth prospective policy values
- the support models and valid-trajectory counts matter
- the simulator-trained policies can be harder for OPE to evaluate because they may push trajectories into less-supported regions

## Action Agreement and Policy Diversity

The action statistics are one of the most informative parts of step 14.

They tell us:

- how often each policy agrees with the logged physician action
- how many unique actions each policy uses
- which actions dominate each policy

## Validation Action Stats

### Offline DDQN

- exact agreement with logged actions: `17.57%`
- unique actions: `27`
- top actions:
  - `0`: `156959`
  - `4`: `31625`
  - `12`: `26248`
  - `2`: `2381`
  - `9`: `1177`

Interpretation:

- the offline policy is relatively conservative
- it leans heavily on action `0`
- it still uses a decent number of distinct actions overall
- its action distribution is broad enough to be nontrivial, but not especially aggressive

### CARE-Sim DDQN

- exact agreement with logged actions: `26.67%`
- unique actions: `31`
- top actions:
  - `2`: `102232`
  - `18`: `39431`
  - `23`: `21880`
  - `3`: `16558`
  - `19`: `8596`

Interpretation:

- CARE-Sim DDQN is the most diverse of the simulator-based policies
- it also has the highest agreement with logged actions among the simulator policies
- this suggests the policy is not just inventing extreme behavior
- it is relatively close to the observed action distribution, while still being more policy-driven than the offline baseline

### MarkovSim DDQN

- exact agreement with logged actions: `5.09%`
- unique actions: `32`
- top actions:
  - `0`: `69100`
  - `4`: `61211`
  - `8`: `37099`
  - `2`: `7586`
  - `3`: `6405`

Interpretation:

- MarkovSim DDQN is the least aligned with the logged physician policy
- it uses the full action space, but the distribution is very different from the others
- this matches the earlier step 13 result that MarkovSim control is the weakest and most degenerate environment for policy learning

### DAG-aware DDQN

- exact agreement with logged actions: `16.70%`
- unique actions: `12`
- top actions:
  - `2`: `138426`
  - `18`: `67512`
  - `19`: `10636`
  - `3`: `3573`
  - `16`: `1346`

Interpretation:

- DAG-aware DDQN is much narrower than CARE-Sim DDQN
- it uses far fewer unique actions
- it is more conservative and more concentrated
- it is also less aligned with the logged policy than CARE-Sim

This narrowness is consistent with what we saw in step 13:

- the DAG-aware control policy is stable
- but it is quite conservative
- it does not fully exploit the action space

## Test Action Stats

The test split shows the same broad pattern.

### Offline DDQN

- exact agreement: `17.95%`
- unique actions: `24`
- top actions:
  - `0`: `155918`
  - `4`: `31432`
  - `12`: `25824`
  - `2`: `2229`
  - `9`: `1124`

### CARE-Sim DDQN

- exact agreement: `26.94%`
- unique actions: `31`
- top actions:
  - `2`: `102684`
  - `18`: `40454`
  - `23`: `20042`
  - `3`: `15017`
  - `19`: `8075`

### MarkovSim DDQN

- exact agreement: `5.18%`
- unique actions: `32`
- top actions:
  - `0`: `66614`
  - `4`: `61009`
  - `8`: `37763`
  - `2`: `7556`
  - `3`: `7364`

### DAG-aware DDQN

- exact agreement: `16.79%`
- unique actions: `15`
- top actions:
  - `2`: `137454`
  - `18`: `66564`
  - `19`: `11031`
  - `3`: `3572`
  - `16`: `1259`

## What the Action Stats Suggest

### CARE-Sim is the most policy-diverse simulator

It has:

- the broadest unique action usage
- the strongest agreement with the logged policy among the simulator-based DDQNs
- a clear nontrivial action distribution

This makes CARE-Sim the strongest simulator-based control benchmark.

### MarkovSim is the least aligned with real behavior

Its action distribution is broad, but not in a clinically meaningful way.

The low agreement percentage and the strong concentration on a few actions suggest the environment is not supporting a good policy learning story.

### DAG-aware is narrower but more principled

It is not as broad as CARE-Sim, but it is also not degenerate.

The fact that it concentrates on a few actions is not necessarily bad:

- it may reflect a genuinely conservative policy
- or it may reflect limited DDQN training budget

Either way, it is a more structured policy than MarkovSim.

### Offline DDQN sits in between

The offline policy:

- has moderate agreement with the logged behavior
- is broader than DAG-aware
- is less action-diverse than CARE-Sim

This makes it a reasonable baseline, but not an obvious winner.

## What the Comparison Means

The most important point in step 14 is that the simulator-trained policies do not all behave the same.

There is a real ordering:

1. CARE-Sim yields the broadest and most behaviorally plausible control policy.
2. DAG-aware yields a structured but conservative policy.
3. MarkovSim yields the least aligned and weakest policy behavior.

The offline DDQN baseline is important because it gives you the “no simulator” reference point.

## Comparing Offline vs Simulator-Based Control

### Offline DDQN

Strengths:

- closest to the logged data in spirit
- stable OPE behavior
- reasonable action diversity

Weaknesses:

- not clearly better than CARE-Sim
- much less interpretable than a simulator-based policy story

### CARE-Sim DDQN

Strengths:

- highest agreement with the logged policy among the simulator-based methods
- broad action usage
- strong control benchmark

Weaknesses:

- OPE estimate is the most negative and the most variable
- may be pushing the evaluator into lower-support regions

### MarkovSim DDQN

Strengths:

- simple baseline
- broad action set

Weaknesses:

- very low logged-policy agreement
- weak control behavior
- poor qualitative match to the other policies

### DAG-aware DDQN

Strengths:

- structured simulator foundation
- moderate agreement with logged policy
- more principled than MarkovSim

Weaknesses:

- narrow action use
- conservative policy
- OPE value is not as favorable as offline DDQN

## How to Read the OPE Results with Caution

The OPE numbers are useful, but they are not ground truth.

You should be careful about overclaiming because:

- the support models may behave differently for each policy
- simulator-based policies can go out of the evaluator’s comfort zone
- the number of valid trajectories differs across policies

That said, the pattern is still informative:

- offline DDQN is stable
- CARE-Sim is the strongest simulator-based policy
- MarkovSim is weak
- DAG-aware is structured but conservative

## Thesis Interpretation

The thesis story from step 14 is not that one policy is a dramatic winner.

Instead, it shows:

- offline RL gives a stable baseline
- simulator-based policies can be evaluated against that baseline
- the simulator choice changes action diversity and action agreement substantially
- CARE-Sim is the strongest environment for policy learning
- DAG-aware is the most principled simulator, but still conservative in the current DDQN setup
- MarkovSim is too coarse to support a strong offline-vs-simulator comparison

## Bottom Line

Step 14 gives you the comparison layer for the thesis:

- **Offline DDQN** is the stable data-driven baseline.
- **CARE-Sim DDQN** is the strongest simulator-based control policy and has the richest action structure.
- **MarkovSim DDQN** is the weakest and least aligned with the logged policy.
- **DAG-aware DDQN** is more principled than MarkovSim, but still conservative and action-narrow.

If you want the shortest thesis sentence:

> Step 14 shows that simulator-based policies are not all equivalent: CARE-Sim supports the strongest and most diverse control behavior, DAG-aware yields a principled but conservative policy, MarkovSim is the weakest control environment, and the offline DDQN remains the clean baseline for comparison.

If you want, I can next assemble these four handoff notes into a short master index file so you have one place to navigate the full step 11 to step 14 story.

## At-a-Glance Summary

### Test-Split Policy Comparison

| Policy | OPE mean | OPE std | Valid trajectories | Action agreement | Unique actions |
| --- | ---: | ---: | ---: | ---: | ---: |
| `offline_ddqn` | `0.0036` | `0.8213` | `9266` | `17.95%` | `24` |
| `caresim_ddqn` | `-2.6655` | `7.7435` | `8204` | `26.94%` | `31` |
| `markovsim_ddqn` | `-0.6813` | `3.5604` | `9158` | `5.18%` | `32` |
| `dagaware_ddqn` | `-1.4975` | `6.0808` | `8492` | `16.79%` | `15` |

### Short Reading

- `offline_ddqn` is the most stable offline baseline and stays closest to full support coverage.
- `caresim_ddqn` has the strongest logged-action agreement and the richest action usage, but the harshest OPE estimate.
- `markovsim_ddqn` has the weakest behavioral alignment with clinicians.
- `dagaware_ddqn` sits between CARE-Sim and MarkovSim: more structured than MarkovSim, but narrower and more conservative than CARE-Sim.
