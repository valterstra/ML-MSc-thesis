# Offline DDQN in CARE-AI

This note explains the Step 14 offline DDQN branch in the same style as the Step 13 DDQN explainer, but with the important difference that the policy is trained on **logged ICU data**, not on simulator rollouts.

Relevant code:
- [step_14_offline_selected.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/scripts/icu_readmit/step_14_offline_selected.py)
- [continuous.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/rl/continuous.py)
- [networks.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/rl/networks.py)
- [observation.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/caresim/control/observation.py)
- [evaluation.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/rl/evaluation.py)
- [readmit.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/caresim/readmit.py)
- [severity.py](/C:/Users/ValterAdmin/Documents/VS%20code%20projects/TemporaryMLthesis/CareAI/src/careai/icu_readmit/caresim/severity.py)

## 1. What Step 14 is

Step 14 trains an **offline DDQN** policy directly from the selected replay dataset.

That means:

- no CARE-Sim rollouts during training
- no simulator interaction loop
- no online exploration in a learned environment

Instead, it learns only from the logged transitions already present in the ICU data.

The role of Step 14 is comparison:

- Step 13 = policy learned in the learned simulator
- Step 14 = policy learned directly from real logged data

Both use the same selected state/action space, so they can be compared fairly.

## 2. What the network takes in and gives out

The policy network is still a Dueling DQN.

It takes in the same Step-16-style observation:

- a fixed window of the most recent `5` `(state, action)` pairs
- `state_dim = 9`
- `action_dim = 5`

So the input size is:

```text
5 * (9 + 5) = 70
```

The output is still:

- `32` Q-values
- one value per action combination

So the network still answers:

```text
Q(observation, action_id)
```

for each action ID in `{0, ..., 31}`.

## 3. What the offline training data looks like

Step 14 first turns the logged ICU dataset into transition tuples:

```text
(o, a, r, o_next, done)
```

where:

- `o` = the current 70-dim observation window
- `a` = the action that was taken in the logged data
- `r` = recomputed reward for Step 14
- `o_next` = the next 70-dim observation window
- `done` = whether the stay ended at that row

The important difference from Step 13 is:

- the tuples come from real logged trajectories
- not from simulator-generated experience

## 4. How the Step 14 reward is defined

Step 14 does **not** use the old parquet reward directly.

It recomputes reward so it matches the current selected-control design:

- non-terminal rows:

```text
r = handcrafted_severity(s_t) - handcrafted_severity(s_t+1)
```

- terminal rows:

```text
r = terminal_readmission_reward(final_state)
```

So the offline policy is trained on the same reward logic as the control branch.

## 5. How one offline DDQN update works

The update is still standard DDQN.

For one sample `(o, a, r, o_next, done)`:

1. Compute the current value:

```text
q_current = Q_online(o, a)
```

2. Build the target:

- if `done = 1`:

```text
target = r
```

- otherwise:

```text
best_next_action = argmax_a Q_online(o_next, a)
target = r + gamma * Q_target(o_next, best_next_action)
```

3. Update `Q_online` so that `q_current` moves closer to `target`.

So the learning rule is the same as Step 13.
The only thing that changes is the source of the transitions.

## 6. What `Q_online` and `Q_target` mean here

They mean exactly the same thing as in Step 13:

- `Q_online` is the network that gets trained
- `Q_target` is the delayed copy used to stabilize targets

`Q_target` is not trained directly.
It is updated by copying weights from `Q_online` every so often.

So the offline branch does **not** need a different Q-network concept.
It needs a different data source.

## 7. What is different from Step 13

Step 13:

- chooses actions in CARE-Sim
- gets the next state from the simulator
- learns a policy from simulated experience

Step 14:

- does not use CARE-Sim for training
- learns from the logged trajectories only
- uses the same observation shape and the same action space

So Step 14 is not a simulator policy.
It is a logged-data policy.

## 8. Why the 5-step history matters

This is the key alignment point.

The offline DDQN in Step 14 uses the same 5-step context as Step 13.

That means both policies see:

- the recent state trajectory
- the recent action trajectory

So the comparison is not:

- ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œsimple state model vs history modelÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â

It is:

- ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Â¦ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œsame history representation, different training sourceÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â

That is the right comparison for the thesis.

## 9. What happens at inference time

Once Step 14 is trained, inference is simple:

1. Build the 70-dim observation window from the current patient history.
2. Feed it into `Q_online`.
3. Get 32 Q-values.
4. Choose `argmax`.
5. Decode the action ID into the 5 binary ICU actions.

So at inference time, the offline policy looks just like the Step 13 policy.
The difference is only how it was learned.

## 10. How Step 14 is evaluated

Step 14 uses offline policy evaluation on held-out logged data.

That means:

- compare the learned offline DDQN against the logged physician actions
- optionally compare it against the Step 13 world-model DDQN
- use doubly robust evaluation support models

So the output is not just the policy network.
It is also an estimate of how that policy would score on the held-out real data.

## 11. The short version

Step 14 is the same DDQN architecture as Step 13, but trained offline on logged ICU trajectories instead of simulator rollouts. It still uses a 5-step history window, still outputs 32 action values, and still trains `Q_online` with a delayed `Q_target`. The real difference is the source of the transitions: Step 13 learns from CARE-Sim, Step 14 learns from the real data.
