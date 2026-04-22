# Selected Causal Markov Simulator Math

This note summarizes `11b` in a compact, readable form.

## Inputs

For one batch:

- `s_t` has shape `(B, T, 9)`
- `a_t` has shape `(B, T, 5)`

Each time step uses the concatenated feature vector:

```text
x_t = [s_t, a_t]
```

## Training Structure

`11b` fits two parts:

1. Six masked Ridge regressions for dynamic next-state prediction
2. One logistic regression for terminal prediction

There is no transformer here.

## Dynamic Transition Models

For each dynamic target `j`:

```text
y_{t+1}^{(j)} = beta_j^T x_t + eps_j
```

The Step 9 causal mask is applied to the action part of `x_t` before fitting.
This is the key constraint in `11b`.

For each target state `j`, we split the input into:

```text
x_t = [state_t, action_t]
```

Then we build a masked version:

```text
x_t^(j) = [state_t, masked_action_t^(j)]
```

where `masked_action_t^(j)` keeps only the action columns allowed for that
specific state `j`.

If action `k` is not allowed to affect state `j`, that action feature is
zeroed for that regression:

```text
x_t[k] = 0
```

Equivalent masked form:

```text
yhat_{t+1}^{(j)} = beta_j^T x_t^(j)
```

where `M_j` is the row of the Step 9 mask for state `j`.

So the mask does not merely "regularize" the Ridge fit.
It changes which action variables are visible to that state-specific model.

## Terminal Model

The terminal outcome is modeled with logistic regression:

```text
P(done_t = 1 | x_t) = sigma(gamma^T x_t)
```

## Dynamic Targets

The six dynamic states are:

- `Hb`
- `BUN`
- `Creatinine`
- `Phosphate`
- `HR`
- `Chloride`

## Step 9 Masked Action Links

The selected causal mask allows only these action effects:

- `Hb` <- `vasopressor`, `ivfluid`, `antibiotic`, `mechvent`
- `BUN` <- `ivfluid`, `diuretic`
- `Creatinine` <- `ivfluid`, `diuretic`
- `Phosphate` <- `ivfluid`, `antibiotic`, `diuretic`
- `HR` <- `vasopressor`, `mechvent`
- `Chloride` <- `ivfluid`, `diuretic`

Static features are still present in the input, but their action effects are zeroed by the mask.

In implementation terms:

- each dynamic state gets its own Ridge model
- before fitting that model, the code copies the same scaled input matrix
- then it sets the disallowed action columns to zero for that target
- the Ridge model is fit only on the masked version of the input

So the causal graph is injected as a hard feature gate, not as a soft penalty.

## Summary

`11b` is a masked linear transition baseline:

- 6 Ridge models for next-state dynamics
- 1 logistic regression for terminal prediction
- 1 fixed Step 9 action mask that blocks disallowed action-to-state effects
