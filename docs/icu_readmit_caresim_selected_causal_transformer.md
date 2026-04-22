# Selected Causal CARE-Sim Transformer

This note lays out the selected causal CARE-Sim path in operational terms.

## Batch Shapes

For one batch in `11a`:

- states: `(B, T, 9)`
- actions: `(B, T, 5)`
- hidden size: `128`

## Step-by-Step Flow

### 1. State projection

```text
(B, T, 9) x (9, 128) -> (B, T, 128)
```

### 2. Action projection

```text
(B, T, 5) x (5, 128) -> (B, T, 128)
```

### 3. Add embeddings

- positional embedding: `(T, 128)`
- time embedding from bloc index: `(B, T, 1) x (1, 128) -> (B, T, 128)`

Token:

```text
e = state_proj + action_proj + pos + time
e: (B, T, 128)
```

### 4. Transformer, repeated 4 times

Each layer preserves shape:

```text
(B, T, 128) -> (B, T, 128)
```

Inside each layer:

- Q, K, V projections:

```text
(B, T, 128) x (128, 128) -> (B, T, 128)
```

- attention output:

```text
(B, T, 128) -> (B, T, 128)
```

- MLP:

```text
(B, T, 128) x (128, 512) -> (B, T, 512)
(B, T, 512) x (512, 128) -> (B, T, 128)
```

### 5. Base next-state head

Because static context is frozen, only 6 dynamic dims are predicted:

```text
(B, T, 128) x (128, 6) -> (B, T, 6)
```

These are inserted into a full next-state tensor:

```text
base_next: (B, T, 9)
```

The 3 static dimensions are copied from the current state.

### 6. Causal action residual

Trainable matrix:

```text
W_causal: (9, 5)
M:        (9, 5) fixed
```

Masked weight:

```text
W_masked = W_causal * M   -> (9, 5)
```

Apply to actions:

```text
(B, T, 5) x (5, 9) -> (B, T, 9)
```

Residual:

```text
resid: (B, T, 9)
```

### 7. Final next-state prediction

```text
next = base_next + resid
(B, T, 9) + (B, T, 9) -> (B, T, 9)
```

### 8. Terminal head

```text
(B, T, 128) x (128, 1) -> (B, T, 1)
```

Then squeeze to:

```text
(B, T)
```

### 9. Loss

- next-state MSE on `(B, T, 9)`, but only dynamic dims count
- terminal BCE on `(B, T)`

After every optimizer step:

```text
W_causal = W_causal * M
```

So forbidden action-state entries are reset to zero every update.

## What Are `B` and `T`?

- `B` = batch size
- `T` = sequence length

Example:

- batch size `32`
- max sequence length `80`

Then:

```text
(B, T, 9) = (32, 80, 9)
```

Meaning:

- 32 patient sequences at once
- 80 time steps per sequence
- 9 state features at each time step

## Where The Causal Mask Acts

`W_causal` is not applied to the transformer hidden state `(B, T, 128)`.

It is applied directly to the raw action vector `(B, T, 5)`.

So the two paths are separate:

1. Transformer path

```text
(B, T, 9) and (B, T, 5)
-> embeddings
-> transformer
-> h of shape (B, T, 128)
-> base next-state head
-> base_next of shape (B, T, 9)
```

2. Causal residual path

```text
actions (B, T, 5)
-> masked linear layer with W_causal of shape (9, 5)
-> resid of shape (B, T, 9)
```

Then they are added:

```text
next = base_next + resid
```

So:

- `W_causal` is trainable
- `M` is fixed
- the actual used weight is `W_causal * M`
- it multiplies the action vector, not the transformer latent

Concretely:

```text
resid_t = (W_causal * M) a_t
```

with:

- `a_t` in `R^5`
- `(W_causal * M)` in `R^(9 x 5)`
- `resid_t` in `R^9`

So the causal layer is a direct action-to-next-state adapter sitting beside the transformer, not inside the transformer hidden stack.

