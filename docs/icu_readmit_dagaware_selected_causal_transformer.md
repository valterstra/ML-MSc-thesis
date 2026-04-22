# Selected Causal DAG-Aware Temporal Transformer

This note lays out the selected causal DAG-aware path in operational terms.

## Batch Shapes

For one batch in `11c`:

- states: `(B, T, 9)`
- actions: `(B, T, 5)`
- hidden size: typically `128`

Selected-track partition:

- dynamic states: `6`
- static confounders: `3`
- actions: `5`

So one time step contains:

- `6` dynamic-state tokens
- `5` action tokens

and the whole sequence also contains:

- `3` static tokens once per patient window

Total token count:

```text
N_tokens = 3 + T * (6 + 5) = 3 + 11T
```

Example with `T = 80`:

```text
N_tokens = 3 + 80 * 11 = 883
```

## Step-by-Step Flow

### 1. Split state into dynamic and static parts

For the selected track, the model uses:

- dynamic indices: `(0, 1, 2, 3, 4, 5)`
- static indices: `(6, 7, 8)`

So:

```text
states: (B, T, 9)
-> dynamic_values: (B, T, 6)
-> static_values:  (B, 3)
```

The static values are taken from the first time step and treated as global context tokens.

### 2. Build node-time tokens

Token layout is fixed by sequence length.

Order:

1. static tokens once:
   - `3` tokens
2. for each time step `t`:
   - `6` state tokens
   - `5` action tokens

So the flattened token value vector is:

```text
token_values =
[ static_0, static_1, static_2,
  state_0_t0, ..., state_5_t0, action_0_t0, ..., action_4_t0,
  state_0_t1, ..., state_5_t1, action_0_t1, ..., action_4_t1,
  ...
]
```

Shape:

```text
token_values: (B, 3 + 11T)
```

This is the main conceptual difference from `11a`.

In `11a`:

- one token = one full time step

In `11c`:

- one token = one scalar variable at one time

### 3. Token embedding

Each scalar token is projected separately:

```text
(B, N_tokens, 1) x (1, 128) -> (B, N_tokens, 128)
```

That is `value_proj`.

Then three learned embeddings are added:

- node identity embedding
- token type embedding
- time embedding

So the token is:

```text
e = value_proj
  + node_embed(node_id)
  + type_embed(token_type)
  + time_proj(log(1 + time))
```

Shapes:

- value projection: `(B, N_tokens, 128)`
- node embedding: `(N_tokens, 128)`
- type embedding: `(N_tokens, 128)`
- time embedding: `(B, N_tokens, 128)`

Then:

```text
e: (B, N_tokens, 128)
```

After that:

- LayerNorm
- Dropout

### 4. Build the two attention masks

This is where the causal structure lives.

The model builds two token-to-token masks:

- `history_mask`
- `action_mask`

Both are square masks of shape:

```text
(N_tokens, N_tokens)
```

and both start fully blocked, then allowed edges are opened.

#### `history_mask`

Used in every transformer layer except the last.

It allows:

- static -> static
- state/action at time `t` -> all static tokens
- state/action at time `t` -> all dynamic-state tokens from times `<= t`
- action token at time `t` -> itself only among same-time actions

It does not allow state tokens to look at action tokens here.

So this phase lets the model build a history representation from:

- static confounders
- past and current dynamic state history

but without direct current-action injection yet.

#### `action_mask`

Used only in the final transformer layer.

It includes everything from `history_mask`, plus:

- state token `j` at time `t` may also attend to current-time action tokens `k` only if the causal mask says `action_k -> state_j`

For the selected track, the fixed dynamic action mask is:

```text
Hb         <- vasopressor, ivfluid, antibiotic, mechvent
BUN        <- ivfluid, diuretic
Creatinine <- ivfluid, diuretic
Phosphate  <- ivfluid, antibiotic, diuretic
HR         <- vasopressor, mechvent
Chloride   <- ivfluid, diuretic
```

So in the final layer, each dynamic state token only gets direct access to its approved current-time actions.

### 5. Transformer stack

The model uses a stack of `nn.TransformerEncoderLayer` blocks.

Each layer preserves shape:

```text
(B, N_tokens, 128) -> (B, N_tokens, 128)
```

Default selected-track architecture:

- `n_layers = 4`
- `n_heads = 8`
- FFN width = `4 * 128 = 512`
- GELU
- pre-LN (`norm_first=True`)

The important part is mask scheduling:

- layers `1 ... L-1`: use `history_mask`
- layer `L`: use `action_mask`

So causality is not added after the transformer. It changes which information can flow inside the final representation layer.

### 6. Gather dynamic-state token hidden states

After the transformer, the model extracts only the hidden states corresponding to dynamic state tokens.

Shape before reshape:

```text
h[:, state_indices.reshape(-1), :] -> (B, T * 6, 128)
```

Then reshape to:

```text
state_hidden: (B, T, 6, 128)
```

This means:

- for each batch item
- for each time step
- for each dynamic state variable
- we now have one hidden vector

### 7. Per-state next-state heads

There is one scalar linear head per dynamic state.

For each dynamic variable `j`:

```text
(B, T, 128) x (128, 1) -> (B, T, 1)
```

After squeeze:

```text
(B, T)
```

Stack all six outputs:

```text
dynamic_next: (B, T, 6)
```

Then assemble the full next-state tensor:

```text
next_state = states.clone()
next_state[..., dynamic_idx] = dynamic_next
next_state[..., static_idx]  = states[..., static_idx]
```

So final next-state shape is:

```text
next_state: (B, T, 9)
```

Static confounders are copied through unchanged.

### 8. Terminal head

The model pools across the six dynamic-state token hidden states at each time step:

```text
terminal_context = mean over dynamic-state dimension
(B, T, 6, 128) -> (B, T, 128)
```

Then applies one linear head:

```text
(B, T, 128) x (128, 1) -> (B, T, 1)
```

After squeeze:

```text
terminal: (B, T)
```

This is a logit. Sigmoid is applied later at inference time.

### 9. Loss

Training loss uses:

- next-state MSE
- terminal BCE with logits
- no reward loss in `11c`

Important detail:

- the loss only counts dynamic state dimensions
- static dimensions are present in `next_state`, but they are masked out of the state loss

So operationally:

```text
loss = w_state * MSE(dynamic next-state only)
     + w_term  * BCE(terminal logits)
```

Padding is also masked out through `src_key_padding_mask`.

## What Are `B`, `T`, and `N_tokens`?

- `B` = batch size
- `T` = sequence length in time steps
- `N_tokens` = number of node-time tokens after flattening

Example with:

- batch size `8`
- sequence length `32`

Then:

```text
states  = (8, 32, 9)
actions = (8, 32, 5)
N_tokens = 3 + 32 * 11 = 355
embeddings = (8, 355, 128)
```

## Where The Causal Mask Acts

This is the critical difference from the old CARE-Sim transformer.

In `11a`:

- the transformer saw time-step tokens containing both state and action
- then a masked linear `action -> next_state` residual was added outside the transformer

So the causal structure acted as an output-side adapter.

In `11c`:

- the model is built from variable tokens
- the attention mask decides which tokens may send information to which others
- current actions are exposed to state-token queries only in the final layer and only along approved causal edges

So causality acts inside the representation-learning path.

Concretely:

### History path

For most layers, a state token at time `t` can use:

- static confounders
- dynamic state history up to `t`

but not current-time action tokens.

### Structural action path

Only in the final layer, a state token for dynamic variable `j` at time `t` can additionally use:

- current-time action tokens `k` such that `action_k -> state_j` is allowed by the fixed causal graph

So if there is no discovered edge:

```text
action_k -> state_j
```

then that action token is blocked from the query for that state token in the final layer.

That is the direct causal injection.

## Why There Are Two Masks Instead of One

This is an implementation choice to reduce leakage.

If allowed and forbidden action tokens all mixed freely across multiple layers, then a forbidden action could still influence a state token indirectly through other action tokens.

So the model does this instead:

- build history first without action mixing into state-token queries
- then open only the approved current-time action edges in the final layer

Also, same-time action tokens only get self-access among actions, not dense action-to-action access.

That makes the structural restriction stricter than a naive DAG-masked attention implementation.

## Padding Behavior

The input batch padding mask starts as:

```text
(B, T)
```

for time steps.

But the transformer operates on node-time tokens, so it is expanded to:

```text
(B, N_tokens)
```

by:

- keeping static tokens always unpadded
- repeating each time-step padding flag across all `11` temporal tokens of that step

So if a time step is padded, all its dynamic-state and action tokens are padded together.

## Prediction Interface

At training time, `forward()` returns:

- `next_state`
- `reward = None`
- `terminal`
- `state_loss_mask`

At inference time, `predict_step()` returns only the last step:

- `next_state[:, -1, :]`
- `sigmoid(terminal[:, -1])`

The ensemble wrapper averages across independently trained members and exposes:

- `next_state_mean`
- `next_state_std`
- `terminal_prob`

With only one ensemble member, `next_state_std` is `NaN` because standard deviation is undefined for `n = 1`.

## How It Is Used As A Simulator

At each environment step:

1. append the chosen action
2. append a placeholder copy of the last state
3. truncate to the model context window if needed
4. run ensemble prediction on the full recent history
5. take the last predicted next state
6. replace the placeholder with that predicted state
7. compute terminal probability and optional reward

So the model is still a temporal world model, but the internal representation is no longer one token per time step. It is a graph-over-time transformer.

## Bottom Line

The old `11a` transformer was:

- time-step tokens
- temporal transformer
- causal action residual added beside the transformer

The new `11c` transformer is:

- node-time tokens
- temporal DAG-aware transformer
- causal structure enforced through masked attention inside the model
- one head per dynamic state
- static confounders as context-only global tokens

That is the cleanest short description of what changed technically.
