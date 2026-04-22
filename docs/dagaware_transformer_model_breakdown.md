# DAG-aware Temporal Transformer World Model

This note gives a concise step-by-step explanation of what the transformer model is doing in our ICU world-model setup. It keeps the standard transformer language of tokens, embeddings, hidden representations, attention, masks, layers, output heads, and loss, but maps those concepts to our specific model.

## 1. Overall goal

The model is a **world model** for ICU trajectories. It takes as input a sequence of patient states and actions and predicts:

- the next state
- the probability of terminal transition

So the model is learning a transition rule of the form:

**history of states + actions + static context -> next state, terminal probability**

## 2. What counts as a token

The model does **not** use one token per time step.

Instead, it uses one token per **scalar variable**. That means:

- each dynamic state variable at time `t` is its own token
- each action variable at time `t` is its own token
- each static confounder is its own token

For the selected setup we have:

- 6 dynamic state variables
- 3 static variables
- 5 action variables

So for a sequence of length `T`, the token layout is:

- 3 static tokens at the front
- then for each time step:
  - 6 dynamic state tokens
  - 5 action tokens

This means the transformer operates on a **node-time sequence** rather than a plain sequence of full state vectors.

## 3. Why this tokenization matters

This tokenization gives the model variable-level control.

Instead of treating the whole patient state at one time as one object, the model can represent:

- which variable is being updated
- which variable is being attended to
- which action is allowed to influence which state

This is what makes the masked, graph-structured design possible.

## 4. State split: dynamic versus static

The state is partitioned into:

- **dynamic state variables**: predicted by the model
- **static confounders**: used as context only

In the selected setup:

- dynamic indices: `(0, 1, 2, 3, 4, 5)`
- static indices: `(6, 7, 8)`

The dynamic variables evolve over time. The static variables are treated as fixed patient context and are copied through unchanged.

## 5. Embedding step

Each token is turned into a vector representation before entering the transformer.

The embedding has four parts:

### a. Value embedding
The scalar value of the token is projected into the model dimension.

### b. Node identity embedding
The model is told which variable the token corresponds to.

### c. Type embedding
The model is told whether the token is:
- static
- dynamic state
- action

### d. Time embedding
The model is told which time step the token belongs to.

So each token starts with a representation that contains:

- its numerical value
- its variable identity
- its role in the system
- its time position

This is the initial hidden representation.

## 6. What attention is doing

Attention is the mechanism that lets one token gather information from other tokens.

In this model, a token is not asking:
- which words in the sentence matter for me?

It is asking:
- which parts of the patient trajectory matter for me?

So a dynamic state token can build its representation from:

- static patient context
- earlier state history
- selected action information

depending on which tokens it is allowed to attend to.

As in a standard transformer, attention works through:

- queries
- keys
- values

The attention scores are computed between tokens, then masked, then turned into weights with softmax, and finally used to mix the value vectors.

## 7. The key transformer benefit in this setup

The transformer lets each token selectively use the relevant context.

That is useful here because the relevant history for one variable may be different from the relevant history for another. A variable does not need to rely on one fixed summary of the whole trajectory. It can attend to the most relevant parts of the allowed history.

## 8. Masks: where structure is imposed

The model uses **masked self-attention**, not unrestricted attention.

So not every token can attend to every other token. The mask defines which information flows are allowed.

In this model there are two main masking ideas:

- temporal-history structure
- action-to-state structure

## 9. History mask

In the earlier layers, the model mainly builds hidden representations from:

- static tokens
- past and current state history

This lets the transformer form a context-aware representation of the patient trajectory before direct action effects are injected into the state tokens.

## 10. Action mask in the final layer

The final transformer layer uses a stricter structural mask.

This mask controls which action tokens are allowed to directly influence which dynamic state tokens. That constraint comes from a fixed binary action-to-state matrix.

So the model is not allowed to learn arbitrary same-step action effects. It can only use direct action-to-state links that are explicitly permitted.

Conceptually, the architecture is doing:

- **earlier layers**: build a representation of history and patient context
- **final layer**: inject direct action effects through the allowed structural edges

## 11. Why this staged design is meaningful

This staged design creates a loose decomposition between:

- baseline temporal dynamics
- direct intervention effects

That is a reasonable design for a clinical world model. The transformer first summarizes how the patient is evolving, then uses the allowed action links to shape the prediction of the next dynamic state.

## 12. Hidden representations

After each transformer layer, each token has an updated hidden representation.

For a dynamic state token, this hidden representation can be interpreted as:

**the model's current summary of everything relevant for predicting that variable, given the allowed context**

After several layers, that representation contains information about:

- static context
- state history
- selected action influence

Those final hidden representations are what the output heads read.

## 13. Output heads for next-state prediction

After the transformer stack, the model extracts the hidden representations corresponding to the dynamic state tokens.

Then it applies a separate output head for each dynamic variable. Each head predicts one scalar next-state value.

These predicted dynamic values are assembled into the next dynamic state.

For the static variables, the model does not predict them. It copies them through unchanged.

So the next-state prediction is:

- dynamic part predicted by the transformer
- static part copied forward

## 14. Terminal prediction

The model also predicts a terminal logit.

This is done by taking the hidden representations of the dynamic state tokens, averaging across the dynamic variables at each time step, and feeding that summary through a terminal head.

After applying a sigmoid, this becomes a terminal probability.

So the model produces two outputs:

- next state
- terminal probability

## 15. What the model is learning

The model is learning a structured transition rule for patient evolution.

More precisely, it is learning:

- how the dynamic state changes over time
- how actions affect that change through allowed edges
- how likely terminal transition is at each step

That is why it is natural to think of the model as a transformer-based world model.

## 16. Loss and training

Training follows the standard neural-network pattern:

1. feed in batched sequences of states and actions
2. run them through the transformer
3. get predicted next states and terminal logits
4. compare predictions to the observed next states and terminal outcomes
5. compute the loss
6. backpropagate
7. update parameters

### State loss
The next-state loss is computed only on the **dynamic state dimensions**.

### Terminal loss
A binary classification loss is used for the terminal prediction.

### Total loss
The total objective is a weighted combination of:

- state prediction loss
- terminal loss

There is no reward head in this selected-track version.

## 17. Padding and variable-length sequences

The model supports variable-length trajectories in batched training.

A padding mask marks which time steps are real and which are padding. Because the transformer works on node-time tokens, the time-level padding mask must be expanded to token level.

This ensures the model does not attend to fake padded positions.

## 18. Ensemble usage

In practice, the full system uses an **ensemble** of these transformer world models.

At inference time:

- the mean prediction across ensemble members is used as the next-state prediction
- the spread across ensemble members is used as an uncertainty estimate

This is useful because the simulator then gets both:

- a predicted transition
- a measure of model uncertainty

## 19. Role inside the simulator

Inside the simulator, the transformer ensemble acts as the learned transition function.

At each simulation step:

- a new action is appended
- the recent context is fed into the ensemble
- the ensemble predicts the next state
- terminal probability and uncertainty are also computed

So the transformer is the engine that drives the simulated patient dynamics.

## 20. Short summary

The model represents an ICU trajectory as a sequence of variable-time tokens rather than whole time-step vectors. Each token is embedded using value, identity, type, and time information. A masked transformer then propagates information across the allowed parts of the trajectory: earlier layers build history-aware representations, and the final layer injects direct action effects through a fixed action-to-state mask. The resulting hidden representations are used to predict the next dynamic state and terminal probability. In the full system, an ensemble of these models provides both next-state predictions and uncertainty estimates for simulation.
