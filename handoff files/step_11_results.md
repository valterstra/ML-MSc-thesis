# Step 11 Results: Simulator Training and What It Means

This note summarizes the three simulator tracks that sit behind step 11 and explains how to read their results.

Strictly speaking, step 11 is a training step, so the direct output is trained model artifacts rather than a final benchmark table. The meaningful comparison comes from the matched step 12 evaluations, which test the trained simulators on held-out data. I am including both here because step 11 and step 12 are part of the same story.

## What Step 11 Is Doing

Step 11 trains three different simulator families on the selected ICU readmission dataset:

- CARE-Sim transformer
- MarkovSim baseline
- DAG-aware transformer

All three are trying to learn the same high-level task:

- take a short history of patient state and actions
- predict the next state transition
- predict terminal / readmission behavior
- preserve static confounders

The difference is in how each model represents time and action dependencies.

## The Three Models

### 1. CARE-Sim Transformer (`11a`)

This is the large transformer baseline.

Training setup:

- `n_models = 5`
- `n_epochs = 30`
- `d_model = 256`
- `n_heads = 8`
- `n_layers = 4`
- `max_seq_len = 80`
- `batch_size = 64`
- `device = cuda`

What it is doing:

- learns a rich temporal representation over the selected ICU trajectory
- uses causal constraints and frozen static context
- is the most expressive of the three step 11 models

What step 11 gives you:

- a heavy-duty transformer ensemble
- strong training signal, but also the most expensive run

### 2. MarkovSim (`11b`)

This is the lightweight baseline.

Training setup:

- not a transformer
- ridge-based / tabular transition model
- training is very fast
- coarse representation of transitions

What it is doing:

- predicts transitions from the current summarized state/action context
- gives a simple benchmark for “how far can we get without a sequence model?”

What step 11 gives you:

- a very fast baseline
- useful for comparison, but much less expressive than either transformer

### 3. DAG-aware Transformer (`11c`)

This is the causally constrained transformer.

Training setup:

- `n_models = 3`
- `n_epochs = 10`
- `d_model = 64`
- `n_heads = 4`
- `n_layers = 2`
- `max_seq_len = 32`
- `batch_size = 8`
- `device = cuda`

What it is doing:

- learns temporal dynamics like the CARE-Sim transformer
- but with a DAG-aware / causally masked structure
- keeps static confounders fixed
- separates dynamic and static state indices explicitly

What step 11 gives you:

- a smaller, more structured transformer
- lower parameter count and lower training cost than CARE-Sim
- useful uncertainty once trained as an ensemble

## How the Results Are Generated

The models are trained in step 11, then evaluated in step 12.

The key metrics from step 12 are:

- one-step next-state MSE
- terminal accuracy
- mean uncertainty
- rollout state MSE
- static-confounder drift

These are the numbers that tell us whether the trained step 11 model is actually usable.

## Side-by-Side Results

The table below uses the matched step 12 reports for each trained simulator.

| Model | Training footprint | One-step test MSE | Terminal accuracy | Mean uncertainty | Rollout last-state MSE | Training time |
|---|---:|---:|---:|---:|---:|---:|
| CARE-Sim transformer | 5 models, 30 epochs, 256-dim | 0.0681 | 0.9536 | 0.0207 | 0.2052 | 5711 s |
| MarkovSim baseline | ridge / tabular | 0.0718 | 0.5429 | 0.2088 | 0.2064 | 23.5 s |
| DAG-aware transformer | 3 models, 10 epochs, 64-dim | 0.0723 | 0.9534 | 0.0145 | 0.2141 | 2328.6 s |

## How to Read the Numbers

### CARE-Sim transformer

CARE-Sim is the strongest raw predictor on one-step next-state error.

What stands out:

- best one-step test MSE among the three
- terminal accuracy is very high
- rollout does not collapse quickly
- uncertainty is finite and modest

Interpretation:

- this is the highest-capacity simulator
- it gives the best pure predictive benchmark
- it is also the most expensive to train and run

This is the model you would cite if the thesis emphasis is “best raw fidelity.”

### MarkovSim

MarkovSim is much cheaper, but it is not competitive on terminal behavior.

What stands out:

- one-step next-state MSE is only slightly worse than the transformers
- terminal accuracy is far lower
- uncertainty is much higher
- rollout error rises steadily over time

Interpretation:

- the model can imitate some transition structure
- but it does not capture the richer temporal / terminal dynamics well
- it is a useful baseline, not a strong final simulator

The important lesson is that one-step MSE alone would overstate how good this model is.

### DAG-aware transformer

DAG-aware is the most interesting thesis model because it trades a small amount of raw fidelity for causal structure and uncertainty discipline.

What stands out:

- one-step next-state MSE is essentially tied with CARE-Sim
- terminal accuracy is also essentially tied with CARE-Sim
- uncertainty is the smallest of the three
- static confounders are preserved exactly
- rollout error is slightly worse than CARE-Sim but still stable

Interpretation:

- this model is compact, structured, and stable
- it behaves like a serious simulator rather than a loose sequence predictor
- the finite uncertainty from the 3-member ensemble makes later control work more meaningful

This is probably the best “scientific story” model in the thesis because it combines:

- decent fidelity
- causal structure
- usable uncertainty

## What the Comparison Suggests

The side-by-side story is:

1. CARE-Sim is the strongest raw predictor.
2. DAG-aware is nearly as accurate, while being more structured and easier to reason about.
3. MarkovSim is fast and simple, but weaker on the clinically important terminal behavior.

So the main tradeoff is not “does the DAG-aware model win on raw MSE?”

It is more like:

- CARE-Sim gives the best benchmark-level fidelity
- DAG-aware gives a better balance of fidelity, structure, and uncertainty
- MarkovSim gives a cheap baseline that is useful mainly as a foil

## Important Caveats

### 1. Step 11 is training, not the final result

The actual performance evidence comes from step 12.

### 2. MarkovSim is not directly comparable in capacity

It is not a transformer, so it should be treated as a simpler baseline rather than a peer architecture.

### 3. One-step MSE is not the whole story

Terminal accuracy and rollout behavior matter just as much.

### 4. Uncertainty only became meaningful for DAG-aware after using an ensemble

The 3-member DAG-aware model is important because it turns uncertainty from a placeholder into a usable signal.

## Bottom Line

If you want the shortest thesis-level interpretation:

- **CARE-Sim** is the best raw predictor.
- **MarkovSim** is the simplest baseline and the weakest on terminal behavior.
- **DAG-aware** is the best balanced model: nearly as accurate as CARE-Sim, structurally cleaner, and more useful for later control because it has meaningful uncertainty.

If you want, this can be turned into the start of a Results chapter with figures, tables, and a more formal thesis voice.
