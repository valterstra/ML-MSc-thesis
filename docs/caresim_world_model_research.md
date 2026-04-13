# CARE-Sim World Model Research
# Transformer-Based Patient Simulator for ICU RL

Research conducted 2026-04-01. Full literature review for building a transformer-based
world model (CARE-Sim) to serve as the environment for the ICU readmission RL agent.

---

## Context

**What we are building:**
- A simulator (world model) that takes a patient's history of states and actions as input
- Predicts the next patient state (and terminal readmission outcome)
- Used as an interactive environment: RL policy proposes action → simulator predicts next state → repeat
- Goal: "virtual clinical trial" -- evaluate RL policy on thousands of simulated patient trajectories

**Data available:**
- ICUdataset.csv (output of step 08): 1.5M transitions, 61,771 stays, 4-hour blocs
- State: ~50 clinical features (vitals, labs, SOFA, Shock Index, etc.)
- Actions: 4 binary drug decisions (diuretic, ivfluid, vasopressor, antibiotic) = 16 combinations
- Outcome: 30-day readmission (binary terminal reward)
- Known causal graph: FCI stability analysis (step 09) confirmed which drugs causally affect which labs

---

## 1. Transformer World Models for RL (General)

### 1.1 Trajectory Transformer (Janner et al., 2021 -- arXiv:2106.02039)

**What it is.** Reframes offline RL entirely as sequence modeling. Trains a single GPT-style
transformer on sequences of the form (s_0, a_0, r_0, s_1, a_1, r_1, ...) interleaved into
a single token stream. Trained with cross-entropy loss to predict the next token.

**Continuous value tokenization -- the key architectural decision.**
Each scalar value is quantized into N discrete bins (typically 100 bins per dimension,
determined by percentile boundaries from training data). For a state with D dimensions,
the transformer sees D tokens per time step, one per feature, followed by A tokens for
the action dimensions, followed by 1 token for the reward. A trajectory of length T
produces T*(D+A+1) tokens total.

**Planning with beam search.** At test time, given the current state, the transformer
generates possible continuations using beam search. Candidate trajectories are scored by
their predicted cumulative reward. Acts as a model-based planner, not just a policy.

**Is it a simulator or a policy?** Both simultaneously. It is a joint model of
(state, action, reward) that can generate trajectories. Can be used as:
(a) a policy -- condition on desired return, sample actions
(b) a dynamics model -- condition on current state+action, sample next state
(c) a planner -- beam search over future trajectories

**Architecture.** Built on minGPT (GPT-2 style). Typical: 4 layers, 4 attention heads,
128-dimensional embeddings. Causal (left-to-right) self-attention.

**Applicability.** Directly applicable. State is a vector of ~50 numerical features --
each gets binned. Action is 4 binary values -- already discrete. The model can generate
next-state predictions given (state_history, action).

---

### 1.2 IRIS (Micheli et al., 2022 -- arXiv:2209.00588)

**What it is.** IRIS = Imagination with autoRegressive transformers and tokenized observations
using a Symbolic encoder. Combines a discrete autoencoder (VQ-VAE style) with an
autoregressive GPT-style transformer as the world model.

**Architecture.** Two components:
1. Discrete tokenizer: Convolutional encoder compresses each observation frame into K=16
   discrete tokens (vocabulary size V=512). Trained as a VQ-VAE.
2. Autoregressive transformer: Takes sequence of all past tokens (observations, actions,
   rewards) and predicts next tokens autoregressively. GPT-style with causal masking.

**RL training loop.**
1. Collect small amount of real environment data.
2. Train tokenizer and transformer world model on real data.
3. Sample imagined trajectories from world model (roll out transformer for H steps).
4. Train RL actor-critic on imagined trajectories.
5. Repeat.

**Results.** State-of-the-art on Atari 100k benchmark (score 1.046, surpasses human on
10 of 26 games).

**Applicability.** Built around discrete tokenization of image observations. For tabular
clinical data, VQ-VAE tokenizer not needed. Embed each feature with linear layer and use
transformer directly. The RL imagination loop is directly applicable.

---

### 1.3 DreamerV1/V2/V3 (Hafner et al., 2019/2021/2023)

**Architecture family.** Uses a Recurrent State Space Model (RSSM), which is GRU-based,
NOT transformer-based.

RSSM:
- Deterministic path: h_t = GRU(h_{t-1}, z_{t-1}, a_{t-1})
- Stochastic path: z_t ~ p(z_t | h_t, o_t) during training; z_t ~ p(z_t | h_t) during imagination

**DreamerV2** adds discrete latent representations (categorical distributions).
32 categorical variables x 32 categories = 1024-bit latent codes.

**DreamerV3** (most general version, handles 150+ diverse tasks). Key improvements:
symlog transformation of inputs, free bits for KL, return normalization.
Can handle proprioceptive (vector) inputs, not just images.

**Imagination-based RL.** RL agent trains entirely in imagined rollouts. Starting from
a real state, unroll H steps forward using h_t, z_t without querying the real environment.

**Is it transformer-based?** No. All Dreamer variants use GRU/RSSM.

---

### 1.4 TransDreamer (Chen et al., 2022 -- arXiv:2202.09462)

**What it is.** Direct replacement of the GRU in Dreamer with a Transformer.
Instead of h_t = GRU(h_{t-1}, ...), uses cross-attention over history:
h_t = Transformer(z_{1:t}, a_{1:t-1}).

**Key difference.** Recurrent state h_t replaced by multi-head attention over all past
latent states. Better long-range memory than GRU (which compresses everything into a
fixed vector). Tested on VisualCoin and DMLab tasks, showing benefit when long-term
memory matters.

**Architecture.** Stochastic latent model unchanged (VQ-VAE or categorical distributions).
Deterministic path uses causal transformer with K attention heads and L layers,
attending back over episode history.

---

### 1.5 GATO (Reed et al., DeepMind 2022 -- arXiv:2205.14135)

**What it is.** Generalist agent trained on 600+ tasks (robotics, Atari, image captioning,
dialogue). Single 1.2B parameter transformer (GPT-3 style), multi-modal tokenization.

**Tokenization.** Images: patches. Text: BPE. Continuous actions/observations: discretized
into 1024 bins. Everything becomes a flat token sequence.

**Relevance.** Existence proof that continuous clinical features can be tokenized (discretized
into 1024 uniform bins) and processed by a standard transformer. However, GATO outputs
actions (policy), not next states (simulator). Not directly usable as a world model.

---

### 1.6 Other Notable Papers (2021-2025)

**Decision Transformer (Chen et al., NeurIPS 2021 -- arXiv:2106.01345).**
Conditions a GPT on (return-to-go, state, action) sequences to produce actions achieving
a desired return. Pure offline RL policy -- NOT a dynamics model/simulator.
Not a world model.

**OTTO (Zhao et al., 2024 -- arXiv:2404.10393).**
Ensemble of transformers (World Transformers) to predict state dynamics and rewards for
offline RL. Generates long-horizon synthetic trajectories by perturbing actions.
Designed for robotics/continuous control, but ensemble-of-transformers approach for
tabular transition modeling is directly applicable.

**Delta-IRIS (Micheli et al., 2024 -- arXiv:2406.19320).**
Improvement of IRIS: encodes frame *deltas* instead of full frames. Reduces token sequence
length dramatically. Order of magnitude faster to train than IRIS.
The delta-encoding idea is excellent for clinical data: patient states change slowly,
so encoding the change from previous state is efficient and clinically meaningful.

---

## 2. Clinical / EHR Transformer Models

### 2.1 medDreamer (Xu et al., 2025 -- arXiv:2505.19785)

**THE MOST DIRECTLY RELEVANT PAPER.**

**What it is.** Model-based RL framework for clinical treatment recommendation, tested on
sepsis and mechanical ventilation. Builds a world model that simulates patient states from
EHR data, then trains an RL policy on a mixture of real and imagined trajectories.

**Architecture.** Two components:
1. World Model with Adaptive Feature Integration (AFI) module: handles irregular temporal
   sampling common in EHRs. Models the missing-data pattern as informative (not just noise).
2. Two-phase policy: Phase 1 trains on real clinical trajectories. Phase 2 trains on a
   mixture of real and imagined (world-model-generated) trajectories.

**Key insight over prior work.** Rather than using coarse discretization and simple
imputation (like Komorowski/Raghu), medDreamer treats missing-data patterns as signals.

**Results.** Tested on two large-scale EHR datasets (including sepsis). Outperforms both
model-free and model-based baselines on clinical outcomes and off-policy evaluation metrics.

**Applicability.** Your data is already on regular 4-hour blocs with KNN imputation applied.
The AFI module is not needed. The overall architecture -- world model trained on EHR
transitions, RL policy trained on imagined rollouts -- is a direct blueprint.

---

### 2.2 AI Clinician (Komorowski et al., Nature Medicine 2018 -- arXiv:1903.02345)

No learned world model. Pure offline FQI/DQN on MIMIC-III. Evaluation via WIS (weighted
importance sampling). No generative patient simulator built.

**Key limitation acknowledged:** Without a simulator, cannot test novel treatment sequences
not seen in the data. This is the gap your transformer world model fills.

---

### 2.3 Raghu et al. (Deep RL for Sepsis, 2017/2018 -- arXiv:1705.08422, 1711.09602)

Also no explicit generative world model. Pure offline RL (DQN) on MIMIC-III.
Policy evaluation via mortality-by-dose-difference analysis.

**Related: Oberst & Sontag (2019, NeurIPS).** Built a generative model of patient
trajectories for sepsis using a GRU-based VAE (Gumbel-MAX SCM). Used for counterfactual
off-policy evaluation, not RL training. Closest prior work to a generative patient simulator
in clinical RL literature before transformers.

---

### 2.4 Relevant Clinical Transformer Architectures

**SMTAFormer (arXiv:2407.11096).** Specifically for ICU readmission prediction (same outcome).
Dual-stream transformer: static features through MLP, temporal features through temporal
transformer, fused via cross-attention. AUC=0.717 on MIMIC-III.
Not a world model, but demonstrates the right architecture for combining static patient
demographics with temporal ICU vitals/labs.

**OmniTFT (arXiv:2511.19485).** Built on Temporal Fusion Transformer for ICU vital signs
AND lab forecasting simultaneously. Handles multi-output nature (predicting ~50 next-state
features). Hierarchical variable selection relevant for your setting.

**iTransformer (arXiv:2310.06625).** Inverts standard attention direction: instead of
attending across time (temporal tokens), attends across variables (variate tokens). Each
clinical variable becomes a token; attention learns cross-variable correlations. For a
world model, this means "how does HR change given current BUN, Creatinine, drug actions?"
-- exactly the causal cross-feature question your FCI analysis answered.

---

### 2.5 VAE/GAN Alternatives

**Oberst & Sontag Gumbel-MAX SCM (NeurIPS 2019).** GRU-based VAE for counterfactual
trajectory generation in sepsis. Most rigorous from a causal standpoint.

**Ensemble of MLPs (PETS/MOPO standard).** 7 neural networks (MLP, 4 hidden layers,
200 units, ELU activations) predict (next_state, reward) given (state, action). Uncertainty
= disagreement between ensemble members. Competitive with transformer approaches on tabular
data, much faster to train. This is the recommended Stage 1 baseline.

---

## 3. Architectures for Tabular Time Series

### The Core Problem

Most transformer world model papers work on images (Atari) or continuous control (MuJoCo).
Our data is different: 50-dimensional numerical vectors, no spatial structure, sequences
of ~20-40 time steps per ICU stay.

### Options for Tabular Sequential Data

| Option | Description | When to use |
|--------|-------------|-------------|
| A: Direct continuous embedding | Linear projection of each feature vector → d_model. Standard causal transformer. MSE loss. | Simplest, recommended starting point |
| B: Trajectory Transformer tokenization | Quantize each feature into 50-100 bins. Each time step = D tokens (one per feature). Cross-entropy loss. | When you want distributional outputs |
| C: iTransformer (variate tokens) | Each variable is a token. Attention over variables, not time. Captures drug-lab interactions. | When cross-variable correlations matter most |
| D: Temporal Fusion Transformer (TFT) | Handles static covariates + temporal inputs + known future inputs. Multi-output. Interpretable variable selection. | Most principled for your data structure |

**Recommendation: Option A** (direct continuous embedding) as starting point.
Option D (TFT) is the most principled if you want to go further.

### Tokenization Strategies by Paper

| Paper | Input type | Strategy |
|-------|-----------|----------|
| Trajectory Transformer | Continuous state+action+reward | Bin each scalar, N=100 bins, cross-entropy |
| IRIS | Image frames | VQ-VAE into K=16 discrete tokens, vocab=512 |
| DreamerV3 | Image or vector | Encoder → symlog → categorical latents |
| GATO | Multi-modal | Images: patches; continuous: 1024 uniform bins |
| SMTAFormer | Static + temporal clinical | MLP(static) + temporal transformer, cross-attention |
| iTransformer | Multivariate time series | Each variable as a token |

---

## 4. The Simulator-RL Loop

### How the RL Agent Interacts with the World Model

**Offline model-based RL training loop (recommended for thesis):**
```
1. Train world model on real data D_real:
   WM: (s_t, a_t, h_t) → (s_{t+1}, r_t, done_t)
   where h_t = transformer hidden state = compressed history

2. Generate imagined trajectories:
   - Sample starting states from D_real
   - For K rollout steps: a_t ~ pi(a|s_t, h_t), s_{t+1} ~ WM(s_t, a_t, h_t)
   - Collect D_model = {(s_t, a_t, r_t, s_{t+1})} from imagined rollouts

3. Train RL policy on mixture:
   D_train = D_real + D_model
   Update pi using FQI, DQN, SAC, etc. on D_train

4. Repeat (fine-tune world model on new real data periodically)
```

**At deployment (decision time):**
```
Given patient's history up to current time t:
1. Feed history (s_1,...,s_t, a_1,...,a_{t-1}) to transformer → h_t
2. For each of 16 possible drug combinations:
   - Simulate K steps forward: WM(s_t, a, h_t) → s_{t+1}, r_t
   - Score trajectory by Q-value or predicted readmission risk
3. Select action a* = argmax Q(s_t, a | h_t)
```

**Four modes of use:**
- (a) Offline trajectory generation: augment training data (MOPO, COMBO, medDreamer) -- recommended
- (b) Online rollouts during training: interleave with real data (Dreamer, TransDreamer)
- (c) Planning at decision time: beam search over future trajectories (Trajectory Transformer)
- (d) All of the above

**For thesis: mode (a) -- offline trajectory generation -- is most tractable.**

---

## 5. Recommended Architecture in Concrete Terms

### Causal Transformer World Model for ICU Patient Simulation

```
Input at time t:
  - History: (s_1, a_1), (s_2, a_2), ..., (s_t, a_t)
  - s_i: 50-dimensional float vector (vitals, labs, SOFA, etc.)
  - a_i: 4-dimensional binary vector (diuretic, ivfluid, vasopressor, antibiotic)

Step 1: Embedding
  - Linear projection: s_i ∈ R^50 → R^d_model  (d_model = 128 or 256)
  - Linear projection: a_i ∈ R^4 → R^d_model
  - Combine: x_i = LayerNorm(W_s * s_i + W_a * a_i + PE_i)
  - PE_i: sinusoidal or learned positional encoding for time step i

Step 2: Causal Transformer (GPT-style)
  - N=4 to 6 transformer layers
  - Each layer: MultiHeadAttention(8 heads) + FFN(4*d_model hidden) + residual + LayerNorm
  - Causal mask: time step t can only attend to t' <= t (no future leakage)
  - Output: h_t ∈ R^d_model (contextual embedding given full history)

Step 3: Output heads
  - Next state:   W_out * h_t → s_{t+1} ∈ R^50  (MSE loss, continuous features)
  - Reward:       W_r   * h_t → r_t ∈ R          (MSE loss, SOFA delta)
  - Terminal:     W_done * h_t → readmit_30d      (BCE loss, only at last time step)

Loss:
  L = L_state + lambda_r * L_reward + lambda_readmit * L_readmit

Practical hyperparameters:
  - d_model = 128 (small) to 256 (medium)
  - n_layers = 4
  - n_heads = 8
  - max_context_length = 40 time steps (~160 hours at 4h blocs)
  - Parameters: ~1M to 4M (trainable on CPU/single GPU)
  - Training time: 3-6 hours on T4 GPU, 24-48 hours CPU
```

---

## 6. Implementation Roadmap

### Stage 1: Ensemble MLP World Model (MOPO-style) -- 1-2 weeks
- 7 MLPs (3 hidden layers, 256 units each)
- Predict (next_state, readmit_terminal) given (state, action)
- Uncertainty = std across 7 ensemble members
- Use as drop-in simulator for existing FQI
- Training time: 2-4 hours on CPU

### Stage 2: Causal Transformer World Model -- 2-4 weeks
- GPT-style, d_model=128, 4 layers, causal masking
- PyTorch nn.TransformerEncoder with causal mask
- Input: sequences of (state_t, action_t) pairs per stay
- Target: state_{t+1} (continuous MSE) + readmit_30d at final step (BCE)
- Causal constraint: drug inputs only flow to FCI-confirmed lab outputs
- Batch by stay, pad shorter stays, mask padding in attention

### Stage 3: RL Integration -- 1 week
- Use trained transformer to generate augmented transitions offline
- Re-run step 11b FQI on augmented data
- Compare OPE metrics vs LightGBM-simulator-based baseline

---

## 7. Key Caveats and Failure Modes

1. **Compounding errors in multi-step rollouts.** Errors compound when rolling out K>1 steps
   with novel action sequences. After 3 steps, model may predict physiologically impossible
   states.
   Mitigation: Limit rollout to 1-3 steps. Apply physiological bounds as hard constraints.
   Use ensemble disagreement to reject high-uncertainty rollouts.

2. **Rare drug combinations unreliable.** 16 action combinations have unequal frequencies.
   Model will have seen thousands of (ivfluid=1, others=0) but almost none of
   (diuretic=1, vasopressor=1, antibiotic=1, ivfluid=0).
   Mitigation: This is expected -- the FCI causal graph provides sanity checks.

3. **Observational distribution, not interventional.** A transformer trained on observational
   EHR data will learn confounded associations (e.g. insulin→glucose wrong sign, same issue
   as step-C LightGBM simulator).
   Mitigation: Use FCI causal graph to constrain which input features are allowed to predict
   which output features (same deconfounding approach applied in step_c_structural_equations_deconfounded.py).

4. **Sparse terminal reward.** readmit_30d is observed only once per stay. Use SOFA delta
   as step reward (rich, frequent), readmit_30d as terminal reward -- same as step 11b setup.

5. **Transformer training instability on short sequences.** Very short stays (6-10 blocs)
   can destabilize training. Pad sequences and use attention masks.

6. **discharge_disposition data leakage.** Known only at discharge. Must be excluded from
   state used during RL rollouts.

7. **Variable-length sequences.** ICU stays have different lengths. Use truncate/pad to
   fixed length with attention masking, or sliding-window approach.

---

## 8. Reference Table

| Paper | arXiv ID | Relevance | Note |
|-------|----------|-----------|------|
| medDreamer | 2505.19785 | HIGHEST -- clinical EHR + world model + RL | Most directly relevant |
| Trajectory Transformer | 2106.02039 | High -- tabular continuous states, beam search | Discretization strategy |
| IRIS | 2209.00588 | Medium -- world model + RL imagination loop | Image-focused, adapt tokenizer |
| Delta-IRIS | 2406.19320 | Medium -- delta encoding for slow-changing states | Good clinical insight |
| COMBO | 2102.08363 | High -- conservative offline MBRL, ensemble world model | Practical gold standard |
| MOPO | 2005.13239 | High -- uncertainty-penalized offline MBRL | Ensemble NN world model |
| OTTO | 2404.10393 | Medium -- ensemble of transformers for offline RL | Tabular transition modeling |
| DreamerV3 | 2301.04104 | Medium -- best general world model, GRU not transformer | Strong baseline |
| iTransformer | 2310.06625 | Medium -- inverted attention for multivariate series | Cross-feature prediction |
| OmniTFT | 2511.19485 | Medium -- TFT for ICU multi-output forecasting | Multi-output next-state |
| SMTAFormer | 2407.11096 | Medium -- ICU readmission + transformer | Same outcome as your project |
| TransDreamer | 2202.09462 | Medium -- transformer replacing GRU in Dreamer | Long-range memory |
| Decision Transformer | 2106.01345 | Low -- policy, not simulator | Offline RL policy only |
| Raghu sepsis RL | 1711.09602 | Background -- no world model | Clinical RL baseline |
| AI Clinician | 1903.02345 | Background -- no world model | Clinical RL baseline |
| Oberst & Sontag | NeurIPS 2019 | Background -- GRU-VAE patient simulator | Pre-transformer clinical sim |
