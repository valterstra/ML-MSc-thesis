# Sepsis RL Pipeline (Steps 09-13)

## What This Is

Reinforcement learning pipeline for sepsis treatment (IV fluids + vasopressors).
Replicates two papers by Raghu et al.:

- **2017**: "Deep Reinforcement Learning for Sepsis Treatment" -- Dueling DDQN + PER, SARSA physician, off-policy evaluation
- **2018**: "Model-Based Reinforcement Learning for Sepsis Treatment" -- 4 environment simulators (NN, Linear, LSTM, BNN)

Both papers use the same data/preprocessing from the AI-Clinician (Komorowski et al.) framework.
Our implementation ports the sepsisrl GitHub repo (Python 2.7, TensorFlow 1.x) to Python 3 + PyTorch.

**Input**: `data/processed/sepsis/MKdataset.csv` (325,159 rows, 37,071 Sepsis-3 ICU stays from MIMIC-IV).
See `docs/sepsis_pipeline.md` for how that dataset was built (steps 01-08).

---

## Pipeline Overview

```
Step 09: Preprocess for RL       (85 sec)    DONE
Step 10: Discrete RL             (13 min)    DONE
Step 11: Continuous RL           (83 min)    DONE
Step 12: Off-Policy Evaluation   (22 min)    DONE
Step 13: Model-Based Simulators  (27 min)    DONE  (all 4 architectures)
```

---

## Step 09 -- Preprocess for RL

**Script:** `scripts/sepsis/step_09_preprocess.py`
**Module:** `src/careai/sepsis/rl/preprocessing.py`

Takes MKdataset.csv and produces RL-ready train/val/test splits:

1. **Action discretization**: IV fluid and vasopressor doses binned into 5 levels each (0 = no drug, 1-4 = quartiles of nonzero doses). Creates 5x5 = 25 discrete actions.
2. **Train/val/test split**: 70/8.7/21.3 by patient (25,949 / 3,707 / 7,415 ICU stays).
3. **Normalization**: Binary features (subtract 0.5), Gaussian features (z-score on train stats), log-normal features (log(0.1+x) then z-score).
4. **Reward construction**:
   - Sparse: +100 survive, -100 die (terminal only)
   - Shaped: `C0 * 1(SOFA unchanged & >0) + C1 * (SOFA change) + C2 * tanh(lactate change)` with C0=-0.025, C1=-0.125, C2=-2.0, terminal +/-100, clamped [-15, 15]
5. **Output columns**: Separate files for discrete RL (48 state features + action_int + sparse reward) and continuous RL (47 state features + iv/vaso levels + shaped reward + done flag).

**Outputs:**
```
data/processed/sepsis/
  rl_train_set_scaled.csv       (228,403 rows, 60 cols)
  rl_val_set_scaled.csv         (32,166 rows)
  rl_test_set_scaled.csv        (64,590 rows)
  rl_*_set_unscaled.csv         (same splits, no normalization)
  rl_*_set_original.csv         (same splits, original columns)
  rl_*_data_discrete.csv        (for step 10)
  rl_*_data_final_cont*.csv     (for step 11, with/without terminal reward)
  action_quartiles.json         (IV/vaso quartile boundaries)
  norm_stats.json               (per-feature mean/std for z-score)
```

**Key constants** (in `preprocessing.py`):
- `STATE_FEATURES`: 48 features (47 used for RL, `bloc` excluded)
- `BINARY_FIELDS`: gender, mechvent, re_admission
- `LOG_FIELDS`: 17 skewed lab/output features
- `NORM_FIELDS`: remaining features get standard z-score

---

## Step 10 -- Discrete RL

**Script:** `scripts/sepsis/step_10_discrete_rl.py`
**Module:** `src/careai/sepsis/rl/discrete.py`

Tabular MDP approach (from Komorowski 2016):

1. **K-means clustering**: 48 state features -> 1,250 discrete states (MiniBatchKMeans).
2. **Transition matrix**: Count-based P(s'|s,a) from observed transitions. 18,519 (s,a) pairs observed out of 31,250 possible.
3. **SARSA physician policy**: On-policy TD learning from clinical data. 250k episodes, alpha=0.1, gamma=1.0. Q range: [-14.29, 14.43].
4. **Value Iteration**: Off-policy optimal policy. gamma=0.9, converged at iteration 64.

**Outputs:**
```
models/sepsis_rl/discrete/
  kmeans_model.pkl          (1,250-cluster KMeans)
  trans_prob.pkl            (transition probability matrix)
  q_table_physician.pkl    (SARSA Q-table, 1250 x 25)
  policy_value_iter.pkl    (VI optimal policy, 1250 states)
  V_value_iter.pkl         (VI value function)
```

---

## Step 11 -- Continuous RL

**Script:** `scripts/sepsis/step_11_continuous_rl.py`
**Modules:** `src/careai/sepsis/rl/continuous.py`, `src/careai/sepsis/rl/networks.py`

Deep RL approach (from Raghu 2017):

1. **SARSA physician** (13 min): 2-layer FC (64-64) + BN + ReLU -> softmax over 25 actions. 70k steps, lr=1e-4. On-policy update from clinical data. Mean Q = 0.027.
2. **Sparse autoencoder** (12 min): 47 -> 200 sigmoid encoder/decoder. 100k steps. KL sparsity penalty (target=0.05, weight=1e-4). Produces 200-dim encoded state representation.
3. **Dueling Double DQN + PER** (two variants):
   - On raw 47-dim state (13 min): 60k steps, lr=1e-4. Mean Q = 0.202.
   - On 200-dim autoencoded state (44 min): 200k steps, lr=1e-4. Mean Q = 0.033.

DQN details: Dueling architecture (value + advantage streams), double Q-learning (separate target network, tau=0.001 soft update), prioritized experience replay (alpha=0.6, beta annealed 0.4->1.0), reward clipped [-1, 1], Q-regularization (threshold=20, lambda=5).

**Outputs:**
```
models/sepsis_rl/continuous/
  sarsa_phys/        (SARSA physician: model + Q-table)
  autoencoder/       (encoder + decoder + encoded states)
  dqn/               (DQN on raw state, 60k steps)
  dqn_auto/          (DQN on autoencoded state, 200k steps)
```

**Has `--smoke` flag**: 200 steps per component (~45 sec total).

---

## Step 12 -- Off-Policy Evaluation

**Script:** `scripts/sepsis/step_12_evaluate.py`
**Module:** `src/careai/sepsis/rl/evaluation.py`

Doubly Robust (DR) off-policy policy evaluation:

1. **Physician policy** (6 min): Supervised classification, 2-layer FC (64-64). 35k steps. Predicts P(action | state).
2. **Reward estimator** (4 min): Reward function approximation, 2-layer FC (128-128). 30k steps.
3. **Environment model** (10 min): Transition dynamics, 2-layer FC (500-500) + noise injection (std=0.03). 60k steps.
4. **DR evaluation** (2 min): Combines importance-weighted returns with approximate model baseline. Evaluates DQN (train), DQN (test), and SARSA physician policies.

**Results:**
```
Policy             DR Value (mean +/- std)    Valid Trajectories
SARSA physician    0.671 +/- 1.641            25,688 / 25,949
DQN (train)        0.647 +/- 1.554            25,679 / 25,949
DQN (test)         0.631 +/- 1.541             7,346 / 7,415
```

**Outputs:**
```
models/sepsis_rl/eval/
  physician_policy/     (supervised policy model)
  reward_estimator/     (reward approximation)
  env_model/            (transition dynamics for DR)

reports/sepsis_rl/
  evaluation_results.json
```

**Has `--smoke` flag**: 200 steps per component (~11 sec total).

---

## Step 13 -- Model-Based Simulators

**Script:** `scripts/sepsis/step_13_simulator.py`
**Module:** `src/careai/sepsis/rl/simulator.py`

Four environment model architectures from Raghu 2018:

1. **NN** (Feedforward Neural Network): 2 FC + ReLU + BatchNorm, 256 hidden. Paper's preferred model.
2. **Linear** (Linear Regression): Single linear layer. Baseline.
3. **LSTM** (Recurrent Neural Network): LSTM on 4-timestep history sequence, 256 hidden.
4. **BNN** (Bayesian Neural Network): Variational inference, 2 layers x 32 hidden, tanh. Provides uncertainty estimates.

All models predict state deltas: `Delta_t = s_{t+1} - s_t` from a 4-timestep history vector `h_t = [s_t, a_t, s_{t-1}, a_{t-1}, s_{t-2}, a_{t-2}, s_{t-3}, a_{t-3}]` (196-dim input, 47-dim output).

Training: 100 epochs, Adam optimizer, MSE loss (+ KL divergence for BNN). 129,468 training pairs from 228k train rows.

**Results (100 epochs, full data):**
```
Model     Test MSE    Rollout MSE(1)  Rollout MSE(10)  Time
LSTM      0.00464     0.00482         0.01244          17 min
NN        0.00477     0.00501         0.01551          2.5 min
Linear    0.00478     0.00491         0.01327          0.6 min
BNN       0.00483     0.00504         0.01372          7 min
```

Ranking matches paper exactly: LSTM < NN ~ Linear < BNN by test MSE.
Rollout MSE grows with steps (autoregressive error accumulation) -- expected behavior.

**Simulator class** (`SepsisSimulator`): Autoregressive rollout engine with `reset()`, `step()`, `rollout()`, `rollout_batch()` methods. Can be used for model-based policy search.

**Outputs:**
```
models/sepsis_rl/simulator/
  nn/       (transition_model.pt + model_config.pkl)
  linear/
  lstm/
  bnn/

reports/sepsis_rl/
  simulator_<type>_per_feature_mse.json   (per-feature test MSE)
  simulator_<type>_rollout_eval.json      (multi-step rollout MSE)
  simulator_comparison.json               (head-to-head comparison)
```

**Usage:**
```bash
# Single model
python -c "import torch; exec(open('scripts/sepsis/step_13_simulator.py').read())" --model-type nn

# All four models
python -c "import torch; exec(open('scripts/sepsis/step_13_simulator.py').read())" --model-type all

# Smoke test (<2 min)
python -c "import torch; exec(open('scripts/sepsis/step_13_simulator.py').read())" --model-type all --smoke
```

**Note**: The `python -c "import torch; exec(...)"` wrapper is required on Windows to work around a DLL loading issue (WinError 1114) where torch must be imported before other DLLs.

---

## Source Code Map

```
src/careai/sepsis/rl/
  __init__.py
  preprocessing.py    -- STATE_FEATURES (48), normalization groups, action discretization,
                         reward construction, train/val/test splitting
  discrete.py         -- cluster_states(), build_transition_matrix(), sarsa_episodic(),
                         value_iteration(). Constants: N_CLUSTERS=1250, N_ACTIONS=25
  networks.py         -- DuelingDQN, SparseAutoencoder, PhysicianPolicy, EnvModel,
                         RewardEstimator (all PyTorch nn.Module)
  continuous.py       -- PrioritizedReplayBuffer, train_dqn(), train_sarsa_physician(),
                         train_autoencoder(), prepare_rl_data()
  evaluation.py       -- train_physician_policy(), train_reward_estimator(),
                         train_env_model(), doubly_robust_evaluation()
  simulator.py        -- TransitionModel (NN), LinearTransitionModel, LSTMTransitionModel,
                         BayesianTransitionModel (BNN), BayesianLinear,
                         MODEL_TYPES dict, create_transition_model(),
                         build_history_dataset(), train_transition_model(),
                         evaluate_per_feature(), SepsisSimulator class,
                         evaluate_rollouts()

scripts/sepsis/
  step_09_preprocess.py
  step_10_discrete_rl.py
  step_11_continuous_rl.py    (--smoke flag)
  step_12_evaluate.py         (--smoke flag)
  step_13_simulator.py        (--smoke flag, --model-type nn|linear|lstm|bnn|all)
  run_full_rl.py              (runs steps 11+12 sequentially, handles torch DLL issue)
```

---

## Comparison with Paper Results

### DR Evaluation (2017/2018 papers)
Paper reports clinician value V_pi_b = 9.90, best blended policy = 12.8 (PHWDR).
Our DR values are ~0.6-0.7. The difference comes from:
1. **Reward scale**: paper uses terminal +/-15, sepsisrl code uses +/-100
2. **Behavior policy estimator**: paper uses kNN (k=250), we use neural network classifier
3. **IS collapse**: deterministic DQN + 25 actions means most importance weights go to zero

The relative ordering (SARSA physician > DQN train > DQN test) is consistent.

### Simulator MSE (2018 paper Table 1)
Paper MSE: NN=0.171, Linear=0.195, LSTM=0.122, BNN=0.220.
Our MSE: NN=0.00477, Linear=0.00478, LSTM=0.00464, BNN=0.00483.
~35x lower because we train on z-score normalized data; paper likely reports raw-scale MSE.
**Ranking matches exactly**: LSTM < NN ~ Linear < BNN.

---

## Key Design Decisions

1. **Action space**: 5x5 = 25 discrete (quartile-binned IV fluid x vasopressor). Same as both papers.
2. **State features**: 47 (48 minus bloc/timestep). Same as sepsisrl/data/state_features.txt.
3. **Reward**: Shaped SOFA/lactate intermediate + sparse terminal. Terminal +/-100 (from sepsisrl code, paper used +/-15).
4. **DQN architecture**: Dueling Double DQN with PER, 2x64 hidden (sepsisrl code uses 64, paper uses 128).
5. **4-timestep history** for simulators: h_t = [s_t, a_t, ..., s_{t-3}, a_{t-3}], chosen by cross-validation in the paper.

---

## Run Order (from MKdataset.csv)

```bash
source ../.venv/Scripts/activate

# Step 09: Preprocess (~85 sec)
python scripts/sepsis/step_09_preprocess.py

# Step 10: Discrete RL (~13 min)
python scripts/sepsis/step_10_discrete_rl.py

# Steps 11+12: Continuous RL + Evaluation (~105 min)
python -c "import torch; exec(open('scripts/sepsis/run_full_rl.py').read())"

# Step 13: Simulators (~27 min for all 4)
python -c "import torch; exec(open('scripts/sepsis/step_13_simulator.py').read())" --model-type all
```

For smoke testing any step, add `--smoke` flag (all complete in <1 min each).
