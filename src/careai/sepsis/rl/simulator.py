"""
Model-based environment simulator for sepsis RL.
Ported from Raghu et al. 2018 "Model-Based Reinforcement Learning for Sepsis Treatment".

The simulator learns a transition model: Delta_t = f(h_t; theta) + epsilon
where h_t = [s_t, a_t, s_{t-1}, a_{t-1}, s_{t-2}, a_{t-2}, s_{t-3}, a_{t-3}]

Key design: uses 4 timesteps of history (current + 3 previous) to capture
temporal patterns in patient physiology.

Four model architectures (from paper Table 1):
  - NN:     2 FC layers + ReLU + BatchNorm (paper's preferred model)
  - Linear: Linear regression baseline
  - LSTM:   Recurrent neural network on the history sequence
  - BNN:    Bayesian neural network with variational inference
"""
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ── Transition Model ──────────────────────────────────────────────────

class TransitionModel(nn.Module):
    """Neural network environment model predicting state deltas.

    Architecture (from Raghu 2018 Section 3.1):
      Input(h_t) -> FC(hidden) -> BN -> ReLU -> FC(hidden) -> BN -> ReLU -> FC(n_state)

    Input: h_t = concat of 4 timesteps of (state, action)
    Output: Delta_t = predicted change in state features
    """

    def __init__(self, n_input, n_output, hidden=256, **kwargs):
        super().__init__()
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, n_output)

    def forward(self, x):
        x = torch.relu(self.bn1(self.fc1(x)))
        x = torch.relu(self.bn2(self.fc2(x)))
        return self.out(x)


class LinearTransitionModel(nn.Module):
    """Linear regression baseline for predicting state deltas.

    From Raghu 2018 Table 1 (LR): MSE=0.195 (worst of the simple models).
    Single linear layer: Delta_t = W * h_t + b.
    """

    def __init__(self, n_input, n_output, **kwargs):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear(x)


class LSTMTransitionModel(nn.Module):
    """LSTM-based transition model predicting state deltas from history sequence.

    From Raghu 2018 Table 1 (RNN): MSE=0.122 (best overall, but poor at early
    timesteps -- paper rejects it in favor of the feedforward NN for consistency).

    Input: history reshaped as (batch, n_history, n_state + n_action) sequence.
    Processes sequence with LSTM, uses final hidden state -> FC -> Delta_t.
    """

    def __init__(self, n_input, n_output, hidden=256, n_history=4,
                 n_lstm_layers=1, dropout=0.0, **kwargs):
        super().__init__()
        # n_input is the flat dim; per-step dim = n_input / n_history
        self.n_history = n_history
        self.step_dim = n_input // n_history
        self.lstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, n_output)

    def forward(self, x):
        # x: (batch, n_input) flat -> reshape to (batch, n_history, step_dim)
        batch_size = x.size(0)
        # History is stored as [s_t, a_t, s_{t-1}, a_{t-1}, ...] (most recent first)
        # Reshape and reverse so LSTM sees oldest-to-newest
        seq = x.view(batch_size, self.n_history, self.step_dim)
        seq = seq.flip(dims=[1])  # oldest first for LSTM
        _, (h_n, _) = self.lstm(seq)
        # h_n: (n_layers, batch, hidden) -> take last layer
        return self.out(h_n[-1])


class BayesianLinear(nn.Module):
    """Linear layer with Gaussian variational posterior over weights.

    Implements local reparameterization trick for efficient sampling.
    Prior: N(0, prior_sigma^2).
    Posterior: q(w) = N(mu_w, sigma_w^2) with sigma_w = log(1 + exp(rho_w)).
    """

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_sigma = prior_sigma

        # Variational parameters
        self.weight_mu = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -3.0))

        # Initialize mu with small random values
        nn.init.xavier_normal_(self.weight_mu)

    def _sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def forward(self, x):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)

        # Reparameterization: w = mu + sigma * epsilon
        if self.training:
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias = self.bias_mu + bias_sigma * torch.randn_like(bias_sigma)
        else:
            # At eval time, use the mean
            weight = self.weight_mu
            bias = self.bias_mu

        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        """KL(q(w) || p(w)) for Gaussian prior and posterior."""
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma = self._sigma(self.bias_rho)

        prior_var = self.prior_sigma ** 2

        # KL for weights
        kl_w = 0.5 * (
            (weight_sigma ** 2 + self.weight_mu ** 2) / prior_var
            - 1.0
            + 2.0 * (np.log(self.prior_sigma) - torch.log(weight_sigma))
        ).sum()

        # KL for biases
        kl_b = 0.5 * (
            (bias_sigma ** 2 + self.bias_mu ** 2) / prior_var
            - 1.0
            + 2.0 * (np.log(self.prior_sigma) - torch.log(bias_sigma))
        ).sum()

        return kl_w + kl_b


class BayesianTransitionModel(nn.Module):
    """Bayesian neural network for predicting state deltas with uncertainty.

    From Raghu 2018 Table 1 (BNN): MSE=0.220 (worst MSE, but provides
    full predictive distribution over Delta_t).

    Architecture: 2 hidden layers, 32 units each, tanh activations.
    Uses variational inference with Gaussian approximate posterior.
    """

    def __init__(self, n_input, n_output, hidden=32, prior_sigma=1.0, **kwargs):
        super().__init__()
        self.fc1 = BayesianLinear(n_input, hidden, prior_sigma)
        self.fc2 = BayesianLinear(hidden, hidden, prior_sigma)
        self.out = BayesianLinear(hidden, n_output, prior_sigma)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.out(x)

    def kl_divergence(self):
        """Total KL divergence across all Bayesian layers."""
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out.kl_divergence()

    def predict_with_uncertainty(self, x, n_samples=20):
        """Sample multiple predictions to estimate mean and variance."""
        self.train()  # enable sampling
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))
        preds = torch.stack(preds, dim=0)  # (n_samples, batch, n_output)
        self.eval()
        return preds.mean(dim=0), preds.var(dim=0)


MODEL_TYPES = {
    "nn": TransitionModel,
    "linear": LinearTransitionModel,
    "lstm": LSTMTransitionModel,
    "bnn": BayesianTransitionModel,
}


def create_transition_model(model_type, n_input, n_output, hidden=256,
                            n_history=4, **kwargs):
    """Factory function to create a transition model by type name."""
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type: {model_type}. Choose from {list(MODEL_TYPES.keys())}")
    cls = MODEL_TYPES[model_type]
    return cls(n_input, n_output, hidden=hidden, n_history=n_history, **kwargs)


# ── History dataset construction ──────────────────────────────────────

def build_history_dataset(df, state_cols, n_history=4):
    """Build (h_t, Delta_t) training pairs with multi-timestep history.

    For each timestep t in a trajectory, constructs:
      h_t = [s_t, a_t, s_{t-1}, a_{t-1}, ..., s_{t-k}, a_{t-k}]
      Delta_t = s_{t+1} - s_t

    Only includes timesteps where:
      - At least n_history timesteps of history are available
      - The next timestep belongs to the same ICU stay

    Args:
        df: DataFrame with state features, action cols, icustayid
        state_cols: list of state feature column names
        n_history: number of timesteps to include (default 4: current + 3 past)

    Returns:
        dict with:
          history: (N, n_history * (n_state + n_action)) array
          deltas: (N, n_state) array
          states: (N, n_state) array (current state, for rollout eval)
          next_states: (N, n_state) array
          icustayids: (N,) array
    """
    logging.info("Building history dataset: %d timesteps of history, %d state features",
                 n_history, len(state_cols))

    n_state = len(state_cols)
    n_action = 2  # IV level, vaso level

    states_raw = df[state_cols].values.astype(np.float32)
    states_raw = np.nan_to_num(states_raw, nan=0.0)

    # Normalized actions: iv_input/4, vaso_input/4
    iv = df["iv_input"].values.astype(np.float32) / 4.0 if "iv_input" in df.columns else np.zeros(len(df), dtype=np.float32)
    vaso = df["vaso_input"].values.astype(np.float32) / 4.0 if "vaso_input" in df.columns else np.zeros(len(df), dtype=np.float32)
    actions_raw = np.column_stack([iv, vaso])

    icuids = df["icustayid"].values

    # Build episode boundaries
    episode_starts = set()
    episode_starts.add(0)
    for i in range(1, len(df)):
        if icuids[i] != icuids[i - 1]:
            episode_starts.add(i)

    # Build history features and delta targets
    histories = []
    deltas = []
    cur_states = []
    next_states = []
    sample_icuids = []

    for i in range(n_history - 1, len(df) - 1):
        # Check: next row is same patient
        if icuids[i + 1] != icuids[i]:
            continue

        # Check: all history rows are from same patient
        valid = True
        for k in range(1, n_history):
            if icuids[i - k] != icuids[i]:
                valid = False
                break
        if not valid:
            continue

        # Build history vector: [s_t, a_t, s_{t-1}, a_{t-1}, ..., s_{t-k+1}, a_{t-k+1}]
        h = []
        for k in range(n_history):
            idx = i - k
            h.append(states_raw[idx])
            h.append(actions_raw[idx])
        h = np.concatenate(h)
        histories.append(h)

        # Delta target
        delta = states_raw[i + 1] - states_raw[i]
        deltas.append(delta)
        cur_states.append(states_raw[i])
        next_states.append(states_raw[i + 1])
        sample_icuids.append(icuids[i])

    histories = np.array(histories, dtype=np.float32)
    deltas = np.array(deltas, dtype=np.float32)
    cur_states = np.array(cur_states, dtype=np.float32)
    next_states = np.array(next_states, dtype=np.float32)
    sample_icuids = np.array(sample_icuids)

    logging.info("  Built %d training pairs (from %d total rows)", len(histories), len(df))
    logging.info("  History vector dim: %d (= %d timesteps x (%d state + %d action))",
                 histories.shape[1], n_history, n_state, n_action)
    logging.info("  Delta vector dim: %d", deltas.shape[1])

    return {
        "history": histories,
        "deltas": deltas,
        "states": cur_states,
        "next_states": next_states,
        "icustayids": sample_icuids,
    }


# ── Training ──────────────────────────────────────────────────────────

def train_transition_model(dataset, n_state, n_action=2, n_history=4,
                           hidden=256, lr=1e-3, batch_size=256,
                           num_epochs=100, val_fraction=0.1,
                           model_type="nn", kl_weight=1e-4,
                           save_dir=None, device="cpu", log_every=10):
    """Train a transition model on (history, delta) pairs.

    Supports all four model types from Raghu 2018:
      - "nn":     Feedforward NN (2 FC + ReLU + BN) — paper's preferred model
      - "linear": Linear regression baseline
      - "lstm":   LSTM on the history sequence
      - "bnn":    Bayesian NN with variational inference

    Args:
        dataset: dict from build_history_dataset()
        n_state: number of state features
        n_action: number of action features (2: IV, vaso)
        n_history: number of history timesteps
        hidden: hidden layer size (ignored for linear; 32 for BNN per paper)
        lr: learning rate
        batch_size: training batch size
        num_epochs: number of training epochs
        val_fraction: fraction of data for validation
        model_type: one of "nn", "linear", "lstm", "bnn"
        kl_weight: KL divergence weight for BNN training (ELBO loss)
        save_dir: where to save the model
        device: PyTorch device
        log_every: log every N epochs

    Returns:
        (model, train_losses, val_losses)
    """
    n_input = n_history * (n_state + n_action)
    n_output = n_state

    # BNN uses 32 hidden units per paper
    if model_type == "bnn" and hidden == 256:
        hidden = 32

    logging.info("Training transition model [%s]:", model_type.upper())
    logging.info("  Input dim: %d, Output dim: %d, Hidden: %d", n_input, n_output, hidden)
    logging.info("  Epochs: %d, Batch: %d, LR: %.1e", num_epochs, batch_size, lr)
    if model_type == "bnn":
        logging.info("  KL weight: %.1e", kl_weight)

    model = create_transition_model(
        model_type, n_input, n_output, hidden=hidden, n_history=n_history,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Train/val split (by sample, not by patient — simpler for this model)
    n = len(dataset["history"])
    n_val = int(n * val_fraction)
    n_train = n - n_val

    perm = np.random.RandomState(42).permutation(n)
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    X_train = torch.FloatTensor(dataset["history"][train_idx]).to(device)
    y_train = torch.FloatTensor(dataset["deltas"][train_idx]).to(device)
    X_val = torch.FloatTensor(dataset["history"][val_idx]).to(device)
    y_val = torch.FloatTensor(dataset["deltas"][val_idx]).to(device)

    logging.info("  Train: %d samples, Val: %d samples", n_train, n_val)

    n_params = sum(p.numel() for p in model.parameters())
    logging.info("  Model parameters: %d", n_params)

    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    best_state_dict = None
    is_bnn = model_type == "bnn"

    t_start = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches = 0

        # Shuffle training data
        shuffle_idx = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            batch_idx = shuffle_idx[start:end]

            pred = model(X_train[batch_idx])
            mse_loss = criterion(pred, y_train[batch_idx])

            if is_bnn:
                kl = model.kl_divergence()
                loss = mse_loss + kl_weight * kl / n_train
            else:
                loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += mse_loss.item()
            n_batches += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

        # Validation (always use mean for BNN)
        model.eval()
        with torch.no_grad():
            val_pred = []
            for start in range(0, n_val, batch_size):
                end = min(start + batch_size, n_val)
                val_pred.append(model(X_val[start:end]))
            val_pred = torch.cat(val_pred, dim=0)
            val_loss = criterion(val_pred, y_val).item()
        val_losses.append(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % log_every == 0:
            elapsed = time.time() - t_start
            remaining = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            logging.info("  Epoch %d/%d (%.0f%%) | train_MSE=%.6f, val_MSE=%.6f | "
                         "best_val=%.6f | elapsed %.0fs | ETA %.0fs",
                         epoch + 1, num_epochs, 100 * (epoch + 1) / num_epochs,
                         train_loss, val_loss, best_val_loss, elapsed, remaining)

    # Load best model
    model.load_state_dict(best_state_dict)
    model.to(device)
    logging.info("Training [%s] complete. Best val MSE=%.6f", model_type.upper(), best_val_loss)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state_dict, f"{save_dir}/transition_model.pt")
        with open(f"{save_dir}/model_config.pkl", "wb") as f:
            pickle.dump({
                "n_input": n_input,
                "n_output": n_output,
                "n_state": n_state,
                "n_action": n_action,
                "n_history": n_history,
                "hidden": hidden,
                "model_type": model_type,
            }, f)
        logging.info("  Model saved to %s", save_dir)

    return model, train_losses, val_losses


# ── Per-feature evaluation ────────────────────────────────────────────

def evaluate_per_feature(model, dataset, state_cols, device="cpu", batch_size=4096):
    """Compute per-feature MSE on the dataset.

    Returns dict mapping feature name -> MSE.
    """
    model.eval()
    X = torch.FloatTensor(dataset["history"]).to(device)
    y = dataset["deltas"]  # numpy, shape (N, n_state)

    # Predict in batches
    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            preds.append(model(X[start:end]).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    # Per-feature MSE
    mse_per_feature = {}
    for i, col in enumerate(state_cols):
        mse = np.mean((preds[:, i] - y[:, i]) ** 2)
        mse_per_feature[col] = float(mse)

    # Overall MSE
    overall_mse = np.mean((preds - y) ** 2)
    mse_per_feature["__overall__"] = float(overall_mse)

    return mse_per_feature


# ── Simulator (rollout engine) ────────────────────────────────────────

class SepsisSimulator:
    """Autoregressive environment simulator for model-based RL.

    Maintains a history buffer and predicts next states using the
    learned transition model.

    Usage:
        sim = SepsisSimulator(model, n_state, n_history=4)
        sim.reset(initial_states, initial_actions)  # provide history
        for t in range(rollout_length):
            action = policy(sim.current_state)
            next_state, delta = sim.step(action)
    """

    def __init__(self, model, n_state, n_action=2, n_history=4, device="cpu"):
        self.model = model
        self.model.eval()
        self.n_state = n_state
        self.n_action = n_action
        self.n_history = n_history
        self.device = device

        # History buffer: list of (state, action) tuples, most recent last
        self.state_history = []
        self.action_history = []

    def reset(self, initial_states, initial_actions):
        """Initialize the simulator with historical states and actions.

        Args:
            initial_states: (n_history, n_state) array — oldest first
            initial_actions: (n_history, n_action) array — oldest first
        """
        assert len(initial_states) == self.n_history, \
            f"Need {self.n_history} initial states, got {len(initial_states)}"
        assert len(initial_actions) == self.n_history, \
            f"Need {self.n_history} initial actions, got {len(initial_actions)}"

        self.state_history = [s.copy() for s in initial_states]
        self.action_history = [a.copy() for a in initial_actions]

    @property
    def current_state(self):
        """Return the most recent state."""
        return self.state_history[-1].copy()

    def _build_history_vector(self):
        """Build the history input vector h_t = [s_t, a_t, s_{t-1}, a_{t-1}, ...]."""
        h = []
        for k in range(self.n_history):
            idx = len(self.state_history) - 1 - k  # most recent first
            h.append(self.state_history[idx])
            h.append(self.action_history[idx])
        return np.concatenate(h).astype(np.float32)

    def step(self, action):
        """Advance one timestep given an action.

        Args:
            action: (n_action,) array — [iv_level/4, vaso_level/4]

        Returns:
            (next_state, delta) — both (n_state,) arrays
        """
        # Build history vector
        h = self._build_history_vector()
        h_tensor = torch.FloatTensor(h).unsqueeze(0).to(self.device)

        # Predict delta
        with torch.no_grad():
            delta = self.model(h_tensor).cpu().numpy().squeeze(0)

        # Compute next state
        current = self.state_history[-1]
        next_state = current + delta

        # Update history (slide window)
        self.state_history.append(next_state.copy())
        self.action_history.append(action.copy())

        # Keep only the last n_history entries
        if len(self.state_history) > self.n_history:
            self.state_history = self.state_history[-self.n_history:]
            self.action_history = self.action_history[-self.n_history:]

        return next_state, delta

    def rollout(self, actions_sequence):
        """Perform a multi-step rollout given a sequence of actions.

        Args:
            actions_sequence: (T, n_action) array of actions

        Returns:
            states: (T+1, n_state) array — initial state + T predicted states
            deltas: (T, n_state) array of predicted deltas
        """
        states = [self.current_state.copy()]
        deltas = []

        for t in range(len(actions_sequence)):
            next_state, delta = self.step(actions_sequence[t])
            states.append(next_state.copy())
            deltas.append(delta.copy())

        return np.array(states), np.array(deltas)

    def rollout_batch(self, initial_states_batch, initial_actions_batch,
                      actions_sequences, n_steps):
        """Batch rollout for multiple patients.

        Args:
            initial_states_batch: (B, n_history, n_state)
            initial_actions_batch: (B, n_history, n_action)
            actions_sequences: (B, n_steps, n_action)
            n_steps: number of rollout steps

        Returns:
            all_states: (B, n_steps+1, n_state)
        """
        B = len(initial_states_batch)
        all_states = np.zeros((B, n_steps + 1, self.n_state), dtype=np.float32)

        for b in range(B):
            self.reset(initial_states_batch[b], initial_actions_batch[b])
            all_states[b, 0] = self.current_state

            for t in range(n_steps):
                next_state, _ = self.step(actions_sequences[b, t])
                all_states[b, t + 1] = next_state

        return all_states


def evaluate_rollouts(model, df, state_cols, n_history=4, n_rollout_steps=10,
                      n_patients=100, seed=42, device="cpu"):
    """Evaluate rollout quality by comparing simulated trajectories to real ones.

    Picks random patients, uses their first n_history states as initialization,
    then rolls out using their ACTUAL actions. Compares predicted vs real states.

    Returns dict with per-step and per-feature MSE.
    """
    logging.info("Evaluating rollouts: %d patients, %d steps each", n_patients, n_rollout_steps)

    n_state = len(state_cols)
    rng = np.random.RandomState(seed)

    states_raw = df[state_cols].values.astype(np.float32)
    states_raw = np.nan_to_num(states_raw, nan=0.0)

    iv = df["iv_input"].values.astype(np.float32) / 4.0 if "iv_input" in df.columns else np.zeros(len(df), dtype=np.float32)
    vaso = df["vaso_input"].values.astype(np.float32) / 4.0 if "vaso_input" in df.columns else np.zeros(len(df), dtype=np.float32)
    actions_raw = np.column_stack([iv, vaso])
    icuids = df["icustayid"].values

    # Find episodes with enough length
    episodes = []
    start = 0
    for i in range(1, len(df)):
        if icuids[i] != icuids[i - 1]:
            length = i - start
            if length >= n_history + n_rollout_steps:
                episodes.append((start, i))
            start = i
    length = len(df) - start
    if length >= n_history + n_rollout_steps:
        episodes.append((start, len(df)))

    logging.info("  %d episodes with length >= %d", len(episodes), n_history + n_rollout_steps)

    if len(episodes) < n_patients:
        n_patients = len(episodes)
        logging.info("  Reduced to %d patients (not enough long episodes)", n_patients)

    selected = rng.choice(len(episodes), n_patients, replace=False)

    sim = SepsisSimulator(model, n_state, n_action=2, n_history=n_history, device=device)

    # Collect per-step MSE
    per_step_mse = np.zeros(n_rollout_steps, dtype=np.float64)
    per_step_feature_mse = np.zeros((n_rollout_steps, n_state), dtype=np.float64)
    n_valid = 0

    for ep_idx in selected:
        ep_start, ep_end = episodes[ep_idx]

        # Initialize with first n_history states
        init_states = states_raw[ep_start:ep_start + n_history]
        init_actions = actions_raw[ep_start:ep_start + n_history]
        sim.reset(init_states, init_actions)

        # Rollout using actual clinician actions
        rollout_actions = actions_raw[ep_start + n_history:ep_start + n_history + n_rollout_steps]
        actual_states = states_raw[ep_start + n_history:ep_start + n_history + n_rollout_steps]

        predicted_states, _ = sim.rollout(rollout_actions)
        # predicted_states[0] = last init state, predicted_states[1:] = predictions
        predicted = predicted_states[1:]  # (n_rollout_steps, n_state)

        for t in range(n_rollout_steps):
            diff = predicted[t] - actual_states[t]
            per_step_mse[t] += np.mean(diff ** 2)
            per_step_feature_mse[t] += diff ** 2

        n_valid += 1

    per_step_mse /= n_valid
    per_step_feature_mse /= n_valid

    logging.info("  Rollout MSE by step:")
    for t in range(n_rollout_steps):
        logging.info("    Step %d: MSE=%.6f", t + 1, per_step_mse[t])

    return {
        "per_step_mse": per_step_mse.tolist(),
        "per_step_feature_mse": per_step_feature_mse.tolist(),
        "n_patients": n_valid,
        "n_steps": n_rollout_steps,
        "state_cols": state_cols,
    }
