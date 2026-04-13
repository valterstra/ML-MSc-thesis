"""
Model-based environment simulator for ICU readmission RL.

Adapted from src/careai/sepsis/rl/simulator.py. Key differences:
  - n_action=5 (5 binary drug flags decoded from integer action 0-31)
    vs sepsis n_action=2 ([iv_level/4, vaso_level/4] continuous)
  - Input state columns: s_* prefix (already z-scored) from rl_dataset_broad.parquet
  - Stay ID column: 'icustayid' (same name as sepsis)
  - Simulator class renamed ICUSimulator

Four model architectures (same as Raghu 2018 Table 1):
  nn:     2 FC + ReLU + BN (preferred model)
  linear: Linear regression baseline
  lstm:   LSTM on history sequence
  bnn:    Bayesian NN with variational inference
"""
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ---------------------------------------------------------------------------
# Transition model architectures (identical to sepsis pipeline)
# ---------------------------------------------------------------------------

class TransitionModel(nn.Module):
    """Feedforward NN: 2 FC + BN + ReLU predicting state deltas."""

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
    """Linear regression baseline for predicting state deltas."""

    def __init__(self, n_input, n_output, **kwargs):
        super().__init__()
        self.linear = nn.Linear(n_input, n_output)

    def forward(self, x):
        return self.linear(x)


class LSTMTransitionModel(nn.Module):
    """LSTM-based transition model on the history sequence."""

    def __init__(self, n_input, n_output, hidden=256, n_history=4,
                 n_lstm_layers=1, dropout=0.0, **kwargs):
        super().__init__()
        self.n_history = n_history
        self.step_dim  = n_input // n_history
        self.lstm = nn.LSTM(
            input_size=self.step_dim,
            hidden_size=hidden,
            num_layers=n_lstm_layers,
            batch_first=True,
            dropout=dropout if n_lstm_layers > 1 else 0.0,
        )
        self.out = nn.Linear(hidden, n_output)

    def forward(self, x):
        batch_size = x.size(0)
        seq = x.view(batch_size, self.n_history, self.step_dim)
        seq = seq.flip(dims=[1])  # oldest first for LSTM
        _, (h_n, _) = self.lstm(seq)
        return self.out(h_n[-1])


class BayesianLinear(nn.Module):
    """Variational Bayesian linear layer (local reparameterization)."""

    def __init__(self, in_features, out_features, prior_sigma=1.0):
        super().__init__()
        self.prior_sigma = prior_sigma
        self.weight_mu  = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -3.0))
        self.bias_mu    = nn.Parameter(torch.zeros(out_features))
        self.bias_rho   = nn.Parameter(torch.full((out_features,), -3.0))
        nn.init.xavier_normal_(self.weight_mu)

    def _sigma(self, rho):
        return torch.log1p(torch.exp(rho))

    def forward(self, x):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma   = self._sigma(self.bias_rho)
        if self.training:
            weight = self.weight_mu + weight_sigma * torch.randn_like(weight_sigma)
            bias   = self.bias_mu   + bias_sigma   * torch.randn_like(bias_sigma)
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return nn.functional.linear(x, weight, bias)

    def kl_divergence(self):
        weight_sigma = self._sigma(self.weight_rho)
        bias_sigma   = self._sigma(self.bias_rho)
        prior_var    = self.prior_sigma ** 2
        kl_w = 0.5 * (
            (weight_sigma ** 2 + self.weight_mu ** 2) / prior_var - 1.0
            + 2.0 * (np.log(self.prior_sigma) - torch.log(weight_sigma))
        ).sum()
        kl_b = 0.5 * (
            (bias_sigma ** 2 + self.bias_mu ** 2) / prior_var - 1.0
            + 2.0 * (np.log(self.prior_sigma) - torch.log(bias_sigma))
        ).sum()
        return kl_w + kl_b


class BayesianTransitionModel(nn.Module):
    """BNN: 2 hidden layers (32 units each), tanh, variational weights."""

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
        return self.fc1.kl_divergence() + self.fc2.kl_divergence() + self.out.kl_divergence()

    def predict_with_uncertainty(self, x, n_samples=20):
        self.train()
        preds = []
        with torch.no_grad():
            for _ in range(n_samples):
                preds.append(self.forward(x))
        preds = torch.stack(preds, dim=0)
        self.eval()
        return preds.mean(dim=0), preds.var(dim=0)


MODEL_TYPES = {
    "nn":     TransitionModel,
    "linear": LinearTransitionModel,
    "lstm":   LSTMTransitionModel,
    "bnn":    BayesianTransitionModel,
}


def create_transition_model(model_type, n_input, n_output, hidden=256, n_history=4, **kwargs):
    if model_type not in MODEL_TYPES:
        raise ValueError(f"Unknown model type '{model_type}'. Choose from {list(MODEL_TYPES.keys())}")
    cls = MODEL_TYPES[model_type]
    return cls(n_input, n_output, hidden=hidden, n_history=n_history, **kwargs)


# ---------------------------------------------------------------------------
# Action decoding utility
# ---------------------------------------------------------------------------

def _decode_actions(actions_int):
    """Decode integer actions 0-31 to 5-bit binary arrays (N, 5)."""
    arr = np.asarray(actions_int, dtype=np.int32)
    return ((arr[:, None] >> np.arange(5)) & 1).astype(np.float32)


# ---------------------------------------------------------------------------
# History dataset construction
# ---------------------------------------------------------------------------

def build_history_dataset(df, state_cols, n_history=4):
    """Build (h_t, Delta_t) training pairs with multi-timestep history.

    For each timestep t, constructs:
      h_t = [s_t, a_t, s_{t-1}, a_{t-1}, ..., s_{t-k}, a_{t-k}]
      Delta_t = s_{t+1} - s_t

    Action at each timestep is decoded to 5-bit binary vector.
    Only includes timesteps with n_history consecutive rows from the same stay.

    Args:
        df:         DataFrame with s_* state cols, 'a' action col, 'icustayid'
        state_cols: list of s_* column names
        n_history:  timesteps of history (default 4: current + 3 past)

    Returns:
        dict with history, deltas, states, next_states, icustayids arrays
    """
    logging.info("Building history dataset: n_history=%d, n_state=%d",
                 n_history, len(state_cols))

    n_state  = len(state_cols)
    n_action = 5

    states_raw = df[state_cols].values.astype(np.float32)
    states_raw = np.nan_to_num(states_raw, nan=0.0)

    actions_raw = _decode_actions(df["a"].values)  # (N, 5) binary
    icuids      = df["icustayid"].values

    histories      = []
    deltas         = []
    cur_states     = []
    next_states_ls = []
    sample_icuids  = []

    for i in range(n_history - 1, len(df) - 1):
        if icuids[i + 1] != icuids[i]:
            continue
        valid = all(icuids[i - k] == icuids[i] for k in range(1, n_history))
        if not valid:
            continue

        h = []
        for k in range(n_history):
            h.append(states_raw[i - k])
            h.append(actions_raw[i - k])
        histories.append(np.concatenate(h))

        deltas.append(states_raw[i + 1] - states_raw[i])
        cur_states.append(states_raw[i])
        next_states_ls.append(states_raw[i + 1])
        sample_icuids.append(icuids[i])

    histories      = np.array(histories,      dtype=np.float32)
    deltas         = np.array(deltas,         dtype=np.float32)
    cur_states     = np.array(cur_states,     dtype=np.float32)
    next_states_ls = np.array(next_states_ls, dtype=np.float32)
    sample_icuids  = np.array(sample_icuids)

    logging.info("  Built %d training pairs from %d rows", len(histories), len(df))
    logging.info("  History vector dim: %d (= %d steps x (%d state + %d action))",
                 histories.shape[1], n_history, n_state, n_action)

    return {
        "history":     histories,
        "deltas":      deltas,
        "states":      cur_states,
        "next_states": next_states_ls,
        "icustayids":  sample_icuids,
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_transition_model(dataset, n_state, n_action=5, n_history=4,
                           hidden=256, lr=1e-3, batch_size=256,
                           num_epochs=100, val_fraction=0.1,
                           model_type="nn", kl_weight=1e-4,
                           save_dir=None, device="cpu", log_every=10):
    """Train a transition model on (history, delta) pairs.

    Same interface as sepsis pipeline. n_action=5 (5 binary drug flags).

    Args:
        dataset:    dict from build_history_dataset()
        n_state:    number of state features (51 for broad)
        n_action:   number of action features (5 binary drug flags)
        n_history:  number of history timesteps
        model_type: 'nn', 'linear', 'lstm', or 'bnn'
    Returns:
        (model, train_losses, val_losses)
    """
    n_input  = n_history * (n_state + n_action)
    n_output = n_state

    if model_type == "bnn" and hidden == 256:
        hidden = 32  # per Raghu 2018

    logging.info("Training transition model [%s]:", model_type.upper())
    logging.info("  n_input=%d, n_output=%d, hidden=%d", n_input, n_output, hidden)
    logging.info("  epochs=%d, batch=%d, lr=%.1e", num_epochs, batch_size, lr)

    model     = create_transition_model(model_type, n_input, n_output,
                                        hidden=hidden, n_history=n_history).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    n     = len(dataset["history"])
    n_val = int(n * val_fraction)
    n_train = n - n_val

    perm    = np.random.RandomState(42).permutation(n)
    train_idx = perm[:n_train]
    val_idx   = perm[n_train:]

    X_train = torch.FloatTensor(dataset["history"][train_idx]).to(device)
    y_train = torch.FloatTensor(dataset["deltas"][train_idx]).to(device)
    X_val   = torch.FloatTensor(dataset["history"][val_idx]).to(device)
    y_val   = torch.FloatTensor(dataset["deltas"][val_idx]).to(device)

    logging.info("  Train: %d samples, Val: %d samples", n_train, n_val)
    logging.info("  Model parameters: %d",
                 sum(p.numel() for p in model.parameters()))

    train_losses   = []
    val_losses     = []
    best_val_loss  = float("inf")
    best_state_dict = None
    is_bnn         = model_type == "bnn"
    t0             = time.time()

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        shuffle_idx = np.random.permutation(n_train)

        for start in range(0, n_train, batch_size):
            end       = min(start + batch_size, n_train)
            batch_idx = shuffle_idx[start:end]

            pred     = model(X_train[batch_idx])
            mse_loss = criterion(pred, y_train[batch_idx])

            if is_bnn:
                loss = mse_loss + kl_weight * model.kl_divergence() / n_train
            else:
                loss = mse_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += mse_loss.item()
            n_batches  += 1

        train_loss = epoch_loss / n_batches
        train_losses.append(train_loss)

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
            best_val_loss   = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % log_every == 0:
            elapsed   = time.time() - t0
            remaining = elapsed / (epoch + 1) * (num_epochs - epoch - 1)
            logging.info(
                "  Epoch %d/%d (%.0f%%) | train_MSE=%.6f val_MSE=%.6f | "
                "best_val=%.6f | elapsed %.0fs | ETA %.0fs",
                epoch + 1, num_epochs, 100 * (epoch + 1) / num_epochs,
                train_loss, val_loss, best_val_loss, elapsed, remaining)

    model.load_state_dict(best_state_dict)
    model.to(device)
    logging.info("Training [%s] complete. Best val MSE=%.6f",
                 model_type.upper(), best_val_loss)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(best_state_dict, os.path.join(save_dir, "transition_model.pt"))
        with open(os.path.join(save_dir, "model_config.pkl"), "wb") as f:
            pickle.dump({
                "n_input":    n_input,
                "n_output":   n_output,
                "n_state":    n_state,
                "n_action":   n_action,
                "n_history":  n_history,
                "hidden":     hidden,
                "model_type": model_type,
            }, f)
        logging.info("  Model saved to %s", save_dir)

    return model, train_losses, val_losses


# ---------------------------------------------------------------------------
# Per-feature evaluation
# ---------------------------------------------------------------------------

def evaluate_per_feature(model, dataset, state_cols, device="cpu", batch_size=4096):
    """Compute per-feature MSE on the dataset.

    Returns dict: feature_name -> MSE, plus '__overall__'.
    """
    model.eval()
    X    = torch.FloatTensor(dataset["history"]).to(device)
    y    = dataset["deltas"]

    preds = []
    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            preds.append(model(X[start:end]).cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    mse_per_feature = {}
    for i, col in enumerate(state_cols):
        mse_per_feature[col] = float(np.mean((preds[:, i] - y[:, i]) ** 2))

    mse_per_feature["__overall__"] = float(np.mean((preds - y) ** 2))
    return mse_per_feature


# ---------------------------------------------------------------------------
# Simulator (rollout engine)
# ---------------------------------------------------------------------------

class ICUSimulator:
    """Autoregressive environment simulator for model-based RL.

    Maintains a history buffer and predicts next states using the
    learned transition model. Adapted from SepsisSimulator with n_action=5.

    Usage:
        sim = ICUSimulator(model, n_state, n_history=4)
        sim.reset(initial_states, initial_actions)
        for t in range(rollout_length):
            action = policy(sim.current_state)  # 5-bit binary array
            next_state, delta = sim.step(action)
    """

    def __init__(self, model, n_state, n_action=5, n_history=4, device="cpu"):
        self.model    = model
        self.model.eval()
        self.n_state  = n_state
        self.n_action = n_action
        self.n_history = n_history
        self.device   = device
        self.state_history  = []
        self.action_history = []

    def reset(self, initial_states, initial_actions):
        """Initialize with n_history historical states and binary actions.

        Args:
            initial_states:  (n_history, n_state) -- oldest first
            initial_actions: (n_history, 5) binary -- oldest first
        """
        assert len(initial_states) == self.n_history
        self.state_history  = [s.copy() for s in initial_states]
        self.action_history = [a.copy() for a in initial_actions]

    @property
    def current_state(self):
        return self.state_history[-1].copy()

    def _build_history_vector(self):
        h = []
        for k in range(self.n_history):
            idx = len(self.state_history) - 1 - k
            h.append(self.state_history[idx])
            h.append(self.action_history[idx])
        return np.concatenate(h).astype(np.float32)

    def step(self, action):
        """Advance one timestep.

        Args:
            action: (5,) binary array [vasopressor, ivfluid, antibiotic, sedation, diuretic]
        Returns:
            (next_state, delta) both (n_state,)
        """
        h        = self._build_history_vector()
        h_tensor = torch.FloatTensor(h).unsqueeze(0).to(self.device)
        with torch.no_grad():
            delta = self.model(h_tensor).cpu().numpy().squeeze(0)

        next_state = self.state_history[-1] + delta

        self.state_history.append(next_state.copy())
        self.action_history.append(action.copy())
        if len(self.state_history) > self.n_history:
            self.state_history  = self.state_history[-self.n_history:]
            self.action_history = self.action_history[-self.n_history:]

        return next_state, delta

    def rollout(self, actions_sequence):
        """Multi-step rollout given a sequence of 5-bit binary actions.

        Args:
            actions_sequence: (T, 5) array
        Returns:
            states: (T+1, n_state), deltas: (T, n_state)
        """
        states = [self.current_state.copy()]
        deltas = []
        for t in range(len(actions_sequence)):
            next_state, delta = self.step(actions_sequence[t])
            states.append(next_state.copy())
            deltas.append(delta.copy())
        return np.array(states), np.array(deltas)


def evaluate_rollouts(model, df, state_cols, n_history=4, n_rollout_steps=10,
                      n_patients=200, seed=42, device="cpu"):
    """Evaluate rollout quality against real patient trajectories.

    Uses actual clinician actions for rollout; compares predicted vs real next states.

    Returns dict with per-step MSE, per-feature MSE, n_patients, state_cols.
    """
    logging.info("Evaluating rollouts: %d patients, %d steps each",
                 n_patients, n_rollout_steps)

    n_state    = len(state_cols)
    rng        = np.random.RandomState(seed)
    states_raw = np.nan_to_num(df[state_cols].values.astype(np.float32), nan=0.0)
    actions_raw = _decode_actions(df["a"].values)  # (N, 5) binary
    icuids     = df["icustayid"].values

    # Find long-enough episodes
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

    logging.info("  %d episodes with length >= %d",
                 len(episodes), n_history + n_rollout_steps)

    if len(episodes) < n_patients:
        n_patients = len(episodes)
        logging.info("  Reduced to %d patients", n_patients)

    selected = rng.choice(len(episodes), n_patients, replace=False)

    sim           = ICUSimulator(model, n_state, n_action=5, n_history=n_history, device=device)
    per_step_mse  = np.zeros(n_rollout_steps, dtype=np.float64)
    per_step_fmse = np.zeros((n_rollout_steps, n_state), dtype=np.float64)
    n_valid       = 0

    for ep_idx in selected:
        ep_start, ep_end = episodes[ep_idx]

        init_states  = states_raw[ep_start:ep_start + n_history]
        init_actions = actions_raw[ep_start:ep_start + n_history]
        sim.reset(init_states, init_actions)

        rollout_actions = actions_raw[ep_start + n_history:ep_start + n_history + n_rollout_steps]
        actual_states   = states_raw[ep_start + n_history:ep_start + n_history + n_rollout_steps]

        predicted_states, _ = sim.rollout(rollout_actions)
        predicted           = predicted_states[1:]

        for t in range(n_rollout_steps):
            diff = predicted[t] - actual_states[t]
            per_step_mse[t]  += np.mean(diff ** 2)
            per_step_fmse[t] += diff ** 2

        n_valid += 1

    per_step_mse  /= n_valid
    per_step_fmse /= n_valid

    logging.info("  Rollout MSE by step:")
    for t in range(n_rollout_steps):
        logging.info("    Step %d: MSE=%.6f", t + 1, per_step_mse[t])

    return {
        "per_step_mse":         per_step_mse.tolist(),
        "per_step_feature_mse": per_step_fmse.tolist(),
        "n_patients":           n_valid,
        "n_steps":              n_rollout_steps,
        "state_cols":           state_cols,
    }
