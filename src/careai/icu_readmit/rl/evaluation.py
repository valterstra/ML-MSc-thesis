"""
Offline policy evaluation for ICU readmission RL.

Adapted from src/careai/sepsis/rl/evaluation.py. Key differences:
  - n_actions=32 (2^5 binary drug combos vs 25 for sepsis)
  - Action encoded as 5-bit binary vector decoded from integer 0-31
    (vasopressor=bit0, ivfluid=bit1, antibiotic=bit2, sedation=bit3, diuretic=bit4)
    vs sepsis which used [iv_level/4, vaso_level/4] 2-feature normalized
  - No orig_data argument -- action encoding is derived from data["actions"] directly
  - value_clip=40 (DR clips at +-40 vs +-20; our Q range is wider with n_actions=32)
"""
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import EnvModel, PhysicianPolicy, RewardEstimator


# ---------------------------------------------------------------------------
# Action encoding
# ---------------------------------------------------------------------------

def decode_actions(actions_int):
    """Decode integer actions 0-31 to 5-bit binary arrays.

    Bit order: vasopressor=0, ivfluid=1, antibiotic=2, sedation=3, diuretic=4

    Args:
        actions_int: (N,) int array, values in [0, 31]
    Returns:
        (N, 5) float32 array of binary drug flags
    """
    arr = np.asarray(actions_int, dtype=np.int32)
    return ((arr[:, None] >> np.arange(5)) & 1).astype(np.float32)


# ---------------------------------------------------------------------------
# Physician policy (supervised pi(a|s))
# ---------------------------------------------------------------------------

def train_physician_policy(data, n_state, n_actions=32, hidden=64,
                           batch_size=64, num_steps=35000, reg_constant=0.1,
                           save_dir=None, device="cpu", log_every=5000):
    """Train supervised policy: pi(a|s) from physician actions.

    Args:
        data: dict from prepare_rl_data()
    Returns:
        trained PhysicianPolicy model
    """
    logging.info("Physician policy: %d steps, hidden=%d, n_actions=%d",
                 num_steps, hidden, n_actions)

    model = PhysicianPolicy(n_state, n_actions, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=reg_constant)
    criterion = nn.CrossEntropyLoss()

    states  = torch.FloatTensor(data["states"]).to(device)
    actions = torch.LongTensor(data["actions"]).to(device)
    n = len(states)
    t0 = time.time()

    for step in range(num_steps):
        model.train()
        idx = np.random.choice(n, batch_size, replace=False)

        logits = model(states[idx])
        loss   = criterion(logits, actions[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            pct       = 100 * (step + 1) / num_steps
            with torch.no_grad():
                acc = (logits.argmax(dim=1) == actions[idx]).float().mean().item()
            logging.info(
                "  Physician policy %d/%d (%.0f%%) | loss=%.4f acc=%.3f | "
                "%.1f steps/s | elapsed %.0fs | ETA %.0fs",
                step + 1, num_steps, pct, loss.item(), acc, steps_sec, elapsed, eta)

    model.eval()
    logging.info("Physician policy training complete")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "physician_policy_model.pt"))

    return model


def predict_physician_probs(model, data, device="cpu", save_path=None):
    """Predict action probabilities for all states.

    Returns:
        (N, n_actions) float32 array of softmax probabilities
    """
    model.eval()
    states    = torch.FloatTensor(data["states"]).to(device)
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            probs = model.predict_proba(states[i:i + 4096])
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_probs, f)

    return all_probs


# ---------------------------------------------------------------------------
# Reward estimator R(s, a)
# ---------------------------------------------------------------------------

def train_reward_estimator(data, n_state, n_action_features=5, hidden=128,
                           batch_size=64, num_steps=30000, lr=1e-3,
                           save_dir=None, device="cpu", log_every=5000):
    """Train reward function approximator R(s, a).

    Action is decoded from integer to 5-bit binary vector internally.
    """
    logging.info("Reward estimator: %d steps", num_steps)

    model     = RewardEstimator(n_state, n_action_features, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    states  = torch.FloatTensor(data["states"]).to(device)
    rewards = torch.FloatTensor(data["rewards"]).to(device)
    actions_bin = torch.FloatTensor(decode_actions(data["actions"])).to(device)
    n = len(states)
    t0 = time.time()

    for step in range(num_steps):
        model.train()
        idx = np.random.choice(n, batch_size, replace=False)

        pred = model(states[idx], actions_bin[idx])
        loss = criterion(pred, rewards[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            pct       = 100 * (step + 1) / num_steps
            logging.info(
                "  Reward est. %d/%d (%.0f%%) | loss=%.4f | "
                "%.1f steps/s | elapsed %.0fs | ETA %.0fs",
                step + 1, num_steps, pct, loss.item(), steps_sec, elapsed, eta)

    model.eval()
    logging.info("Reward estimator training complete")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "reward_estimator_model.pt"))

    return model


def predict_rewards(model, data, device="cpu", save_path=None):
    """Predict rewards for all state-action pairs."""
    model.eval()
    states      = torch.FloatTensor(data["states"]).to(device)
    actions_bin = torch.FloatTensor(decode_actions(data["actions"])).to(device)
    all_rewards = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            r = model(states[i:i + 4096], actions_bin[i:i + 4096])
            all_rewards.append(r.cpu().numpy())
    all_rewards = np.concatenate(all_rewards, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_rewards, f)

    return all_rewards


# ---------------------------------------------------------------------------
# Environment model P(s'|s, a)
# ---------------------------------------------------------------------------

def train_env_model(data, n_state, n_action_features=5, hidden=500,
                    batch_size=32, num_steps=60000, lr=1e-3, noise_std=0.03,
                    save_dir=None, device="cpu", log_every=5000):
    """Train transition dynamics model: P(s'|s, a).

    Predicts state delta (s' - s) on non-terminal transitions.
    Gaussian noise added to states during training for robustness.
    """
    logging.info("Environment model: %d steps, hidden=%d", num_steps, hidden)

    model     = EnvModel(n_state, n_action_features, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    states      = torch.FloatTensor(data["states"]).to(device)
    next_states = torch.FloatTensor(data["next_states"]).to(device)
    dones       = torch.FloatTensor(data["done"]).to(device)
    actions_bin = torch.FloatTensor(decode_actions(data["actions"])).to(device)
    deltas      = next_states - states

    # Train only on non-terminal transitions
    non_terminal = (data["done"] == 0).astype(bool)
    nt_indices   = np.where(non_terminal)[0]
    logging.info("  %d non-terminal transitions for training", len(nt_indices))

    t0 = time.time()

    for step in range(num_steps):
        model.train()
        idx = np.random.choice(nt_indices, batch_size, replace=False)

        noisy_states = states[idx] + torch.randn_like(states[idx]) * noise_std
        pred_delta   = model(noisy_states, actions_bin[idx])
        loss         = criterion(pred_delta, deltas[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            pct       = 100 * (step + 1) / num_steps
            logging.info(
                "  Env model %d/%d (%.0f%%) | loss=%.6f | "
                "%.1f steps/s | elapsed %.0fs | ETA %.0fs",
                step + 1, num_steps, pct, loss.item(), steps_sec, elapsed, eta)

    model.eval()
    logging.info("Environment model training complete")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(save_dir, "env_model.pt"))

    return model


def predict_next_states(model, data, device="cpu", save_path=None):
    """Predict next states for all transitions."""
    model.eval()
    states      = torch.FloatTensor(data["states"]).to(device)
    actions_bin = torch.FloatTensor(decode_actions(data["actions"])).to(device)
    all_next    = []
    with torch.no_grad():
        for i in range(0, len(states), 4096):
            delta  = model(states[i:i + 4096], actions_bin[i:i + 4096])
            next_s = states[i:i + 4096] + delta
            all_next.append(next_s.cpu().numpy())
    all_next = np.concatenate(all_next, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_next, f)

    return all_next


# ---------------------------------------------------------------------------
# Doubly Robust off-policy evaluation
# ---------------------------------------------------------------------------

def doubly_robust_evaluation(data, agent_actions, agent_q_values,
                             physician_probs, reward_estimates,
                             gamma=0.99, value_clip=40.0):
    """Doubly Robust off-policy value estimation.

    Adapted from sepsis pipeline. value_clip=40 (wider than sepsis's 20)
    because our Q-value range with 32 actions and +-15 terminal is larger.

    Args:
        data:             dict from prepare_rl_data()
        agent_actions:    (N,) int array -- greedy actions from the RL agent
        agent_q_values:   (N, 32) float array -- Q-values from the RL agent
        physician_probs:  (N, 32) float array -- pi(a|s) from physician policy
        reward_estimates: (N,) float array -- R(s,a) from reward estimator
        gamma:            discount factor (default 0.99)
        value_clip:       clip trajectory values outside +-value_clip

    Returns:
        dict with mean, std, n_trajectories, n_valid, values array
    """
    logging.info("Doubly Robust evaluation: gamma=%.2f, value_clip=%.1f",
                 gamma, value_clip)

    actions_taken = data["actions"]
    rewards       = data["rewards"]
    done          = data["done"]
    n             = len(actions_taken)

    # Build trajectory boundaries from done flags
    trajectories = []
    start = 0
    for i in range(1, n):
        if done[i - 1] == 1.0:
            trajectories.append((start, i))
            start = i
    if start < n:
        trajectories.append((start, n))

    logging.info("  %d trajectories found", len(trajectories))

    trajectory_values = []

    for traj_start, traj_end in trajectories:
        v_dr = 0.0

        for t in range(traj_end - 1, traj_start - 1, -1):
            a_phys  = int(actions_taken[t])
            a_agent = int(agent_actions[t])

            prob_phys = physician_probs[t, a_phys]
            if prob_phys < 1e-6:
                prob_phys = 1e-6

            rho = (1.0 / prob_phys) if (a_agent == a_phys) else 0.0

            q_agent = agent_q_values[t, a_agent]
            r_est   = reward_estimates[t]
            r_act   = rewards[t]

            # DR update: V = R_est + rho * (R_actual + gamma*V - Q_agent)
            v_dr = r_est + rho * (r_act + gamma * v_dr - q_agent)

        if abs(v_dr) <= value_clip:
            trajectory_values.append(v_dr)

    trajectory_values = np.array(trajectory_values)
    mean_val = trajectory_values.mean()
    std_val  = trajectory_values.std()

    logging.info("  DR value: mean=%.4f, std=%.4f, n_valid=%d/%d",
                 mean_val, std_val, len(trajectory_values), len(trajectories))

    return {
        "mean":           float(mean_val),
        "std":            float(std_val),
        "n_trajectories": len(trajectories),
        "n_valid":        len(trajectory_values),
        "values":         trajectory_values,
    }
