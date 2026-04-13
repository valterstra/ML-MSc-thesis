"""
Offline policy evaluation for sepsis RL.
Ported from sepsisrl/eval/ notebooks.

Handles: physician policy learning, reward estimation, environment model,
doubly robust off-policy evaluation.
"""
import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import PhysicianPolicy, RewardEstimator, EnvModel


def train_physician_policy(data, n_state, n_actions=25, hidden=64,
                           batch_size=64, num_steps=35000, reg_constant=0.1,
                           save_dir=None, device="cpu", log_every=5000):
    """Train supervised policy: pi(a|s) from physician actions.

    Ported from sepsisrl/eval/physician_policy_tf.ipynb.

    Returns (model, action_probs_dict) where action_probs_dict has
    train/val/test probability arrays.
    """
    logging.info("Physician policy: %d steps, hidden=%d", num_steps, hidden)

    model = PhysicianPolicy(n_state, n_actions, hidden).to(device)
    optimizer = optim.Adam(model.parameters(), weight_decay=reg_constant)
    criterion = nn.CrossEntropyLoss()

    states = torch.FloatTensor(data["states"]).to(device)
    actions = torch.LongTensor(data["actions"]).to(device)
    n = len(states)
    t_train_start = time.time()

    for step in range(num_steps):
        model.train()
        idx = np.random.choice(n, batch_size, replace=False)
        batch_states = states[idx]
        batch_actions = actions[idx]

        logits = model(batch_states)
        loss = criterion(logits, batch_actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_train_start
            steps_sec = (step + 1) / elapsed
            remaining = (num_steps - step - 1) / steps_sec
            pct = 100 * (step + 1) / num_steps
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc = (preds == batch_actions).float().mean().item()
            logging.info("  Physician policy step %d/%d (%.0f%%) | loss=%.4f, acc=%.3f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, loss.item(), acc, steps_sec, elapsed, remaining)

    model.eval()
    logging.info("Physician policy training complete")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/physician_policy_model.pt")

    return model


def predict_physician_probs(model, data, device="cpu", save_path=None):
    """Predict action probabilities for all states."""
    model.eval()
    states = torch.FloatTensor(data["states"]).to(device)
    all_probs = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, len(states), bs):
            probs = model.predict_proba(states[i:i + bs])
            all_probs.append(probs.cpu().numpy())
    all_probs = np.concatenate(all_probs, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_probs, f)

    return all_probs


def train_reward_estimator(data, orig_data, n_state, n_actions=25,
                           hidden=128, batch_size=64, num_steps=30000,
                           lr=1e-3, save_dir=None, device="cpu",
                           log_every=5000):
    """Train reward function approximator R(s, a).

    Ported from sepsisrl/eval/reward_estimator_new.ipynb.
    Uses SOFA/lactate shaped rewards for non-terminal states,
    and binary classification for terminal states.
    """
    logging.info("Reward estimator: %d steps", num_steps)

    model = RewardEstimator(n_state, n_action_features=2, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # Prepare data: state + normalized action (iv/4, vaso/4)
    states = torch.FloatTensor(data["states"]).to(device)
    rewards = torch.FloatTensor(data["rewards"]).to(device)

    # Normalize action features to [0, 1] range
    iv_input = orig_data.get("iv_input", np.zeros(len(data["states"])))
    vaso_input = orig_data.get("vaso_input", np.zeros(len(data["states"])))
    actions = torch.FloatTensor(np.column_stack([
        iv_input.astype(float) / 4.0,
        vaso_input.astype(float) / 4.0,
    ])).to(device)

    n = len(states)
    t_train_start = time.time()

    for step in range(num_steps):
        model.train()
        idx = np.random.choice(n, batch_size, replace=False)

        pred = model(states[idx], actions[idx])
        loss = criterion(pred, rewards[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_train_start
            steps_sec = (step + 1) / elapsed
            remaining = (num_steps - step - 1) / steps_sec
            pct = 100 * (step + 1) / num_steps
            logging.info("  Reward est. step %d/%d (%.0f%%) | loss=%.4f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, loss.item(), steps_sec, elapsed, remaining)

    model.eval()
    logging.info("Reward estimator training complete")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/reward_estimator_model.pt")

    return model


def predict_rewards(model, data, orig_data, device="cpu", save_path=None):
    """Predict rewards for all state-action pairs."""
    model.eval()
    states = torch.FloatTensor(data["states"]).to(device)
    iv_input = orig_data.get("iv_input", np.zeros(len(data["states"])))
    vaso_input = orig_data.get("vaso_input", np.zeros(len(data["states"])))
    actions = torch.FloatTensor(np.column_stack([
        iv_input.astype(float) / 4.0,
        vaso_input.astype(float) / 4.0,
    ])).to(device)

    all_rewards = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, len(states), bs):
            r = model(states[i:i + bs], actions[i:i + bs])
            all_rewards.append(r.cpu().numpy())
    all_rewards = np.concatenate(all_rewards, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_rewards, f)

    return all_rewards


def train_env_model(data, orig_data, n_state, hidden=500, batch_size=32,
                    num_steps=60000, lr=1e-3, noise_std=0.03,
                    save_dir=None, device="cpu", log_every=5000):
    """Train transition dynamics model: P(s'|s,a).

    Ported from sepsisrl/eval/env_model_regression_for_eval.ipynb.
    Predicts state DELTA, adds to current state.
    """
    logging.info("Environment model: %d steps, hidden=%d", num_steps, hidden)

    model = EnvModel(n_state, n_action_features=2, hidden=hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    states = torch.FloatTensor(data["states"]).to(device)
    next_states = torch.FloatTensor(data["next_states"]).to(device)
    dones = torch.FloatTensor(data["done"]).to(device)

    iv_input = orig_data.get("iv_input", np.zeros(len(data["states"])))
    vaso_input = orig_data.get("vaso_input", np.zeros(len(data["states"])))
    actions = torch.FloatTensor(np.column_stack([
        iv_input.astype(float) / 4.0,
        vaso_input.astype(float) / 4.0,
    ])).to(device)

    # Target: state delta (next - current)
    deltas = next_states - states
    n = len(states)

    # Only train on non-terminal transitions
    non_terminal = (dones == 0).cpu().numpy().astype(bool)
    nt_indices = np.where(non_terminal)[0]
    logging.info("  %d non-terminal transitions for training", len(nt_indices))

    t_train_start = time.time()
    for step in range(num_steps):
        model.train()
        idx = np.random.choice(nt_indices, batch_size, replace=False)

        # Add Gaussian noise to states for robustness
        noisy_states = states[idx] + torch.randn_like(states[idx]) * noise_std

        pred_delta = model(noisy_states, actions[idx])
        loss = criterion(pred_delta, deltas[idx])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_train_start
            steps_sec = (step + 1) / elapsed
            remaining = (num_steps - step - 1) / steps_sec
            pct = 100 * (step + 1) / num_steps
            logging.info("  Env model step %d/%d (%.0f%%) | loss=%.6f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, loss.item(), steps_sec, elapsed, remaining)

    model.eval()
    logging.info("Environment model training complete")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/env_model.pt")

    return model


def predict_next_states(model, data, orig_data, device="cpu", save_path=None):
    """Predict next states for all transitions."""
    model.eval()
    states = torch.FloatTensor(data["states"]).to(device)
    iv_input = orig_data.get("iv_input", np.zeros(len(data["states"])))
    vaso_input = orig_data.get("vaso_input", np.zeros(len(data["states"])))
    actions = torch.FloatTensor(np.column_stack([
        iv_input.astype(float) / 4.0,
        vaso_input.astype(float) / 4.0,
    ])).to(device)

    all_next = []
    bs = 4096
    with torch.no_grad():
        for i in range(0, len(states), bs):
            delta = model(states[i:i + bs], actions[i:i + bs])
            next_s = states[i:i + bs] + delta
            all_next.append(next_s.cpu().numpy())
    all_next = np.concatenate(all_next, axis=0)

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(all_next, f)

    return all_next


def doubly_robust_evaluation(data, agent_actions, agent_q_values,
                             physician_probs, reward_estimates,
                             gamma=0.99, value_clip=20.0):
    """Doubly Robust off-policy value estimation.

    Ported from sepsisrl/eval/doubly_robust.ipynb.

    Args:
        data: dict from prepare_rl_data() with icustayid info
        agent_actions: array of agent policy actions per timestep
        agent_q_values: array of agent Q-values (n_timesteps x n_actions)
        physician_probs: array of physician policy probs (n_timesteps x n_actions)
        reward_estimates: array of estimated rewards per timestep
        gamma: discount factor
        value_clip: clip trajectory values outside [-clip, clip]

    Returns:
        dict with mean, std, per-trajectory values
    """
    logging.info("Doubly Robust evaluation: gamma=%.2f", gamma)

    actions_taken = data["actions"]  # physician's actual actions
    rewards = data["rewards"]
    done = data["done"]
    n = len(actions_taken)

    # Build trajectory boundaries
    trajectories = []
    start = 0
    for i in range(1, n):
        if done[i - 1] == 1.0:
            trajectories.append((start, i))
            start = i
    if start < n:
        trajectories.append((start, n))

    logging.info("  %d trajectories", len(trajectories))

    trajectory_values = []

    for traj_start, traj_end in trajectories:
        # Process backward through trajectory
        v_dr = 0.0

        for t in range(traj_end - 1, traj_start - 1, -1):
            a_phys = int(actions_taken[t])
            a_agent = int(agent_actions[t])

            # Importance weight
            prob_phys = physician_probs[t, a_phys]
            if prob_phys < 1e-6:
                prob_phys = 1e-6  # avoid division by zero

            # rho = 1/pi(a|s) if agent agrees with physician, else 0
            if a_agent == a_phys:
                rho = 1.0 / prob_phys
            else:
                rho = 0.0

            # Agent's Q-value for its chosen action
            q_agent = agent_q_values[t, a_agent]

            # Estimated reward
            r_est = reward_estimates[t]
            r_actual = rewards[t]

            # DR recursion
            v_dr = r_est + rho * (r_actual + gamma * v_dr - q_agent)

        # Clip extreme values
        if abs(v_dr) <= value_clip:
            trajectory_values.append(v_dr)

    trajectory_values = np.array(trajectory_values)
    mean_val = trajectory_values.mean()
    std_val = trajectory_values.std()

    logging.info("  DR value: mean=%.4f, std=%.4f, n_valid=%d/%d",
                 mean_val, std_val, len(trajectory_values), len(trajectories))

    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "n_trajectories": len(trajectories),
        "n_valid": len(trajectory_values),
        "values": trajectory_values,
    }
