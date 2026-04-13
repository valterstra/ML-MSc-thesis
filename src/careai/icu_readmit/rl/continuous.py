"""
Continuous-state RL for ICU readmission.

Adapted from src/careai/sepsis/rl/continuous.py. Key differences:
  - Input from rl_dataset_broad.parquet (s_*, s_next_* columns precomputed)
  - n_actions=32 (2^5 binary drug combos vs 25 for sepsis)
  - clip_reward=False -- preserve terminal ±15 signal
    (sepsis clipped to ±1 which would collapse our terminal reward)
  - No autoencoder -- state already well-conditioned (z-scored, clipped to ±5)
  - Checkpoints saved every checkpoint_every steps
"""
import logging
import os
import pickle
import time
from collections import namedtuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import DuelingDQN

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Identical to sepsis pipeline implementation.
    """

    def __init__(self, capacity, alpha=0.6, epsilon=0.01):
        self.capacity  = capacity
        self.alpha     = alpha
        self.epsilon   = epsilon
        self.buffer    = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos  = 0
        self.size = 0

    def add(self, transition, priority=None):
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0

        if self.size < self.capacity:
            self.buffer.append(transition)
            self.size += 1
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.9):
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        samples = [self.buffer[i] for i in indices]

        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(td) + self.epsilon) ** self.alpha


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_rl_data(df, state_cols, next_state_cols):
    """Extract numpy arrays from the precomputed rl_dataset parquet.

    Args:
        df:              DataFrame filtered to a single split
        state_cols:      list of s_* column names
        next_state_cols: list of s_next_* column names (same order as state_cols)

    Returns:
        dict with keys: states, actions, rewards, next_states, done
    """
    states      = df[state_cols].values.astype(np.float32)
    next_states = df[next_state_cols].values.astype(np.float32)
    actions     = df["a"].values.astype(int)
    rewards     = df["r"].values.astype(np.float32)
    done        = df["done"].values.astype(np.float32)

    return {
        "states":      states,
        "actions":     actions,
        "rewards":     rewards,
        "next_states": next_states,
        "done":        done,
    }


# ---------------------------------------------------------------------------
# DDQN training
# ---------------------------------------------------------------------------

def train_dqn(data, n_state, n_actions=32, hidden=128, leaky_slope=0.01,
              lr=1e-4, gamma=0.99, tau=0.001, batch_size=32, num_steps=100000,
              reward_threshold=20, reg_lambda=5.0,
              per_alpha=0.6, per_epsilon=0.01, beta_start=0.9,
              save_dir=None, checkpoint_every=20000,
              device="cpu", log_every=5000):
    """Train Dueling Double DQN with Prioritized Experience Replay.

    Adapted from sepsis pipeline. Key change: no reward clipping to [-1,1]
    so the terminal ±15 signal is preserved.

    Args:
        data:             dict from prepare_rl_data()
        n_state:          number of state features
        n_actions:        action space size (32)
        reward_threshold: Q-value regularisation ceiling (20, slightly above ±15)
        save_dir:         directory for model + checkpoints
    Returns:
        (main_net, actions_train, q_values_train)
    """
    logging.info("DDQN: %d steps, n_state=%d, n_actions=%d, lr=%.1e, gamma=%.2f, hidden=%d",
                 num_steps, n_state, n_actions, lr, gamma, hidden)

    main_net   = DuelingDQN(n_state, n_actions, hidden, leaky_slope).to(device)
    target_net = DuelingDQN(n_state, n_actions, hidden, leaky_slope).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=lr)

    # Fill replay buffer
    n = len(data["states"])
    buffer = PrioritizedReplayBuffer(n, alpha=per_alpha, epsilon=per_epsilon)
    for i in range(n):
        t = Transition(
            data["states"][i],
            data["actions"][i],
            data["rewards"][i],
            data["next_states"][i],
            data["done"][i],
        )
        buffer.add(t, abs(data["rewards"][i]) + per_epsilon)

    logging.info("  Replay buffer: %d transitions", buffer.size)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    losses = []
    t0 = time.time()

    for step in range(num_steps):
        main_net.train()
        samples, indices, is_weights = buffer.sample(batch_size, beta=beta_start)
        is_weights = is_weights.to(device)

        states      = torch.FloatTensor(np.array([s.state      for s in samples])).to(device)
        actions     = torch.LongTensor( [s.action     for s in samples]).to(device)
        rewards     = torch.FloatTensor([s.reward     for s in samples]).to(device)
        next_states = torch.FloatTensor(np.array([s.next_state for s in samples])).to(device)
        dones       = torch.FloatTensor([s.done       for s in samples]).to(device)

        # Current Q
        q_values = main_net(states)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target: action from main, value from target
        with torch.no_grad():
            next_q_main   = main_net(next_states)
            next_actions  = next_q_main.argmax(dim=1)
            next_q_target = target_net(next_states)
            next_q        = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)
            next_q        = next_q.clamp(-reward_threshold, reward_threshold)
            target        = rewards + gamma * next_q * (1 - dones)

        td_error = target - q_taken

        # PER-weighted MSE loss + Q-value regularisation
        loss  = (is_weights * td_error.pow(2)).mean()
        reg   = torch.clamp(q_values.abs() - reward_threshold, min=0).sum()
        loss += reg_lambda * reg / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_net.parameters(), 10.0)
        optimizer.step()

        buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        # Soft target update
        for p_m, p_t in zip(main_net.parameters(), target_net.parameters()):
            p_t.data.copy_(tau * p_m.data + (1 - tau) * p_t.data)

        losses.append(loss.item())

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            avg_loss  = np.mean(losses[-log_every:])
            pct       = 100 * (step + 1) / num_steps
            logging.info("  DDQN %d/%d (%.0f%%) | loss=%.4f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, avg_loss, steps_sec, elapsed, eta)

        if save_dir and checkpoint_every and (step + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_{step+1}.pt")
            torch.save(main_net.state_dict(), ckpt_path)
            logging.info("  Checkpoint saved: %s", ckpt_path)

    # Final policy + Q-values over full training set
    main_net.eval()
    with torch.no_grad():
        all_states = torch.FloatTensor(data["states"]).to(device)
        all_q = []
        for i in range(0, len(all_states), 4096):
            all_q.append(main_net(all_states[i:i+4096]).cpu().numpy())
        all_q      = np.concatenate(all_q, axis=0)
        all_actions = all_q.argmax(axis=1)

    logging.info("DDQN training complete. Mean Q=%.4f", all_q.mean())

    if save_dir:
        torch.save(main_net.state_dict(), os.path.join(save_dir, "dqn_model.pt"))
        with open(os.path.join(save_dir, "dqn_actions.pkl"), "wb") as f:
            pickle.dump(all_actions, f)
        with open(os.path.join(save_dir, "dqn_q_values.pkl"), "wb") as f:
            pickle.dump(all_q, f)
        with open(os.path.join(save_dir, "dqn_losses.pkl"), "wb") as f:
            pickle.dump(losses, f)
        logging.info("  Model saved to %s", save_dir)

    return main_net, all_actions, all_q


# ---------------------------------------------------------------------------
# SARSA physician baseline
# ---------------------------------------------------------------------------

def train_sarsa_physician(data, n_state, n_actions=32, hidden=128,
                          lr=1e-4, gamma=0.99, tau=0.001, batch_size=32,
                          num_steps=80000, reward_threshold=20, reg_lambda=5.0,
                          per_alpha=0.6, per_epsilon=0.01, beta_start=0.9,
                          save_dir=None, checkpoint_every=20000,
                          device="cpu", log_every=5000):
    """Train SARSA to learn the physician's Q-function (on-policy).

    Same architecture as DDQN but uses physician's actual next action
    instead of argmax. Provides a baseline for off-policy evaluation.

    Args:
        data: dict from prepare_rl_data()
    Returns:
        (main_net, actions_train, q_values_train)
    """
    logging.info("SARSA physician: %d steps, n_state=%d, n_actions=%d",
                 num_steps, n_state, n_actions)

    main_net   = DuelingDQN(n_state, n_actions, hidden, 0.01).to(device)
    target_net = DuelingDQN(n_state, n_actions, hidden, 0.01).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=lr)

    # Precompute physician's next action for each transition
    n = len(data["states"])
    next_actions_arr = np.zeros(n, dtype=int)
    for i in range(n - 1):
        if data["done"][i] == 0:
            next_actions_arr[i] = data["actions"][i + 1]

    buffer = PrioritizedReplayBuffer(n, alpha=per_alpha, epsilon=per_epsilon)
    for i in range(n):
        t = Transition(
            data["states"][i],
            data["actions"][i],
            data["rewards"][i],
            data["next_states"][i],
            data["done"][i],
        )
        buffer.add(t, abs(data["rewards"][i]) + per_epsilon)

    logging.info("  Replay buffer: %d transitions", buffer.size)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    losses = []
    t0 = time.time()

    for step in range(num_steps):
        main_net.train()
        samples, indices, is_weights = buffer.sample(batch_size, beta=beta_start)
        is_weights = is_weights.to(device)

        states      = torch.FloatTensor(np.array([s.state      for s in samples])).to(device)
        actions     = torch.LongTensor( [s.action     for s in samples]).to(device)
        rewards     = torch.FloatTensor([s.reward     for s in samples]).to(device)
        next_states = torch.FloatTensor(np.array([s.next_state for s in samples])).to(device)
        dones       = torch.FloatTensor([s.done       for s in samples]).to(device)

        q_values = main_net(states)
        q_taken  = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            next_q            = target_net(next_states)
            phys_next_actions = torch.LongTensor(
                [next_actions_arr[idx] for idx in indices]
            ).to(device)
            next_q_taken = next_q.gather(1, phys_next_actions.unsqueeze(1)).squeeze(1)
            next_q_taken = next_q_taken.clamp(-reward_threshold, reward_threshold)
            target = rewards + gamma * next_q_taken * (1 - dones)

        td_error = target - q_taken
        loss  = (is_weights * td_error.pow(2)).mean()
        reg   = torch.clamp(q_values.abs() - reward_threshold, min=0).sum()
        loss += reg_lambda * reg / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_net.parameters(), 10.0)
        optimizer.step()

        buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        for p_m, p_t in zip(main_net.parameters(), target_net.parameters()):
            p_t.data.copy_(tau * p_m.data + (1 - tau) * p_t.data)

        losses.append(loss.item())

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            pct       = 100 * (step + 1) / num_steps
            logging.info("  SARSA %d/%d (%.0f%%) | loss=%.4f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, loss.item(), steps_sec, elapsed, eta)

        if save_dir and checkpoint_every and (step + 1) % checkpoint_every == 0:
            ckpt_path = os.path.join(save_dir, f"checkpoint_{step+1}.pt")
            torch.save(main_net.state_dict(), ckpt_path)
            logging.info("  Checkpoint saved: %s", ckpt_path)

    # Final Q-values
    main_net.eval()
    with torch.no_grad():
        all_states = torch.FloatTensor(data["states"]).to(device)
        all_q = []
        for i in range(0, len(all_states), 4096):
            all_q.append(main_net(all_states[i:i+4096]).cpu().numpy())
        all_q       = np.concatenate(all_q, axis=0)
        all_actions = all_q.argmax(axis=1)

    logging.info("SARSA physician complete. Mean Q=%.4f", all_q.mean())

    if save_dir:
        torch.save(main_net.state_dict(), os.path.join(save_dir, "sarsa_phys_model.pt"))
        with open(os.path.join(save_dir, "phys_actions.pkl"), "wb") as f:
            pickle.dump(all_actions, f)
        with open(os.path.join(save_dir, "phys_q_values.pkl"), "wb") as f:
            pickle.dump(all_q, f)
        with open(os.path.join(save_dir, "sarsa_losses.pkl"), "wb") as f:
            pickle.dump(losses, f)

    return main_net, all_actions, all_q


# ---------------------------------------------------------------------------
# Inference utility
# ---------------------------------------------------------------------------

def compute_q_values(model, states_np, device="cpu", batch_size=4096):
    """Compute Q-values for a numpy state array in batches.

    Returns:
        q_values:  (n, n_actions) numpy array
        actions:   (n,) argmax actions
    """
    model.eval()
    all_q = []
    with torch.no_grad():
        states_t = torch.FloatTensor(states_np).to(device)
        for i in range(0, len(states_t), batch_size):
            all_q.append(model(states_t[i:i+batch_size]).cpu().numpy())
    all_q = np.concatenate(all_q, axis=0)
    return all_q, all_q.argmax(axis=1)
