"""
Continuous-state RL for sepsis treatment.
Ported from sepsisrl/continuous/ notebooks.

Handles: DQN training (Double Dueling with PER), autoencoder training,
SARSA physician baseline.
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

from .networks import DuelingDQN, SparseAutoencoder

Transition = namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])

# ── Prioritized Experience Replay ─────────────────────────────────────

class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer.

    Ported from sepsisrl/continuous/q_network.ipynb.
    """

    def __init__(self, capacity, alpha=0.6, epsilon=0.01):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
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

        # Importance sampling weights
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()

        return samples, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(td) + self.epsilon) ** self.alpha


# ── Dataset preparation ───────────────────────────────────────────────

def prepare_rl_data(df, state_cols, device="cpu"):
    """Extract state/action/reward/next_state/done from DataFrame.

    Returns dict with numpy arrays ready for RL training.
    """
    states = df[state_cols].values.astype(np.float32)
    # Replace NaN with 0 (after normalization, 0 is roughly the mean)
    states = np.nan_to_num(states, nan=0.0)

    actions = df["action_id"].values.astype(int)
    rewards = df["reward"].values.astype(np.float32)
    icuids = df["icustayid"].values

    # Build next_state and done arrays
    n = len(df)
    next_states = np.zeros_like(states)
    done = np.zeros(n, dtype=np.float32)

    for i in range(n - 1):
        if icuids[i + 1] == icuids[i]:
            next_states[i] = states[i + 1]
        else:
            done[i] = 1.0  # terminal

    done[-1] = 1.0  # last row is always terminal

    return {
        "states": states,
        "actions": actions,
        "rewards": rewards,
        "next_states": next_states,
        "done": done,
    }


# ── DQN Training ──────────────────────────────────────────────────────

def train_dqn(data, n_state, n_actions=25, hidden=128, leaky_slope=0.01,
              lr=1e-4, gamma=0.99, tau=0.001, batch_size=32, num_steps=60000,
              reward_threshold=20, reg_lambda=5.0, clip_reward=True,
              per_alpha=0.6, per_epsilon=0.01, beta_start=0.9,
              save_dir=None, device="cpu", log_every=5000):
    """Train Dueling Double DQN with Prioritized Experience Replay.

    Ported from sepsisrl/continuous/q_network.ipynb.

    Args:
        data: dict from prepare_rl_data()
        n_state: number of state features
        save_dir: directory to save model checkpoints
    Returns:
        (main_net, actions_train, q_values_train)
    """
    logging.info("DQN training: %d steps, lr=%.1e, gamma=%.2f, hidden=%d",
                 num_steps, lr, gamma, hidden)

    main_net = DuelingDQN(n_state, n_actions, hidden, leaky_slope).to(device)
    target_net = DuelingDQN(n_state, n_actions, hidden, leaky_slope).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=lr)

    # Build replay buffer
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
        priority = abs(data["rewards"][i]) + per_epsilon
        buffer.add(t, priority)

    logging.info("  Replay buffer filled: %d transitions", buffer.size)

    losses = []
    t_train_start = time.time()
    for step in range(num_steps):
        main_net.train()
        samples, indices, is_weights = buffer.sample(batch_size, beta=beta_start)
        is_weights = is_weights.to(device)

        states = torch.FloatTensor(np.array([s.state for s in samples])).to(device)
        actions = torch.LongTensor([s.action for s in samples]).to(device)
        rewards = torch.FloatTensor([s.reward for s in samples]).to(device)
        next_states = torch.FloatTensor(np.array([s.next_state for s in samples])).to(device)
        dones = torch.FloatTensor([s.done for s in samples]).to(device)

        if clip_reward:
            rewards = rewards.clamp(-1.0, 1.0)

        # Current Q-values
        q_values = main_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN: select action from main, evaluate with target
        with torch.no_grad():
            next_q_main = main_net(next_states)
            next_actions = next_q_main.argmax(dim=1)
            next_q_target = target_net(next_states)
            next_q = next_q_target.gather(1, next_actions.unsqueeze(1)).squeeze(1)

            # Clip target Q-values
            next_q = next_q.clamp(-reward_threshold, reward_threshold)

            target = rewards + gamma * next_q * (1 - dones)

        # TD error
        td_error = target - q_taken

        # PER weighted loss
        loss = (is_weights * td_error.pow(2)).mean()

        # Regularization: penalize Q-values that exceed threshold
        reg = torch.clamp(q_values.abs() - reward_threshold, min=0).sum()
        loss += reg_lambda * reg / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_net.parameters(), 10.0)
        optimizer.step()

        # Update priorities
        buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        # Soft update target network
        for p_main, p_target in zip(main_net.parameters(), target_net.parameters()):
            p_target.data.copy_(tau * p_main.data + (1 - tau) * p_target.data)

        losses.append(loss.item())

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_train_start
            steps_sec = (step + 1) / elapsed
            remaining = (num_steps - step - 1) / steps_sec
            avg_loss = np.mean(losses[-log_every:])
            pct = 100 * (step + 1) / num_steps
            logging.info("  DQN step %d/%d (%.0f%%) | loss=%.4f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, avg_loss, steps_sec, elapsed, remaining)

    # Extract final policy and Q-values
    main_net.eval()
    with torch.no_grad():
        all_states = torch.FloatTensor(data["states"]).to(device)
        # Process in batches to avoid OOM
        all_q = []
        bs = 4096
        for i in range(0, len(all_states), bs):
            batch = all_states[i:i + bs]
            all_q.append(main_net(batch).cpu().numpy())
        all_q = np.concatenate(all_q, axis=0)
        all_actions = all_q.argmax(axis=1)

    logging.info("DQN training complete. Mean Q=%.4f", all_q.mean())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(main_net.state_dict(), f"{save_dir}/dqn_model.pt")
        with open(f"{save_dir}/dqn_actions.pkl", "wb") as f:
            pickle.dump(all_actions, f)
        with open(f"{save_dir}/dqn_q_values.pkl", "wb") as f:
            pickle.dump(all_q, f)
        logging.info("  Model saved to %s", save_dir)

    return main_net, all_actions, all_q


def train_sarsa_physician(data, n_state, n_actions=25, hidden=128,
                          lr=1e-4, gamma=0.99, tau=0.001, batch_size=32,
                          num_steps=70000, reward_threshold=20, reg_lambda=5.0,
                          per_alpha=0.6, per_epsilon=0.01, beta_start=0.9,
                          save_dir=None, device="cpu", log_every=5000):
    """Train SARSA to learn physician Q-function (on-policy).

    Same architecture as DQN but uses physician's actual next-action
    instead of max Q.

    Ported from sepsisrl/continuous/sarsa_physician.ipynb.
    """
    logging.info("SARSA physician: %d steps, lr=%.1e", num_steps, lr)

    main_net = DuelingDQN(n_state, n_actions, hidden, 0.01).to(device)
    target_net = DuelingDQN(n_state, n_actions, hidden, 0.01).to(device)
    target_net.load_state_dict(main_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(main_net.parameters(), lr=lr)

    # Build next-action array (physician's actual action at t+1)
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

    t_train_start = time.time()
    for step in range(num_steps):
        main_net.train()
        samples, indices, is_weights = buffer.sample(batch_size, beta=beta_start)
        is_weights = is_weights.to(device)

        states = torch.FloatTensor(np.array([s.state for s in samples])).to(device)
        actions = torch.LongTensor([s.action for s in samples]).to(device)
        rewards = torch.FloatTensor([s.reward for s in samples]).to(device)
        next_states = torch.FloatTensor(np.array([s.next_state for s in samples])).to(device)
        dones = torch.FloatTensor([s.done for s in samples]).to(device)

        rewards = rewards.clamp(-1.0, 1.0)

        q_values = main_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # SARSA: use physician's actual next action (not max)
        with torch.no_grad():
            next_q = target_net(next_states)
            # Get physician's next actions for these samples
            sample_next_actions = torch.LongTensor(
                [next_actions_arr[idx] for idx in indices]
            ).to(device)
            next_q_taken = next_q.gather(1, sample_next_actions.unsqueeze(1)).squeeze(1)
            next_q_taken = next_q_taken.clamp(-reward_threshold, reward_threshold)
            target = rewards + gamma * next_q_taken * (1 - dones)

        td_error = target - q_taken
        loss = (is_weights * td_error.pow(2)).mean()
        reg = torch.clamp(q_values.abs() - reward_threshold, min=0).sum()
        loss += reg_lambda * reg / batch_size

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(main_net.parameters(), 10.0)
        optimizer.step()

        buffer.update_priorities(indices, td_error.detach().cpu().numpy())

        for p_main, p_target in zip(main_net.parameters(), target_net.parameters()):
            p_target.data.copy_(tau * p_main.data + (1 - tau) * p_target.data)

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_train_start
            steps_sec = (step + 1) / elapsed
            remaining = (num_steps - step - 1) / steps_sec
            pct = 100 * (step + 1) / num_steps
            logging.info("  SARSA step %d/%d (%.0f%%) | loss=%.4f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, loss.item(), steps_sec, elapsed, remaining)

    # Extract Q-values
    main_net.eval()
    with torch.no_grad():
        all_states = torch.FloatTensor(data["states"]).to(device)
        all_q = []
        bs = 4096
        for i in range(0, len(all_states), bs):
            all_q.append(main_net(all_states[i:i + bs]).cpu().numpy())
        all_q = np.concatenate(all_q, axis=0)
        all_actions = all_q.argmax(axis=1)

    logging.info("SARSA physician complete. Mean Q=%.4f", all_q.mean())

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(main_net.state_dict(), f"{save_dir}/sarsa_phys_model.pt")
        with open(f"{save_dir}/phys_actions.pkl", "wb") as f:
            pickle.dump(all_actions, f)
        with open(f"{save_dir}/phys_q_values.pkl", "wb") as f:
            pickle.dump(all_q, f)

    return main_net, all_actions, all_q


def train_autoencoder(data, n_input, n_hidden=200, lr=1e-4,
                      num_steps=100000, batch_size=100,
                      sparsity_target=0.05, kl_weight=0.0001,
                      save_dir=None, device="cpu", log_every=10000):
    """Train sparse autoencoder for state compression.

    Ported from sepsisrl/continuous/autoencoder.ipynb.
    """
    logging.info("Autoencoder: %d -> %d, %d steps", n_input, n_hidden, num_steps)

    model = SparseAutoencoder(n_input, n_hidden).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    states = torch.FloatTensor(data["states"]).to(device)
    n = len(states)

    t_train_start = time.time()
    for step in range(num_steps):
        model.train()
        idx = np.random.choice(n, batch_size, replace=False)
        batch = states[idx]

        recon, z = model(batch)

        # MSE reconstruction loss
        mse_loss = nn.MSELoss()(recon, batch)

        # KL divergence sparsity penalty
        rho_hat = z.mean(dim=0)
        rho = sparsity_target
        kl = rho * torch.log(rho / (rho_hat + 1e-8)) + \
             (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-8))
        kl_loss = kl.sum()

        loss = mse_loss + kl_weight * kl_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (step + 1) % log_every == 0:
            elapsed = time.time() - t_train_start
            steps_sec = (step + 1) / elapsed
            remaining = (num_steps - step - 1) / steps_sec
            pct = 100 * (step + 1) / num_steps
            logging.info("  AE step %d/%d (%.0f%%) | MSE=%.6f, KL=%.4f | %.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps, pct, mse_loss.item(), kl_loss.item(),
                         steps_sec, elapsed, remaining)

    # Encode all data
    model.eval()
    with torch.no_grad():
        encoded = []
        bs = 4096
        for i in range(0, n, bs):
            batch = states[i:i + bs]
            _, z = model(batch)
            encoded.append(z.cpu().numpy())
        encoded = np.concatenate(encoded, axis=0)

    logging.info("Autoencoder complete. Encoded shape: %s", encoded.shape)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        torch.save(model.state_dict(), f"{save_dir}/autoencoder_model.pt")
        with open(f"{save_dir}/encoded_states.pkl", "wb") as f:
            pickle.dump(encoded, f)

    return model, encoded
