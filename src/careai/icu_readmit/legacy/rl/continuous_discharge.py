"""
Joint DDQN + SARSA for two-phase ICU-readmission MDP with discharge action.

Architecture
------------
Two separate DuelingDQN networks share NO parameters but are trained jointly
in the same loop from one replay buffer:

  Q_drug      (n_state=5, n_actions=16)  -- in-stay drug decisions (phase=0)
  Q_discharge (n_state=5, n_actions=3)   -- discharge destination  (phase=1)

Cross-phase Bellman backup
--------------------------
  Regular in-stay (phase=0, next_is_discharge=0):
      target = r + gamma * max_a Q_drug_target(s')          [DDQN]

  Last in-stay bloc (phase=0, next_is_discharge=1):
      target = r + gamma * max_a Q_discharge_target(s')     [cross-phase]
      --> drug policy at final bloc is shaped by discharge value

  Discharge terminal (phase=1, done=1):
      target = r                                             [no bootstrap]

This means Q_drug learns to steer the patient toward states where
Q_discharge predicts a good discharge outcome (low readmission risk).

Inputs
------
  data dict from prepare_discharge_data(), containing:
    states, actions, rewards, next_states, done,
    phase, next_is_discharge, next_a_physician

Outputs
-------
  models/icu_readmit/tier2_discharge/
    ddqn/
      dqn_drug_model.pt          drug Q-network
      dqn_discharge_model.pt     discharge Q-network
      dqn_drug_actions.pkl       greedy drug actions (train)
      dqn_drug_q_values.pkl      drug Q-matrix (train)
      dqn_discharge_actions.pkl  greedy discharge actions (discharge rows only)
      dqn_discharge_q_values.pkl discharge Q-matrix (discharge rows only)
      dqn_losses.pkl             per-step total loss list
    sarsa_phys/
      (same pattern, prefix phys_)
"""

import logging
import os
import pickle
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from .networks import DuelingDQN


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_discharge_data(df, state_cols, next_state_cols):
    """Extract numpy arrays from the discharge-augmented parquet.

    Args:
        df:              DataFrame for one split (train / val / test)
        state_cols:      list of s_* column names
        next_state_cols: matching s_next_* column names

    Returns:
        dict with keys: states, actions, rewards, next_states, done,
                        phase, next_is_discharge, next_a_physician
    """
    return {
        "states":              df[state_cols].values.astype(np.float32),
        "next_states":         df[next_state_cols].values.astype(np.float32),
        "actions":             df["a"].values.astype(np.int64),
        "rewards":             df["r"].values.astype(np.float32),
        "done":                df["done"].values.astype(np.float32),
        "phase":               df["phase"].values.astype(np.int64),
        "next_is_discharge":   df["next_is_discharge"].values.astype(np.int64),
        "next_a_physician":    df["next_a_physician"].values.astype(np.int64),
    }


# ---------------------------------------------------------------------------
# Prioritized Experience Replay
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Identical to continuous.py -- reproduced here to keep module self-contained."""

    def __init__(self, capacity, alpha=0.6, epsilon=0.01):
        self.capacity  = capacity
        self.alpha     = alpha
        self.epsilon   = epsilon
        self.buffer    = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos  = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done,
            phase, next_is_discharge, next_a_physician, priority=None):
        entry = (state, action, reward, next_state, done,
                 phase, next_is_discharge, next_a_physician)
        if priority is None:
            priority = self.priorities[:self.size].max() if self.size > 0 else 1.0
        if self.size < self.capacity:
            self.buffer.append(entry)
            self.size += 1
        else:
            self.buffer[self.pos] = entry
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.9):
        probs   = self.priorities[:self.size] ** self.alpha
        probs  /= probs.sum()
        indices = np.random.choice(self.size, batch_size, p=probs, replace=False)
        batch   = [self.buffer[i] for i in indices]
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return batch, indices, torch.FloatTensor(weights)

    def update_priorities(self, indices, td_errors):
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = (abs(float(td)) + self.epsilon) ** self.alpha


def _unpack_batch(batch, device):
    """Unpack a list of buffer entries into GPU tensors."""
    states            = torch.FloatTensor(np.array([b[0] for b in batch])).to(device)
    actions           = torch.LongTensor( [b[1] for b in batch]).to(device)
    rewards           = torch.FloatTensor([b[2] for b in batch]).to(device)
    next_states       = torch.FloatTensor(np.array([b[3] for b in batch])).to(device)
    dones             = torch.FloatTensor([b[4] for b in batch]).to(device)
    phases            = torch.LongTensor( [b[5] for b in batch]).to(device)
    nid               = torch.BoolTensor( [bool(b[6]) for b in batch]).to(device)
    next_a_phys       = torch.LongTensor( [b[7] for b in batch]).to(device)
    return states, actions, rewards, next_states, dones, phases, nid, next_a_phys


# ---------------------------------------------------------------------------
# Joint target computation (shared by DDQN and SARSA)
# ---------------------------------------------------------------------------

def _compute_targets(rewards, next_states, dones, phases, nid,
                     main_drug, target_drug, main_discharge, target_discharge,
                     gamma, reward_threshold, device,
                     next_actions_override=None):
    """
    Compute Bellman targets for all transitions in one batch.

    Three cases handled:
      1. phase=0, nid=False  -> bootstrap from Q_drug_target (DDQN) or next_a_override (SARSA)
      2. phase=0, nid=True   -> bootstrap from Q_discharge_target (cross-phase)
      3. phase=1, done=1     -> no bootstrap (multiplied out by (1-done))

    Args:
        next_actions_override: if not None, use these actions for SARSA bootstrap (phase=0 only)
    """
    with torch.no_grad():
        batch_size = len(rewards)
        next_q_bootstrap = torch.zeros(batch_size, device=device)

        # --- Case 1: regular in-stay (phase=0, nid=False) -------------------
        mask_regular = (phases == 0) & (~nid)
        if mask_regular.any():
            ns_r = next_states[mask_regular]
            if next_actions_override is not None:
                na_r = next_actions_override[mask_regular]
                nq_r = target_drug(ns_r).gather(1, na_r.unsqueeze(1)).squeeze(1)
            else:
                # Double DQN: action from main, value from target
                na_r = main_drug(ns_r).argmax(dim=1)
                nq_r = target_drug(ns_r).gather(1, na_r.unsqueeze(1)).squeeze(1)
            next_q_bootstrap[mask_regular] = nq_r

        # --- Case 2: last in-stay bloc (phase=0, nid=True) ------------------
        mask_last = (phases == 0) & nid
        if mask_last.any():
            ns_l    = next_states[mask_last]
            na_l    = main_discharge(ns_l).argmax(dim=1)
            nq_l    = target_discharge(ns_l).gather(1, na_l.unsqueeze(1)).squeeze(1)
            next_q_bootstrap[mask_last] = nq_l

        # Case 3 (phase=1, done=1): next_q_bootstrap stays 0; zeroed by (1-done) anyway

        next_q_bootstrap = next_q_bootstrap.clamp(-reward_threshold, reward_threshold)
        targets = rewards + gamma * next_q_bootstrap * (1 - dones)

    return targets


# ---------------------------------------------------------------------------
# Joint DDQN training
# ---------------------------------------------------------------------------

def train_joint_dqn(data, n_state, n_drug_actions=16, n_discharge_actions=3,
                    hidden=128, leaky_slope=0.01,
                    lr=1e-4, gamma=0.99, tau=0.001, batch_size=32,
                    num_steps=100000, reward_threshold=20, reg_lambda=5.0,
                    per_alpha=0.6, per_epsilon=0.01, beta_start=0.9,
                    save_dir=None, checkpoint_every=20000,
                    device="cpu", log_every=5000):
    """
    Train joint Dueling DDQN with cross-phase Bellman backup.

    Args:
        data:                dict from prepare_discharge_data()
        n_state:             state dimension (5 for Tier 2)
        n_drug_actions:      drug action space size (16)
        n_discharge_actions: discharge action space size (3)
        save_dir:            output directory

    Returns:
        (main_drug, main_discharge,
         drug_actions_train, drug_q_train,
         discharge_actions_train, discharge_q_train)
    """
    logging.info("Joint DDQN: %d steps | n_state=%d | drug=%d actions | "
                 "discharge=%d actions | lr=%.1e | device=%s",
                 num_steps, n_state, n_drug_actions, n_discharge_actions, lr, device)

    # Two independent networks + target copies
    main_drug        = DuelingDQN(n_state, n_drug_actions,      hidden, leaky_slope).to(device)
    target_drug      = DuelingDQN(n_state, n_drug_actions,      hidden, leaky_slope).to(device)
    main_discharge   = DuelingDQN(n_state, n_discharge_actions, hidden, leaky_slope).to(device)
    target_discharge = DuelingDQN(n_state, n_discharge_actions, hidden, leaky_slope).to(device)

    target_drug.load_state_dict(main_drug.state_dict());      target_drug.eval()
    target_discharge.load_state_dict(main_discharge.state_dict()); target_discharge.eval()

    opt_drug      = optim.Adam(main_drug.parameters(),      lr=lr)
    opt_discharge = optim.Adam(main_discharge.parameters(), lr=lr)

    # Fill replay buffer
    n = len(data["states"])
    buf = PrioritizedReplayBuffer(n, alpha=per_alpha, epsilon=per_epsilon)
    for i in range(n):
        buf.add(
            data["states"][i], data["actions"][i], data["rewards"][i],
            data["next_states"][i], data["done"][i],
            data["phase"][i], data["next_is_discharge"][i], data["next_a_physician"][i],
            priority=abs(float(data["rewards"][i])) + per_epsilon,
        )

    logging.info("  Replay buffer: %d transitions (%d phase-0, %d phase-1)",
                 buf.size,
                 int((data["phase"] == 0).sum()),
                 int((data["phase"] == 1).sum()))

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    losses = []
    t0     = time.time()

    for step in range(num_steps):
        main_drug.train();      main_discharge.train()
        batch, indices, is_weights = buf.sample(batch_size, beta=beta_start)
        is_weights = is_weights.to(device)

        states, actions, rewards, next_states, dones, phases, nid, _ = \
            _unpack_batch(batch, device)

        # ---- Targets -------------------------------------------------------
        targets = _compute_targets(
            rewards, next_states, dones, phases, nid,
            main_drug, target_drug, main_discharge, target_discharge,
            gamma, reward_threshold, device,
        )

        # ---- Drug loss (phase=0 rows) ---------------------------------------
        mask0 = phases == 0
        loss  = torch.tensor(0.0, device=device)
        td_combined = torch.zeros(batch_size, device=device)

        # BatchNorm1d requires >= 2 samples; skip phase if too few in this batch
        if mask0.sum() >= 2:
            q0      = main_drug(states[mask0])
            q_taken0 = q0.gather(1, actions[mask0].unsqueeze(1)).squeeze(1)
            td0     = targets[mask0] - q_taken0
            loss0   = (is_weights[mask0] * td0.pow(2)).mean()
            reg0    = torch.clamp(q0.abs() - reward_threshold, min=0).sum()
            loss0  += reg_lambda * reg0 / batch_size
            loss    = loss + loss0
            td_combined[mask0] = td0.detach()

        # ---- Discharge loss (phase=1 rows) ----------------------------------
        mask1 = phases == 1
        if mask1.sum() >= 2:
            q1      = main_discharge(states[mask1])
            # Actions for phase=1 are discharge categories 0-2; clamp for safety
            a1      = actions[mask1].clamp(0, n_discharge_actions - 1)
            q_taken1 = q1.gather(1, a1.unsqueeze(1)).squeeze(1)
            td1     = targets[mask1] - q_taken1
            loss1   = (is_weights[mask1] * td1.pow(2)).mean()
            reg1    = torch.clamp(q1.abs() - reward_threshold, min=0).sum()
            loss1  += reg_lambda * reg1 / batch_size
            loss    = loss + loss1
            td_combined[mask1] = td1.detach()

        opt_drug.zero_grad()
        opt_discharge.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(main_drug.parameters(),      10.0)
        nn.utils.clip_grad_norm_(main_discharge.parameters(), 10.0)
        opt_drug.step()
        opt_discharge.step()

        buf.update_priorities(indices, td_combined.cpu().numpy())

        # Soft target updates
        for pm, pt in zip(main_drug.parameters(),      target_drug.parameters()):
            pt.data.copy_(tau * pm.data + (1 - tau) * pt.data)
        for pm, pt in zip(main_discharge.parameters(), target_discharge.parameters()):
            pt.data.copy_(tau * pm.data + (1 - tau) * pt.data)

        losses.append(loss.item())

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            avg_loss  = float(np.mean(losses[-log_every:]))
            logging.info("  Joint-DDQN %d/%d (%.0f%%) | loss=%.4f | "
                         "%.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps,
                         100 * (step + 1) / num_steps,
                         avg_loss, steps_sec, elapsed, eta)

        if save_dir and checkpoint_every and (step + 1) % checkpoint_every == 0:
            torch.save(main_drug.state_dict(),
                       os.path.join(save_dir, f"checkpoint_drug_{step+1}.pt"))
            torch.save(main_discharge.state_dict(),
                       os.path.join(save_dir, f"checkpoint_discharge_{step+1}.pt"))
            logging.info("  Checkpoint saved at step %d", step + 1)

    # ---- Final Q-values ----------------------------------------------------
    main_drug.eval();  main_discharge.eval()
    all_states = torch.FloatTensor(data["states"]).to(device)

    with torch.no_grad():
        drug_q_list, disch_q_list = [], []
        for i in range(0, len(all_states), 4096):
            drug_q_list.append( main_drug(all_states[i:i+4096]).cpu().numpy())
            disch_q_list.append(main_discharge(all_states[i:i+4096]).cpu().numpy())
        drug_q_all  = np.concatenate(drug_q_list,  axis=0)
        disch_q_all = np.concatenate(disch_q_list, axis=0)

    drug_actions  = drug_q_all.argmax(axis=1)
    disch_actions = disch_q_all.argmax(axis=1)

    # Report on discharge rows only (phase=1)
    phase1_mask = data["phase"] == 1
    if phase1_mask.any():
        dq_phase1 = disch_q_all[phase1_mask]
        da_phase1 = dq_phase1.argmax(axis=1)
        logging.info("Discharge Q (phase-1 rows): mean=%.4f | "
                     "action dist: %s",
                     dq_phase1.mean(),
                     str(np.bincount(da_phase1, minlength=n_discharge_actions).tolist()))

    logging.info("Joint DDQN complete. Drug mean Q=%.4f | Discharge mean Q=%.4f",
                 drug_q_all.mean(), disch_q_all.mean())

    if save_dir:
        torch.save(main_drug.state_dict(),
                   os.path.join(save_dir, "dqn_drug_model.pt"))
        torch.save(main_discharge.state_dict(),
                   os.path.join(save_dir, "dqn_discharge_model.pt"))
        with open(os.path.join(save_dir, "dqn_drug_actions.pkl"),       "wb") as f: pickle.dump(drug_actions,  f)
        with open(os.path.join(save_dir, "dqn_drug_q_values.pkl"),      "wb") as f: pickle.dump(drug_q_all,   f)
        with open(os.path.join(save_dir, "dqn_discharge_actions.pkl"),  "wb") as f: pickle.dump(disch_actions, f)
        with open(os.path.join(save_dir, "dqn_discharge_q_values.pkl"), "wb") as f: pickle.dump(disch_q_all,  f)
        with open(os.path.join(save_dir, "dqn_losses.pkl"),             "wb") as f: pickle.dump(losses,       f)
        logging.info("  Models saved to %s", save_dir)

    return (main_drug, main_discharge,
            drug_actions, drug_q_all,
            disch_actions, disch_q_all)


# ---------------------------------------------------------------------------
# Joint SARSA physician baseline
# ---------------------------------------------------------------------------

def train_joint_sarsa_physician(data, n_state, n_drug_actions=16, n_discharge_actions=3,
                                 hidden=128, leaky_slope=0.01,
                                 lr=1e-4, gamma=0.99, tau=0.001, batch_size=32,
                                 num_steps=80000, reward_threshold=20, reg_lambda=5.0,
                                 per_alpha=0.6, per_epsilon=0.01, beta_start=0.9,
                                 save_dir=None, checkpoint_every=20000,
                                 device="cpu", log_every=5000):
    """
    Train joint SARSA to learn the physician's Q-function (on-policy).

    Uses precomputed next_a_physician for all phase-0 bootstrap steps,
    including the cross-phase handoff where next_a_physician is the
    observed discharge category for the last in-stay bloc.

    Returns:
        (main_drug, main_discharge,
         drug_actions, drug_q, discharge_actions, discharge_q)
    """
    logging.info("Joint SARSA physician: %d steps | n_state=%d | "
                 "drug=%d | discharge=%d | device=%s",
                 num_steps, n_state, n_drug_actions, n_discharge_actions, device)

    main_drug        = DuelingDQN(n_state, n_drug_actions,      hidden, leaky_slope).to(device)
    target_drug      = DuelingDQN(n_state, n_drug_actions,      hidden, leaky_slope).to(device)
    main_discharge   = DuelingDQN(n_state, n_discharge_actions, hidden, leaky_slope).to(device)
    target_discharge = DuelingDQN(n_state, n_discharge_actions, hidden, leaky_slope).to(device)

    target_drug.load_state_dict(main_drug.state_dict());          target_drug.eval()
    target_discharge.load_state_dict(main_discharge.state_dict()); target_discharge.eval()

    opt_drug      = optim.Adam(main_drug.parameters(),      lr=lr)
    opt_discharge = optim.Adam(main_discharge.parameters(), lr=lr)

    n = len(data["states"])
    buf = PrioritizedReplayBuffer(n, alpha=per_alpha, epsilon=per_epsilon)
    for i in range(n):
        buf.add(
            data["states"][i], data["actions"][i], data["rewards"][i],
            data["next_states"][i], data["done"][i],
            data["phase"][i], data["next_is_discharge"][i], data["next_a_physician"][i],
            priority=abs(float(data["rewards"][i])) + per_epsilon,
        )

    logging.info("  Replay buffer: %d transitions", buf.size)

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    losses = []
    t0     = time.time()

    for step in range(num_steps):
        main_drug.train();  main_discharge.train()
        batch, indices, is_weights = buf.sample(batch_size, beta=beta_start)
        is_weights = is_weights.to(device)

        states, actions, rewards, next_states, dones, phases, nid, next_a_phys = \
            _unpack_batch(batch, device)

        # SARSA: use physician's observed next action for phase=0 bootstrap
        # (for last in-stay blocs, next_a_physician = observed discharge category)
        targets = _compute_targets(
            rewards, next_states, dones, phases, nid,
            main_drug, target_drug, main_discharge, target_discharge,
            gamma, reward_threshold, device,
            next_actions_override=next_a_phys,
        )

        mask0  = phases == 0
        mask1  = phases == 1
        loss   = torch.tensor(0.0, device=device)
        td_combined = torch.zeros(batch_size, device=device)

        # BatchNorm1d requires >= 2 samples; skip phase if too few in this batch
        if mask0.sum() >= 2:
            q0       = main_drug(states[mask0])
            q_taken0 = q0.gather(1, actions[mask0].unsqueeze(1)).squeeze(1)
            td0      = targets[mask0] - q_taken0
            loss0    = (is_weights[mask0] * td0.pow(2)).mean()
            reg0     = torch.clamp(q0.abs() - reward_threshold, min=0).sum()
            loss0   += reg_lambda * reg0 / batch_size
            loss     = loss + loss0
            td_combined[mask0] = td0.detach()

        if mask1.sum() >= 2:
            q1       = main_discharge(states[mask1])
            a1       = actions[mask1].clamp(0, n_discharge_actions - 1)
            q_taken1 = q1.gather(1, a1.unsqueeze(1)).squeeze(1)
            td1      = targets[mask1] - q_taken1
            loss1    = (is_weights[mask1] * td1.pow(2)).mean()
            reg1     = torch.clamp(q1.abs() - reward_threshold, min=0).sum()
            loss1   += reg_lambda * reg1 / batch_size
            loss     = loss + loss1
            td_combined[mask1] = td1.detach()

        opt_drug.zero_grad()
        opt_discharge.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(main_drug.parameters(),      10.0)
        nn.utils.clip_grad_norm_(main_discharge.parameters(), 10.0)
        opt_drug.step()
        opt_discharge.step()

        buf.update_priorities(indices, td_combined.cpu().numpy())

        for pm, pt in zip(main_drug.parameters(),      target_drug.parameters()):
            pt.data.copy_(tau * pm.data + (1 - tau) * pt.data)
        for pm, pt in zip(main_discharge.parameters(), target_discharge.parameters()):
            pt.data.copy_(tau * pm.data + (1 - tau) * pt.data)

        losses.append(loss.item())

        if (step + 1) % log_every == 0:
            elapsed   = time.time() - t0
            steps_sec = (step + 1) / elapsed
            eta       = (num_steps - step - 1) / steps_sec
            logging.info("  Joint-SARSA %d/%d (%.0f%%) | loss=%.4f | "
                         "%.1f steps/s | elapsed %.0fs | ETA %.0fs",
                         step + 1, num_steps,
                         100 * (step + 1) / num_steps,
                         float(np.mean(losses[-log_every:])),
                         steps_sec, elapsed, eta)

        if save_dir and checkpoint_every and (step + 1) % checkpoint_every == 0:
            torch.save(main_drug.state_dict(),
                       os.path.join(save_dir, f"checkpoint_drug_{step+1}.pt"))
            torch.save(main_discharge.state_dict(),
                       os.path.join(save_dir, f"checkpoint_discharge_{step+1}.pt"))

    # Final Q-values
    main_drug.eval();  main_discharge.eval()
    all_states = torch.FloatTensor(data["states"]).to(device)
    with torch.no_grad():
        dq_list, dischq_list = [], []
        for i in range(0, len(all_states), 4096):
            dq_list.append(   main_drug(all_states[i:i+4096]).cpu().numpy())
            dischq_list.append(main_discharge(all_states[i:i+4096]).cpu().numpy())
        drug_q_all  = np.concatenate(dq_list,    axis=0)
        disch_q_all = np.concatenate(dischq_list, axis=0)

    drug_actions  = drug_q_all.argmax(axis=1)
    disch_actions = disch_q_all.argmax(axis=1)

    logging.info("Joint SARSA complete. Drug mean Q=%.4f | Discharge mean Q=%.4f",
                 drug_q_all.mean(), disch_q_all.mean())

    if save_dir:
        torch.save(main_drug.state_dict(),
                   os.path.join(save_dir, "sarsa_drug_model.pt"))
        torch.save(main_discharge.state_dict(),
                   os.path.join(save_dir, "sarsa_discharge_model.pt"))
        with open(os.path.join(save_dir, "phys_drug_actions.pkl"),       "wb") as f: pickle.dump(drug_actions,  f)
        with open(os.path.join(save_dir, "phys_drug_q_values.pkl"),      "wb") as f: pickle.dump(drug_q_all,   f)
        with open(os.path.join(save_dir, "phys_discharge_actions.pkl"),  "wb") as f: pickle.dump(disch_actions, f)
        with open(os.path.join(save_dir, "phys_discharge_q_values.pkl"), "wb") as f: pickle.dump(disch_q_all,  f)
        with open(os.path.join(save_dir, "sarsa_losses.pkl"),            "wb") as f: pickle.dump(losses,       f)

    return (main_drug, main_discharge,
            drug_actions, drug_q_all,
            disch_actions, disch_q_all)


# ---------------------------------------------------------------------------
# Inference utilities
# ---------------------------------------------------------------------------

def compute_q_values(model, states_np, device="cpu", batch_size=4096):
    """Compute Q-values for a numpy state array. Works for either network."""
    model.eval()
    all_q = []
    with torch.no_grad():
        st = torch.FloatTensor(states_np).to(device)
        for i in range(0, len(st), batch_size):
            all_q.append(model(st[i:i+batch_size]).cpu().numpy())
    all_q = np.concatenate(all_q, axis=0)
    return all_q, all_q.argmax(axis=1)
