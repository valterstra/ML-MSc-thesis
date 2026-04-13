"""Offline Dueling DDQN with Prioritized Experience Replay for V3 RL.

Trains on the fixed offline dataset produced by preprocessing.py.
No environment interaction -- pure batch RL from observed transitions.

Architecture (matches sepsis Step 11):
  Input (19) -> FC(128) -> BN -> LeakyReLU -> FC(128) -> BN -> LeakyReLU
      -> Value stream:     FC(64) -> FC(1)
      -> Advantage stream: FC(64) -> FC(256)
  Q(s, a) = V(s) + A(s, a) - mean_a(A(s, a))

Action space: 256 (all 2^8 binary drug combinations)
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# !! torch must be imported BEFORE numpy/pandas on Windows.
# numpy's C extensions load runtime DLLs that block torch's c10.dll loading
# when torch is imported later. The fix: add torch/lib to the DLL search path
# and import torch first.
def _fix_torch_dll() -> None:
    if sys.platform != "win32" or not hasattr(os, "add_dll_directory"):
        return
    for p in sys.path:
        candidate = Path(p) / "torch" / "lib"
        if candidate.is_dir():
            os.add_dll_directory(str(candidate))
            return

_fix_torch_dll()

import torch
import torch.nn as nn
import torch.optim as optim

import json
import logging

import numpy as np
import pandas as pd

from careai.sepsis_v3.preprocessing import N_ACTIONS, STATE_FEATURES

log = logging.getLogger(__name__)

N_STATE = len(STATE_FEATURES)   # 19
GAMMA   = 0.99
LR      = 1e-3
BATCH   = 256
TAU     = 0.005    # soft target update rate
UPDATE_TARGET_EVERY = 200   # steps between hard/soft target syncs


# ---------------------------------------------------------------------------
# Network
# ---------------------------------------------------------------------------

class DuelingDQN(nn.Module):
    """Dueling DQN: separate value and advantage streams."""

    def __init__(self, n_state: int = N_STATE, n_actions: int = N_ACTIONS) -> None:
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(n_state, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.01),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64),
            nn.LeakyReLU(0.01),
            nn.Linear(64, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.shared(x)
        v = self.value_stream(h)                  # (B, 1)
        a = self.advantage_stream(h)              # (B, N_ACTIONS)
        q = v + (a - a.mean(dim=1, keepdim=True)) # (B, N_ACTIONS)
        return q


# ---------------------------------------------------------------------------
# Prioritized Replay Buffer (offline / fixed dataset)
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """Offline PER: data is fixed; priorities updated after each batch.

    Uses alpha=0.6 for priority exponent, beta=0.4 for IS correction.
    """

    def __init__(self, size: int, alpha: float = 0.6) -> None:
        self.size  = size
        self.alpha = alpha
        self.priorities = np.ones(size, dtype=np.float32)

    def sample(
        self, batch_size: int, beta: float = 0.4
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (indices, IS weights, normalized probs)."""
        probs = self.priorities ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(self.size, size=batch_size, replace=False, p=probs)
        # Importance-sampling weights
        weights = (self.size * probs[indices]) ** (-beta)
        weights /= weights.max()
        return indices, weights.astype(np.float32), probs

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray) -> None:
        self.priorities[indices] = np.abs(td_errors) + 1e-6


# ---------------------------------------------------------------------------
# Offline DDQN trainer
# ---------------------------------------------------------------------------

class OfflineDDQN:
    """Offline Dueling DDQN trained on a fixed transition dataset."""

    def __init__(self, device: str = "cpu") -> None:
        self.device = torch.device(device)
        self.q_net      = DuelingDQN().to(self.device)
        self.q_target   = DuelingDQN().to(self.device)
        self.q_target.load_state_dict(self.q_net.state_dict())
        self.q_target.eval()
        self.optimizer  = optim.Adam(self.q_net.parameters(), lr=LR)
        self.loss_fn    = nn.SmoothL1Loss(reduction="none")
        self.step_count = 0
        self.train_losses: list[float] = []

    # ── Data loading ──────────────────────────────────────────────────────

    @staticmethod
    def _load_split(csv_path: str | Path) -> tuple[
        np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
    ]:
        """Load a preprocessed RL CSV and return (s, a, r, s', done) arrays."""
        df = pd.read_csv(csv_path)
        s_cols  = ["s_"  + f for f in STATE_FEATURES]
        sn_cols = ["sn_" + f for f in STATE_FEATURES]
        s   = df[s_cols].values.astype(np.float32)
        sn  = df[sn_cols].values.astype(np.float32)
        a   = df["action_id"].values.astype(np.int64)
        r   = df["reward"].values.astype(np.float32)
        done = df["done"].values.astype(np.float32)
        return s, a, r, sn, done

    # ── Training ──────────────────────────────────────────────────────────

    def fit(
        self,
        train_csv: str | Path,
        val_csv: str | Path,
        n_steps: int = 50_000,
        log_every: int = 5_000,
    ) -> "OfflineDDQN":
        """Train the DDQN for n_steps gradient updates."""
        log.info("Loading training data...")
        s, a, r, sn, done = self._load_split(train_csv)
        n_train = len(s)
        log.info("Training on %d transitions for %d steps", n_train, n_steps)

        buffer = PrioritizedReplayBuffer(n_train)

        beta_start, beta_end = 0.4, 1.0

        for step in range(1, n_steps + 1):
            beta = beta_start + (beta_end - beta_start) * step / n_steps
            idx, weights, _ = buffer.sample(BATCH, beta)

            s_b  = torch.tensor(s[idx],    device=self.device)
            a_b  = torch.tensor(a[idx],    device=self.device)
            r_b  = torch.tensor(r[idx],    device=self.device)
            sn_b = torch.tensor(sn[idx],   device=self.device)
            d_b  = torch.tensor(done[idx], device=self.device)
            w_b  = torch.tensor(weights,   device=self.device)

            # Double DQN target: select action with online net, evaluate with target
            with torch.no_grad():
                best_a_next  = self.q_net(sn_b).argmax(dim=1)
                q_next       = self.q_target(sn_b)
                q_next_best  = q_next.gather(1, best_a_next.unsqueeze(1)).squeeze(1)
                target        = r_b + GAMMA * q_next_best * (1.0 - d_b)

            self.q_net.train()
            q_pred = self.q_net(s_b).gather(1, a_b.unsqueeze(1)).squeeze(1)
            td_err = target - q_pred.detach().cpu().numpy()

            elementwise_loss = self.loss_fn(q_pred, target)
            loss = (w_b * elementwise_loss).mean()

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.q_net.parameters(), 1.0)
            self.optimizer.step()

            # Update priorities
            buffer.update_priorities(idx, td_err)

            # Soft target update
            if step % UPDATE_TARGET_EVERY == 0:
                for tp, op in zip(
                    self.q_target.parameters(), self.q_net.parameters()
                ):
                    tp.data.copy_(TAU * op.data + (1.0 - TAU) * tp.data)

            self.train_losses.append(float(loss.item()))
            self.step_count += 1

            if step % log_every == 0:
                mean_loss = np.mean(self.train_losses[-log_every:])
                log.info("Step %d/%d | loss=%.6f", step, n_steps, mean_loss)

        log.info("Training complete. Steps: %d", self.step_count)
        return self

    # ── Inference ─────────────────────────────────────────────────────────

    def predict_actions(self, df: pd.DataFrame) -> np.ndarray:
        """Return recommended action_id for each row in df."""
        s_cols = ["s_" + f for f in STATE_FEATURES]
        s = torch.tensor(
            df[s_cols].values.astype(np.float32), device=self.device
        )
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(s)
        return q.argmax(dim=1).cpu().numpy()

    def q_values(self, df: pd.DataFrame) -> np.ndarray:
        """Return Q-value matrix (n_rows, 256) for all actions."""
        s_cols = ["s_" + f for f in STATE_FEATURES]
        s = torch.tensor(
            df[s_cols].values.astype(np.float32), device=self.device
        )
        self.q_net.eval()
        with torch.no_grad():
            q = self.q_net(s)
        return q.cpu().numpy()

    # ── Persistence ───────────────────────────────────────────────────────

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.q_net.state_dict(), model_dir / "q_net.pt")
        torch.save(self.q_target.state_dict(), model_dir / "q_target.pt")
        meta = {
            "n_state":      N_STATE,
            "n_actions":    N_ACTIONS,
            "gamma":        GAMMA,
            "lr":           LR,
            "batch":        BATCH,
            "tau":          TAU,
            "step_count":   self.step_count,
            "state_features": STATE_FEATURES,
            "train_loss_final": float(np.mean(self.train_losses[-1000:]))
                                  if self.train_losses else None,
        }
        (model_dir / "metadata.json").write_text(json.dumps(meta, indent=2))
        log.info("Saved DDQN to %s", model_dir)

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cpu") -> "OfflineDDQN":
        model_dir = Path(model_dir)
        obj = cls(device=device)
        obj.q_net.load_state_dict(
            torch.load(model_dir / "q_net.pt", map_location=device)
        )
        obj.q_target.load_state_dict(
            torch.load(model_dir / "q_target.pt", map_location=device)
        )
        obj.q_net.eval()
        obj.q_target.eval()
        meta = json.loads((model_dir / "metadata.json").read_text())
        obj.step_count = meta.get("step_count", 0)
        log.info("Loaded DDQN from %s (%d steps)", model_dir, obj.step_count)
        return obj
