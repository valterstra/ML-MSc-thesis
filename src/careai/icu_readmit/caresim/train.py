"""
Training utilities for CareSimGPT world model.

Loss function:
    L = w_state * MSE(next_state) + w_reward * MSE(reward) + w_term * BCE(terminal)

Only non-padding positions contribute to the loss (padding mask applied).

EnsembleTrainer trains N independent CareSimGPT models on the same data.
Ensemble uncertainty = standard deviation of predictions across the N models.
"""
from __future__ import annotations

import os
import json
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import CareSimGPT
from .dataset import ICUSequenceDataset, collate_sequences


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------

def compute_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_state: float = 1.0,
    w_reward: float = 0.5,
    w_term: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute masked world model loss.

    Masks out padding positions so they don't contribute to the gradient.

    Args:
        pred  : output of CareSimGPT.forward() -- dict with next_state, reward, terminal
        batch : collated batch from collate_sequences()
        w_*   : loss weights

    Returns:
        total_loss : scalar tensor (differentiable)
        breakdown  : dict with individual loss values (detached floats, for logging)
    """
    # --- Next-state MSE (only real positions) ---
    ns_pred = pred["next_state"]                            # (B, T, state_dim)

    # True where positions are REAL (not padding)
    # Move to same device as predictions (batch dict stays on CPU from DataLoader)
    real_mask = ~batch["src_key_padding_mask"].to(ns_pred.device)  # (B, T) bool
    ns_true = batch["next_states"].to(ns_pred.device)       # (B, T, state_dim)
    state_err = ((ns_pred - ns_true) ** 2)                  # (B, T, state_dim)
    state_loss_mask = pred.get("state_loss_mask", None)
    if state_loss_mask is not None:
        state_loss_mask = state_loss_mask.to(ns_pred.device).view(1, 1, -1)
        denom = state_loss_mask.sum().clamp(min=1.0)
        state_err = (state_err * state_loss_mask).sum(dim=-1) / denom
    else:
        state_err = state_err.mean(dim=-1)                  # (B, T)  -- mean over features
    loss_state = (state_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    # --- Reward MSE (optional head) ---
    if pred["reward"] is None or w_reward <= 0:
        loss_reward = torch.zeros((), device=ns_pred.device)
    else:
        r_pred = pred["reward"]                                 # (B, T)
        r_true = batch["rewards"].to(r_pred.device)             # (B, T)
        reward_err = (r_pred - r_true) ** 2
        loss_reward = (reward_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    # --- Terminal BCE ---
    t_pred = pred["terminal"]                               # (B, T) logits
    t_true = batch["terminals"].to(t_pred.device)           # (B, T) float 0/1
    terminal_err = F.binary_cross_entropy_with_logits(t_pred, t_true, reduction="none")
    loss_term = (terminal_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    total = w_state * loss_state + w_reward * loss_reward + w_term * loss_term

    breakdown = {
        "state": loss_state.item(),
        "reward": loss_reward.item(),
        "terminal": loss_term.item(),
        "total": total.item(),
    }
    return total, breakdown


# ---------------------------------------------------------------------------
# Single-model training
# ---------------------------------------------------------------------------

def train_epoch(
    model: CareSimGPT,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    w_state: float = 1.0,
    w_reward: float = 0.5,
    w_term: float = 0.5,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    """Run one training epoch. Returns average loss breakdown."""
    model.train()
    totals = {"state": 0.0, "reward": 0.0, "terminal": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        # Move to device
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        time_steps = batch["time_steps"].to(device)

        optimizer.zero_grad()
        pred = model(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
        loss, breakdown = compute_loss(pred, batch, w_state, w_reward, w_term)

        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        model.enforce_causal_mask()   # no-op when use_causal_constraints=False

        for k, v in breakdown.items():
            totals[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: CareSimGPT,
    loader: DataLoader,
    device: torch.device,
    w_state: float = 1.0,
    w_reward: float = 0.5,
    w_term: float = 0.5,
) -> dict[str, float]:
    """Run one validation epoch. Returns average loss breakdown."""
    model.eval()
    totals = {"state": 0.0, "reward": 0.0, "terminal": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        time_steps = batch["time_steps"].to(device)

        pred = model(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
        _, breakdown = compute_loss(pred, batch, w_state, w_reward, w_term)

        for k, v in breakdown.items():
            totals[k] += v
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train_model(
    model: CareSimGPT,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    n_epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_dir: str | None = None,
    w_state: float = 1.0,
    w_reward: float = 0.5,
    w_term: float = 0.5,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    """Train a single CareSimGPT model.

    Args:
        model       : CareSimGPT instance
        train_loader: DataLoader with collate_sequences
        val_loader  : optional validation DataLoader
        n_epochs    : number of epochs
        lr          : learning rate (Adam)
        device      : torch.device (auto-detected if None)
        save_dir    : directory to save best model checkpoint
        w_*         : loss weights
        grad_clip   : gradient clipping norm
        verbose     : print progress

    Returns:
        history : list of dicts (one per epoch) with train/val losses
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_losses = train_epoch(model, train_loader, optimizer, device, w_state, w_reward, w_term, grad_clip)
        scheduler.step()

        entry = {"epoch": epoch, "train": train_losses}

        if val_loader is not None:
            val_losses = eval_epoch(model, val_loader, device, w_state, w_reward, w_term)
            entry["val"] = val_losses

            if save_dir is not None and val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        history.append(entry)

        if verbose:
            elapsed = time.time() - t0
            tr = train_losses
            line = (
                f"Epoch {epoch:3d}/{n_epochs}"
                f"  train loss: {tr['total']:.4f}"
                f"  (state={tr['state']:.4f}, reward={tr['reward']:.4f}, term={tr['terminal']:.4f})"
            )
            if val_loader is not None and "val" in entry:
                vl = entry["val"]
                line += f"  | val: {vl['total']:.4f}"
            line += f"  [{elapsed:.1f}s]"
            print(line)

    return history


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

class EnsembleTrainer:
    """Train N independent CareSimGPT models for uncertainty quantification.

    Usage:
        trainer = EnsembleTrainer(n_models=5, model_kwargs={...})
        trainer.train(train_loader, val_loader, n_epochs=30, save_dir="models/caresim/")
        trainer.save("models/caresim/")
    """

    def __init__(self, n_models: int = 5, model_kwargs: dict | None = None):
        self.n_models = n_models
        self.model_kwargs = model_kwargs or {}
        self.models: list[CareSimGPT] = []

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        n_epochs: int = 30,
        lr: float = 1e-3,
        device: torch.device | None = None,
        save_dir: str | None = None,
        verbose: bool = True,
        **loss_kwargs,
    ) -> list[list[dict]]:
        """Train each ensemble member independently.

        Returns: list of history dicts (one per model).
        """
        all_histories = []
        for i in range(self.n_models):
            if verbose:
                print(f"\n=== Training ensemble member {i+1}/{self.n_models} ===")
            model = CareSimGPT(**self.model_kwargs)
            member_save_dir = os.path.join(save_dir, f"member_{i}") if save_dir else None
            history = train_model(
                model, train_loader, val_loader, n_epochs, lr, device, member_save_dir,
                verbose=verbose, **loss_kwargs
            )
            self.models.append(model)
            all_histories.append(history)
        return all_histories

    def save(self, save_dir: str):
        """Save all ensemble members and their config."""
        os.makedirs(save_dir, exist_ok=True)
        for i, model in enumerate(self.models):
            member_dir = os.path.join(save_dir, f"member_{i}")
            os.makedirs(member_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(member_dir, "model.pt"))

        # Save config
        config = {
            "n_models": self.n_models,
            "model_kwargs": self.model_kwargs,
        }
        with open(os.path.join(save_dir, "ensemble_config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Ensemble saved to {save_dir}")

    @classmethod
    def load(
        cls,
        save_dir: str,
        device: torch.device | None = None,
        prefer_best: bool = True,
    ) -> "EnsembleTrainer":
        """Load a saved ensemble.

        Args:
            save_dir    : ensemble directory with member_i subfolders
            device      : torch device for loading
            prefer_best : if True, load member_i/best_model.pt when present,
                          otherwise fall back to member_i/model.pt
        """
        with open(os.path.join(save_dir, "ensemble_config.json")) as f:
            config = json.load(f)
        trainer = cls(n_models=config["n_models"], model_kwargs=config["model_kwargs"])
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for i in range(config["n_models"]):
            model = CareSimGPT(**config["model_kwargs"])
            member_dir = os.path.join(save_dir, f"member_{i}")
            best_path = os.path.join(member_dir, "best_model.pt")
            final_path = os.path.join(member_dir, "model.pt")
            checkpoint_path = best_path if prefer_best and os.path.exists(best_path) else final_path
            state_dict = torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True,
            )
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            trainer.models.append(model)
        which = "best_model.pt when available" if prefer_best else "model.pt only"
        print(f"Loaded ensemble ({config['n_models']} models) from {save_dir} using {which}")
        return trainer
