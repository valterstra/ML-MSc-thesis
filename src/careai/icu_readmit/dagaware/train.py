"""
Training utilities for the DAG-aware temporal world model.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .model import DAGAwareTemporalWorldModel


def compute_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_state: float = 1.0,
    w_reward: float = 0.0,
    w_term: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    ns_pred = pred["next_state"]
    real_mask = ~batch["src_key_padding_mask"].to(ns_pred.device)
    ns_true = batch["next_states"].to(ns_pred.device)

    state_err = (ns_pred - ns_true) ** 2
    state_loss_mask = pred.get("state_loss_mask", None)
    if state_loss_mask is not None:
        state_loss_mask = state_loss_mask.to(ns_pred.device).view(1, 1, -1)
        denom = state_loss_mask.sum().clamp(min=1.0)
        state_err = (state_err * state_loss_mask).sum(dim=-1) / denom
    else:
        state_err = state_err.mean(dim=-1)
    loss_state = (state_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    loss_reward = torch.zeros((), device=ns_pred.device)
    if pred["reward"] is not None and w_reward > 0:
        r_pred = pred["reward"]
        r_true = batch["rewards"].to(r_pred.device)
        reward_err = (r_pred - r_true) ** 2
        loss_reward = (reward_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    t_pred = pred["terminal"]
    t_true = batch["terminals"].to(t_pred.device)
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


def train_epoch(
    model: DAGAwareTemporalWorldModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    w_state: float = 1.0,
    w_reward: float = 0.0,
    w_term: float = 0.5,
    grad_clip: float = 1.0,
) -> dict[str, float]:
    model.train()
    totals = {"state": 0.0, "reward": 0.0, "terminal": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        time_steps = batch["time_steps"].to(device)

        optimizer.zero_grad()
        pred = model(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
        loss, breakdown = compute_loss(pred, batch, w_state=w_state, w_reward=w_reward, w_term=w_term)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        for key, value in breakdown.items():
            totals[key] += value
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_epoch(
    model: DAGAwareTemporalWorldModel,
    loader: DataLoader,
    device: torch.device,
    w_state: float = 1.0,
    w_reward: float = 0.0,
    w_term: float = 0.5,
) -> dict[str, float]:
    model.eval()
    totals = {"state": 0.0, "reward": 0.0, "terminal": 0.0, "total": 0.0}
    n_batches = 0

    for batch in loader:
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        time_steps = batch["time_steps"].to(device)

        pred = model(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
        _, breakdown = compute_loss(pred, batch, w_state=w_state, w_reward=w_reward, w_term=w_term)

        for key, value in breakdown.items():
            totals[key] += value
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in totals.items()}


def train_model(
    model: DAGAwareTemporalWorldModel,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    n_epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    save_dir: str | None = None,
    w_state: float = 1.0,
    w_reward: float = 0.0,
    w_term: float = 0.5,
    grad_clip: float = 1.0,
    verbose: bool = True,
) -> list[dict]:
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)

    history = []
    best_val_loss = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_losses = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            w_state=w_state,
            w_reward=w_reward,
            w_term=w_term,
            grad_clip=grad_clip,
        )
        scheduler.step()
        entry = {"epoch": epoch, "train": train_losses}

        if val_loader is not None:
            val_losses = eval_epoch(
                model,
                val_loader,
                device,
                w_state=w_state,
                w_reward=w_reward,
                w_term=w_term,
            )
            entry["val"] = val_losses
            if save_dir is not None and val_losses["total"] < best_val_loss:
                best_val_loss = val_losses["total"]
                os.makedirs(save_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pt"))

        history.append(entry)
        if verbose:
            elapsed = time.time() - t0
            line = (
                f"Epoch {epoch:3d}/{n_epochs}"
                f"  train loss: {train_losses['total']:.4f}"
                f"  (state={train_losses['state']:.4f}, reward={train_losses['reward']:.4f}, term={train_losses['terminal']:.4f})"
            )
            if "val" in entry:
                line += f"  | val: {entry['val']['total']:.4f}"
            line += f"  [{elapsed:.1f}s]"
            print(line)

    return history


class EnsembleTrainer:
    """Train and save an ensemble of DAG-aware temporal models."""

    def __init__(self, n_models: int = 5, model_kwargs: dict | None = None):
        self.n_models = n_models
        self.model_kwargs = model_kwargs or {}
        self.models: list[DAGAwareTemporalWorldModel] = []

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
        histories = []
        for index in range(self.n_models):
            if verbose:
                print(f"\n=== Training ensemble member {index + 1}/{self.n_models} ===")
            model = DAGAwareTemporalWorldModel(**self.model_kwargs)
            member_save_dir = os.path.join(save_dir, f"member_{index}") if save_dir else None
            history = train_model(
                model,
                train_loader,
                val_loader,
                n_epochs=n_epochs,
                lr=lr,
                device=device,
                save_dir=member_save_dir,
                verbose=verbose,
                **loss_kwargs,
            )
            self.models.append(model)
            histories.append(history)
        return histories

    def save(self, save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        for index, model in enumerate(self.models):
            member_dir = os.path.join(save_dir, f"member_{index}")
            os.makedirs(member_dir, exist_ok=True)
            torch.save(model.state_dict(), os.path.join(member_dir, "model.pt"))

        config = {"n_models": self.n_models, "model_kwargs": self.model_kwargs}
        with open(os.path.join(save_dir, "ensemble_config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        print(f"Ensemble saved to {save_dir}")

    @classmethod
    def load(
        cls,
        save_dir: str,
        device: torch.device | None = None,
        prefer_best: bool = True,
    ) -> "EnsembleTrainer":
        with open(os.path.join(save_dir, "ensemble_config.json"), encoding="utf-8") as f:
            config = json.load(f)
        trainer = cls(n_models=config["n_models"], model_kwargs=config["model_kwargs"])
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        for index in range(config["n_models"]):
            model = DAGAwareTemporalWorldModel(**config["model_kwargs"])
            member_dir = os.path.join(save_dir, f"member_{index}")
            best_path = os.path.join(member_dir, "best_model.pt")
            final_path = os.path.join(member_dir, "model.pt")
            checkpoint_path = best_path if prefer_best and os.path.exists(best_path) else final_path
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.to(device)
            model.eval()
            trainer.models.append(model)
        which = "best_model.pt when available" if prefer_best else "model.pt only"
        print(f"Loaded ensemble ({config['n_models']} models) from {save_dir} using {which}")
        return trainer
