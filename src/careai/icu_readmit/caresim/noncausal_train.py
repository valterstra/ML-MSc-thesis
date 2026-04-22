from __future__ import annotations

import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .noncausal_model import NonCausalCareSimTransformer


def compute_noncausal_loss(
    pred: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
    w_state: float = 1.0,
    w_term: float = 0.2,
    w_readmit: float = 0.2,
) -> tuple[torch.Tensor, dict[str, float]]:
    next_state_pred = pred["next_state"]
    next_state_true = batch["next_states"].to(next_state_pred.device)
    real_mask = ~batch["src_key_padding_mask"].to(next_state_pred.device)

    if not torch.isfinite(next_state_pred).all():
        raise ValueError("Non-finite values in predicted next_state")
    if not torch.isfinite(next_state_true).all():
        raise ValueError("Non-finite values in target next_state")

    state_err = ((next_state_pred - next_state_true) ** 2).mean(dim=-1)
    loss_state = (state_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    terminal_pred = pred["terminal"]
    terminal_true = batch["terminals"].to(terminal_pred.device)
    if not torch.isfinite(terminal_pred).all():
        raise ValueError("Non-finite values in predicted terminal logits")
    terminal_err = F.binary_cross_entropy_with_logits(terminal_pred, terminal_true, reduction="none")
    loss_term = (terminal_err * real_mask.float()).sum() / real_mask.float().sum().clamp(min=1)

    readmit_pred = pred["readmit"]
    readmit_true = batch["readmit"].to(readmit_pred.device)
    if not torch.isfinite(readmit_pred).all():
        raise ValueError("Non-finite values in predicted readmission logits")
    loss_readmit = F.binary_cross_entropy_with_logits(readmit_pred, readmit_true)

    total = w_state * loss_state + w_term * loss_term + w_readmit * loss_readmit
    if not torch.isfinite(total):
        raise ValueError("Non-finite total loss in non-causal CARE-Sim training")
    breakdown = {
        "state": float(loss_state.detach().item()),
        "terminal": float(loss_term.detach().item()),
        "readmit": float(loss_readmit.detach().item()),
        "total": float(total.detach().item()),
    }
    return total, breakdown


def _run_epoch(
    model: NonCausalCareSimTransformer,
    loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
    w_state: float,
    w_term: float,
    w_readmit: float,
    grad_clip: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)

    totals = {"state": 0.0, "terminal": 0.0, "readmit": 0.0, "total": 0.0}
    n_batches = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        states = batch["states"].to(device)
        actions = batch["actions"].to(device)
        mask = batch["src_key_padding_mask"].to(device)
        time_steps = batch["time_steps"].to(device)

        with torch.set_grad_enabled(training):
            pred = model(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
            loss, breakdown = compute_noncausal_loss(pred, batch, w_state=w_state, w_term=w_term, w_readmit=w_readmit)
            if training:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

        for key, value in breakdown.items():
            totals[key] += value
        n_batches += 1

    return {key: value / max(n_batches, 1) for key, value in totals.items()}


def train_noncausal_model(
    model: NonCausalCareSimTransformer,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    n_epochs: int,
    lr: float = 1e-3,
    device: torch.device | None = None,
    w_state: float = 1.0,
    w_term: float = 0.2,
    w_readmit: float = 0.2,
    grad_clip: float = 1.0,
    verbose: bool = True,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))

    history: list[dict] = []
    best_state = None
    best_val = float("inf")

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()
        train_losses = _run_epoch(
            model=model,
            loader=train_loader,
            device=device,
            optimizer=optimizer,
            w_state=w_state,
            w_term=w_term,
            w_readmit=w_readmit,
            grad_clip=grad_clip,
            max_batches=max_train_batches,
        )
        scheduler.step()

        entry = {"epoch": epoch, "train": train_losses}
        if val_loader is not None:
            with torch.no_grad():
                val_losses = _run_epoch(
                    model=model,
                    loader=val_loader,
                    device=device,
                    optimizer=None,
                    w_state=w_state,
                    w_term=w_term,
                    w_readmit=w_readmit,
                    grad_clip=grad_clip,
                    max_batches=max_val_batches,
                )
            entry["val"] = val_losses
            if val_losses["total"] < best_val:
                best_val = val_losses["total"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append(entry)
        if verbose:
            line = (
                f"Epoch {epoch:3d}/{n_epochs} "
                f"train={train_losses['total']:.4f} "
                f"(state={train_losses['state']:.4f}, term={train_losses['terminal']:.4f}, readmit={train_losses['readmit']:.4f})"
            )
            if "val" in entry:
                line += f" | val={entry['val']['total']:.4f}"
            line += f" [{time.time() - t0:.1f}s]"
            print(line, flush=True)

    return history, best_state
