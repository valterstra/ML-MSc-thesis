"""
CareSimEnsemble: ensemble of trained CareSimGPT models.

Wraps multiple models trained by EnsembleTrainer and provides:
  predict()     : mean prediction + uncertainty (std across ensemble)
  uncertainty() : per-feature std as out-of-distribution signal

The ensemble is used by CareSimEnvironment as the actual simulator backend.
"""
from __future__ import annotations

import torch
import numpy as np
from .model import CareSimGPT
from .train import EnsembleTrainer


class CareSimEnsemble:
    """Inference-only ensemble wrapper for a trained EnsembleTrainer.

    Args:
        models : list of trained CareSimGPT models (moved to eval mode)
        device : torch.device for inference
    """

    def __init__(self, models: list[CareSimGPT], device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.models = [m.to(device).eval() for m in models]
        self.n_models = len(models)
        self.max_seq_len = min((m.max_seq_len for m in self.models), default=0)

    @classmethod
    def from_trainer(cls, trainer: EnsembleTrainer, device: torch.device | None = None) -> "CareSimEnsemble":
        return cls(trainer.models, device=device)

    @classmethod
    def from_dir(cls, save_dir: str, device: torch.device | None = None) -> "CareSimEnsemble":
        trainer = EnsembleTrainer.load(save_dir, device=device)
        return cls(trainer.models, device=device)

    @torch.no_grad()
    def predict(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run all models and return mean + std predictions.

        Args:
            states               : (B, T, state_dim)
            actions              : (B, T, action_dim)
            src_key_padding_mask : (B, T) bool or None

        Returns dict with:
            next_state_mean : (B, T, state_dim) -- ensemble mean
            next_state_std  : (B, T, state_dim) -- ensemble std (uncertainty)
            reward_mean     : (B, T) or None
            reward_std      : (B, T) or None
            terminal_prob   : (B, T)            -- mean sigmoid of terminal logits
        """
        states = states.to(self.device)
        actions = actions.to(self.device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)
        if time_steps is not None:
            time_steps = time_steps.to(self.device)

        all_ns, all_rw, all_term = [], [], []
        for model in self.models:
            out = model(states, actions, src_key_padding_mask=src_key_padding_mask, time_steps=time_steps)
            all_ns.append(out["next_state"])
            if out["reward"] is not None:
                all_rw.append(out["reward"])
            all_term.append(torch.sigmoid(out["terminal"]))

        ns_stack = torch.stack(all_ns, dim=0)    # (N, B, T, state_dim)
        term_stack = torch.stack(all_term, dim=0) # (N, B, T)
        if all_rw:
            rw_stack = torch.stack(all_rw, dim=0)    # (N, B, T)
            reward_mean = rw_stack.mean(dim=0)
            reward_std = rw_stack.std(dim=0)
        else:
            reward_mean = None
            reward_std = None

        return {
            "next_state_mean": ns_stack.mean(dim=0),
            "next_state_std": ns_stack.std(dim=0),
            "reward_mean": reward_mean,
            "reward_std": reward_std,
            "terminal_prob": term_stack.mean(dim=0),
        }

    @torch.no_grad()
    def predict_last_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Predict only the LAST time step (for simulator rollout).

        Args:
            states  : (B, T, state_dim)
            actions : (B, T, action_dim)

        Returns dict with:
            next_state_mean : (B, state_dim)
            next_state_std  : (B, state_dim)
            reward_mean     : (B,) or None
            terminal_prob   : (B,)
        """
        out = self.predict(states, actions, time_steps=time_steps)
        return {
            "next_state_mean": out["next_state_mean"][:, -1, :],
            "next_state_std": out["next_state_std"][:, -1, :],
            "reward_mean": None if out["reward_mean"] is None else out["reward_mean"][:, -1],
            "terminal_prob": out["terminal_prob"][:, -1],
        }

    def uncertainty_score(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Scalar uncertainty score for the last step: mean std across state features.

        High scores signal the model is making predictions outside its training distribution
        (e.g., rare drug combinations or extreme lab values).

        Returns:
            uncertainty : (B,) -- mean std across state_dim at the last position
        """
        out = self.predict_last_step(states, actions, time_steps=time_steps)
        return out["next_state_std"].mean(dim=-1)   # (B,)
