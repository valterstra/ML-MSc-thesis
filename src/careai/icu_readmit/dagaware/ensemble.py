"""
Inference-only ensemble wrapper for the DAG-aware temporal world model.
"""
from __future__ import annotations

import torch

from .model import DAGAwareTemporalWorldModel
from .train import EnsembleTrainer


class DAGAwareEnsemble:
    def __init__(self, models: list[DAGAwareTemporalWorldModel], device: torch.device | None = None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.models = [model.to(device).eval() for model in models]
        self.n_models = len(models)
        self.max_seq_len = min((model.max_seq_len for model in self.models), default=0)

    @classmethod
    def from_trainer(cls, trainer: EnsembleTrainer, device: torch.device | None = None) -> "DAGAwareEnsemble":
        return cls(trainer.models, device=device)

    @classmethod
    def from_dir(cls, save_dir: str, device: torch.device | None = None) -> "DAGAwareEnsemble":
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
        states = states.to(self.device)
        actions = actions.to(self.device)
        if src_key_padding_mask is not None:
            src_key_padding_mask = src_key_padding_mask.to(self.device)
        if time_steps is not None:
            time_steps = time_steps.to(self.device)

        all_ns, all_term = [], []
        for model in self.models:
            out = model(states, actions, src_key_padding_mask=src_key_padding_mask, time_steps=time_steps)
            all_ns.append(out["next_state"])
            all_term.append(torch.sigmoid(out["terminal"]))

        ns_stack = torch.stack(all_ns, dim=0)
        term_stack = torch.stack(all_term, dim=0)
        return {
            "next_state_mean": ns_stack.mean(dim=0),
            "next_state_std": ns_stack.std(dim=0),
            "reward_mean": None,
            "reward_std": None,
            "terminal_prob": term_stack.mean(dim=0),
        }

    @torch.no_grad()
    def predict_last_step(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        out = self.predict(states, actions, time_steps=time_steps)
        return {
            "next_state_mean": out["next_state_mean"][:, -1, :],
            "next_state_std": out["next_state_std"][:, -1, :],
            "reward_mean": None,
            "terminal_prob": out["terminal_prob"][:, -1],
        }

    def uncertainty_score(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        out = self.predict_last_step(states, actions, time_steps=time_steps)
        return out["next_state_std"].mean(dim=-1)
