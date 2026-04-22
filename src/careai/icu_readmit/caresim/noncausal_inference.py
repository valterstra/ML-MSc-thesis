from __future__ import annotations

import json
from pathlib import Path

import torch

from .noncausal_model import NonCausalCareSimTransformer


class NonCausalCareSimModel:
    """Inference wrapper for a trained non-causal CARE-Sim model."""

    def __init__(
        self,
        model: NonCausalCareSimTransformer,
        config: dict,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device).eval()
        self.config = config
        self.state_cols = list(config["state_cols"])
        self.next_state_cols = list(config["next_state_cols"])
        self.action_cols = list(config["action_cols"])
        self.dynamic_state_names = [c.removeprefix("s_next_") for c in self.next_state_cols]
        self.dynamic_state_idx = [self.state_cols.index(f"s_{name}") for name in self.dynamic_state_names]
        self.static_state_idx = [i for i in range(len(self.state_cols)) if i not in self.dynamic_state_idx]
        self.max_seq_len = int(config["model_kwargs"]["max_seq_len"])

    @classmethod
    def from_dir(cls, model_dir: str, device: torch.device | None = None, which: str = "best") -> "NonCausalCareSimModel":
        model_root = Path(model_dir)
        with open(model_root / "train_config.json", "r", encoding="utf-8") as f:
            config = json.load(f)

        model = NonCausalCareSimTransformer(**config["model_kwargs"])
        if which == "best" and (model_root / "best_model.pt").exists():
            checkpoint = model_root / "best_model.pt"
        else:
            checkpoint = model_root / "model.pt"

        map_location = device or torch.device("cpu")
        state_dict = torch.load(checkpoint, map_location=map_location)
        model.load_state_dict(state_dict)
        return cls(model=model, config=config, device=device)

    def _assemble_full_next_state(self, current_states: torch.Tensor, next_state_dynamic: torch.Tensor) -> torch.Tensor:
        full_next = current_states.clone()
        full_next[..., self.dynamic_state_idx] = next_state_dynamic
        return full_next

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

        raw = self.model(states, actions, src_key_padding_mask=src_key_padding_mask, time_steps=time_steps)
        next_state_full = self._assemble_full_next_state(states, raw["next_state"])
        return {
            "next_state_mean": next_state_full,
            "next_state_dynamic_mean": raw["next_state"],
            "terminal_prob": torch.sigmoid(raw["terminal"]),
            "readmit_prob": torch.sigmoid(raw["readmit"]),
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
            "next_state_dynamic_mean": out["next_state_dynamic_mean"][:, -1, :],
            "terminal_prob": out["terminal_prob"][:, -1],
            "readmit_prob": out["readmit_prob"],
        }
