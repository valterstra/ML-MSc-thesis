from __future__ import annotations

import json
from pathlib import Path

import torch
import joblib
import numpy as np
import pandas as pd


class LightGBMReadmitModel:
    """Readmission risk model on the selected transformed terminal state."""

    def __init__(
        self,
        model,
        feature_names: list[str],
        state_feature_names: list[str],
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.feature_names = list(feature_names)
        self.state_feature_names = list(state_feature_names)
        self.device = device or torch.device("cpu")
        self._feature_idx = [self.state_feature_names.index(name) for name in self.feature_names]

    @classmethod
    def from_dir(
        cls,
        model_dir: str,
        state_feature_names: list[str],
        device: torch.device | None = None,
    ) -> "LightGBMReadmitModel":
        model_path = Path(model_dir) / "terminal_readmit_selected.joblib"
        config_path = Path(model_dir) / "terminal_readmit_selected_config.json"
        payload = joblib.load(model_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        return cls(
            model=model,
            feature_names=list(cfg["feature_names"]),
            state_feature_names=state_feature_names,
            device=device,
        )

    def to(self, device: torch.device) -> "LightGBMReadmitModel":
        self.device = device
        return self

    def predict_proba(self, states: torch.Tensor) -> torch.Tensor:
        """Return P(readmit=1) for batched states shaped (..., state_dim)."""
        states = states.detach().to("cpu").float()
        flat = states.reshape(-1, states.shape[-1]).numpy()
        x = pd.DataFrame(flat[:, self._feature_idx], columns=self.feature_names)
        prob = self.model.predict_proba(x)[:, 1]
        out = torch.tensor(prob, dtype=torch.float32, device=self.device)
        return out.reshape(states.shape[:-1])

    def terminal_reward(self, states: torch.Tensor, reward_scale: float = 15.0) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (reward, p_readmit) on terminal states."""
        p_readmit = self.predict_proba(states)
        reward = float(reward_scale) - 2.0 * float(reward_scale) * p_readmit
        return reward, p_readmit
