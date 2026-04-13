from __future__ import annotations

import json
from pathlib import Path

import torch
import joblib
import numpy as np


HANDCRAFTED_SELECTED_WEIGHTS = {
    "Hb": 0.28,
    "BUN": 0.24,
    "Creatinine": 0.24,
    "Phosphate": 0.10,
    "HR": 0.09,
    "Chloride": 0.05,
}


class HandcraftedSelectedSeverity:
    """Handcrafted severity index on the selected transformed state space.

    Expects the simulator state to already be in the replay/model space:
      clipped, partly log-transformed, and z-scored on the ICU training cohort.

    Interpretation:
      - low Hb is worse
      - high BUN / Creatinine / Phosphate / HR are worse
      - Chloride is penalized for deviation from the ICU cohort centre
    """

    def __init__(
        self,
        state_feature_names: list[str],
        weights: dict[str, float] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.state_feature_names = list(state_feature_names)
        self.weights = dict(weights or HANDCRAFTED_SELECTED_WEIGHTS)
        self.device = device or torch.device("cpu")
        self._idx = {name: self.state_feature_names.index(name) for name in self.weights.keys()}

    def to(self, device: torch.device) -> "HandcraftedSelectedSeverity":
        self.device = device
        return self

    def score(self, states: torch.Tensor) -> torch.Tensor:
        states = states.to(self.device).float()
        score = torch.zeros(states.shape[:-1], dtype=torch.float32, device=self.device)
        score = score + float(self.weights["Hb"]) * torch.relu(-states[..., self._idx["Hb"]])
        score = score + float(self.weights["BUN"]) * torch.relu(states[..., self._idx["BUN"]])
        score = score + float(self.weights["Creatinine"]) * torch.relu(states[..., self._idx["Creatinine"]])
        score = score + float(self.weights["Phosphate"]) * torch.relu(states[..., self._idx["Phosphate"]])
        score = score + float(self.weights["HR"]) * torch.relu(states[..., self._idx["HR"]])
        score = score + float(self.weights["Chloride"]) * torch.abs(states[..., self._idx["Chloride"]])
        return score


class RidgeSeveritySurrogate:
    """Frozen ridge surrogate mapping selected dynamic state to a severity score."""

    def __init__(
        self,
        coefficients: np.ndarray,
        intercept: float,
        feature_names: list[str],
        scaler_params: dict[str, dict],
        target_clip: tuple[float, float] | None = None,
        state_feature_names: list[str] | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.feature_names = list(feature_names)
        self.intercept = float(intercept)
        self.scaler_params = scaler_params
        self.target_clip = tuple(target_clip) if target_clip is not None else None
        self.state_feature_names = list(state_feature_names or feature_names)
        self.device = device or torch.device("cpu")

        coef = torch.tensor(np.asarray(coefficients, dtype=np.float32), dtype=torch.float32, device=self.device)
        self._coef = coef.reshape(-1)
        self._feature_idx = [self.state_feature_names.index(name) for name in self.feature_names]

    @classmethod
    def from_dir(
        cls,
        model_dir: str,
        state_feature_names: list[str],
        device: torch.device | None = None,
    ) -> "RidgeSeveritySurrogate":
        model_path = Path(model_dir) / "ridge_sofa_surrogate.joblib"
        config_path = Path(model_dir) / "severity_surrogate_config.json"
        payload = joblib.load(model_path)
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        model = payload["model"] if isinstance(payload, dict) and "model" in payload else payload
        return cls(
            coefficients=np.asarray(model.coef_, dtype=np.float32),
            intercept=float(model.intercept_),
            feature_names=list(cfg["feature_names"]),
            scaler_params=dict(cfg["scaler_params"]),
            target_clip=tuple(cfg.get("target_clip", [])) if cfg.get("target_clip") is not None else None,
            state_feature_names=state_feature_names,
            device=device,
        )

    def to(self, device: torch.device) -> "RidgeSeveritySurrogate":
        self.device = device
        self._coef = self._coef.to(device)
        return self

    def _transform_feature(self, values: torch.Tensor, name: str) -> torch.Tensor:
        params = self.scaler_params[name]
        clip_lo, clip_hi = params["clip"]
        x = values.clamp(float(clip_lo), float(clip_hi))
        if params.get("log1p", False):
            x = torch.log1p(x)
        mean = float(params["mean"])
        std = float(params["std"])
        return (x - mean) / max(std, 1e-8)

    def score(self, states: torch.Tensor) -> torch.Tensor:
        """Compute severity score for batched states shaped (..., state_dim)."""
        states = states.to(self.device).float()
        transformed = []
        for coef_idx, state_idx in enumerate(self._feature_idx):
            name = self.feature_names[coef_idx]
            transformed.append(self._transform_feature(states[..., state_idx], name))
        x = torch.stack(transformed, dim=-1)
        score = (x * self._coef).sum(dim=-1) + self.intercept
        if self.target_clip is not None:
            score = score.clamp(float(self.target_clip[0]), float(self.target_clip[1]))
        return score


def load_severity_model(
    mode: str,
    state_feature_names: list[str],
    model_dir: str | None = None,
    device: torch.device | None = None,
):
    if mode == "handcrafted":
        return HandcraftedSelectedSeverity(state_feature_names=state_feature_names, device=device)
    if mode == "surrogate":
        if not model_dir:
            raise ValueError("model_dir is required for surrogate severity mode")
        return RidgeSeveritySurrogate.from_dir(
            model_dir=model_dir,
            state_feature_names=state_feature_names,
            device=device,
        )
    raise ValueError(f"Unsupported severity mode: {mode!r}")
