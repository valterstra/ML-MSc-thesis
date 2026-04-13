from __future__ import annotations

import json
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch
import numpy as np
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler


STATE_FEATURE_NAMES = [
    "Hb",
    "BUN",
    "Creatinine",
    "Phosphate",
    "HR",
    "Chloride",
    "age",
    "charlson_score",
    "prior_ed_visits_6m",
]
ACTION_FEATURE_NAMES = ["vasopressor", "ivfluid", "antibiotic", "diuretic", "mechvent"]
STATE_COLS = [f"s_{name}" for name in STATE_FEATURE_NAMES]
ACTION_COLS = [f"{name}_b" for name in ACTION_FEATURE_NAMES]
NEXT_STATE_COLS = [f"s_next_{name}" for name in STATE_FEATURE_NAMES]

DYNAMIC_STATE_NAMES = STATE_FEATURE_NAMES[:6]
STATIC_STATE_NAMES = STATE_FEATURE_NAMES[6:]
DYNAMIC_STATE_IDX = tuple(range(6))
STATIC_STATE_IDX = tuple(range(6, 9))
ACTION_FEATURE_OFFSET = len(STATE_COLS)

# Rows = state dims, cols = actions. Matches the selected causal transformer mask.
SELECTED_CAUSAL_ACTION_MASK = np.array([
    [1, 1, 1, 0, 1],  # Hb
    [0, 1, 0, 1, 0],  # BUN
    [0, 1, 0, 1, 0],  # Creatinine
    [0, 1, 1, 1, 0],  # Phosphate
    [1, 0, 0, 0, 1],  # HR
    [0, 1, 0, 1, 0],  # Chloride
    [0, 0, 0, 0, 0],  # age
    [0, 0, 0, 0, 0],  # charlson_score
    [0, 0, 0, 0, 0],  # prior_ed_visits_6m
], dtype=np.float32)


@dataclass
class MarkovSimConfig:
    ridge_alpha: float = 1.0
    terminal_c: float = 1.0
    max_iter: int = 1000


class MarkovSimEnsemble:
    """Deterministic selected-track Markov simulator with residual std uncertainty."""

    def __init__(
        self,
        transition_models: list[Ridge],
        terminal_model: LogisticRegression,
        feature_scaler: StandardScaler,
        next_state_std: np.ndarray,
        action_mask_matrix: np.ndarray,
        config: MarkovSimConfig,
        device: torch.device | None = None,
    ):
        self.transition_models = transition_models
        self.terminal_model = terminal_model
        self.feature_scaler = feature_scaler
        self.next_state_std = np.asarray(next_state_std, dtype=np.float32).reshape(-1)
        self.action_mask_matrix = np.asarray(action_mask_matrix, dtype=np.float32)
        self.config = config
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_seq_len = 1
        self.n_models = 1

    @property
    def state_dim(self) -> int:
        return len(STATE_COLS)

    @property
    def action_dim(self) -> int:
        return len(ACTION_COLS)

    @property
    def dynamic_dim(self) -> int:
        return len(DYNAMIC_STATE_IDX)

    def _scale_features(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        x = np.concatenate([states, actions], axis=-1).astype(np.float32, copy=False)
        flat = x.reshape(-1, x.shape[-1])
        return self.feature_scaler.transform(flat)

    def _masked_transition_features(self, x_scaled: np.ndarray, dynamic_idx: int) -> np.ndarray:
        x_masked = x_scaled.copy()
        action_mask = self.action_mask_matrix[dynamic_idx]
        disallowed = np.where(action_mask < 0.5)[0]
        if len(disallowed):
            x_masked[:, ACTION_FEATURE_OFFSET + disallowed] = 0.0
        return x_masked

    def _predict_flat_np(self, states: np.ndarray, actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_scaled = self._scale_features(states, actions)
        dyn_pred = np.zeros((x_scaled.shape[0], self.dynamic_dim), dtype=np.float32)
        for dyn_pos, state_idx in enumerate(DYNAMIC_STATE_IDX):
            dyn_pred[:, dyn_pos] = self.transition_models[dyn_pos].predict(
                self._masked_transition_features(x_scaled, state_idx)
            ).astype(np.float32)
        term_prob = self.terminal_model.predict_proba(x_scaled)[:, 1].astype(np.float32)
        return dyn_pred, term_prob

    @torch.no_grad()
    def predict(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
        time_steps: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        del time_steps  # unused in the Markov baseline
        states_np = states.detach().cpu().numpy().astype(np.float32)
        actions_np = actions.detach().cpu().numpy().astype(np.float32)

        B, T, _ = states_np.shape
        dyn_pred, term_prob = self._predict_flat_np(states_np.reshape(-1, self.state_dim), actions_np.reshape(-1, self.action_dim))

        next_state = states_np.copy()
        next_state[..., DYNAMIC_STATE_IDX] = dyn_pred.reshape(B, T, self.dynamic_dim)

        next_state_std = np.zeros_like(next_state, dtype=np.float32)
        next_state_std[..., DYNAMIC_STATE_IDX] = self.next_state_std.reshape(1, 1, -1)

        out = {
            "next_state_mean": torch.tensor(next_state, dtype=torch.float32, device=self.device),
            "next_state_std": torch.tensor(next_state_std, dtype=torch.float32, device=self.device),
            "reward_mean": None,
            "reward_std": None,
            "terminal_prob": torch.tensor(term_prob.reshape(B, T), dtype=torch.float32, device=self.device),
        }

        if src_key_padding_mask is not None:
            mask = src_key_padding_mask.to(self.device)
            out["next_state_mean"] = out["next_state_mean"].masked_fill(mask.unsqueeze(-1), 0.0)
            out["next_state_std"] = out["next_state_std"].masked_fill(mask.unsqueeze(-1), 0.0)
            out["terminal_prob"] = out["terminal_prob"].masked_fill(mask, 0.0)

        return out

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

    def save(self, save_dir: str) -> None:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        bundle = {
            "transition_models": self.transition_models,
            "terminal_model": self.terminal_model,
            "feature_scaler": self.feature_scaler,
            "next_state_std": self.next_state_std,
            "action_mask_matrix": self.action_mask_matrix,
            "config": self.config,
        }
        with open(path / "markovsim_bundle.pkl", "wb") as f:
            pickle.dump(bundle, f)
        meta = {
            "state_cols": STATE_COLS,
            "action_cols": ACTION_COLS,
            "next_state_cols": NEXT_STATE_COLS,
            "dynamic_state_idx": list(DYNAMIC_STATE_IDX),
            "static_state_idx": list(STATIC_STATE_IDX),
            "dynamic_state_names": DYNAMIC_STATE_NAMES,
            "static_state_names": STATIC_STATE_NAMES,
            "action_feature_names": ACTION_FEATURE_NAMES,
            "causal_action_mask": self.action_mask_matrix.tolist(),
            "ridge_alpha": self.config.ridge_alpha,
            "terminal_c": self.config.terminal_c,
            "max_iter": self.config.max_iter,
        }
        with open(path / "markovsim_config.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    @classmethod
    def from_dir(cls, save_dir: str, device: torch.device | None = None) -> "MarkovSimEnsemble":
        path = Path(save_dir)
        with open(path / "markovsim_bundle.pkl", "rb") as f:
            bundle = pickle.load(f)
        return cls(
            transition_models=bundle["transition_models"],
            terminal_model=bundle["terminal_model"],
            feature_scaler=bundle["feature_scaler"],
            next_state_std=bundle["next_state_std"],
            action_mask_matrix=bundle["action_mask_matrix"],
            config=bundle["config"],
            device=device,
        )
