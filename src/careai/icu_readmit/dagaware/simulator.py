"""
Gym-like simulator wrapper for the DAG-aware temporal ensemble.
"""
from __future__ import annotations

import torch

from .ensemble import DAGAwareEnsemble
from ..caresim.readmit import LightGBMReadmitModel
from ..caresim.severity import RidgeSeveritySurrogate


class DAGAwareEnvironment:
    def __init__(
        self,
        ensemble: DAGAwareEnsemble,
        max_steps: int = 20,
        uncertainty_threshold: float = 1.0,
        device: torch.device | None = None,
        severity_model: RidgeSeveritySurrogate | None = None,
        terminal_outcome_model: LightGBMReadmitModel | None = None,
        terminal_reward_scale: float = 15.0,
    ):
        self.ensemble = ensemble
        self.max_steps = max_steps
        self.uncertainty_threshold = uncertainty_threshold
        self.device = device or ensemble.device
        self.severity_model = severity_model.to(self.device) if severity_model is not None else None
        self.terminal_outcome_model = terminal_outcome_model.to(self.device) if terminal_outcome_model is not None else None
        self.terminal_reward_scale = float(terminal_reward_scale)

        self._states: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self._time_steps: torch.Tensor | None = None
        self._step_count: int = 0
        self._done: torch.Tensor | None = None

    def reset(
        self,
        seed_states: torch.Tensor,
        seed_actions: torch.Tensor,
        seed_time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._states = seed_states.to(self.device).float()
        self._actions = seed_actions.to(self.device).float()
        if seed_time_steps is None:
            seq_len = seed_states.shape[1]
            seed_time_steps = (
                torch.arange(seq_len, device=self.device, dtype=torch.float32)
                .unsqueeze(0)
                .repeat(seed_states.shape[0], 1)
            )
        self._time_steps = seed_time_steps.to(self.device).float()
        self._step_count = 0
        self._done = torch.zeros(seed_states.shape[0], dtype=torch.bool, device=self.device)
        return self._states[:, -1, :]

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        assert self._states is not None, "Call reset() before step()"
        batch_size = self._states.shape[0]
        action = action.to(self.device).float()

        new_action = action.unsqueeze(1)
        actions_extended = torch.cat([self._actions, new_action], dim=1)
        next_time = self._time_steps[:, -1:] + 1.0
        time_extended = torch.cat([self._time_steps, next_time], dim=1)
        last_state = self._states[:, -1:, :]
        states_extended = torch.cat([self._states, last_state], dim=1)

        context_len = max(int(self.ensemble.max_seq_len), 1)
        states_context = states_extended[:, -context_len:, :]
        actions_context = actions_extended[:, -context_len:, :]
        time_context = time_extended[:, -context_len:]
        pred = self.ensemble.predict_last_step(states_context, actions_context, time_steps=time_context)

        next_state = pred["next_state_mean"]
        uncertainty = pred["next_state_std"].mean(dim=-1)
        if self.severity_model is not None:
            current_state = self._states[:, -1, :]
            reward = self.severity_model.score(current_state) - self.severity_model.score(next_state)
        else:
            reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        terminal_prob = pred["terminal_prob"]
        done = (terminal_prob > 0.5) | self._done

        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = torch.ones_like(done)
        newly_done = done & (~self._done)
        terminal_reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        p_readmit = torch.full((batch_size,), float("nan"), dtype=torch.float32, device=self.device)
        if self.terminal_outcome_model is not None and newly_done.any():
            reward_add, p_term = self.terminal_outcome_model.terminal_reward(
                next_state[newly_done],
                reward_scale=self.terminal_reward_scale,
            )
            terminal_reward[newly_done] = reward_add
            p_readmit[newly_done] = p_term
            reward = reward + terminal_reward

        self._states = states_extended.clone()
        self._states[:, -1, :] = next_state
        self._actions = actions_extended
        self._time_steps = time_extended
        if self._states.shape[1] > context_len:
            self._states = self._states[:, -context_len:, :]
            self._actions = self._actions[:, -context_len:, :]
            self._time_steps = self._time_steps[:, -context_len:]
        self._done = done

        info = {
            "uncertainty": uncertainty.cpu().numpy(),
            "terminal_prob": terminal_prob.cpu().numpy(),
            "uncertain_flag": (uncertainty > self.uncertainty_threshold).cpu().numpy(),
            "step": self._step_count,
            "reward_source": "severity" if self.severity_model is not None else "zero",
            "terminal_reward": terminal_reward.cpu().numpy(),
            "terminal_p_readmit": p_readmit.cpu().numpy(),
        }
        return next_state, reward, done, info

    def current_state(self) -> torch.Tensor:
        return self._states[:, -1, :]

    def rollout(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        horizon = actions.shape[1]
        all_states, all_rewards, all_dones, all_unc = [], [], [], []
        for step in range(horizon):
            action_t = actions[:, step, :]
            next_state, reward, done, info = self.step(action_t)
            all_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(done)
            all_unc.append(torch.from_numpy(info["uncertainty"]))
        return {
            "states": torch.stack(all_states, dim=1),
            "rewards": torch.stack(all_rewards, dim=1),
            "dones": torch.stack(all_dones, dim=1),
            "uncertainty": torch.stack(all_unc, dim=1),
        }
