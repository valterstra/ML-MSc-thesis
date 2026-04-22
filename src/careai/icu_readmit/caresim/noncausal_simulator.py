from __future__ import annotations

import torch

from .noncausal_inference import NonCausalCareSimModel


class NonCausalCareSimEnvironment:
    """Single-model rollout wrapper for the non-causal CARE-Sim branch."""

    def __init__(
        self,
        model: NonCausalCareSimModel,
        max_steps: int = 20,
        device: torch.device | None = None,
    ) -> None:
        self.model = model
        self.max_steps = int(max_steps)
        self.device = device or model.device
        self._states: torch.Tensor | None = None
        self._actions: torch.Tensor | None = None
        self._time_steps: torch.Tensor | None = None
        self._done: torch.Tensor | None = None
        self._step_count = 0

    def reset(
        self,
        seed_states: torch.Tensor,
        seed_actions: torch.Tensor,
        seed_time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self._states = seed_states.to(self.device).float()
        self._actions = seed_actions.to(self.device).float()
        if seed_time_steps is None:
            seq_len = self._states.shape[1]
            seed_time_steps = torch.arange(seq_len, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(self._states.shape[0], 1)
        self._time_steps = seed_time_steps.to(self.device).float()
        self._done = torch.zeros(self._states.shape[0], dtype=torch.bool, device=self.device)
        self._step_count = 0
        return self._states[:, -1, :]

    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        assert self._states is not None and self._actions is not None and self._time_steps is not None

        batch_size = self._states.shape[0]
        action = action.to(self.device).float()

        new_action = action.unsqueeze(1)
        actions_extended = torch.cat([self._actions, new_action], dim=1)
        next_time = self._time_steps[:, -1:] + 1.0
        time_extended = torch.cat([self._time_steps, next_time], dim=1)
        last_state = self._states[:, -1:, :]
        states_extended = torch.cat([self._states, last_state], dim=1)

        context_len = max(int(self.model.max_seq_len), 1)
        pred = self.model.predict_last_step(
            states_extended[:, -context_len:, :],
            actions_extended[:, -context_len:, :],
            time_steps=time_extended[:, -context_len:],
        )

        next_state = pred["next_state_mean"]
        terminal_prob = pred["terminal_prob"]
        readmit_prob = pred["readmit_prob"]
        done = (terminal_prob > 0.5) | self._done

        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = torch.ones_like(done)

        self._states = states_extended.clone()
        self._states[:, -1, :] = next_state
        self._actions = actions_extended
        self._time_steps = time_extended
        if self._states.shape[1] > context_len:
            self._states = self._states[:, -context_len:, :]
            self._actions = self._actions[:, -context_len:, :]
            self._time_steps = self._time_steps[:, -context_len:]
        self._done = done

        reward = torch.zeros(batch_size, dtype=torch.float32, device=self.device)
        info = {
            "terminal_prob": terminal_prob.detach().cpu().numpy(),
            "readmit_prob": readmit_prob.detach().cpu().numpy(),
            "step": self._step_count,
        }
        return next_state, reward, done, info
