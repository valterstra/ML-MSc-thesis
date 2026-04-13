"""
CareSimEnvironment: gym-like interface wrapping the CareSimEnsemble.

This is the virtual clinical trial environment described in slide 6 of the proposal.

The RL agent interacts with the simulator by:
  1. reset(seed_states, seed_actions) -- feed a patient's history (real observations up to now)
  2. step(action) -- propose a drug combination; simulator predicts next state + reward + done
  3. Repeat for H steps to generate a full simulated trajectory

The simulator maintains the full history internally so the GPT transformer can attend
over all past context when predicting the next state.

Uncertainty flag:
  After each step, info["uncertainty"] = mean std across ensemble (float).
  If this exceeds uncertainty_threshold, the prediction is flagged as unreliable.

Single-patient (B=1) use is the default for interactive use.
Batched rollout (B>1) is supported for offline trajectory augmentation.
"""
from __future__ import annotations

import torch
import numpy as np
from .ensemble import CareSimEnsemble
from .readmit import LightGBMReadmitModel
from .severity import RidgeSeveritySurrogate


class CareSimEnvironment:
    """Transformer-based ICU patient simulator.

    Args:
        ensemble             : trained CareSimEnsemble
        max_steps            : maximum rollout length (beyond seed history)
        uncertainty_threshold: flag predictions where ensemble std > this value
        device               : torch.device for inference
    """

    def __init__(
        self,
        ensemble: CareSimEnsemble,
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

        # Internal state
        self._states: torch.Tensor | None = None    # (B, T, state_dim)
        self._actions: torch.Tensor | None = None   # (B, T, action_dim)
        self._time_steps: torch.Tensor | None = None  # (B, T)
        self._step_count: int = 0
        self._done: torch.Tensor | None = None      # (B,) bool

    def reset(
        self,
        seed_states: torch.Tensor,
        seed_actions: torch.Tensor,
        seed_time_steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Initialize the simulator with a patient's observed history.

        Args:
            seed_states  : (B, T_seed, state_dim) -- real observed states
            seed_actions : (B, T_seed, action_dim) -- real actions taken
                           The last action corresponds to the CURRENT decision point.

        Returns:
            current_state : (B, state_dim) -- the last observed state
        """
        self._states = seed_states.to(self.device).float()
        self._actions = seed_actions.to(self.device).float()
        if seed_time_steps is None:
            T_seed = seed_states.shape[1]
            seed_time_steps = torch.arange(T_seed, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(seed_states.shape[0], 1)
        self._time_steps = seed_time_steps.to(self.device).float()
        self._step_count = 0
        B = seed_states.shape[0]
        self._done = torch.zeros(B, dtype=torch.bool, device=self.device)

        return self._states[:, -1, :]   # current state = last observed

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """Take one step: propose a drug combination, get next patient state.

        Args:
            action : (B, action_dim) float -- binary drug flags for next bloc

        Returns:
            next_state  : (B, state_dim)
            reward      : (B,)
            done        : (B,) bool
            info        : dict with uncertainty, terminal_prob
        """
        assert self._states is not None, "Call reset() before step()"

        B = self._states.shape[0]
        action = action.to(self.device).float()

        # Append the new action to history
        # The new state will be filled in after prediction
        new_action = action.unsqueeze(1)                   # (B, 1, action_dim)
        actions_extended = torch.cat([self._actions, new_action], dim=1)  # (B, T+1, action_dim)
        next_time = self._time_steps[:, -1:] + 1.0
        time_extended = torch.cat([self._time_steps, next_time], dim=1)   # (B, T+1)

        # Repeat last known state as placeholder for the new position
        last_state = self._states[:, -1:, :]               # (B, 1, state_dim)
        states_extended = torch.cat([self._states, last_state], dim=1)  # (B, T+1, state_dim)

        # Predict next state using the full history
        context_len = max(int(self.ensemble.max_seq_len), 1)
        states_context = states_extended[:, -context_len:, :]
        actions_context = actions_extended[:, -context_len:, :]
        time_context = time_extended[:, -context_len:]
        pred = self.ensemble.predict_last_step(states_context, actions_context, time_steps=time_context)

        next_state = pred["next_state_mean"]               # (B, state_dim)
        uncertainty = pred["next_state_std"].mean(dim=-1)  # (B,)
        if self.severity_model is not None:
            current_state = self._states[:, -1, :]
            reward = self.severity_model.score(current_state) - self.severity_model.score(next_state)
        elif pred["reward_mean"] is not None:
            reward = pred["reward_mean"]                   # (B,)
        else:
            reward = torch.zeros(B, dtype=torch.float32, device=self.device)
        terminal_prob = pred["terminal_prob"]              # (B,)
        done = (terminal_prob > 0.5) | self._done          # (B,) -- once done, stays done

        # Check max steps
        self._step_count += 1
        if self._step_count >= self.max_steps:
            done = torch.ones_like(done)
        newly_done = done & (~self._done)
        terminal_reward = torch.zeros(B, dtype=torch.float32, device=self.device)
        p_readmit = torch.full((B,), float("nan"), dtype=torch.float32, device=self.device)
        if self.terminal_outcome_model is not None and newly_done.any():
            reward_add, p_term = self.terminal_outcome_model.terminal_reward(
                next_state[newly_done],
                reward_scale=self.terminal_reward_scale,
            )
            terminal_reward[newly_done] = reward_add
            p_readmit[newly_done] = p_term
            reward = reward + terminal_reward

        # Update internal history
        self._states = states_extended.clone()
        self._states[:, -1, :] = next_state               # fill in the predicted state
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
            "reward_source": "severity" if self.severity_model is not None else ("head" if pred["reward_mean"] is not None else "zero"),
            "terminal_reward": terminal_reward.cpu().numpy(),
            "terminal_p_readmit": p_readmit.cpu().numpy(),
        }

        return next_state, reward, done, info

    def current_state(self) -> torch.Tensor:
        """Return the most recently predicted (or seeded) state."""
        return self._states[:, -1, :]

    def rollout(
        self,
        actions: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Execute a fixed action sequence without re-seeding.

        Useful for evaluating a specific drug treatment plan over H steps.

        Args:
            actions : (B, H, action_dim) -- action sequence to execute

        Returns dict with:
            states      : (B, H, state_dim) -- predicted states at each step
            rewards     : (B, H)
            dones       : (B, H) bool
            uncertainty : (B, H)
        """
        H = actions.shape[1]
        all_states, all_rewards, all_dones, all_unc = [], [], [], []

        for t in range(H):
            a_t = actions[:, t, :]
            next_state, reward, done, info = self.step(a_t)
            all_states.append(next_state)
            all_rewards.append(reward)
            all_dones.append(done)
            all_unc.append(torch.from_numpy(info["uncertainty"]))

        return {
            "states": torch.stack(all_states, dim=1),      # (B, H, state_dim)
            "rewards": torch.stack(all_rewards, dim=1),    # (B, H)
            "dones": torch.stack(all_dones, dim=1),        # (B, H) bool
            "uncertainty": torch.stack(all_unc, dim=1),    # (B, H)
        }
