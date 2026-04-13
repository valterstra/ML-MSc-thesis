"""Simulator-based DDQN training for the causal Markov baseline."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import asdict, dataclass
import logging
import random

import torch
import numpy as np
import torch.nn.functional as F

from ...rl.networks import DuelingDQN
from ...caresim.readmit import LightGBMReadmitModel
from ...caresim.severity import RidgeSeveritySurrogate
from ..model import MarkovSimEnsemble
from ..simulator import MarkovSimEnvironment
from .actions import ACTION_GRID, action_tensor, build_action_grid
from .evaluation import SeedEpisode
from .observation import ObservationBuilder


@dataclass
class DQNConfig:
    observation_window: int = 5
    rollout_steps: int = 5
    gamma: float = 0.99
    lr: float = 1e-4
    batch_size: int = 64
    replay_capacity: int = 20000
    warmup_steps: int = 500
    train_steps: int = 20000
    target_sync_every: int = 250
    epsilon_start: float = 1.0
    epsilon_end: float = 0.10
    epsilon_decay_steps: int = 20000
    uncertainty_penalty: float = 0.25
    uncertainty_threshold: float = 1.0
    log_every: int = 100
    seed: int = 42


def _epsilon(step: int, config: DQNConfig) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    frac = step / max(config.epsilon_decay_steps, 1)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def _sample_action(model: DuelingDQN, obs: np.ndarray, epsilon: float, device: torch.device, rng: np.random.Generator) -> int:
    n_actions = model.adv_out.out_features
    if rng.random() < epsilon:
        return int(rng.integers(0, n_actions))
    model.eval()
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(obs_t)
    return int(q_values.argmax(dim=1).item())


def _greedy_action(model: DuelingDQN, obs: np.ndarray, device: torch.device) -> int:
    model.eval()
    with torch.no_grad():
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        q_values = model(obs_t)
    return int(q_values.argmax(dim=1).item())


def train_ddqn(
    ensemble: MarkovSimEnsemble,
    episodes: list[SeedEpisode],
    config: DQNConfig,
    device: torch.device,
    severity_model: RidgeSeveritySurrogate | None = None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> tuple[DuelingDQN, dict]:
    if not episodes:
        raise ValueError("No training episodes were provided")

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    rng = np.random.default_rng(config.seed)

    action_dim = int(episodes[0].seed_actions.shape[1])
    state_dim = int(episodes[0].seed_states.shape[1])
    action_grid = ACTION_GRID if action_dim == ACTION_GRID.shape[1] else build_action_grid(action_dim)
    obs_builder = ObservationBuilder(window_len=config.observation_window, state_dim=state_dim, action_dim=action_dim)
    obs_dim = obs_builder.obs_dim
    online = DuelingDQN(n_input=obs_dim, n_actions=len(action_grid), hidden=128).to(device)
    target = copy.deepcopy(online).to(device)
    target.eval()
    optimizer = torch.optim.Adam(online.parameters(), lr=config.lr)
    replay: deque[tuple[np.ndarray, int, float, np.ndarray, float]] = deque(maxlen=config.replay_capacity)

    metrics = {
        "losses": [],
        "episode_returns": [],
        "episode_lengths": [],
        "episode_uncertainty": [],
        "epsilon": [],
        "config": asdict(config),
    }

    env = MarkovSimEnvironment(
        ensemble,
        max_steps=config.rollout_steps,
        uncertainty_threshold=config.uncertainty_threshold,
        device=device,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=terminal_reward_scale,
    )

    for step in range(config.train_steps):
        episode = episodes[step % len(episodes)]
        seed_states_t = torch.tensor(episode.seed_states, dtype=torch.float32, device=device).unsqueeze(0)
        seed_actions_t = torch.tensor(episode.seed_actions, dtype=torch.float32, device=device).unsqueeze(0)
        seed_time_steps_t = torch.tensor(episode.seed_time_steps, dtype=torch.float32, device=device).unsqueeze(0)
        env.reset(seed_states_t, seed_actions_t, seed_time_steps=seed_time_steps_t)
        obs = obs_builder.reset(episode.seed_states, episode.seed_actions)

        total_reward = 0.0
        total_uncertainty = 0.0
        rollout_len = 0
        epsilon = _epsilon(step, config)

        for _ in range(config.rollout_steps):
            action_id = _sample_action(online, obs, epsilon, device, rng)
            action = action_tensor(action_id, device=device, action_dim=action_dim)
            next_state, reward, done_tensor, info = env.step(action)

            reward_value = float(reward[0].item())
            uncertainty_value = float(info["uncertainty"][0])
            shaped_reward = reward_value - config.uncertainty_penalty * uncertainty_value
            done = float(bool(done_tensor[0].item()))

            next_obs = obs_builder.append(
                next_state[0].detach().cpu().numpy(),
                np.asarray(action_grid[action_id], dtype=np.float32),
            )
            replay.append((obs.copy(), int(action_id), shaped_reward, next_obs.copy(), done))

            obs = next_obs
            total_reward += shaped_reward
            total_uncertainty += uncertainty_value
            rollout_len += 1
            if done:
                break

        metrics["episode_returns"].append(total_reward)
        metrics["episode_lengths"].append(rollout_len)
        metrics["episode_uncertainty"].append(total_uncertainty / max(rollout_len, 1))
        metrics["epsilon"].append(epsilon)

        if len(replay) < max(config.batch_size, config.warmup_steps):
            continue

        batch = random.sample(replay, config.batch_size)
        obs_batch = torch.tensor(np.stack([row[0] for row in batch]), dtype=torch.float32, device=device)
        action_batch = torch.tensor([row[1] for row in batch], dtype=torch.long, device=device)
        reward_batch = torch.tensor([row[2] for row in batch], dtype=torch.float32, device=device)
        next_obs_batch = torch.tensor(np.stack([row[3] for row in batch]), dtype=torch.float32, device=device)
        done_batch = torch.tensor([row[4] for row in batch], dtype=torch.float32, device=device)

        online.train()
        q_values = online(obs_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = online(next_obs_batch).argmax(dim=1)
            next_q = target(next_obs_batch).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = reward_batch + config.gamma * (1.0 - done_batch) * next_q

        loss = F.smooth_l1_loss(q_values, target_q)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(online.parameters(), max_norm=5.0)
        optimizer.step()
        metrics["losses"].append(float(loss.item()))

        if (step + 1) % config.target_sync_every == 0:
            target.load_state_dict(online.state_dict())

        if config.log_every > 0 and ((step + 1) % config.log_every == 0 or step == 0):
            recent_returns = metrics["episode_returns"][-config.log_every:]
            recent_unc = metrics["episode_uncertainty"][-config.log_every:]
            recent_lengths = metrics["episode_lengths"][-config.log_every:]
            recent_losses = metrics["losses"][-config.log_every:] if metrics["losses"] else []
            logging.info(
                "DDQN step %d/%d | eps=%.3f | replay=%d | mean_return=%.4f | mean_unc=%.4f | mean_len=%.2f | mean_loss=%s",
                step + 1,
                config.train_steps,
                epsilon,
                len(replay),
                float(np.mean(recent_returns)) if recent_returns else float("nan"),
                float(np.mean(recent_unc)) if recent_unc else float("nan"),
                float(np.mean(recent_lengths)) if recent_lengths else float("nan"),
                f"{float(np.mean(recent_losses)):.4f}" if recent_losses else "n/a",
            )

    return online.eval(), metrics


def greedy_policy_from_model(model: DuelingDQN, device: torch.device):
    def _policy(env, obs_builder, current_obs, episode):
        return {"action_id": _greedy_action(model, current_obs, device=device)}

    return _policy
