"""Simple non-contextual multi-armed bandit baseline for ICU control."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path

import numpy as np
import torch


def build_action_grid(action_dim: int) -> np.ndarray:
    return np.array(
        [[(action_id >> bit) & 1 for bit in range(action_dim)] for action_id in range(2 ** action_dim)],
        dtype=np.float32,
    )


@dataclass
class BanditConfig:
    rollout_steps: int = 5
    train_steps: int = 5000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 5000
    uncertainty_penalty: float = 0.25
    log_every: int = 100
    seed: int = 42


def _epsilon(step: int, config: BanditConfig) -> float:
    if step >= config.epsilon_decay_steps:
        return config.epsilon_end
    frac = step / max(config.epsilon_decay_steps, 1)
    return config.epsilon_start + frac * (config.epsilon_end - config.epsilon_start)


def _sample_action(values: np.ndarray, epsilon: float, rng: np.random.Generator) -> int:
    if rng.random() < epsilon:
        return int(rng.integers(0, len(values)))
    return int(np.argmax(values))


def train_bandit(
    env,
    episodes,
    config: BanditConfig,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Train a simple epsilon-greedy sample-average bandit on simulator rewards."""
    if not episodes:
        raise ValueError("No training episodes were provided")

    rng = np.random.default_rng(config.seed)
    action_dim = int(episodes[0].seed_actions.shape[1])
    action_grid = build_action_grid(action_dim)
    n_actions = int(action_grid.shape[0])
    action_values = np.zeros(n_actions, dtype=np.float64)
    action_counts = np.zeros(n_actions, dtype=np.int64)

    metrics = {
        "episode_returns": [],
        "episode_lengths": [],
        "episode_uncertainty": [],
        "epsilon": [],
        "action_values": [],
        "action_counts": [],
        "config": asdict(config),
    }

    for step in range(config.train_steps):
        episode = episodes[step % len(episodes)]
        seed_states_t = torch.tensor(episode.seed_states, dtype=torch.float32, device=device).unsqueeze(0)
        seed_actions_t = torch.tensor(episode.seed_actions, dtype=torch.float32, device=device).unsqueeze(0)
        seed_time_steps_t = torch.tensor(episode.seed_time_steps, dtype=torch.float32, device=device).unsqueeze(0)
        env.reset(seed_states_t, seed_actions_t, seed_time_steps=seed_time_steps_t)

        epsilon = _epsilon(step, config)
        total_reward = 0.0
        total_uncertainty = 0.0
        rollout_len = 0

        for _ in range(config.rollout_steps):
            action_id = _sample_action(action_values, epsilon, rng)
            action = torch.tensor(action_grid[action_id], dtype=torch.float32, device=device).unsqueeze(0)
            _, reward, done_tensor, info = env.step(action)

            reward_value = float(reward[0].item())
            uncertainty_value = float(info["uncertainty"][0])
            shaped_reward = reward_value - config.uncertainty_penalty * uncertainty_value

            action_counts[action_id] += 1
            action_values[action_id] += (shaped_reward - action_values[action_id]) / action_counts[action_id]

            total_reward += shaped_reward
            total_uncertainty += uncertainty_value
            rollout_len += 1
            if bool(done_tensor[0].item()):
                break

        metrics["episode_returns"].append(total_reward)
        metrics["episode_lengths"].append(rollout_len)
        metrics["episode_uncertainty"].append(total_uncertainty / max(rollout_len, 1))
        metrics["epsilon"].append(epsilon)

        if config.log_every > 0 and ((step + 1) % config.log_every == 0 or step == 0):
            recent_returns = metrics["episode_returns"][-config.log_every:]
            recent_unc = metrics["episode_uncertainty"][-config.log_every:]
            recent_lengths = metrics["episode_lengths"][-config.log_every:]
            logging.info(
                "Bandit step %d/%d | eps=%.3f | mean_return=%.4f | mean_unc=%.4f | mean_len=%.2f | best_action=%d | best_value=%.4f",
                step + 1,
                config.train_steps,
                epsilon,
                float(np.mean(recent_returns)),
                float(np.mean(recent_unc)),
                float(np.mean(recent_lengths)),
                int(np.argmax(action_values)),
                float(np.max(action_values)),
            )

    metrics["action_values"] = [float(x) for x in action_values]
    metrics["action_counts"] = [int(x) for x in action_counts]
    return action_values.astype(np.float32), action_counts, metrics


def greedy_policy_from_values(action_values: np.ndarray):
    """Return a stateless policy that always picks the current best arm."""
    best_action = int(np.argmax(np.asarray(action_values, dtype=np.float32)))

    def _policy(env, obs_builder, current_obs, episode):
        return {"action_id": best_action}

    return _policy


def save_bandit_artifacts(
    action_values: np.ndarray,
    action_counts: np.ndarray,
    metrics: dict,
    config: BanditConfig,
    out_dir: Path,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "bandit_model.json"
    payload = {
        "action_values": [float(x) for x in action_values],
        "action_counts": [int(x) for x in action_counts],
        "n_actions": int(len(action_values)),
        "config": asdict(config),
    }
    with open(model_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    with open(out_dir / "bandit_train_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(out_dir / "bandit_train_config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(config), f, indent=2)
    return model_path


def load_bandit_model(model_path: str) -> np.ndarray:
    with open(model_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return np.asarray(payload["action_values"], dtype=np.float32)
