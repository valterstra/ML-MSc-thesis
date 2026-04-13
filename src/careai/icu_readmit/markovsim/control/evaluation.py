"""Seed episode loading and simulator-based policy evaluation for MarkovSim."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Protocol

import torch
import numpy as np
import pandas as pd

from ...caresim.dataset import BLOC_COL, SPLIT_COL, STAY_COL
from ...caresim.readmit import LightGBMReadmitModel
from ...caresim.severity import RidgeSeveritySurrogate
from ..model import ACTION_COLS, MarkovSimEnsemble, STATE_COLS
from ..simulator import MarkovSimEnvironment
from .actions import ACTION_GRID, action_tensor, build_action_grid, encode_action_vec
from .observation import ObservationBuilder
from .planner import PlannerConfig, planner_action


@dataclass
class SeedEpisode:
    split: str
    stay_id: int
    seed_states: np.ndarray
    seed_actions: np.ndarray
    seed_time_steps: np.ndarray
    initial_action_id: int
    state_cols: list[str]
    action_cols: list[str]


class PolicyFn(Protocol):
    def __call__(
        self,
        env: MarkovSimEnvironment,
        obs_builder: ObservationBuilder,
        current_obs: np.ndarray,
        episode: SeedEpisode,
    ) -> dict: ...


def load_seed_episodes(
    data_path: str,
    split: str,
    history_len: int,
    max_episodes: int | None,
    seed: int,
) -> list[SeedEpisode]:
    use_cols = [STAY_COL, BLOC_COL, SPLIT_COL, *STATE_COLS, *ACTION_COLS]
    df = pd.read_parquet(data_path, columns=use_cols)
    df = df[df[SPLIT_COL] == split].copy()

    episodes: list[SeedEpisode] = []
    for stay_id, stay_df in df.groupby(STAY_COL, sort=False):
        stay_df = stay_df.sort_values(BLOC_COL).reset_index(drop=True)
        if len(stay_df) < history_len:
            continue
        seed_rows = stay_df.iloc[:history_len]
        seed_actions = seed_rows[ACTION_COLS].to_numpy(dtype=np.float32, copy=True)
        episodes.append(SeedEpisode(
            split=split,
            stay_id=int(stay_id),
            seed_states=seed_rows[STATE_COLS].to_numpy(dtype=np.float32, copy=True),
            seed_actions=seed_actions,
            seed_time_steps=seed_rows[BLOC_COL].to_numpy(dtype=np.float32, copy=True),
            initial_action_id=encode_action_vec(seed_actions[-1], action_dim=len(ACTION_COLS)),
            state_cols=list(STATE_COLS),
            action_cols=list(ACTION_COLS),
        ))

    if max_episodes is not None and len(episodes) > max_episodes:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(episodes), size=max_episodes, replace=False)
        episodes = [episodes[i] for i in idx]
    return episodes


def make_repeat_last_policy() -> PolicyFn:
    def _policy(env: MarkovSimEnvironment, obs_builder: ObservationBuilder, current_obs: np.ndarray, episode: SeedEpisode) -> dict:
        return {"action_id": int(episode.initial_action_id)}

    return _policy


def make_random_policy(rng: np.random.Generator, action_dim: int | None = None) -> PolicyFn:
    def _policy(env: MarkovSimEnvironment, obs_builder: ObservationBuilder, current_obs: np.ndarray, episode: SeedEpisode) -> dict:
        dim = int(action_dim or episode.seed_actions.shape[1])
        return {"action_id": int(rng.integers(0, 2 ** dim))}

    return _policy


def make_planner_policy(ensemble: MarkovSimEnsemble, planner_config: PlannerConfig, device: torch.device) -> PolicyFn:
    def _policy(env: MarkovSimEnvironment, obs_builder: ObservationBuilder, current_obs: np.ndarray, episode: SeedEpisode) -> dict:
        return planner_action(
            ensemble=ensemble,
            seed_states=env._states,
            seed_actions=env._actions,
            seed_time_steps=env._time_steps,
            config=planner_config,
            device=device,
            action_dim=episode.seed_actions.shape[1],
        )

    return _policy


def evaluate_policies(
    ensemble: MarkovSimEnsemble,
    episodes: list[SeedEpisode],
    policies: dict[str, PolicyFn],
    planner_config: PlannerConfig,
    rollout_steps: int,
    observation_window: int,
    uncertainty_threshold: float,
    uncertainty_penalty: float,
    device: torch.device,
    severity_model: RidgeSeveritySurrogate | None = None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
    progress_every: int = 0,
) -> tuple[dict[str, dict], pd.DataFrame]:
    rows: list[dict] = []
    total_policy_runs = len(episodes) * len(policies)
    completed_runs = 0

    for episode in episodes:
        action_dim = int(episode.seed_actions.shape[1])
        action_grid = ACTION_GRID if action_dim == ACTION_GRID.shape[1] else build_action_grid(action_dim)
        seed_states_t = torch.tensor(episode.seed_states, dtype=torch.float32, device=device).unsqueeze(0)
        seed_actions_t = torch.tensor(episode.seed_actions, dtype=torch.float32, device=device).unsqueeze(0)
        seed_time_steps_t = torch.tensor(episode.seed_time_steps, dtype=torch.float32, device=device).unsqueeze(0)

        for policy_name, policy_fn in policies.items():
            env = MarkovSimEnvironment(
                ensemble,
                max_steps=rollout_steps,
                uncertainty_threshold=uncertainty_threshold,
                device=device,
                severity_model=severity_model,
                terminal_outcome_model=terminal_outcome_model,
                terminal_reward_scale=terminal_reward_scale,
            )
            env.reset(seed_states_t, seed_actions_t, seed_time_steps=seed_time_steps_t)
            obs_builder = ObservationBuilder(
                window_len=observation_window,
                state_dim=episode.seed_states.shape[1],
                action_dim=episode.seed_actions.shape[1],
            )
            current_obs = obs_builder.reset(episode.seed_states, episode.seed_actions)

            discounted_return = 0.0
            raw_reward_total = 0.0
            uncertainty_total = 0.0
            action_ids: list[int] = []
            done = False
            last_terminal_prob = 0.0

            for step_idx in range(rollout_steps):
                decision = policy_fn(env, obs_builder, current_obs, episode)
                action_id = int(decision["action_id"] if "action_id" in decision else decision["best_action_id"])
                action = action_tensor(action_id, device=device, action_dim=action_dim)
                next_state, reward, done_tensor, info = env.step(action)

                reward_value = float(reward[0].item())
                uncertainty_value = float(info["uncertainty"][0])
                penalized_reward = reward_value - uncertainty_penalty * uncertainty_value
                last_terminal_prob = float(info["terminal_prob"][0])
                done = bool(done_tensor[0].item())

                discounted_return += (planner_config.gamma ** step_idx) * penalized_reward
                raw_reward_total += reward_value
                uncertainty_total += uncertainty_value
                action_ids.append(action_id)
                current_obs = obs_builder.append(
                    next_state[0].detach().cpu().numpy(),
                    np.asarray(action_grid[action_id], dtype=np.float32),
                )

                rows.append({
                    "split": episode.split,
                    "stay_id": episode.stay_id,
                    "policy": policy_name,
                    "step": step_idx + 1,
                    "action_id": action_id,
                    "reward": reward_value,
                    "uncertainty": uncertainty_value,
                    "penalized_reward": penalized_reward,
                    "terminal_prob": last_terminal_prob,
                    "terminal_p_readmit": float(info["terminal_p_readmit"][0]) if not np.isnan(info["terminal_p_readmit"][0]) else np.nan,
                    "terminal_reward": float(info["terminal_reward"][0]),
                    "done": done,
                })

                if done:
                    break

            rows.append({
                "split": episode.split,
                "stay_id": episode.stay_id,
                "policy": policy_name,
                "step": 0,
                "action_trace": ",".join(str(a) for a in action_ids),
                "rollout_steps": len(action_ids),
                "discounted_return": discounted_return,
                "raw_reward_total": raw_reward_total,
                "mean_uncertainty": (uncertainty_total / max(len(action_ids), 1)),
                "terminated": done,
                "last_terminal_prob": last_terminal_prob,
            })

            completed_runs += 1
            if progress_every > 0 and (completed_runs % progress_every == 0 or completed_runs == total_policy_runs):
                logging.info(
                    "Policy eval progress %d/%d (%.0f%%) | split=%s | latest_policy=%s | stay=%s",
                    completed_runs,
                    total_policy_runs,
                    100.0 * completed_runs / max(total_policy_runs, 1),
                    episode.split,
                    policy_name,
                    episode.stay_id,
                )

    traces = pd.DataFrame(rows)
    terminal_rows = traces[traces["step"] == 0].copy()
    step_rows = traces[traces["step"] > 0].copy()
    summary: dict[str, dict] = {}
    for policy_name, policy_df in terminal_rows.groupby("policy"):
        policy_steps = step_rows[step_rows["policy"] == policy_name]
        action_counts = (
            policy_steps["action_id"].value_counts().sort_index().to_dict()
            if not policy_steps.empty and "action_id" in policy_steps
            else {}
        )
        first_actions = (
            policy_steps.sort_values(["stay_id", "step"]).groupby("stay_id", as_index=False).first()
            if not policy_steps.empty
            else pd.DataFrame(columns=["action_id"])
        )
        first_action_counts = (
            first_actions["action_id"].value_counts().sort_index().to_dict()
            if not first_actions.empty and "action_id" in first_actions
            else {}
        )
        summary[policy_name] = {
            "episodes": int(len(policy_df)),
            "mean_discounted_return": float(policy_df["discounted_return"].mean()),
            "mean_raw_reward_total": float(policy_df["raw_reward_total"].mean()),
            "mean_uncertainty": float(policy_df["mean_uncertainty"].mean()),
            "termination_rate": float(policy_df["terminated"].mean()),
            "mean_rollout_steps": float(policy_df["rollout_steps"].mean()),
            "mean_last_terminal_prob": float(policy_df["last_terminal_prob"].mean()),
            "std_discounted_return": float(policy_df["discounted_return"].std(ddof=0)),
            "p25_discounted_return": float(policy_df["discounted_return"].quantile(0.25)),
            "p50_discounted_return": float(policy_df["discounted_return"].quantile(0.50)),
            "p75_discounted_return": float(policy_df["discounted_return"].quantile(0.75)),
            "mean_step_reward": float(policy_steps["reward"].mean()) if not policy_steps.empty else 0.0,
            "mean_step_penalized_reward": float(policy_steps["penalized_reward"].mean()) if not policy_steps.empty else 0.0,
            "mean_step_terminal_prob": float(policy_steps["terminal_prob"].mean()) if not policy_steps.empty else 0.0,
            "action_counts": {str(int(k)): int(v) for k, v in action_counts.items()},
            "first_action_counts": {str(int(k)): int(v) for k, v in first_action_counts.items()},
        }

    return summary, traces
