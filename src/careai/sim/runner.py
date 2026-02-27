"""Simulation runner for Sim v1."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .env_bandit_v1 import BanditEnvV1
from .policies import BasePolicy


@dataclass(frozen=True)
class SimulationOutputs:
    episodes_df: pd.DataFrame
    metrics_df: pd.DataFrame


def run_policies(
    env: BanditEnvV1,
    policies: list[BasePolicy],
    n_episodes: int,
) -> SimulationOutputs:
    episode_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []

    for policy in policies:
        rewards: list[float] = []
        ys: list[int] = []
        action_costs: list[float] = []
        p_hats: list[float] = []
        actions: list[str] = []

        for ep in range(n_episodes):
            state = env.reset()
            action = policy.choose_action(env._state, env.model)  # uses full row for observed policy
            res = env.step(action)
            rewards.append(res.reward)
            ys.append(res.y)
            action_costs.append(res.action_cost)
            p_hats.append(res.p_hat)
            actions.append(res.action)
            episode_rows.append(
                {
                    "policy": policy.name,
                    "episode": ep,
                    "action": res.action,
                    "readmitted_30d": res.y,
                    "reward": res.reward,
                    "action_cost": res.action_cost,
                    "p_hat": res.p_hat,
                }
            )

        actions_ser = pd.Series(actions)
        metric_rows.append(
            {
                "policy": policy.name,
                "n_episodes": n_episodes,
                "mean_reward": float(np.mean(rewards)),
                "readmit_rate": float(np.mean(ys)),
                "mean_action_cost": float(np.mean(action_costs)),
                "mean_p_hat": float(np.mean(p_hats)),
                "high_support_rate": float((actions_ser == "A_HIGH_SUPPORT").mean()),
                "low_support_rate": float((actions_ser == "A_LOW_SUPPORT").mean()),
            }
        )

    episodes_df = pd.DataFrame(episode_rows)
    metrics_df = pd.DataFrame(metric_rows).sort_values("mean_reward", ascending=False).reset_index(drop=True)
    return SimulationOutputs(episodes_df=episodes_df, metrics_df=metrics_df)

