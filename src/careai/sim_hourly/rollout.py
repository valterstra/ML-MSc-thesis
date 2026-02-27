from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dynamics import DynamicsModel
from .policies import HourlyPolicy
from .readmission import ReadmissionModel, summarize_trajectory_for_readmit


@dataclass(frozen=True)
class RolloutOutputs:
    trajectories: pd.DataFrame
    policy_metrics: pd.DataFrame


def rollout_policies(
    start_states: pd.DataFrame,
    dynamics: DynamicsModel,
    readmit: ReadmissionModel,
    policies: list[HourlyPolicy],
    max_steps: int,
    n_rollouts: int,
    done_threshold: float,
    seed: int,
) -> RolloutOutputs:
    rng = np.random.default_rng(seed)
    traj_rows: list[dict[str, float | int | str]] = []
    metric_rows: list[dict[str, float | int | str]] = []
    starts = start_states[dynamics.state_cols].to_numpy(dtype=float)

    for pol in policies:
        risks: list[float] = []
        lengths: list[int] = []
        done_rates: list[int] = []
        for ridx in range(n_rollouts):
            s = starts[int(rng.integers(0, len(starts)))].copy()
            states = [s.copy()]
            actions: list[np.ndarray] = []
            terminated = 0
            for t in range(max_steps):
                a = pol.choose_action(rng).astype(float)
                ns, done_prob = dynamics.predict_next(s, a)
                actions.append(a.copy())
                states.append(ns.copy())
                traj_rows.append(
                    {
                        "policy": pol.name,
                        "rollout_id": ridx,
                        "t": t,
                        "done_prob": done_prob,
                        "a_t_vaso": float(a[0]),
                        "a_t_vent": float(a[1]),
                        "a_t_crrt": float(a[2]),
                        **{c: float(s[j]) for j, c in enumerate(dynamics.state_cols)},
                        **{f"next_{c}": float(ns[j]) for j, c in enumerate(dynamics.state_cols)},
                    }
                )
                s = ns
                if done_prob >= done_threshold:
                    terminated = 1
                    break
            feats = summarize_trajectory_for_readmit(states=states, actions=actions, state_cols=dynamics.state_cols)
            risk = readmit.predict_prob(feats)
            risks.append(risk)
            lengths.append(len(actions))
            done_rates.append(terminated)
        metric_rows.append(
            {
                "policy": pol.name,
                "n_rollouts": n_rollouts,
                "mean_pred_readmit_risk": float(np.mean(risks)),
                "std_pred_readmit_risk": float(np.std(risks)),
                "mean_rollout_hours": float(np.mean(lengths)),
                "termination_rate": float(np.mean(done_rates)),
            }
        )

    return RolloutOutputs(trajectories=pd.DataFrame(traj_rows), policy_metrics=pd.DataFrame(metric_rows))

