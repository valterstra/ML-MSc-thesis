from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .dynamics import DynamicsModel
from .readmission import ReadmissionModel, summarize_trajectory_for_readmit


@dataclass(frozen=True)
class CounterfactualResult:
    trajectories: pd.DataFrame
    risks: pd.DataFrame


def rollout_fixed_action_paths(
    initial_state: np.ndarray,
    state_cols: list[str],
    dynamics: DynamicsModel,
    readmit: ReadmissionModel,
    max_steps: int,
) -> CounterfactualResult:
    policies: dict[str, np.ndarray] = {
        "no_treatment": np.array([0, 0, 0], dtype=float),
        "always_vaso": np.array([1, 0, 0], dtype=float),
        "always_vent": np.array([0, 1, 0], dtype=float),
        "always_crrt": np.array([0, 0, 1], dtype=float),
        "all_treatments": np.array([1, 1, 1], dtype=float),
    }

    traj_rows: list[dict[str, float | int | str]] = []
    risk_rows: list[dict[str, float | str]] = []

    for name, act in policies.items():
        s = initial_state.copy()
        states = [s.copy()]
        actions: list[np.ndarray] = []
        for t in range(max_steps):
            ns, done_prob = dynamics.predict_next(s, act)
            actions.append(act.copy())
            states.append(ns.copy())
            traj_rows.append(
                {
                    "policy": name,
                    "t": t,
                    "done_prob": float(done_prob),
                    "a_t_vaso": float(act[0]),
                    "a_t_vent": float(act[1]),
                    "a_t_crrt": float(act[2]),
                    **{c: float(s[i]) for i, c in enumerate(state_cols)},
                    **{f"next_{c}": float(ns[i]) for i, c in enumerate(state_cols)},
                }
            )
            s = ns

        features = summarize_trajectory_for_readmit(states=states, actions=actions, state_cols=state_cols)
        risk_rows.append({"policy": name, "pred_readmit_risk": float(readmit.predict_prob(features))})

    return CounterfactualResult(trajectories=pd.DataFrame(traj_rows), risks=pd.DataFrame(risk_rows))

