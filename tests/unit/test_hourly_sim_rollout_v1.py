from __future__ import annotations

import numpy as np
import pandas as pd

from careai.sim_hourly.data import prepare_hourly_data
from careai.sim_hourly.dynamics import fit_dynamics_model
from careai.sim_hourly.policies import build_policies
from careai.sim_hourly.readmission import fit_readmission_model
from careai.sim_hourly.rollout import rollout_policies


def test_hourly_rollout_smoke() -> None:
    trans = pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "split": ["train", "train", "train", "train", "test", "test"],
            "t": [0, 1, 0, 1, 0, 1],
            "done": [0, 1, 0, 1, 0, 1],
            "s_t_sofa": [1, 2, 2, 3, 1, 2],
            "s_t_mbp": [70, 68, 75, 73, 72, 71],
            "s_t_age": [60, 60, 70, 70, 65, 65],
            "a_t_vaso": [0, 1, 0, 1, 0, 0],
            "a_t_vent": [0, 0, 1, 1, 0, 0],
            "a_t_crrt": [0, 0, 0, 1, 0, 0],
        }
    )
    epi = pd.DataFrame(
        {
            "split": ["train", "valid", "test"],
            "readmit_30d": [0, 1, 0],
            "episode_hours": [10, 20, 12],
            "last_s_t_sofa": [1, 2, 1],
            "last_s_t_mbp": [70, 65, 72],
            "last_s_t_age": [60, 70, 65],
            "mean_s_t_sofa": [1.1, 2.0, 1.3],
            "mean_s_t_mbp": [70, 66, 72],
            "mean_s_t_age": [60, 70, 65],
            "min_s_t_mbp": [68, 60, 70],
            "max_s_t_sofa": [2, 4, 2],
            "delta_s_t_sofa": [1, 2, 1],
            "delta_s_t_mbp": [-2, -5, -1],
            "frac_vaso": [0.1, 0.8, 0.2],
            "frac_vent": [0.1, 0.6, 0.1],
            "frac_crrt": [0.0, 0.5, 0.0],
            "any_vaso": [1, 1, 1],
            "any_vent": [1, 1, 1],
            "any_crrt": [0, 1, 0],
            "last_s_t_heart_rate": [80, 90, 82],
            "last_s_t_resp_rate": [16, 20, 17],
            "last_s_t_spo2": [98, 95, 97],
            "last_s_t_creatinine": [1.0, 2.0, 1.1],
            "last_s_t_bun": [20, 45, 22],
            "last_s_t_charlson": [2, 6, 3],
            "mean_s_t_heart_rate": [82, 92, 84],
            "mean_s_t_resp_rate": [16.5, 21, 17.2],
            "mean_s_t_spo2": [98, 95, 97],
            "mean_s_t_creatinine": [1.1, 1.9, 1.2],
            "mean_s_t_bun": [21, 44, 23],
            "mean_s_t_charlson": [2, 6, 3],
        }
    )
    cfg = {
        "state": {"features_numeric": ["s_t_sofa", "s_t_mbp", "s_t_age"]},
        "actions": {"components": ["a_t_vaso", "a_t_vent", "a_t_crrt"]},
        "dynamics_model": {"alpha": 1.0},
        "readmission_model": {
            "max_iter": 200,
            "c": 1.0,
            "random_state": 42,
            "features": [c for c in epi.columns if c not in {"split", "readmit_30d"}],
        },
        "simulation": {"max_steps": 5, "n_rollouts_per_policy": 5, "done_threshold": 0.5, "seed": 1},
    }
    hourly = prepare_hourly_data(trans, cfg)
    dyn = fit_dynamics_model(hourly.train, cfg, hourly.state_cols, hourly.action_cols)
    readm = fit_readmission_model(epi, cfg)
    policies = build_policies(hourly.train[hourly.action_cols].to_numpy(dtype=float))
    out = rollout_policies(hourly.eval_starts, dyn, readm, policies[:2], max_steps=5, n_rollouts=5, done_threshold=0.5, seed=1)
    assert len(out.policy_metrics) == 2
    assert set(out.policy_metrics["policy"]) == {"observed_random", "no_treatment"}
    assert np.isfinite(out.policy_metrics["mean_pred_readmit_risk"]).all()
    assert len(out.trajectories) > 0

