from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.sim.data_prep import prepare_sim_data
from careai.sim.env_bandit_v1 import BanditEnvV1
from careai.sim.outcome_model import fit_outcome_model
from careai.sim.policies import AlwaysLowPolicy
from careai.sim.reporting import summary_payload
from careai.sim.runner import run_policies
from careai.sim.weights import build_sim_training_weights


def test_weighted_smoke_small() -> None:
    df = pd.DataFrame(
        {
            "a_t": [
                "A_LOW_SUPPORT",
                "A_HIGH_SUPPORT",
                "A_LOW_SUPPORT",
                "A_HIGH_SUPPORT",
                "A_LOW_SUPPORT",
                "A_HIGH_SUPPORT",
            ],
            "within_30d_next_admit": [0, 1, 0, 1, 0, 1],
            "split": ["train", "train", "train", "train", "test", "test"],
            "s_t_lace": [4, 10, 5, 11, 6, 12],
            "s_t_age": [50, 75, 52, 77, 54, 79],
            "s_t_comorbidity": [1, 3, 1, 3, 1, 3],
            "s_t_acuity": [1, 3, 1, 3, 1, 3],
            "s_t_length": [1, 3, 1, 3, 1, 3],
            "s_t_physical_status": [2, 8, 2, 8, 2, 8],
            "s_t_admission_type": ["ELECTIVE", "EMERGENCY", "ELECTIVE", "EMERGENCY", "ELECTIVE", "EMERGENCY"],
        }
    )
    cfg = {
        "data": {"allowed_actions": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT"], "eval_split": "test"},
        "model": {
            "features_numeric": [
                "s_t_lace",
                "s_t_age",
                "s_t_comorbidity",
                "s_t_acuity",
                "s_t_length",
                "s_t_physical_status",
            ],
            "features_categorical": ["s_t_admission_type"],
            "max_iter": 1000,
            "C": 1.0,
            "weight_include_actions": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT"],
            "weight_clip_percentiles": [0.01, 0.99],
            "weight_stabilized": True,
            "weight_treatment_high_label": "A_HIGH_SUPPORT",
        },
        "reward": {"cost_low": 0.0, "cost_high": 0.05},
        "simulation": {"n_episodes": 5, "seed": 1},
        "policies": {"risk_threshold": 0.3},
    }
    prep = prepare_sim_data(df, cfg)
    wres = build_sim_training_weights(prep.train_df, cfg)
    model = fit_outcome_model(prep.train_df, cfg, sample_weight=wres.sample_weight)
    env = BanditEnvV1(prep.eval_df, model, cfg, seed=1)
    out = run_policies(env, [AlwaysLowPolicy()], n_episodes=5)
    payload = summary_payload(cfg, out.metrics_df, weighting=wres.diagnostics)
    assert len(out.metrics_df) == 1
    assert payload["weighting"]["n_rows_weighted"] == 4

