from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import yaml


def test_sim_hourly_script_smoke(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    transitions = tmp_path / "transitions.csv"
    episodes = tmp_path / "episodes.csv"
    out_dir = tmp_path / "out"
    cfg_path = tmp_path / "cfg.yaml"

    pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2", "e2", "e3", "e3"],
            "split": ["train", "train", "train", "train", "test", "test"],
            "t": [0, 1, 0, 1, 0, 1],
            "done": [0, 1, 0, 1, 0, 1],
            "s_t_sofa": [1, 2, 2, 3, 1, 2],
            "s_t_mbp": [70, 68, 75, 73, 72, 71],
            "s_t_heart_rate": [80, 82, 84, 86, 81, 83],
            "s_t_resp_rate": [16, 18, 17, 19, 16, 18],
            "s_t_spo2": [98, 97, 97, 96, 98, 97],
            "s_t_gcs": [15, 14, 13, 12, 15, 14],
            "s_t_urine_output_rate": [1.2, 1.1, 0.9, 0.8, 1.3, 1.2],
            "s_t_oxygen_delivery": [2.0, 3.0, 4.0, 4.0, 1.0, 2.0],
            "s_t_creatinine": [1.0, 1.1, 1.2, 1.3, 1.0, 1.1],
            "s_t_bun": [20, 22, 24, 26, 21, 23],
            "s_t_age": [60, 60, 70, 70, 65, 65],
            "s_t_charlson": [2, 2, 4, 4, 3, 3],
            "a_t_vaso": [0, 1, 0, 1, 0, 0],
            "a_t_vent": [0, 0, 1, 1, 0, 0],
            "a_t_crrt": [0, 0, 0, 1, 0, 0],
        }
    ).to_csv(transitions, index=False)

    pd.DataFrame(
        {
            "split": ["train", "valid", "test"],
            "readmit_30d": [0, 1, 0],
            "episode_hours": [10, 20, 12],
            "last_s_t_sofa": [1, 2, 1],
            "last_s_t_mbp": [70, 65, 72],
            "last_s_t_heart_rate": [80, 90, 82],
            "last_s_t_resp_rate": [16, 20, 17],
            "last_s_t_spo2": [98, 95, 97],
            "last_s_t_gcs": [15, 12, 14],
            "last_s_t_urine_output_rate": [1.2, 0.8, 1.1],
            "last_s_t_oxygen_delivery": [2.5, 4.0, 1.5],
            "last_s_t_creatinine": [1.0, 2.0, 1.1],
            "last_s_t_bun": [20, 45, 22],
            "last_s_t_age": [60, 70, 65],
            "last_s_t_charlson": [2, 6, 3],
            "mean_s_t_sofa": [1.1, 2.0, 1.3],
            "mean_s_t_mbp": [70, 66, 72],
            "mean_s_t_heart_rate": [82, 92, 84],
            "mean_s_t_resp_rate": [16.5, 21, 17.2],
            "mean_s_t_spo2": [98, 95, 97],
            "mean_s_t_gcs": [14.5, 12.5, 14.5],
            "mean_s_t_urine_output_rate": [1.15, 0.85, 1.2],
            "mean_s_t_oxygen_delivery": [2.5, 4.0, 1.5],
            "mean_s_t_creatinine": [1.1, 1.9, 1.2],
            "mean_s_t_bun": [21, 44, 23],
            "mean_s_t_age": [60, 70, 65],
            "mean_s_t_charlson": [2, 6, 3],
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
        }
    ).to_csv(episodes, index=False)

    cfg = {
        "input": {"transitions_path": str(transitions), "episode_table_path": str(episodes)},
        "output": {"report_dir": str(out_dir), "prefix": "hourly_sim"},
        "state": {"features_numeric": ["s_t_sofa", "s_t_mbp", "s_t_heart_rate", "s_t_resp_rate", "s_t_spo2", "s_t_gcs", "s_t_urine_output_rate", "s_t_oxygen_delivery", "s_t_creatinine", "s_t_bun", "s_t_age", "s_t_charlson"]},
        "actions": {"components": ["a_t_vaso", "a_t_vent", "a_t_crrt"], "policies": ["observed_random", "no_treatment"]},
        "simulation": {"max_steps": 8, "n_rollouts_per_policy": 5, "seed": 42, "done_threshold": 0.5},
        "dynamics_model": {"kind": "ridge", "alpha": 1.0},
        "readmission_model": {
            "max_iter": 200,
            "c": 1.0,
            "random_state": 42,
            "features": [
                "episode_hours",
                "last_s_t_sofa",
                "last_s_t_mbp",
                "last_s_t_heart_rate",
                "last_s_t_resp_rate",
                "last_s_t_spo2",
                "last_s_t_gcs",
                "last_s_t_urine_output_rate",
                "last_s_t_oxygen_delivery",
                "last_s_t_creatinine",
                "last_s_t_bun",
                "last_s_t_age",
                "last_s_t_charlson",
                "mean_s_t_sofa",
                "mean_s_t_mbp",
                "mean_s_t_heart_rate",
                "mean_s_t_resp_rate",
                "mean_s_t_spo2",
                "mean_s_t_gcs",
                "mean_s_t_urine_output_rate",
                "mean_s_t_oxygen_delivery",
                "mean_s_t_creatinine",
                "mean_s_t_bun",
                "mean_s_t_age",
                "mean_s_t_charlson",
                "min_s_t_mbp",
                "max_s_t_sofa",
                "delta_s_t_sofa",
                "delta_s_t_mbp",
                "frac_vaso",
                "frac_vent",
                "frac_crrt",
                "any_vaso",
                "any_vent",
                "any_crrt",
            ],
        },
    }
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    cmd = [sys.executable, str(root / "scripts" / "sim" / "run_hourly_sim.py"), "--config", str(cfg_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "hourly_sim_summary.json").exists()
    assert (out_dir / "hourly_sim_policy_metrics.csv").exists()
    assert (out_dir / "hourly_sim_trajectories.csv").exists()
