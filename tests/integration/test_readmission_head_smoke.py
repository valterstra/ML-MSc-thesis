from __future__ import annotations

import pandas as pd

from careai.readmission.episode_builder_v1 import build_episode_table_v1
from careai.readmission.model_v1 import train_readmission_baseline
from careai.readmission.qa_v1 import generate_episode_qa_v1


def test_readmission_head_smoke_end_to_end() -> None:
    trans = pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2", "e2", "e3"],
            "patient_id": [1, 1, 2, 2, 3],
            "index_hadm_id": [10, 10, 20, 20, 30],
            "split": ["train", "train", "valid", "valid", "test"],
            "index_admittime": ["2020-01-01"] * 5,
            "index_dischtime": ["2020-01-02"] * 5,
            "t": [0, 1, 0, 1, 0],
            "a_t": ["A_NONE", "A_VASO", "A_NONE", "A_VENT", "A_NONE"],
            "a_t_vaso": [0, 1, 0, 0, 0],
            "a_t_vent": [0, 0, 0, 1, 0],
            "a_t_crrt": [0, 0, 0, 0, 0],
            "s_t_sofa": [1, 3, 2, 2, 1],
            "s_t_mbp": [70, 65, 80, 78, 75],
            "s_t_heart_rate": [80, 85, 78, 80, 82],
            "s_t_resp_rate": [16, 18, 15, 15, 16],
            "s_t_spo2": [98, 97, 99, 98, 97],
            "s_t_creatinine": [1.1, 1.2, 0.9, 0.9, 1.0],
            "s_t_bun": [12, 15, 10, 11, 12],
            "s_t_age": [65, 65, 50, 50, 55],
            "s_t_charlson": [2, 2, 1, 1, 1],
        }
    )
    labels = pd.DataFrame(
        {
            "patient_id": [1, 2, 3],
            "index_hadm_id": [10, 20, 30],
            "days_to_next_admit": [14.0, 45.0, None],
            "readmit_30d": [1, 0, 0],
        }
    )
    epi = build_episode_table_v1(trans, labels)
    qa = generate_episode_qa_v1(epi)
    assert qa["ok"]
    cfg = {
        "model": {"kind": "logreg", "max_iter": 200, "c": 1.0, "random_state": 42},
        "features": {
            "include": [
                c
                for c in epi.columns
                if c
                not in {"episode_id", "patient_id", "index_hadm_id", "split", "readmit_30d", "index_admittime", "index_dischtime", "days_to_next_admit"}
            ]
        },
    }
    out = train_readmission_baseline(epi, cfg)
    assert "metrics" in out

