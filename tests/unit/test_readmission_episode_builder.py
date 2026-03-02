from __future__ import annotations

import pandas as pd

from careai.readmission.episode_builder import build_episode_table


def test_episode_builder_aggregates_and_joins_labels() -> None:
    trans = pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2"],
            "patient_id": [1, 1, 2],
            "index_hadm_id": [10, 10, 20],
            "split": ["train", "train", "test"],
            "index_admittime": ["2020-01-01", "2020-01-01", "2020-01-05"],
            "index_dischtime": ["2020-01-02", "2020-01-02", "2020-01-06"],
            "t": [0, 1, 0],
            "a_t": ["A_NONE", "A_VASO", "A_VENT"],
            "a_t_vaso": [0, 1, 0],
            "a_t_vent": [0, 0, 1],
            "a_t_crrt": [0, 0, 0],
            "s_t_sofa": [1, 3, 2],
            "s_t_mbp": [70, 65, 80],
            "s_t_heart_rate": [80, 85, 78],
            "s_t_resp_rate": [16, 18, 15],
            "s_t_spo2": [98, 97, 99],
            "s_t_gcs": [15, 14, 15],
            "s_t_urine_output_rate": [1.2, 1.1, 1.5],
            "s_t_oxygen_delivery": [2.0, 3.0, 1.0],
            "s_t_creatinine": [1.1, 1.2, 0.9],
            "s_t_bun": [12, 15, 10],
            "s_t_age": [65, 65, 50],
            "s_t_charlson": [2, 2, 1],
        }
    )
    labels = pd.DataFrame(
        {
            "patient_id": [1, 2],
            "index_hadm_id": [10, 20],
            "days_to_next_admit": [14.0, None],
            "readmit_30d": [1, 0],
        }
    )
    epi = build_episode_table(trans, labels)
    assert len(epi) == 2
    e1 = epi[epi["episode_id"] == "e1"].iloc[0]
    assert int(e1["episode_hours"]) == 2
    assert float(e1["last_s_t_sofa"]) == 3.0
    assert float(e1["last_s_t_gcs"]) == 14.0
    assert float(e1["mean_s_t_urine_output_rate"]) == 1.15
    assert float(e1["delta_s_t_sofa"]) == 2.0
    assert int(e1["readmit_30d"]) == 1
