from __future__ import annotations

import pandas as pd

from careai.transitions.build_transition_v3_hourly import build_transitions_v3_hourly


def test_done_is_terminal_step_only() -> None:
    cfg = {
        "outcome": {"sofa_jump_threshold": 2},
        "action": {"classes": {"A_NONE": [0, 0, 0]}},
        "schema": {"version": "transition_v3_mimiccode_hourly", "source_dataset": "test"},
        "sample_tag": "full",
    }
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 10, 20],
            "stay_id": [100, 100, 200],
            "hr": [0, 1, 0],
            "starttime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-02 00:00"]),
            "endtime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-02 01:00"]),
            "s_t_sofa": [1, 2, 1],
            "s_t_mbp": [70, 69, 72],
            "s_t_heart_rate": [80, 82, 75],
            "s_t_resp_rate": [16, 17, 14],
            "s_t_spo2": [98, 97, 99],
            "s_t_creatinine": [1.1, 1.2, 1.0],
            "s_t_bun": [12, 14, 11],
            "s_t_age": [65, 65, 50],
            "s_t_charlson": [2, 2, 1],
            "a_t_vaso": [0, 0, 0],
            "a_t_vent": [0, 0, 0],
            "a_t_crrt": [0, 0, 0],
        }
    )
    out = build_transitions_v3_hourly(df, cfg)
    done_by_stay = out.groupby("episode_id")["done"].apply(list).to_dict()
    assert done_by_stay["100"] == [0, 1]
    assert done_by_stay["200"] == [1]

