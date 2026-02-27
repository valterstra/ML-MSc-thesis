from __future__ import annotations

import pandas as pd

from careai.transitions.build_transition_v3_hourly import build_transitions_v3_hourly


def _cfg() -> dict:
    return {
        "outcome": {"sofa_jump_threshold": 2},
        "action": {
            "classes": {
                "A_NONE": [0, 0, 0],
                "A_VASO": [1, 0, 0],
                "A_VENT": [0, 1, 0],
                "A_CRRT": [0, 0, 1],
                "A_VASO_VENT": [1, 1, 0],
                "A_VASO_CRRT": [1, 0, 1],
                "A_VENT_CRRT": [0, 1, 1],
                "A_VASO_VENT_CRRT": [1, 1, 1],
            }
        },
        "schema": {"version": "transition_v3_mimiccode_hourly", "source_dataset": "test"},
        "sample_tag": "full",
    }


def test_action_mapping_composes_expected_labels() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 1],
            "hadm_id": [10, 10, 10],
            "stay_id": [100, 100, 100],
            "hr": [0, 1, 2],
            "starttime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 02:00"]),
            "endtime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-01 03:00"]),
            "s_t_sofa": [1, 1, 1],
            "s_t_mbp": [70, 69, 68],
            "s_t_heart_rate": [80, 82, 84],
            "s_t_resp_rate": [16, 17, 18],
            "s_t_spo2": [98, 97, 96],
            "s_t_creatinine": [1.1, 1.2, 1.3],
            "s_t_bun": [12, 14, 16],
            "s_t_age": [65, 65, 65],
            "s_t_charlson": [2, 2, 2],
            "a_t_vaso": [1, 0, 1],
            "a_t_vent": [0, 1, 1],
            "a_t_crrt": [0, 0, 1],
        }
    )
    out = build_transitions_v3_hourly(df, _cfg())
    assert out["a_t"].tolist() == ["A_VASO", "A_VENT", "A_VASO_VENT_CRRT"]

