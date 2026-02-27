from __future__ import annotations

import pandas as pd

from careai.contracts.validators_v3 import validate_transition_v3_contract
from careai.qa.transition_report_v3 import generate_transition_v3_qa
from careai.transitions.build_transition_v3_hourly import build_transitions_v3_hourly
from careai.transitions.split import assign_subject_splits


def test_v3_transition_build_from_fixture() -> None:
    cfg = {
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
        "sample_tag": "sample_2pct",
    }
    src = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 10, 20],
            "stay_id": [100, 100, 200],
            "hr": [0, 1, 0],
            "starttime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-02 00:00"]),
            "endtime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00", "2020-01-02 01:00"]),
            "s_t_sofa": [1, 3, 2],
            "s_t_mbp": [70, 69, 71],
            "s_t_heart_rate": [80, 82, 76],
            "s_t_resp_rate": [16, 17, 15],
            "s_t_spo2": [98, 97, 99],
            "s_t_creatinine": [1.1, 1.2, 1.0],
            "s_t_bun": [12, 14, 11],
            "s_t_age": [65, 65, 50],
            "s_t_charlson": [2, 2, 1],
            "a_t_vaso": [1, 0, 0],
            "a_t_vent": [0, 1, 0],
            "a_t_crrt": [0, 0, 0],
        }
    )
    out = build_transitions_v3_hourly(src, cfg)
    out = assign_subject_splits(out, train=0.7, valid=0.15, test=0.15, seed=42)
    report = validate_transition_v3_contract(out)
    assert report["ok"], report["errors"]
    qa = generate_transition_v3_qa(out)
    assert qa["rows"] == len(out)
    assert qa["episodes"] == 2

