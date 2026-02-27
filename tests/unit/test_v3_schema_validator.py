from __future__ import annotations

import pandas as pd

from careai.contracts.validators_v3 import validate_transition_v3_contract
from careai.transitions.build_transition_v3_hourly import build_transitions_v3_hourly


def test_v3_validator_passes_on_valid_output() -> None:
    cfg = {
        "outcome": {"sofa_jump_threshold": 2},
        "action": {"classes": {"A_NONE": [0, 0, 0]}},
        "schema": {"version": "transition_v3_mimiccode_hourly", "source_dataset": "test"},
        "sample_tag": "full",
    }
    src = pd.DataFrame(
        {
            "subject_id": [1, 1],
            "hadm_id": [10, 10],
            "stay_id": [100, 100],
            "hr": [0, 1],
            "starttime": pd.to_datetime(["2020-01-01 00:00", "2020-01-01 01:00"]),
            "endtime": pd.to_datetime(["2020-01-01 01:00", "2020-01-01 02:00"]),
            "s_t_sofa": [1, 3],
            "s_t_mbp": [70, 69],
            "s_t_heart_rate": [80, 82],
            "s_t_resp_rate": [16, 17],
            "s_t_spo2": [98, 97],
            "s_t_creatinine": [1.1, 1.2],
            "s_t_bun": [12, 14],
            "s_t_age": [65, 65],
            "s_t_charlson": [2, 2],
            "a_t_vaso": [0, 0],
            "a_t_vent": [0, 0],
            "a_t_crrt": [0, 0],
        }
    )
    out = build_transitions_v3_hourly(src, cfg)
    out["split"] = "train"
    report = validate_transition_v3_contract(out, sofa_jump_threshold=2)
    assert report["ok"], report["errors"]
