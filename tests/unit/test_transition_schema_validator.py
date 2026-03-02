from __future__ import annotations

import pandas as pd

from careai.contracts.validators import validate_transition_contract
from careai.transitions.build_transition import build_transitions


def test_transition_validator_passes_on_valid_output() -> None:
    cfg = {
        "outcome": {"sofa_jump_threshold": 2},
        "action": {"classes": {"A_NONE": [0, 0, 0]}},
        "schema": {"version": "transition_hourly_mimiccode", "source_dataset": "test"},
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
            "s_t_gcs": [15, 14],
            "s_t_urine_output_rate": [1.2, 1.1],
            "s_t_oxygen_delivery": [2.0, 3.0],
            "s_t_creatinine": [1.1, 1.2],
            "s_t_bun": [12, 14],
            "s_t_age": [65, 65],
            "s_t_charlson": [2, 2],
            "a_t_vaso": [0, 0],
            "a_t_vent": [0, 0],
            "a_t_crrt": [0, 0],
        }
    )
    out = build_transitions(src, cfg)
    out["split"] = "train"
    report = validate_transition_contract(out, sofa_jump_threshold=2)
    assert report["ok"], report["errors"]
