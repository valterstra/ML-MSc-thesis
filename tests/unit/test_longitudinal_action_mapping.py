from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.build_longitudinal_v1 import build_longitudinal_from_transitions_v2


def test_longitudinal_action_mapping_and_padding() -> None:
    cfg = {
        "features": {"state_numeric": ["s_t_lace", "s_t_age", "s_t_comorbidity", "s_t_acuity", "s_t_length", "s_t_physical_status"]},
        "actions": {"mapping": {"A_LOW_SUPPORT": 0, "A_HIGH_SUPPORT": 1, "A_TERMINAL": 2, "A_UNKNOWN": 3}, "padding_id": -1},
        "split": {"mapping": {"train": 0, "valid": 1, "test": 2}},
    }
    df = pd.DataFrame(
        {
            "episode_id": ["e1", "e1", "e2"],
            "patient_id": [1, 1, 2],
            "t": [0, 1, 0],
            "split": ["train", "train", "test"],
            "index_admittime": ["2020-01-01", "2020-01-03", "2020-02-01"],
            "index_dischtime": ["2020-01-02", "2020-01-04", "2020-02-02"],
            "a_t": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT", "SOMETHING_NEW"],
            "y_t1": [1, 0, 0],
            "done": [0, 1, 1],
            "s_t_lace": [10, 9, 4],
            "s_t_age": [60, 60, 50],
            "s_t_comorbidity": [2, 2, 1],
            "s_t_acuity": [3, 2, 1],
            "s_t_length": [2, 2, 1],
            "s_t_physical_status": [5, 4, 2],
        }
    )
    out = build_longitudinal_from_transitions_v2(df, cfg)
    a = out.tensors["A_action"]

    # e1 has two valid steps, no padding.
    assert a[0, 0] == 0
    assert a[0, 1] == 1
    # e2 unknown action falls back to A_UNKNOWN id.
    assert a[1, 0] == 3
    # second step for e2 is padding.
    assert a[1, 1] == -1

