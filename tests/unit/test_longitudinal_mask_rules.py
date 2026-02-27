from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.build_longitudinal_v1 import build_longitudinal_from_transitions_v2


def test_longitudinal_mask_is_prefix_valid() -> None:
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
            "index_admittime": ["2020-01-01", "2020-01-05", "2020-02-01"],
            "index_dischtime": ["2020-01-02", "2020-01-06", "2020-02-02"],
            "a_t": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT", "A_UNKNOWN"],
            "y_t1": [1, 0, 0],
            "done": [0, 1, 1],
            "s_t_lace": [1, 2, 3],
            "s_t_age": [10, 10, 20],
            "s_t_comorbidity": [1, 1, 2],
            "s_t_acuity": [1, 1, 2],
            "s_t_length": [1, 1, 2],
            "s_t_physical_status": [1, 1, 2],
        }
    )
    out = build_longitudinal_from_transitions_v2(df, cfg)
    m = out.tensors["M_valid"]
    assert m[0].tolist() == [1, 1]
    assert m[1].tolist() == [1, 0]

