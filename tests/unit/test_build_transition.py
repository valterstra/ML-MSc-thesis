from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.build_transition_v1 import build_transitions


def _cfg() -> dict:
    return {
        "horizon": {"readmit_days": 30},
        "schema": {"version": "transition_v1", "source_dataset": "test"},
        "sample_tag": "sample_2pct",
    }


def test_build_transitions_with_and_without_30d_readmit() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 11, 20],
            "admittime": ["2020-01-01", "2020-01-20", "2020-02-01"],
            "dischtime": ["2020-01-05", "2020-01-25", "2020-02-05"],
            "admission_type": ["EMERGENCY", "URGENT", "ELECTIVE"],
            "discharge_location": ["HOME", "HOME", "HOME"],
            "anchor_age": [70, 70, 60],
            "Length": [2, 2, 1],
            "Acuity": [3, 3, 0],
            "Comorbidity": [2, 2, 1],
            "LACE": [10, 11, 3],
            "labevents": [5, 6, 2],
        }
    )
    out = build_transitions(df, _cfg())
    row_10 = out[out["index_hadm_id"] == 10].iloc[0]
    row_11 = out[out["index_hadm_id"] == 11].iloc[0]
    assert row_10["within_30d_next_admit"] == 1
    assert pd.notna(row_10["s_t1_lace"])
    assert row_11["within_30d_next_admit"] == 0
    assert pd.isna(row_11["s_t1_lace"])

