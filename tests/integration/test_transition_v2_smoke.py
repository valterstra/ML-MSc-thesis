from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.contracts.validators_v2 import validate_transition_v2_contract
from careai.transitions.build_transition_v2_multi import build_transitions_v2
from careai.transitions.split import assign_subject_splits


def test_transition_v2_small_build() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2],
            "hadm_id": [10, 11, 20],
            "admittime": ["2020-01-01", "2020-01-20", "2020-02-01"],
            "dischtime": ["2020-01-05", "2020-01-25", "2020-02-05"],
            "admission_type": ["EMERGENCY", "URGENT", "ELECTIVE"],
            "discharge_location": ["HOME", "HOME HEALTH CARE", "HOME"],
            "anchor_age": [70, 70, 60],
            "Length": [2, 2, 1],
            "Acuity": [3, 3, 0],
            "Comorbidity": [2, 2, 1],
            "LACE": [10, 11, 3],
            "labevents": [5, 6, 2],
        }
    )
    cfg = {
        "horizon": {"readmit_days": 30},
        "schema": {"version": "transition_v2_multi", "source_dataset": "test"},
        "sample_tag": "sample_2pct",
    }
    out = build_transitions_v2(df, cfg)
    out = assign_subject_splits(out, train=0.7, valid=0.15, test=0.15, seed=42)
    report = validate_transition_v2_contract(out)
    assert report["ok"]

