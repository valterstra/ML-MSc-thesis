from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.contracts.validators import validate_transition_contract
from careai.qa.transition_report_v1 import generate_transition_qa
from careai.transitions.build_transition_v1 import build_transitions
from careai.transitions.split import assign_subject_splits


def test_end_to_end_small_dataframe() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3],
            "hadm_id": [10, 11, 20, 30],
            "admittime": ["2020-01-01", "2020-01-20", "2020-02-01", "2020-03-01"],
            "dischtime": ["2020-01-05", "2020-01-25", "2020-02-05", "2020-03-05"],
            "admission_type": ["EMERGENCY", "URGENT", "ELECTIVE", "URGENT"],
            "discharge_location": ["HOME", "HOME", "HOME", "HOME"],
            "anchor_age": [70, 70, 60, 50],
            "Length": [2, 2, 1, 2],
            "Acuity": [3, 3, 0, 3],
            "Comorbidity": [2, 2, 1, 1],
            "LACE": [10, 11, 3, 6],
            "labevents": [5, 6, 2, 1],
        }
    )
    cfg = {
        "horizon": {"readmit_days": 30},
        "schema": {"version": "transition_v1", "source_dataset": "test"},
        "sample_tag": "sample_2pct",
    }
    out = build_transitions(df, cfg)
    out = assign_subject_splits(out, train=0.7, valid=0.15, test=0.15, seed=42)
    report = validate_transition_contract(out)
    qa = generate_transition_qa(out)
    assert report["ok"]
    assert qa["rows"] == len(out)

