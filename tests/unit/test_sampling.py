from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.sampling import subject_level_sample


def test_subject_level_sample_deterministic() -> None:
    df = pd.DataFrame(
        {
            "subject_id": [1, 1, 2, 3, 4, 5],
            "hadm_id": [10, 11, 20, 30, 40, 50],
        }
    )
    a = subject_level_sample(df, fraction=0.4, seed=42)
    b = subject_level_sample(df, fraction=0.4, seed=42)
    assert a["subject_id"].tolist() == b["subject_id"].tolist()
    assert a["hadm_id"].tolist() == b["hadm_id"].tolist()

