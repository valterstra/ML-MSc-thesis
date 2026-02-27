from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.split import assign_subject_splits


def test_assign_subject_splits_has_multiple_buckets() -> None:
    df = pd.DataFrame({"patient_id": list(range(1, 1001))})
    out = assign_subject_splits(df, train=0.7, valid=0.15, test=0.15, seed=42)
    counts = out["split"].value_counts().to_dict()
    assert "train" in counts and counts["train"] > 0
    assert "valid" in counts and counts["valid"] > 0
    assert "test" in counts and counts["test"] > 0

