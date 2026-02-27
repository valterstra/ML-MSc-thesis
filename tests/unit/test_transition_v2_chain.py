from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.episode_chain import assign_episode_ids_and_steps


def test_episode_steps_increment_and_reset() -> None:
    df = pd.DataFrame(
        {
            "patient_id": [1, 1, 1, 2, 2],
            "index_admittime": pd.to_datetime(["2020-01-01", "2020-01-10", "2020-02-20", "2020-03-01", "2020-03-10"]),
            "index_hadm_id": [10, 11, 12, 20, 21],
            "done": [0, 1, 1, 0, 1],
        }
    )
    out = assign_episode_ids_and_steps(df)
    p1 = out[out["patient_id"] == 1].sort_values("index_admittime")
    assert p1["episode_step"].tolist() == [0, 1, 0]

