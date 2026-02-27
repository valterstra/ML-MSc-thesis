from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.transitions.build_longitudinal_v1 import build_longitudinal_from_transitions_v2


def _cfg() -> dict:
    return {
        "features": {
            "state_numeric": [
                "s_t_lace",
                "s_t_age",
                "s_t_comorbidity",
                "s_t_acuity",
                "s_t_length",
                "s_t_physical_status",
            ]
        },
        "actions": {"mapping": {"A_LOW_SUPPORT": 0, "A_HIGH_SUPPORT": 1, "A_TERMINAL": 2, "A_UNKNOWN": 3}, "padding_id": -1},
        "split": {"mapping": {"train": 0, "valid": 1, "test": 2}},
    }


def _df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "episode_id": ["p1_ep0", "p1_ep0", "p2_ep0"],
            "patient_id": [1, 1, 2],
            "t": [0, 1, 0],
            "split": ["train", "train", "test"],
            "index_admittime": ["2020-01-01", "2020-01-05", "2020-02-01"],
            "index_dischtime": ["2020-01-02", "2020-01-06", "2020-02-02"],
            "a_t": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT", "A_UNKNOWN"],
            "y_t1": [1, 0, 0],
            "done": [0, 1, 1],
            "s_t_lace": [10, 8, 3],
            "s_t_age": [70, 70, 55],
            "s_t_comorbidity": [2, 2, 1],
            "s_t_acuity": [3, 2, 1],
            "s_t_length": [2, 2, 1],
            "s_t_physical_status": [5, 4, 2],
        }
    )


def test_longitudinal_builder_shapes_and_dtypes() -> None:
    out = build_longitudinal_from_transitions_v2(_df(), _cfg())
    x = out.tensors["X_state"]
    m = out.tensors["M_valid"]
    a = out.tensors["A_action"]
    y = out.tensors["Y_next"]
    d = out.tensors["D_done"]
    s = out.tensors["split"]

    assert x.shape == (2, 2, 6)
    assert m.shape == (2, 2)
    assert a.shape == (2, 2)
    assert y.shape == (2, 2)
    assert d.shape == (2, 2)
    assert s.shape == (2,)

    assert x.dtype == np.float32
    assert m.dtype == np.uint8
    assert a.dtype == np.int16
    assert y.dtype == np.uint8
    assert d.dtype == np.uint8
    assert s.dtype == np.int8

