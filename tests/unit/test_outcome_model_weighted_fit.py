from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.sim.outcome_model import fit_outcome_model


def test_fit_outcome_model_with_sample_weight() -> None:
    train_df = pd.DataFrame(
        {
            "a_t": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT", "A_LOW_SUPPORT", "A_HIGH_SUPPORT"],
            "within_30d_next_admit": [0, 1, 0, 1],
            "s_t_lace": [4, 10, 5, 11],
            "s_t_age": [50, 75, 52, 77],
            "s_t_comorbidity": [1, 3, 1, 3],
            "s_t_acuity": [1, 3, 1, 3],
            "s_t_length": [1, 3, 1, 3],
            "s_t_physical_status": [2, 8, 2, 8],
            "s_t_admission_type": ["ELECTIVE", "EMERGENCY", "ELECTIVE", "EMERGENCY"],
        }
    )
    cfg = {
        "model": {
            "features_numeric": [
                "s_t_lace",
                "s_t_age",
                "s_t_comorbidity",
                "s_t_acuity",
                "s_t_length",
                "s_t_physical_status",
            ],
            "features_categorical": ["s_t_admission_type"],
            "max_iter": 1000,
            "C": 1.0,
        }
    }
    w = np.array([1.0, 1.5, 1.0, 1.5], dtype=float)
    model = fit_outcome_model(train_df, cfg, sample_weight=w)
    assert model.model is not None
    assert len(model.feature_columns) > 0

