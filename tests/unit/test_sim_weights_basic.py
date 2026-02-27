from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.sim.weights import build_sim_training_weights


def test_build_sim_training_weights_basic_properties() -> None:
    df = pd.DataFrame(
        {
            "a_t": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT", "A_LOW_SUPPORT", "A_HIGH_SUPPORT", "A_UNKNOWN"],
            "s_t_lace": [5, 9, 6, 10, 7],
            "s_t_age": [50, 70, 55, 72, 60],
            "s_t_comorbidity": [1, 3, 1, 3, 2],
            "s_t_acuity": [1, 3, 1, 3, 2],
            "s_t_length": [1, 3, 1, 3, 2],
            "s_t_physical_status": [2, 8, 2, 8, 4],
            "s_t_admission_type": ["ELECTIVE", "EMERGENCY", "ELECTIVE", "EMERGENCY", "URGENT"],
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
            "weight_include_actions": ["A_LOW_SUPPORT", "A_HIGH_SUPPORT"],
            "weight_clip_percentiles": [0.01, 0.99],
            "weight_stabilized": True,
            "weight_treatment_high_label": "A_HIGH_SUPPORT",
        }
    }
    out = build_sim_training_weights(df, cfg)
    assert out.sample_weight.shape == (len(df),)
    assert out.propensity.shape == (len(df),)
    assert np.isfinite(out.sample_weight).all()
    assert (out.sample_weight > 0).all()
    # rows outside include_actions keep neutral weights and no propensity.
    assert out.sample_weight[-1] == 1.0
    assert np.isnan(out.propensity[-1])
    assert out.diagnostics["n_rows_weighted"] == 4

