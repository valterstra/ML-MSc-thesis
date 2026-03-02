from __future__ import annotations

import pandas as pd

from careai.readmission.model import train_readmission_baseline


def test_readmission_model_trains_and_returns_metrics() -> None:
    rows = []
    for i in range(30):
        split = "train" if i < 20 else ("valid" if i < 25 else "test")
        y = 1 if i % 5 == 0 else 0
        rows.append(
            {
                "episode_id": f"e{i}",
                "patient_id": i,
                "index_hadm_id": 1000 + i,
                "split": split,
                "readmit_30d": y,
                "episode_hours": 10 + i,
                "last_s_t_sofa": float(i % 6),
                "last_s_t_mbp": 70.0 + (i % 10),
                "last_s_t_heart_rate": 80.0 + (i % 7),
                "last_s_t_resp_rate": 16.0 + (i % 5),
                "last_s_t_spo2": 96.0 + (i % 3),
                "last_s_t_gcs": 13.0 + (i % 3),
                "last_s_t_urine_output_rate": 0.5 + (i % 4) * 0.1,
                "last_s_t_oxygen_delivery": 1.0 + (i % 3),
                "last_s_t_creatinine": 1.0 + (i % 4) * 0.1,
                "last_s_t_bun": 12.0 + (i % 6),
                "last_s_t_age": 50.0 + (i % 15),
                "last_s_t_charlson": float(i % 4),
                "mean_s_t_sofa": float(i % 6),
                "mean_s_t_mbp": 70.0 + (i % 10),
                "mean_s_t_heart_rate": 80.0 + (i % 7),
                "mean_s_t_resp_rate": 16.0 + (i % 5),
                "mean_s_t_spo2": 96.0 + (i % 3),
                "mean_s_t_gcs": 13.0 + (i % 3),
                "mean_s_t_urine_output_rate": 0.6 + (i % 4) * 0.1,
                "mean_s_t_oxygen_delivery": 1.5 + (i % 3),
                "mean_s_t_creatinine": 1.0 + (i % 4) * 0.1,
                "mean_s_t_bun": 12.0 + (i % 6),
                "mean_s_t_age": 50.0 + (i % 15),
                "mean_s_t_charlson": float(i % 4),
                "min_s_t_mbp": 60.0 + (i % 10),
                "max_s_t_sofa": float((i % 6) + 1),
                "delta_s_t_sofa": float((i % 3) - 1),
                "delta_s_t_mbp": float((i % 5) - 2),
                "frac_vaso": float((i % 4) / 4.0),
                "frac_vent": float((i % 3) / 3.0),
                "frac_crrt": float((i % 5) / 5.0),
                "any_vaso": int(i % 2 == 0),
                "any_vent": int(i % 3 == 0),
                "any_crrt": int(i % 5 == 0),
            }
        )
    df = pd.DataFrame(rows)
    cfg = {
        "model": {"kind": "logreg", "max_iter": 200, "c": 1.0, "random_state": 42},
        "features": {
            "include": [
                c
                for c in df.columns
                if c
                not in {"episode_id", "patient_id", "index_hadm_id", "split", "readmit_30d"}
            ]
        },
    }
    out = train_readmission_baseline(df, cfg)
    assert out["metrics"]["valid"]["n"] > 0
    assert out["metrics"]["test"]["n"] > 0
