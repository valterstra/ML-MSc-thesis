"""Extract v1 admission state from stage-02 columns."""

from __future__ import annotations

import pandas as pd


def add_state_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["s_t_length"] = pd.to_numeric(out["Length"], errors="coerce")
    out["s_t_acuity"] = pd.to_numeric(out["Acuity"], errors="coerce")
    out["s_t_comorbidity"] = pd.to_numeric(out["Comorbidity"], errors="coerce")
    out["s_t_lace"] = pd.to_numeric(out["LACE"], errors="coerce")
    out["s_t_physical_status"] = pd.to_numeric(out["labevents"], errors="coerce").fillna(0)
    out["s_t_age"] = pd.to_numeric(out["anchor_age"], errors="coerce")
    out["s_t_admission_type"] = out["admission_type"].astype(str)
    out["s_t_discharge_location"] = out["discharge_location"].astype(str)
    return out

