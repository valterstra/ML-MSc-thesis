"""Censoring and readmission horizon logic."""

from __future__ import annotations

import pandas as pd


def within_horizon_days(dischtime: pd.Series, next_admittime: pd.Series, days: int) -> pd.Series:
    delta = (next_admittime - dischtime).dt.total_seconds() / 86400.0
    return ((delta > 0) & (delta <= float(days))).astype(int)

