"""Observed action mapping from discharge destination (v1)."""

from __future__ import annotations

import re

import pandas as pd

A_LOW_SUPPORT = "A_LOW_SUPPORT"
A_HIGH_SUPPORT = "A_HIGH_SUPPORT"
A_TERMINAL = "A_TERMINAL"
A_UNKNOWN = "A_UNKNOWN"

ACTION_SOURCE = "discharge_location_v1"
ALLOWED_ACTIONS = {A_LOW_SUPPORT, A_HIGH_SUPPORT, A_TERMINAL, A_UNKNOWN}

LOW_SUPPORT_LABELS = {
    "HOME",
    "AGAINST ADVICE",
}

HIGH_SUPPORT_LABELS = {
    "HOME HEALTH CARE",
    "SKILLED NURSING FACILITY",
    "REHAB",
    "CHRONIC/LONG TERM ACUTE CARE",
    "ACUTE HOSPITAL",
    "PSYCH FACILITY",
    "OTHER FACILITY",
}

TERMINAL_LABELS = {
    "DIED",
    "HOSPICE",
}


def normalize_discharge_label(value: object) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip().upper()
    s = re.sub(r"\s+", " ", s)
    if s in {"", "NAN", "NONE", "NULL"}:
        return ""
    return s


def map_discharge_to_action(label: str) -> str:
    if not label:
        return A_UNKNOWN
    if label in LOW_SUPPORT_LABELS:
        return A_LOW_SUPPORT
    if label in HIGH_SUPPORT_LABELS:
        return A_HIGH_SUPPORT
    if label in TERMINAL_LABELS:
        return A_TERMINAL
    return A_UNKNOWN


def derive_action_from_discharge(series: pd.Series) -> pd.DataFrame:
    raw_norm = series.apply(normalize_discharge_label)
    out = pd.DataFrame(
        {
            "a_t": raw_norm.apply(map_discharge_to_action),
            "a_t_source": ACTION_SOURCE,
            "a_t_raw_discharge_location": raw_norm,
        }
    )
    return out

