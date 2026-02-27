"""Deterministic subject-level sampling."""

from __future__ import annotations

import hashlib

import pandas as pd


def _uniform_from_key(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return bucket / float(2**64)


def subject_level_sample(df: pd.DataFrame, fraction: float, seed: int) -> pd.DataFrame:
    if fraction >= 1.0:
        return df.copy()
    if fraction <= 0.0:
        return df.iloc[0:0].copy()

    subjects = pd.Series(df["subject_id"].dropna().unique())
    keep_subjects = subjects[
        subjects.apply(lambda sid: _uniform_from_key(f"sample:{seed}:{int(sid)}") < fraction)
    ]
    return df[df["subject_id"].isin(keep_subjects)].copy()
