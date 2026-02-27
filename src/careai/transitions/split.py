"""Deterministic subject-level split assignment."""

from __future__ import annotations

import hashlib

import pandas as pd


def _u01(seed: int, subject_id: int) -> float:
    digest = hashlib.sha256(f"split:{seed}:{subject_id}".encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return bucket / float(2**64)


def assign_subject_splits(
    df: pd.DataFrame,
    train: float,
    valid: float,
    test: float,
    seed: int,
) -> pd.DataFrame:
    if abs((train + valid + test) - 1.0) > 1e-8:
        raise ValueError("split fractions must sum to 1.0")
    out = df.copy()

    def pick(subject_id: int) -> str:
        u = _u01(seed, int(subject_id))
        if u < train:
            return "train"
        if u < train + valid:
            return "valid"
        return "test"

    split_map = {int(s): pick(int(s)) for s in out["patient_id"].dropna().unique()}
    out["split"] = out["patient_id"].map(split_map)
    return out
