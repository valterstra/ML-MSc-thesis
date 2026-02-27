"""Load configuration and input datasets."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from careai.contracts.transition_schema import REQUIRED_INPUT_STAGE02_COLUMNS


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_from_config(config_path: Path, relative_or_abs: str) -> Path:
    p = Path(relative_or_abs)
    if p.is_absolute():
        return p
    return (config_path.parent / p).resolve()


def load_stage02(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["admittime", "dischtime"])
    missing = [c for c in REQUIRED_INPUT_STAGE02_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Stage-02 input missing required columns: {missing}")
    return df

