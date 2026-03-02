"""Extract hourly ICU state-action rows from MIMIC-derived tables using psql COPY."""

from __future__ import annotations

import io
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _psql_args(db_cfg: dict[str, Any], copy_sql: str) -> list[str]:
    return [
        "psql",
        "-h",
        str(db_cfg["host"]),
        "-p",
        str(db_cfg["port"]),
        "-d",
        str(db_cfg["name"]),
        "-v",
        "ON_ERROR_STOP=1",
        "-c",
        copy_sql,
    ]


def _psql_env(db_cfg: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    user_env = str(db_cfg.get("user_env", "PGUSER"))
    password_env = str(db_cfg.get("password_env", "PGPASSWORD"))
    if user_env in os.environ:
        env["PGUSER"] = os.environ[user_env]
    if password_env in os.environ:
        env["PGPASSWORD"] = os.environ[password_env]
    return env


def _render_copy_sql(sql_query: str) -> str:
    query = " ".join(sql_query.split()).rstrip(";")
    return f"COPY ({query}) TO STDOUT WITH CSV HEADER"


def extract_hourly_from_postgres(cfg: dict[str, Any], sql_file: Path, limit_stays: int | None = None) -> pd.DataFrame:
    sql_query = sql_file.read_text(encoding="utf-8")
    stay_filter = ""
    if limit_stays is not None and int(limit_stays) > 0:
        stay_filter = (
            "AND ih.stay_id IN ("
            "SELECT stay_id FROM mimiciv_derived.icustay_hourly "
            f"GROUP BY stay_id ORDER BY stay_id LIMIT {int(limit_stays)})"
        )
    sql_query = sql_query.replace("/*__STAY_FILTER__*/", stay_filter)
    cmd = _psql_args(cfg["db"], _render_copy_sql(sql_query))
    proc = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
        env=_psql_env(cfg["db"]),
    )
    if proc.returncode != 0:
        raise RuntimeError(f"psql extraction failed ({proc.returncode}): {proc.stderr.strip()}")
    df = pd.read_csv(io.StringIO(proc.stdout))

    for col in ["starttime", "endtime"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    for col in ["subject_id", "hadm_id", "stay_id", "hr"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    hr_min = int(cfg["time"].get("hr_min", 0))
    hr_max = cfg["time"].get("hr_max", None)
    if "hr" in df.columns:
        df = df[df["hr"] >= hr_min].copy()
        if hr_max is not None:
            df = df[df["hr"] <= int(hr_max)].copy()
    return df.reset_index(drop=True)
