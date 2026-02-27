"""Build 30-day readmission labels from admissions history in Postgres."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _psql_env(db_cfg: dict[str, Any]) -> dict[str, str]:
    env = dict(os.environ)
    user_env = str(db_cfg.get("user_env", "PGUSER"))
    password_env = str(db_cfg.get("password_env", "PGPASSWORD"))
    if user_env in os.environ:
        env["PGUSER"] = os.environ[user_env]
    if password_env in os.environ:
        env["PGPASSWORD"] = os.environ[password_env]
    return env


def _psql_args(db_cfg: dict[str, Any], sql_text: str) -> list[str]:
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
        sql_text,
    ]


def load_readmission_labels_from_db(cfg: dict[str, Any], episode_df: pd.DataFrame) -> pd.DataFrame:
    db_cfg = cfg["db"]
    schemas = cfg["db"]["schemas"]
    days = int(cfg["label"]["readmit_days"])

    hosp_typed = str(schemas["hosp_typed"])
    hosp_fallback = str(schemas["hosp_fallback"])

    # Use typed schema when present; fallback to casted source schema.
    sql = f"""
    COPY (
      WITH adm AS (
        SELECT
          CAST(subject_id AS INTEGER) AS subject_id,
          CAST(hadm_id AS INTEGER) AS hadm_id,
          CAST(CAST(admittime AS TEXT) AS TIMESTAMP) AS admittime,
          CAST(CAST(dischtime AS TEXT) AS TIMESTAMP) AS dischtime
        FROM {hosp_typed}.admissions
      ),
      nxt AS (
        SELECT
          a.subject_id,
          a.hadm_id,
          a.dischtime AS index_dischtime,
          MIN(b.admittime) AS next_admittime
        FROM adm a
        LEFT JOIN adm b
          ON b.subject_id = a.subject_id
         AND b.admittime > a.dischtime
        GROUP BY a.subject_id, a.hadm_id, a.dischtime
      )
      SELECT
        subject_id AS patient_id,
        hadm_id AS index_hadm_id,
        index_dischtime,
        next_admittime,
        CASE
          WHEN next_admittime IS NULL THEN NULL
          ELSE EXTRACT(EPOCH FROM (next_admittime - index_dischtime))/86400.0
        END AS days_to_next_admit,
        CASE
          WHEN next_admittime IS NULL THEN 0
          WHEN EXTRACT(EPOCH FROM (next_admittime - index_dischtime))/86400.0 > 0
           AND EXTRACT(EPOCH FROM (next_admittime - index_dischtime))/86400.0 <= {days}
            THEN 1
          ELSE 0
        END AS readmit_30d
      FROM nxt
    ) TO STDOUT WITH CSV HEADER
    """

    args = _psql_args(db_cfg, sql)
    env = _psql_env(db_cfg)
    proc = subprocess.run(args, check=False, capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        # Fallback query directly on fallback schema with safe casts.
        sql2 = f"""
        COPY (
          WITH adm AS (
            SELECT
              CAST(subject_id AS INTEGER) AS subject_id,
              CAST(hadm_id AS INTEGER) AS hadm_id,
              CAST(CAST(admittime AS TEXT) AS TIMESTAMP) AS admittime,
              CAST(CAST(dischtime AS TEXT) AS TIMESTAMP) AS dischtime
            FROM {hosp_fallback}.admissions
          ),
          nxt AS (
            SELECT
              a.subject_id,
              a.hadm_id,
              a.dischtime AS index_dischtime,
              MIN(b.admittime) AS next_admittime
            FROM adm a
            LEFT JOIN adm b
              ON b.subject_id = a.subject_id
             AND b.admittime > a.dischtime
            GROUP BY a.subject_id, a.hadm_id, a.dischtime
          )
          SELECT
            subject_id AS patient_id,
            hadm_id AS index_hadm_id,
            index_dischtime,
            next_admittime,
            CASE
              WHEN next_admittime IS NULL THEN NULL
              ELSE EXTRACT(EPOCH FROM (next_admittime - index_dischtime))/86400.0
            END AS days_to_next_admit,
            CASE
              WHEN next_admittime IS NULL THEN 0
              WHEN EXTRACT(EPOCH FROM (next_admittime - index_dischtime))/86400.0 > 0
               AND EXTRACT(EPOCH FROM (next_admittime - index_dischtime))/86400.0 <= {days}
                THEN 1
              ELSE 0
            END AS readmit_30d
          FROM nxt
        ) TO STDOUT WITH CSV HEADER
        """
        args2 = _psql_args(db_cfg, sql2)
        proc = subprocess.run(args2, check=False, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            raise RuntimeError(f"Failed readmission label query in fallback schema: {proc.stderr.strip()}")

    labels = pd.read_csv(pd.io.common.StringIO(proc.stdout))
    labels["patient_id"] = pd.to_numeric(labels["patient_id"], errors="coerce").astype("Int64")
    labels["index_hadm_id"] = pd.to_numeric(labels["index_hadm_id"], errors="coerce").astype("Int64")
    labels["readmit_30d"] = pd.to_numeric(labels["readmit_30d"], errors="coerce").fillna(0).astype(int)
    labels["days_to_next_admit"] = pd.to_numeric(labels["days_to_next_admit"], errors="coerce")
    return labels
