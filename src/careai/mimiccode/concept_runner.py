"""Run selected MIMIC-code concept SQL scripts on Postgres."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any


REQUIRED_SUBSET_SQL = [
    "demographics/icustay_times.sql",
    "demographics/icustay_hourly.sql",
    "demographics/weight_durations.sql",
    "demographics/age.sql",
    "demographics/icustay_detail.sql",
    "measurement/bg.sql",
    "measurement/chemistry.sql",
    "measurement/enzyme.sql",
    "measurement/complete_blood_count.sql",
    "measurement/gcs.sql",
    "measurement/urine_output.sql",
    "measurement/urine_output_rate.sql",
    "measurement/vitalsign.sql",
    "measurement/oxygen_delivery.sql",
    "measurement/ventilator_setting.sql",
    "comorbidity/charlson.sql",
    "medication/dobutamine.sql",
    "medication/dopamine.sql",
    "medication/epinephrine.sql",
    "medication/norepinephrine.sql",
    "medication/phenylephrine.sql",
    "medication/vasopressin.sql",
    "medication/milrinone.sql",
    "treatment/ventilation.sql",
    "treatment/crrt.sql",
    "score/sofa.sql",
    "medication/vasoactive_agent.sql",
]


def _psql_base_args(db_cfg: dict[str, Any]) -> list[str]:
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


def _run_sql_text(db_cfg: dict[str, Any], sql_text: str, cwd: Path | None = None) -> None:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sql", delete=False, encoding="utf-8") as tf:
        tf.write(sql_text)
        temp_path = Path(tf.name)
    try:
        subprocess.run(
            _psql_base_args(db_cfg) + ["-f", str(temp_path)],
            check=True,
            env=_psql_env(db_cfg),
            cwd=str(cwd) if cwd is not None else None,
        )
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except OSError:
            pass


def _table_exists(db_cfg: dict[str, Any], schema: str, table_name: str) -> bool:
    query = f"SELECT to_regclass('{schema}.{table_name}') IS NOT NULL;"
    proc = subprocess.run(
        _psql_base_args(db_cfg) + ["-tA", "-c", query],
        check=True,
        capture_output=True,
        text=True,
        env=_psql_env(db_cfg),
    )
    return proc.stdout.strip().lower() == "t"


def _rewrite_sql_for_schemas(sql_text: str, schemas: dict[str, Any]) -> str:
    hosp = str(schemas.get("hosp", "mimiciv_hosp"))
    icu = str(schemas.get("icu", "mimiciv_icu"))
    derived = str(schemas.get("derived", "mimiciv_derived"))
    out = sql_text
    out = out.replace("mimiciv_hosp.", f"{hosp}.")
    out = out.replace("mimiciv_icu.", f"{icu}.")
    out = out.replace("mimiciv_derived.", f"{derived}.")
    return out


def ensure_typed_hosp_schema(cfg: dict[str, Any]) -> str:
    """Create a compatibility hosp schema with typed admissions/labevents views."""
    db_cfg = cfg["db"]
    schemas = cfg["db"]["schemas"]
    source_hosp = "mimiciv_hosp"
    target_hosp = str(schemas.get("hosp", source_hosp))
    if target_hosp == source_hosp:
        return source_hosp

    sql = f"""
    CREATE SCHEMA IF NOT EXISTS {target_hosp};

    DROP VIEW IF EXISTS {target_hosp}.labevents;
    DROP VIEW IF EXISTS {target_hosp}.diagnoses_icd;
    DROP VIEW IF EXISTS {target_hosp}.admissions;
    DROP VIEW IF EXISTS {target_hosp}.patients;

    CREATE OR REPLACE VIEW {target_hosp}.patients AS
    SELECT
      CAST(subject_id AS INTEGER) AS subject_id,
      gender,
      CAST(anchor_age AS SMALLINT) AS anchor_age,
      CAST(anchor_year AS SMALLINT) AS anchor_year,
      anchor_year_group,
      CASE
        WHEN NULLIF(BTRIM(CAST(dod AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(dod AS TEXT) AS DATE)
      END AS dod
    FROM {source_hosp}.patients;

    CREATE OR REPLACE VIEW {target_hosp}.admissions AS
    SELECT
      CAST(subject_id AS INTEGER) AS subject_id,
      CAST(hadm_id AS INTEGER) AS hadm_id,
      CASE
        WHEN NULLIF(BTRIM(CAST(admittime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(admittime AS TEXT) AS TIMESTAMP)
      END AS admittime,
      CASE
        WHEN NULLIF(BTRIM(CAST(dischtime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(dischtime AS TEXT) AS TIMESTAMP)
      END AS dischtime,
      CASE
        WHEN NULLIF(BTRIM(CAST(deathtime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(deathtime AS TEXT) AS TIMESTAMP)
      END AS deathtime,
      admission_type,
      admit_provider_id,
      admission_location,
      discharge_location,
      insurance,
      language,
      marital_status,
      race,
      CASE
        WHEN NULLIF(BTRIM(CAST(edregtime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(edregtime AS TEXT) AS TIMESTAMP)
      END AS edregtime,
      CASE
        WHEN NULLIF(BTRIM(CAST(edouttime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(edouttime AS TEXT) AS TIMESTAMP)
      END AS edouttime,
      CAST(hospital_expire_flag AS SMALLINT) AS hospital_expire_flag
    FROM {source_hosp}.admissions;

    CREATE OR REPLACE VIEW {target_hosp}.diagnoses_icd AS
    SELECT
      CAST(subject_id AS INTEGER) AS subject_id,
      CASE
        WHEN NULLIF(BTRIM(CAST(hadm_id AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(hadm_id AS TEXT) AS INTEGER)
      END AS hadm_id,
      seq_num,
      icd_code,
      CAST(icd_version AS SMALLINT) AS icd_version
    FROM {source_hosp}.diagnoses_icd;

    CREATE OR REPLACE VIEW {target_hosp}.labevents AS
    SELECT
      CASE
        WHEN NULLIF(BTRIM(CAST(labevent_id AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(labevent_id AS TEXT) AS BIGINT)
      END AS labevent_id,
      CAST(subject_id AS INTEGER) AS subject_id,
      CASE
        WHEN NULLIF(BTRIM(CAST(hadm_id AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(hadm_id AS TEXT) AS INTEGER)
      END AS hadm_id,
      CASE
        WHEN NULLIF(BTRIM(CAST(specimen_id AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(specimen_id AS TEXT) AS BIGINT)
      END AS specimen_id,
      CASE
        WHEN NULLIF(BTRIM(CAST(itemid AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(itemid AS TEXT) AS INTEGER)
      END AS itemid,
      order_provider_id,
      CASE
        WHEN NULLIF(BTRIM(CAST(charttime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(charttime AS TEXT) AS TIMESTAMP)
      END AS charttime,
      CASE
        WHEN NULLIF(BTRIM(CAST(storetime AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(storetime AS TEXT) AS TIMESTAMP)
      END AS storetime,
      value,
      CASE
        WHEN NULLIF(BTRIM(CAST(valuenum AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(valuenum AS TEXT) AS DOUBLE PRECISION)
      END AS valuenum,
      valueuom,
      CASE
        WHEN NULLIF(BTRIM(CAST(ref_range_lower AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(ref_range_lower AS TEXT) AS DOUBLE PRECISION)
      END AS ref_range_lower,
      CASE
        WHEN NULLIF(BTRIM(CAST(ref_range_upper AS TEXT)), '') IS NULL THEN NULL
        ELSE CAST(CAST(ref_range_upper AS TEXT) AS DOUBLE PRECISION)
      END AS ref_range_upper,
      flag,
      priority,
      comments
    FROM {source_hosp}.labevents;
    """
    _run_sql_text(db_cfg, sql)
    return target_hosp


def run_mimiccode_concepts(cfg: dict[str, Any], mimiccode_repo: Path) -> list[str]:
    concepts_root = mimiccode_repo / "mimic-iv" / "concepts_postgres"
    if not concepts_root.exists():
        raise FileNotFoundError(f"MIMIC-code concepts path not found: {concepts_root}")

    db_cfg = cfg["db"]
    ensure_typed_hosp_schema(cfg)
    mode = str(cfg["concepts"].get("build_mode", "required_subset"))
    rebuild_existing = bool(cfg["concepts"].get("rebuild_existing", False))
    derived_schema = str(cfg["db"]["schemas"].get("derived", "mimiciv_derived"))
    executed: list[str] = []

    cmd_base = _psql_base_args(db_cfg)
    env = _psql_env(db_cfg)

    if mode == "full":
        target = concepts_root / "postgres-make-concepts.sql"
        sql_text = _rewrite_sql_for_schemas(target.read_text(encoding="utf-8"), cfg["db"]["schemas"])
        _run_sql_text(db_cfg, sql_text, cwd=concepts_root)
        executed.append(str(target))
        return executed

    func_sql = concepts_root / "postgres-functions.sql"
    _run_sql_text(db_cfg, func_sql.read_text(encoding="utf-8"), cwd=concepts_root)
    executed.append(str(func_sql))

    for rel in REQUIRED_SUBSET_SQL:
        target = concepts_root / rel
        table_name = Path(rel).stem
        if (not rebuild_existing) and _table_exists(db_cfg, derived_schema, table_name):
            continue
        raw = target.read_text(encoding="utf-8")
        sql_text = _rewrite_sql_for_schemas(raw, cfg["db"]["schemas"])
        _run_sql_text(db_cfg, sql_text, cwd=concepts_root)
        executed.append(str(target))
    return executed
