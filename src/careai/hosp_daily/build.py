"""Hospital-stay daily transition dataset — orchestrator + all build steps."""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import psycopg2

from careai.hosp_daily.drug_lists import DRUG_CLASSES
from careai.io.write_outputs import write_csv, write_json
from careai.transitions.sampling import subject_level_sample
from careai.transitions.split import assign_subject_splits

log = logging.getLogger(__name__)

SCHEMA_VERSION = "hosp_daily_v1"

# ---------------------------------------------------------------------------
# itemid → column name mapping
# ---------------------------------------------------------------------------
ITEMID_TO_COL: dict[int, str] = {
    50912: "creatinine",   51006: "bun",
    50983: "sodium",       50971: "potassium",
    50882: "bicarbonate",  50868: "anion_gap",
    50931: "glucose",      51222: "hemoglobin",
    51301: "wbc",          51265: "platelets",
    50960: "magnesium",    50893: "calcium",
    50970: "phosphate",    51237: "inr",
    50885: "bilirubin",    51244: "lymphocytes",
    51256: "neutrophils",  50862: "albumin",
    50813: "lactate_raw",
}

TIER1_COLS = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "glucose", "hemoglobin", "wbc", "platelets", "magnesium", "calcium",
    "phosphate",
]
TIER2_COLS = ["inr", "bilirubin", "nlr"]
TIER3_ALBUMIN = ["albumin"]

# ---------------------------------------------------------------------------
# ICU unit set + careunit grouping
# ---------------------------------------------------------------------------
ICU_UNITS = {
    "Medical Intensive Care Unit (MICU)",
    "Surgical Intensive Care Unit (SICU)",
    "Cardiac Vascular Intensive Care Unit (CVICU)",
    "Coronary Care Unit (CCU)",
    "Neuro Surgical Intensive Care Unit (Neuro SICU)",
    "Medical/Surgical Intensive Care Unit (MICU/SICU)",
}


def careunit_to_group(unit: str | None) -> str:
    if unit is None:
        return "other"
    if unit in ICU_UNITS or "Intensive Care" in unit:
        return "icu"
    if "Emergency Department" in unit:
        return "ed"
    if any(x in unit for x in ["Med/Surg", "Medicine"]):
        return "medicine"
    if "Labor" in unit or "Obstetric" in unit:
        return "obstetrics"
    if "Hematology" in unit or "Oncology" in unit:
        return "oncology"
    if "Neurology" in unit:
        return "neurology"
    if "Psychiatry" in unit:
        return "psychiatry"
    if any(x in unit for x in ["Surgery", "Vascular", "Transplant"]):
        return "surgery"
    return "other"


# ---------------------------------------------------------------------------
# Service grouping
# ---------------------------------------------------------------------------
SVC_MAP: dict[str, set[str]] = {
    "medicine":    {"MED", "CMED", "NMED", "OMED", "GU"},
    "surgery":     {"SURG", "NSURG", "CSURG", "VSURG", "TSURG", "PSURG",
                    "ORTHO", "ENT", "EYE"},
    "icu_svc":     {"MICU", "SICU", "TSICU", "CSRU", "CCU"},
    "psychiatry":  {"PSYCH"},
    "obstetrics":  {"OBS", "GYN"},
    "trauma":      {"TRAUM"},
}

_SVC_LOOKUP: dict[str, str] = {}
for _group, _codes in SVC_MAP.items():
    for _code in _codes:
        _SVC_LOOKUP[_code] = _group


def service_to_group(svc: str | None) -> str:
    if svc is None:
        return "other"
    return _SVC_LOOKUP.get(svc, "other")


# ---------------------------------------------------------------------------
# Charlson ICD-10 fallback mapping
# ---------------------------------------------------------------------------
_CHARLSON_ICD10: dict[str, tuple[str, int]] = {}

_CHARLSON_PREFIXES: list[tuple[list[str], str, int]] = [
    (["I21", "I22", "I25.2"], "mi", 1),
    (["I50", "I11.0", "I13.0", "I13.2", "I42"], "chf", 1),
    (["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1"], "pvd", 1),
    (["I60", "I61", "I62", "I63", "I64", "I65", "I66", "G45", "G46"], "cvd", 1),
    (["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"], "dementia", 1),
    (["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47", "J60",
      "J61", "J62", "J63", "J64", "J65", "J66", "J67"], "copd", 1),
    (["M05", "M06", "M32", "M33", "M34", "M35.3"], "rheumatic", 1),
    (["K25", "K26", "K27", "K28"], "pud", 1),
    (["K70.0", "K70.1", "K70.2", "K70.3", "K73", "K74"], "mild_liver", 1),
    (["E10.0", "E10.1", "E10.9", "E11.0", "E11.1", "E11.9",
      "E13.0", "E13.1", "E13.9"], "dm_uncomplicated", 1),
    (["E10.2", "E10.3", "E10.4", "E10.5", "E11.2", "E11.3",
      "E11.4", "E11.5", "E13.2", "E13.3", "E13.4", "E13.5"], "dm_complicated", 2),
    (["G81", "G82", "G04.1"], "hemiplegia", 2),
    (["N03", "N05", "N18", "N19", "N25.0"], "renal", 2),
    (["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C70", "C71", "C72",
      "C73", "C74", "C75", "C76", "C81", "C82", "C83", "C84", "C85",
      "C88", "C90", "C91", "C92", "C93", "C94", "C95", "C96"], "cancer", 2),
    (["K72", "K76.6", "K76.7"], "severe_liver", 3),
    (["C77", "C78", "C79", "C80"], "metastatic", 6),
    (["B20", "B21", "B22", "B24"], "aids", 6),
]


def _build_charlson_lookup() -> dict[str, tuple[str, int]]:
    lookup: dict[str, tuple[str, int]] = {}
    for prefixes, category, weight in _CHARLSON_PREFIXES:
        for pfx in prefixes:
            lookup[pfx] = (category, weight)
    return lookup


_CHARLSON_LOOKUP = _build_charlson_lookup()


def _compute_charlson_from_icd(conn: Any, schema_hosp: str) -> pd.DataFrame:
    """Fallback: compute Charlson score from diagnoses_icd."""
    log.info("  Computing Charlson score from diagnoses_icd (fallback)")
    sql = f"""
        SELECT hadm_id, icd_code, icd_version
        FROM {schema_hosp}.diagnoses_icd
    """
    diag = pd.read_sql(sql, conn)
    # Keep only ICD-10
    diag = diag[diag["icd_version"] == 10].copy()
    diag["icd_code"] = diag["icd_code"].str.strip()

    records: list[dict[str, Any]] = []
    for hadm_id, grp in diag.groupby("hadm_id"):
        found_categories: dict[str, int] = {}
        for code in grp["icd_code"]:
            for length in range(len(code), 2, -1):
                prefix = code[:length]
                if prefix in _CHARLSON_LOOKUP:
                    cat, weight = _CHARLSON_LOOKUP[prefix]
                    if cat not in found_categories or weight > found_categories[cat]:
                        found_categories[cat] = weight
                    break
        records.append({
            "hadm_id": hadm_id,
            "charlson_score": sum(found_categories.values()),
        })

    if not records:
        return pd.DataFrame(columns=["hadm_id", "charlson_score"])
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _hadm_filter(spine: pd.DataFrame, alias: str = "") -> str:
    """Return a SQL fragment to filter by hadm_ids in the spine.

    Returns empty string if spine is large (>50k admissions) to avoid
    sending huge IN clauses. For sample builds this keeps SQL fast.
    """
    hadm_ids = spine["hadm_id"].unique()
    if len(hadm_ids) > 50_000:
        return ""
    prefix = f"{alias}." if alias else ""
    csv = ",".join(str(int(h)) for h in hadm_ids)
    return f" AND {prefix}hadm_id::bigint IN ({csv})"


def _get_conn(cfg: dict[str, Any]) -> Any:
    db = cfg["db"]
    kwargs: dict[str, Any] = {
        "host": db["host"],
        "port": db["port"],
        "dbname": db["name"],
    }
    user = os.environ.get(db["user_env"])
    password = os.environ.get(db["password_env"])
    if user:
        kwargs["user"] = user
    if password:
        kwargs["password"] = password
    return psycopg2.connect(**kwargs)


def _table_exists(conn: Any, schema: str, table: str) -> bool:
    cur = conn.cursor()
    cur.execute(
        "SELECT 1 FROM information_schema.tables "
        "WHERE table_schema = %s AND table_name = %s LIMIT 1",
        (schema, table),
    )
    result = cur.fetchone() is not None
    cur.close()
    return result


# ===================================================================
# Step 1: SPINE
# ===================================================================

def step1_spine(conn: Any, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build one row per (hadm_id × calendar day)."""
    log.info("Step 1: Building spine")
    schema_hosp = cfg["schemas"]["hosp"]
    min_age = cfg["cohort"]["min_age"]

    sql = f"""
        SELECT
            a.hadm_id,
            a.subject_id,
            a.admittime,
            a.dischtime,
            a.deathtime,
            a.admission_type,
            a.admission_location,
            a.hospital_expire_flag,
            p.gender,
            CAST(p.anchor_age AS INTEGER) + (EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - CAST(p.anchor_year AS INTEGER))
                AS age_at_admit
        FROM {schema_hosp}.admissions a
        JOIN (
            SELECT
                CAST(subject_id AS INTEGER) AS subject_id,
                gender,
                anchor_age,
                anchor_year
            FROM {schema_hosp}.patients
        ) p ON CAST(a.subject_id AS INTEGER) = p.subject_id
        WHERE CAST(p.anchor_age AS INTEGER) + (EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP)) - CAST(p.anchor_year AS INTEGER))
              >= {min_age}
          AND a.dischtime IS NOT NULL
    """
    adm = pd.read_sql(sql, conn)
    log.info("  Admissions loaded: %d", len(adm))

    adm["admit_date"] = pd.to_datetime(adm["admittime"]).dt.normalize()
    adm["disch_date"] = pd.to_datetime(adm["dischtime"]).dt.normalize()

    # Filter out same-day stays (LOS < 1 day)
    adm = adm[adm["disch_date"] > adm["admit_date"]].copy()
    log.info("  After removing same-day stays: %d admissions", len(adm))

    rows: list[dict[str, Any]] = []
    for _, r in adm.iterrows():
        dates = pd.date_range(r["admit_date"], r["disch_date"], freq="D")
        for i, d in enumerate(dates):
            rows.append({
                "hadm_id": r["hadm_id"],
                "subject_id": r["subject_id"],
                "calendar_date": d.date(),
                "day_of_stay": i,
                "is_last_day": int(d == r["disch_date"]),
                "admittime": r["admittime"],
                "dischtime": r["dischtime"],
                "hospital_expire_flag": r["hospital_expire_flag"],
                "age_at_admit": r["age_at_admit"],
                "gender": r["gender"],
                "admission_type": r["admission_type"],
                "admission_location": r["admission_location"],
            })

    spine = pd.DataFrame(rows)
    spine["hadm_id"] = spine["hadm_id"].astype("int64")
    spine["subject_id"] = spine["subject_id"].astype("int64")
    log.info("  Spine rows: %d", len(spine))
    return spine


# ===================================================================
# Step 2: STATIC FEATURES
# ===================================================================

def step2_static(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    """Join DRG severity/mortality and Charlson score."""
    log.info("Step 2: Static features")
    schema_hosp = cfg["schemas"]["hosp"]
    schema_derived = cfg["schemas"]["derived"]

    # DRG
    drg_sql = f"""
        SELECT hadm_id,
               MAX(drg_severity) AS drg_severity,
               MAX(drg_mortality) AS drg_mortality
        FROM {schema_hosp}.drgcodes
        WHERE drg_type = 'APR'
        GROUP BY hadm_id
    """
    drg = pd.read_sql(drg_sql, conn)
    spine = spine.merge(drg, on="hadm_id", how="left")

    # Charlson — try derived table first, fall back to ICD computation
    charlson_source = "mimiciv_derived.charlson"
    if _table_exists(conn, schema_derived, "charlson"):
        log.info("  Using %s.charlson table", schema_derived)
        charlson_sql = f"""
            SELECT hadm_id, charlson_comorbidity_index AS charlson_score
            FROM {schema_derived}.charlson
        """
        charlson = pd.read_sql(charlson_sql, conn)
    else:
        charlson_source = "diagnoses_icd (ICD-10 fallback)"
        charlson = _compute_charlson_from_icd(conn, schema_hosp)

    spine = spine.merge(charlson, on="hadm_id", how="left")
    log.info("  Charlson source: %s", charlson_source)
    return spine, charlson_source


# ===================================================================
# Step 3: LOCATION STATE
# ===================================================================

def step3_location(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    """Daily careunit_group, is_icu, days_in_current_unit from transfers."""
    log.info("Step 3: Location state")
    schema_hosp = cfg["schemas"]["hosp"]

    sql = f"""
        SELECT hadm_id, careunit, intime, outtime
        FROM {schema_hosp}.transfers
        WHERE hadm_id IS NOT NULL
          AND eventtype != 'discharge'
    """
    transfers = pd.read_sql(sql, conn)
    transfers["indate"] = pd.to_datetime(transfers["intime"]).dt.normalize()
    transfers["outdate"] = pd.to_datetime(
        transfers["outtime"].fillna(transfers["intime"] + pd.Timedelta(days=1))
    ).dt.normalize()

    # Build a lookup: for each hadm_id, sorted transfer rows
    transfer_groups = {}
    for hadm_id, grp in transfers.groupby("hadm_id"):
        transfer_groups[hadm_id] = grp.sort_values("intime").reset_index(drop=True)

    careunit_groups = []
    is_icu_flags = []
    days_in_unit = []

    for _, row in spine.iterrows():
        hadm_id = row["hadm_id"]
        day_d = pd.Timestamp(row["calendar_date"]).normalize()

        grp = transfer_groups.get(hadm_id)
        if grp is None:
            careunit_groups.append("other")
            is_icu_flags.append(0)
            days_in_unit.append(0)
            continue

        # Find transfer active at day_d: indate <= day_d < outdate
        mask = (grp["indate"] <= day_d) & (day_d < grp["outdate"])
        matched = grp[mask]

        if matched.empty:
            # Try last transfer before day_d
            before = grp[grp["indate"] <= day_d]
            if not before.empty:
                best = before.iloc[-1]
            else:
                careunit_groups.append("other")
                is_icu_flags.append(0)
                days_in_unit.append(0)
                continue
        else:
            best = matched.iloc[-1]  # latest intime if multiple overlaps

        unit = best["careunit"]
        grp_name = careunit_to_group(unit)
        careunit_groups.append(grp_name)
        is_icu_flags.append(int(grp_name == "icu"))
        days_in_unit.append((day_d - best["indate"]).days)

    spine["careunit_group"] = careunit_groups
    spine["is_icu"] = is_icu_flags
    spine["days_in_current_unit"] = days_in_unit

    log.info("  Location state assigned")
    return spine


# ===================================================================
# Step 4: SERVICE STATE
# ===================================================================

def step4_service(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    """Daily curr_service_group from services table."""
    log.info("Step 4: Service state")
    schema_hosp = cfg["schemas"]["hosp"]

    sql = f"""
        SELECT hadm_id, curr_service, transfertime
        FROM {schema_hosp}.services
    """
    services = pd.read_sql(sql, conn)
    services["svc_date"] = pd.to_datetime(services["transfertime"]).dt.normalize()
    services = services.sort_values(["hadm_id", "transfertime"])

    svc_groups = {}
    for hadm_id, grp in services.groupby("hadm_id"):
        svc_groups[hadm_id] = grp.reset_index(drop=True)

    svc_results = []
    for _, row in spine.iterrows():
        hadm_id = row["hadm_id"]
        day_d = pd.Timestamp(row["calendar_date"]).normalize()

        grp = svc_groups.get(hadm_id)
        if grp is None:
            svc_results.append("other")
            continue

        before = grp[grp["svc_date"] <= day_d]
        if before.empty:
            svc_results.append("other")
        else:
            svc_results.append(service_to_group(before.iloc[-1]["curr_service"]))

    spine["curr_service_group"] = svc_results
    log.info("  Service state assigned")
    return spine


# ===================================================================
# Step 5: LAB STATE
# ===================================================================

def step5_labs(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    """Query labevents (aggregated in PostgreSQL), pivot, forward-fill."""
    log.info("Step 5: Lab state")
    schema_hosp = cfg["schemas"]["hosp"]
    itemids = list(ITEMID_TO_COL.keys())
    itemid_csv = ",".join(str(i) for i in itemids)

    hadm_filt = _hadm_filter(spine)
    sql = f"""
        SELECT
            hadm_id,
            DATE(charttime) AS lab_date,
            itemid,
            AVG(CAST(valuenum AS DOUBLE PRECISION)) AS daily_value,
            COUNT(*) AS n_draws
        FROM {schema_hosp}.labevents
        WHERE hadm_id IS NOT NULL
          AND valuenum IS NOT NULL
          AND NULLIF(BTRIM(CAST(valuenum AS TEXT)), '') IS NOT NULL
          AND itemid::integer IN ({itemid_csv})
          {hadm_filt}
        GROUP BY hadm_id, DATE(charttime), itemid
    """
    labs_raw = pd.read_sql(sql, conn)
    labs_raw["hadm_id"] = labs_raw["hadm_id"].astype("int64")
    labs_raw["itemid"] = labs_raw["itemid"].astype("int64")
    log.info("  Lab rows (aggregated): %d", len(labs_raw))

    # Map itemid → column name
    labs_raw["col_name"] = labs_raw["itemid"].map(ITEMID_TO_COL)

    # Count distinct labs per (hadm_id, lab_date)
    n_labs = (
        labs_raw.groupby(["hadm_id", "lab_date"])["itemid"]
        .nunique()
        .reset_index()
        .rename(columns={"itemid": "n_labs_today"})
    )

    # Pivot to wide
    labs_wide = labs_raw.pivot_table(
        index=["hadm_id", "lab_date"],
        columns="col_name",
        values="daily_value",
        aggfunc="mean",
    ).reset_index()

    # Compute NLR
    if "neutrophils" in labs_wide.columns and "lymphocytes" in labs_wide.columns:
        labs_wide["nlr"] = (
            labs_wide["neutrophils"] / labs_wide["lymphocytes"].replace(0, np.nan)
        ).clip(upper=50)
        labs_wide.drop(columns=["neutrophils", "lymphocytes"], inplace=True)
    else:
        labs_wide["nlr"] = np.nan

    # Compute lactate_elevated
    lactate_threshold = cfg.get("lactate_threshold", 2.0)
    if "lactate_raw" in labs_wide.columns:
        labs_wide["lactate_elevated"] = (
            labs_wide["lactate_raw"] > lactate_threshold
        ).astype(int)
        labs_wide.drop(columns=["lactate_raw"], inplace=True)
    else:
        labs_wide["lactate_elevated"] = np.nan

    # Merge n_labs_today
    labs_wide = labs_wide.merge(n_labs, on=["hadm_id", "lab_date"], how="left")

    # Merge onto spine
    spine["calendar_date_dt"] = pd.to_datetime(spine["calendar_date"])
    labs_wide["lab_date"] = pd.to_datetime(labs_wide["lab_date"])
    spine = spine.merge(
        labs_wide,
        left_on=["hadm_id", "calendar_date_dt"],
        right_on=["hadm_id", "lab_date"],
        how="left",
    )
    spine.drop(columns=["lab_date", "calendar_date_dt"], inplace=True)

    # Forward-fill within each hadm_id
    fill_limits = cfg.get("lab_fill_limits", {})
    tier1_limit = fill_limits.get("tier1", 2)
    tier2_limit = fill_limits.get("tier2", 3)
    albumin_limit = fill_limits.get("albumin", 5)

    spine.sort_values(["hadm_id", "day_of_stay"], inplace=True)

    for cols, limit in [
        (TIER1_COLS, tier1_limit),
        (TIER2_COLS, tier2_limit),
        (TIER3_ALBUMIN, albumin_limit),
    ]:
        present = [c for c in cols if c in spine.columns]
        if present:
            spine[present] = spine.groupby("hadm_id")[present].ffill(limit=limit)

    # lactate_elevated: no forward-fill, fill NaN with 0
    if "lactate_elevated" in spine.columns:
        spine["lactate_elevated"] = spine["lactate_elevated"].fillna(0).astype(int)

    # n_labs_today: fill missing with 0
    if "n_labs_today" in spine.columns:
        spine["n_labs_today"] = spine["n_labs_today"].fillna(0).astype(int)

    log.info("  Lab state merged and forward-filled")
    return spine


# ===================================================================
# Step 6: INFECTION STATE
# ===================================================================

def step6_infection(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    """Culture flags from microbiologyevents."""
    log.info("Step 6: Infection state")
    schema_hosp = cfg["schemas"]["hosp"]

    hadm_filt = _hadm_filter(spine)
    sql = f"""
        SELECT
            hadm_id,
            DATE(COALESCE(charttime, chartdate)) AS culture_date,
            MAX(CASE WHEN org_name IS NOT NULL THEN 1 ELSE 0 END) AS positive_today,
            MAX(CASE WHEN org_name IS NOT NULL
                     AND spec_type_desc = 'BLOOD CULTURE' THEN 1 ELSE 0 END)
                AS blood_positive_today,
            COUNT(*) AS culture_count
        FROM {schema_hosp}.microbiologyevents
        WHERE hadm_id IS NOT NULL
          {hadm_filt}
        GROUP BY hadm_id, DATE(COALESCE(charttime, chartdate))
    """
    micro = pd.read_sql(sql, conn)
    micro["hadm_id"] = micro["hadm_id"].astype("int64")
    log.info("  Microbiology rows: %d", len(micro))

    micro["culture_ordered_today"] = 1
    micro["culture_date"] = pd.to_datetime(micro["culture_date"])

    spine["calendar_date_dt"] = pd.to_datetime(spine["calendar_date"])
    spine = spine.merge(
        micro[["hadm_id", "culture_date", "positive_today",
               "blood_positive_today", "culture_ordered_today"]],
        left_on=["hadm_id", "calendar_date_dt"],
        right_on=["hadm_id", "culture_date"],
        how="left",
    )
    spine.drop(columns=["culture_date", "calendar_date_dt"], inplace=True)

    for col in ["culture_ordered_today", "positive_today", "blood_positive_today"]:
        spine[col] = spine[col].fillna(0).astype(int)

    # Cumulative flags: latch to 1 once positive
    spine.sort_values(["hadm_id", "day_of_stay"], inplace=True)
    spine["positive_culture_cumulative"] = (
        spine.groupby("hadm_id")["positive_today"].cummax()
    )
    spine["blood_culture_positive_cumulative"] = (
        spine.groupby("hadm_id")["blood_positive_today"].cummax()
    )
    spine.drop(columns=["positive_today", "blood_positive_today"], inplace=True)

    log.info("  Infection state assigned")
    return spine


# ===================================================================
# Step 7: ACTIONS
# ===================================================================

def step7_actions(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Drug class flags, antibiotic start/stop, ICU transitions, discharge."""
    log.info("Step 7: Actions")
    schema_hosp = cfg["schemas"]["hosp"]

    # --- 7a: Prescriptions --------------------------------------------------
    log.info("  7a: Prescriptions")
    hadm_filt = _hadm_filter(spine)
    rx_sql = f"""
        SELECT hadm_id, drug, starttime, stoptime
        FROM {schema_hosp}.prescriptions
        WHERE hadm_id IS NOT NULL
          AND drug_type = 'MAIN'
          AND starttime IS NOT NULL
          {hadm_filt}
    """
    rx = pd.read_sql(rx_sql, conn)
    log.info("  Prescription rows: %d", len(rx))

    rx["start_date"] = pd.to_datetime(rx["starttime"]).dt.normalize()
    rx["stop_date"] = pd.to_datetime(rx["stoptime"]).dt.normalize()
    # If no stoptime, treat as single-day prescription
    rx["stop_date"] = rx["stop_date"].fillna(rx["start_date"] + pd.Timedelta(days=1))

    # Classify each prescription row
    compiled_patterns = {
        cls: [re.compile(p, re.IGNORECASE) for p in patterns]
        for cls, patterns in DRUG_CLASSES.items()
    }

    def classify_drug(drug_name: str) -> list[str]:
        classes = []
        for cls, patterns in compiled_patterns.items():
            if any(p.search(drug_name) for p in patterns):
                classes.append(cls)
        return classes

    rx["drug_classes"] = rx["drug"].apply(classify_drug)

    # Expand prescriptions to daily records
    # For efficiency, build per-(hadm_id, date) sets of active classes
    active_classes: dict[tuple[int, Any], set[str]] = {}
    ab_started: dict[tuple[int, Any], int] = {}
    ab_stopped: dict[tuple[int, Any], int] = {}

    for _, r in rx.iterrows():
        if not r["drug_classes"]:
            continue
        hadm_id = r["hadm_id"]
        dates = pd.date_range(r["start_date"], r["stop_date"], freq="D", inclusive="left")
        for d in dates:
            key = (hadm_id, d.date())
            if key not in active_classes:
                active_classes[key] = set()
            active_classes[key].update(r["drug_classes"])

        if "antibiotic" in r["drug_classes"]:
            start_key = (hadm_id, r["start_date"].date())
            ab_started[start_key] = 1
            stop_key = (hadm_id, r["stop_date"].date())
            ab_stopped[stop_key] = 1

    # Map onto spine
    for cls in DRUG_CLASSES:
        col = f"{cls}_active"
        spine[col] = spine.apply(
            lambda row: int(
                cls in active_classes.get(
                    (row["hadm_id"], row["calendar_date"]), set()
                )
            ),
            axis=1,
        )

    spine["antibiotic_started"] = spine.apply(
        lambda row: ab_started.get((row["hadm_id"], row["calendar_date"]), 0),
        axis=1,
    )
    spine["antibiotic_stopped"] = spine.apply(
        lambda row: ab_stopped.get((row["hadm_id"], row["calendar_date"]), 0),
        axis=1,
    )

    # n_active_drug_classes
    drug_class_cols = [f"{cls}_active" for cls in DRUG_CLASSES]
    spine["n_active_drug_classes"] = spine[drug_class_cols].sum(axis=1)

    # --- 7b: Care intensity transitions from transfers -----------------------
    log.info("  7b: Care intensity transitions")
    xfer_sql = f"""
        SELECT hadm_id, careunit, eventtype, intime, outtime
        FROM {schema_hosp}.transfers
        WHERE hadm_id IS NOT NULL
    """
    xfers = pd.read_sql(xfer_sql, conn)
    xfers["indate"] = pd.to_datetime(xfers["intime"]).dt.normalize()

    # discharge events
    discharge_events = (
        xfers[xfers["eventtype"] == "discharge"]
        .groupby(["hadm_id", "indate"])
        .size()
        .reset_index()
        .rename(columns={0: "_count"})
    )
    discharge_set = set(zip(discharge_events["hadm_id"], discharge_events["indate"]))

    # ICU escalation / stepdown: consecutive transfers
    non_discharge = xfers[xfers["eventtype"] != "discharge"].sort_values(
        ["hadm_id", "intime"]
    )

    icu_esc_set: set[tuple[int, Any]] = set()
    icu_step_set: set[tuple[int, Any]] = set()

    for hadm_id, grp in non_discharge.groupby("hadm_id"):
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            prev_icu = careunit_to_group(grp.iloc[i - 1]["careunit"]) == "icu"
            curr_icu = careunit_to_group(grp.iloc[i]["careunit"]) == "icu"
            day_key = (hadm_id, grp.iloc[i]["indate"])
            if curr_icu and not prev_icu:
                icu_esc_set.add(day_key)
            elif not curr_icu and prev_icu:
                icu_step_set.add(day_key)

    spine["calendar_date_ts"] = pd.to_datetime(spine["calendar_date"]).dt.normalize()
    spine["icu_escalation"] = spine.apply(
        lambda r: int((r["hadm_id"], r["calendar_date_ts"]) in icu_esc_set), axis=1
    )
    spine["icu_stepdown"] = spine.apply(
        lambda r: int((r["hadm_id"], r["calendar_date_ts"]) in icu_step_set), axis=1
    )
    spine["discharged"] = spine.apply(
        lambda r: int((r["hadm_id"], r["calendar_date_ts"]) in discharge_set), axis=1
    )
    spine.drop(columns=["calendar_date_ts"], inplace=True)

    log.info("  Actions assigned")
    return spine


# ===================================================================
# Step 8: LABEL + SPLIT + OUTPUT
# ===================================================================

def step8_label_split_output(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame, charlson_source: str,
    sample_only: bool = False,
) -> pd.DataFrame:
    """Readmit_30d label, train/valid/test split, write CSV + manifest."""
    log.info("Step 8: Label, split, output")
    schema_hosp = cfg["schemas"]["hosp"]

    # --- Readmission label ---
    readmit_sql = f"""
        SELECT a1.hadm_id,
            CASE WHEN MIN(a2.admittime::timestamp) <= a1.dischtime::timestamp + INTERVAL '30 days'
                 THEN 1 ELSE 0 END AS readmit_30d
        FROM {schema_hosp}.admissions a1
        LEFT JOIN {schema_hosp}.admissions a2
            ON a1.subject_id = a2.subject_id
            AND a2.admittime::timestamp > a1.dischtime::timestamp
        GROUP BY a1.hadm_id, a1.dischtime
    """
    readmit = pd.read_sql(readmit_sql, conn)
    readmit["hadm_id"] = readmit["hadm_id"].astype("int64")
    spine = spine.merge(readmit, on="hadm_id", how="left")
    spine["readmit_30d"] = spine["readmit_30d"].fillna(0).astype(int)

    # --- Split ---
    split_cfg = cfg["split"]
    # assign_subject_splits expects a "patient_id" column
    spine["patient_id"] = spine["subject_id"]
    spine = assign_subject_splits(
        spine,
        train=split_cfg["train"],
        valid=split_cfg["valid"],
        test=split_cfg["test"],
        seed=split_cfg["seed"],
    )
    spine.drop(columns=["patient_id"], inplace=True)

    # --- Schema version + timestamp ---
    build_ts = datetime.now(timezone.utc).isoformat()
    spine["schema_version"] = SCHEMA_VERSION
    spine["build_timestamp_utc"] = build_ts

    # --- Column ordering ---
    desired_order = [
        # Episode identifiers
        "hadm_id", "subject_id", "day_of_stay", "calendar_date", "is_last_day",
        # Episode frame (static)
        "admittime", "dischtime", "hospital_expire_flag",
        "age_at_admit", "gender",
        "admission_type", "admission_location",
        "drg_severity", "drg_mortality", "charlson_score",
        # Location
        "careunit_group", "is_icu", "days_in_current_unit", "curr_service_group",
        # Labs Tier 1
        "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
        "glucose", "hemoglobin", "wbc", "platelets", "magnesium", "calcium",
        "phosphate",
        # Labs Tier 2
        "inr", "bilirubin", "nlr",
        # Labs Tier 3
        "albumin", "lactate_elevated",
        # Lab meta
        "n_labs_today",
        # Infection
        "culture_ordered_today", "positive_culture_cumulative",
        "blood_culture_positive_cumulative",
        # Actions
        "antibiotic_active", "anticoagulant_active", "diuretic_active",
        "steroid_active", "insulin_active", "opioid_active",
        "antibiotic_started", "antibiotic_stopped",
        "icu_escalation", "icu_stepdown", "discharged",
        "n_active_drug_classes",
        # Label + split
        "readmit_30d", "split", "schema_version", "build_timestamp_utc",
    ]
    present_cols = [c for c in desired_order if c in spine.columns]
    extra_cols = [c for c in spine.columns if c not in desired_order]
    if extra_cols:
        log.warning("  Unexpected extra columns (will be dropped): %s", extra_cols)
    spine = spine[present_cols]

    # --- Write outputs ---
    project_root = Path(cfg.get("_project_root", "."))
    out_dir = project_root / cfg["output"]["dir"]
    out_path = out_dir / cfg["output"]["filename"]
    sample_path = out_dir / cfg["output"]["sample_filename"]
    manifest_path = project_root / "reports" / "hosp_daily" / "build_manifest.json"

    if sample_only:
        # In sample-only mode the spine IS the sample — write it directly
        log.info("  Writing sample dataset: %s", sample_path)
        write_csv(spine, sample_path)
        sample_df = spine
    else:
        log.info("  Writing full dataset: %s", out_path)
        write_csv(spine, out_path)

        # Sample
        sample_cfg = cfg["sample"]
        n_subjects = spine["subject_id"].nunique()
        fraction = sample_cfg["n_episodes"] / n_subjects if n_subjects > 0 else 1.0
        sample_df = subject_level_sample(
            spine, fraction=fraction, seed=sample_cfg["seed"]
        )
        log.info(
            "  Sample: %d rows (%d episodes)",
            len(sample_df),
            sample_df["hadm_id"].nunique(),
        )
        write_csv(sample_df, sample_path)

    # Manifest
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "build_timestamp_utc": build_ts,
        "row_count": len(spine),
        "column_count": len(spine.columns),
        "columns": list(spine.columns),
        "hadm_id_count": int(spine["hadm_id"].nunique()),
        "subject_id_count": int(spine["subject_id"].nunique()),
        "split_counts": spine["split"].value_counts().to_dict(),
        "sample_row_count": len(sample_df),
        "sample_episode_count": int(sample_df["hadm_id"].nunique()),
        "charlson_source": charlson_source,
        "sample_only_build": sample_only,
        "output_path": str(out_path) if not sample_only else None,
        "sample_path": str(sample_path),
    }
    write_json(manifest, manifest_path)
    log.info("  Manifest written: %s", manifest_path)

    return spine


# ===================================================================
# Pipeline orchestrator
# ===================================================================

ALL_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]


def _sample_spine_early(
    spine: pd.DataFrame, n_episodes: int, seed: int
) -> pd.DataFrame:
    """Down-sample spine to ~n_episodes admissions (by subject) for fast builds."""
    n_subjects = spine["subject_id"].nunique()
    if n_subjects == 0:
        return spine
    # Estimate fraction: we want ~n_episodes *admissions*, sample by subject
    n_hadm = spine["hadm_id"].nunique()
    episodes_per_subject = n_hadm / n_subjects
    target_subjects = int(n_episodes / episodes_per_subject) + 1
    fraction = min(target_subjects / n_subjects, 1.0)
    sampled = subject_level_sample(spine, fraction=fraction, seed=seed)
    log.info(
        "  Early sample: %d subjects, %d admissions, %d rows",
        sampled["subject_id"].nunique(),
        sampled["hadm_id"].nunique(),
        len(sampled),
    )
    return sampled


def run_pipeline(
    cfg: dict[str, Any],
    steps: list[int] | None = None,
    dry_run: bool = False,
    sample_only: bool = False,
) -> pd.DataFrame | None:
    """Run the full (or partial) build pipeline.

    Args:
        sample_only: If True, down-sample to ~5k episodes right after
            Step 1 so all subsequent steps run much faster. Useful for
            testing the pipeline before committing to a full build.
    """
    steps = steps or ALL_STEPS
    conn = _get_conn(cfg)
    log.info("Connected to PostgreSQL %s:%s/%s", cfg["db"]["host"],
             cfg["db"]["port"], cfg["db"]["name"])

    try:
        spine = None
        charlson_source = "unknown"

        if 1 in steps:
            spine = step1_spine(conn, cfg)
            if dry_run:
                log.info("Dry-run: spine has %d rows, %d admissions",
                         len(spine), spine["hadm_id"].nunique())
                return spine
            if sample_only:
                sample_cfg = cfg["sample"]
                spine = _sample_spine_early(
                    spine,
                    n_episodes=sample_cfg["n_episodes"],
                    seed=sample_cfg["seed"],
                )

        if 2 in steps:
            spine, charlson_source = step2_static(conn, cfg, spine)

        if 3 in steps:
            spine = step3_location(conn, cfg, spine)

        if 4 in steps:
            spine = step4_service(conn, cfg, spine)

        if 5 in steps:
            spine = step5_labs(conn, cfg, spine)

        if 6 in steps:
            spine = step6_infection(conn, cfg, spine)

        if 7 in steps:
            spine = step7_actions(conn, cfg, spine)

        if 8 in steps:
            spine = step8_label_split_output(
                conn, cfg, spine, charlson_source,
                sample_only=sample_only,
            )

        log.info("Pipeline complete.")
        return spine
    finally:
        conn.close()
