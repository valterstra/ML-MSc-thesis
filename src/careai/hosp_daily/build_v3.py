"""Hospital-stay clean triplet dataset v3.

V3 is the V2 pipeline with 6 targeted changes:

  1. Labs:        FIRST draw per day (DISTINCT ON, not AVG GROUP BY).
                 first_lab_charttime stored per row for temporal auditing.
  2. No fill:    Forward-fill removed entirely. Labs are NaN if not measured.
  3. Panel gate: Rows kept only when ALL 13 core labs are measured on that day.
  4. Action filter: A row is DROPPED if any of the 5 ATE drugs had a new
                 prescription start BEFORE first_lab_charttime on that day.
                 Pre-existing drugs (started before this calendar day) are
                 always valid — no timing check needed.
  5. Triplets:   Each output row is (state_T, action_T, next_state_T+1).
                 next_* columns pre-joined in the same row.
                 The same rules 3+4 apply to day T+1.
  6. Deltas:     Lab deltas computed as next_X - X (not from forward-fill).

Everything else from V2 is preserved: all extra labs, eMAR, IV route,
discharge meds, care transitions, DRG, demographics, partial SOFA, NLR, etc.
Extra columns are NaN when not measured — they never gate row inclusion.

Expected output: ~1.02M triplet rows.
Schema version: hosp_daily_v3
"""

from __future__ import annotations

import json
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

SCHEMA_VERSION = "hosp_daily_v3"

# ---------------------------------------------------------------------------
# Lab itemids — identical to V2 (all labs queried; only PANEL_COLS gate rows)
# ---------------------------------------------------------------------------
ITEMID_TO_COL: dict[int, str] = {
    # Core panel (13) — all must be present for row inclusion
    50912: "creatinine",   51006: "bun",
    50983: "sodium",       50971: "potassium",
    50882: "bicarbonate",  50868: "anion_gap",
    50931: "glucose",      51222: "hemoglobin",
    51301: "wbc",          51265: "platelets",
    50960: "magnesium",    50893: "calcium",
    50970: "phosphate",
    # Extra labs — included when measured, NaN otherwise
    51237: "inr",          50885: "bilirubin",
    51275: "ptt",          50862: "albumin",
    50813: "lactate_raw",
    51244: "lymphocytes",  51256: "neutrophils",
    50889: "crp",
    51003: "troponin_t",   51002: "troponin_i",
    50963: "bnp",
    50861: "alt",          50878: "ast",
}

PANEL_COLS: list[str] = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
    "magnesium",
]

# All continuous lab columns eligible for delta computation
LAB_COLS: list[str] = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
    "magnesium", "inr", "bilirubin", "ptt", "albumin", "nlr",
    "crp", "troponin_t", "troponin_i", "bnp", "alt", "ast",
]

ATE_DRUGS = ["antibiotic", "anticoagulant", "diuretic", "steroid", "insulin"]

# Route values treated as intravenous (same as V2)
IV_ROUTE_KEYWORDS = {
    "IV", "IV DRIP", "IV BOLUS", "IVPB", "IV PUSH",
    "IV CONT", "IVPB DRIP", "IV LOCK", "IV INFUSION",
}

# eMAR event_txt values confirming actual administration (same as V2)
EMAR_ADMINISTERED = {
    "Administered", "Delayed Administered",
    "Administered Bolus from IV Drip", "Administered in Other Location",
    "Partial Administered",
}

# ---------------------------------------------------------------------------
# ICU unit classification (unchanged from V2)
# ---------------------------------------------------------------------------
_ICU_UNITS = {
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
    if unit in _ICU_UNITS or "Intensive Care" in unit:
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


_SVC_MAP: dict[str, set[str]] = {
    "medicine":   {"MED", "CMED", "NMED", "OMED", "GU"},
    "surgery":    {"SURG", "NSURG", "CSURG", "VSURG", "TSURG", "PSURG",
                   "ORTHO", "ENT", "EYE"},
    "icu_svc":    {"MICU", "SICU", "TSICU", "CSRU", "CCU"},
    "psychiatry": {"PSYCH"},
    "obstetrics": {"OBS", "GYN"},
    "trauma":     {"TRAUM"},
}
_SVC_LOOKUP: dict[str, str] = {
    code: group for group, codes in _SVC_MAP.items() for code in codes
}


def service_to_group(svc: str | None) -> str:
    if svc is None:
        return "other"
    return _SVC_LOOKUP.get(svc, "other")


# ---------------------------------------------------------------------------
# Partial SOFA (unchanged from V2 — handles NaN gracefully)
# ---------------------------------------------------------------------------
def compute_partial_sofa(df: pd.DataFrame) -> pd.Series:
    """3-component lab-based SOFA (renal + hepatic + coagulation).
    Returns NaN only when ALL three components are unmeasured.
    In V3, bilirubin/INR are optional so this may be 1-2 components.
    """
    parts: list[pd.Series] = []
    if "creatinine" in df.columns:
        cr = df["creatinine"]
        renal = np.select(
            [cr.isna(), cr < 1.2, cr < 2.0, cr < 3.5, cr < 5.0],
            [np.nan,    0,        1,        2,        3], default=4,
        )
        parts.append(pd.Series(renal, index=df.index, dtype="float64"))
    if "bilirubin" in df.columns:
        bili = df["bilirubin"]
        hepatic = np.select(
            [bili.isna(), bili < 1.2, bili < 2.0, bili < 6.0, bili < 12.0],
            [np.nan,      0,          1,          2,           3], default=4,
        )
        parts.append(pd.Series(hepatic, index=df.index, dtype="float64"))
    if "platelets" in df.columns:
        plt = df["platelets"]
        coag = np.select(
            [plt.isna(), plt >= 150, plt >= 100, plt >= 50, plt >= 20],
            [np.nan,     0,          1,          2,         3], default=4,
        )
        parts.append(pd.Series(coag, index=df.index, dtype="float64"))
    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")
    return pd.concat(parts, axis=1).sum(axis=1, min_count=1)


# ---------------------------------------------------------------------------
# Charlson ICD-10 fallback (unchanged from V2)
# ---------------------------------------------------------------------------
_CHARLSON_PREFIXES: list[tuple[list[str], str, int]] = [
    (["I21", "I22", "I25.2"], "mi", 1),
    (["I50", "I11.0", "I13.0", "I13.2", "I42"], "chf", 1),
    (["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1"], "pvd", 1),
    (["I60", "I61", "I62", "I63", "I64", "I65", "I66", "G45", "G46"], "cvd", 1),
    (["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"], "dementia", 1),
    (["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
      "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"], "copd", 1),
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
_CHARLSON_LOOKUP: dict[str, tuple[str, int]] = {
    pfx: (cat, w)
    for prefixes, cat, w in _CHARLSON_PREFIXES
    for pfx in prefixes
}


def _compute_charlson_from_icd(conn: Any, schema_hosp: str) -> pd.DataFrame:
    log.info("  Computing Charlson from diagnoses_icd (ICD-10 fallback)")
    sql = f"SELECT hadm_id, icd_code, icd_version FROM {schema_hosp}.diagnoses_icd"
    diag = pd.read_sql(sql, conn)
    diag = diag[diag["icd_version"] == 10].copy()
    diag["icd_code"] = diag["icd_code"].str.strip()
    records: list[dict] = []
    for hadm_id, grp in diag.groupby("hadm_id"):
        found: dict[str, int] = {}
        for code in grp["icd_code"]:
            for length in range(len(code), 2, -1):
                pfx = code[:length]
                if pfx in _CHARLSON_LOOKUP:
                    cat, weight = _CHARLSON_LOOKUP[pfx]
                    if cat not in found or weight > found[cat]:
                        found[cat] = weight
                    break
        records.append({"hadm_id": hadm_id, "charlson_score": sum(found.values())})
    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["hadm_id", "charlson_score"]
    )


# ---------------------------------------------------------------------------
# DB helpers (unchanged from V2)
# ---------------------------------------------------------------------------
def _get_conn(cfg: dict[str, Any]) -> Any:
    db = cfg["db"]
    kwargs: dict[str, Any] = {
        "host": db["host"], "port": db["port"], "dbname": db["name"],
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
        "WHERE table_schema=%s AND table_name=%s LIMIT 1",
        (schema, table),
    )
    result = cur.fetchone() is not None
    cur.close()
    return result


def _hadm_filter(spine: pd.DataFrame, alias: str = "") -> str:
    hadm_ids = spine["hadm_id"].unique()
    if len(hadm_ids) > 50_000:
        return ""
    prefix = f"{alias}." if alias else ""
    csv = ",".join(str(int(h)) for h in hadm_ids)
    return f" AND {prefix}hadm_id::bigint IN ({csv})"


# ===========================================================================
# Step 1: SPINE
# ===========================================================================

def step1_spine(conn: Any, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build admission-day spine (one row per hadm_id × calendar_date)."""
    log.info("Step 1: Building spine")
    schema_hosp = cfg["schemas"]["hosp"]
    min_age     = cfg["cohort"]["min_age"]

    sql = f"""
        SELECT
            a.hadm_id, a.subject_id,
            a.admittime, a.dischtime, a.deathtime,
            a.admission_type, a.admission_location,
            a.discharge_location, a.hospital_expire_flag,
            a.insurance, a.marital_status, a.race, a.language,
            p.gender, p.dod,
            CAST(p.anchor_age AS INTEGER)
                + (EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP))
                   - CAST(p.anchor_year AS INTEGER)) AS age_at_admit
        FROM {schema_hosp}.admissions a
        JOIN (
            SELECT CAST(subject_id AS INTEGER) AS subject_id,
                   gender, anchor_age, anchor_year, dod
            FROM {schema_hosp}.patients
        ) p ON CAST(a.subject_id AS INTEGER) = p.subject_id
        WHERE
            CAST(p.anchor_age AS INTEGER)
                + (EXTRACT(YEAR FROM CAST(a.admittime AS TIMESTAMP))
                   - CAST(p.anchor_year AS INTEGER)) >= {min_age}
          AND a.dischtime IS NOT NULL
    """
    adm = pd.read_sql(sql, conn)
    adm["admit_date"] = pd.to_datetime(adm["admittime"]).dt.normalize()
    adm["disch_date"] = pd.to_datetime(adm["dischtime"]).dt.normalize()
    adm = adm[adm["disch_date"] > adm["admit_date"]].copy()
    log.info("  Admissions (multi-day stays): %d", len(adm))

    rows = []
    for _, r in adm.iterrows():
        dates = pd.date_range(r["admit_date"], r["disch_date"], freq="D")
        for i, d in enumerate(dates):
            rows.append({
                "hadm_id":              r["hadm_id"],
                "subject_id":           r["subject_id"],
                "calendar_date":        d.date(),
                "day_of_stay":          i,
                "is_last_day":          int(d == r["disch_date"]),
                "admittime":            r["admittime"],
                "dischtime":            r["dischtime"],
                "hospital_expire_flag": r["hospital_expire_flag"],
                "age_at_admit":         r["age_at_admit"],
                "gender":               r["gender"],
                "admission_type":       r["admission_type"],
                "admission_location":   r["admission_location"],
                "discharge_location":   r["discharge_location"],
                "insurance":            r["insurance"],
                "marital_status":       r["marital_status"],
                "race":                 r["race"],
                "language":             r["language"],
                "dod":                  r["dod"],
            })

    spine = pd.DataFrame(rows)
    spine["hadm_id"]    = spine["hadm_id"].astype("int64")
    spine["subject_id"] = spine["subject_id"].astype("int64")
    log.info("  Spine rows: %d", len(spine))
    return spine


# ===========================================================================
# Step 2: STATIC FEATURES
# ===========================================================================

def step2_static(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame):
    """Add charlson_score and DRG severity/mortality."""
    log.info("Step 2: Static features")
    schema_hosp    = cfg["schemas"]["hosp"]
    schema_derived = cfg["schemas"]["derived"]

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

    if _table_exists(conn, schema_derived, "charlson"):
        ch_sql = f"""
            SELECT hadm_id, charlson_comorbidity_index AS charlson_score
            FROM {schema_derived}.charlson
        """
        charlson = pd.read_sql(ch_sql, conn)
        source = "mimiciv_derived.charlson"
    else:
        charlson = _compute_charlson_from_icd(conn, schema_hosp)
        source = "diagnoses_icd (ICD-10 fallback)"

    spine = spine.merge(charlson, on="hadm_id", how="left")
    log.info("  Charlson source: %s", source)
    return spine, source


# ===========================================================================
# Step 3: LABS — FIRST DRAW, NO FORWARD-FILL, PANEL FILTER
# ===========================================================================

def step3_labs(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    """Query first lab draw per day; pivot; filter to full-panel days.

    KEY V3 CHANGE:
    - DISTINCT ON keeps the earliest charttime per (hadm_id, date, itemid)
    - No forward-fill applied at any tier
    - Rows dropped where any of the 13 PANEL_COLS is NaN
    - first_lab_charttime stored: minimum draw_time across all labs on that day
    """
    log.info("Step 3: Labs (first draw, no forward-fill, panel filter)")
    schema_hosp = cfg["schemas"]["hosp"]
    itemid_csv  = ",".join(str(i) for i in ITEMID_TO_COL)
    hadm_filt   = _hadm_filter(spine)

    # DISTINCT ON: one row per (hadm_id, date, itemid), earliest charttime.
    # NOTE: charttime and itemid are TEXT in MIMIC — cast explicitly.
    sql = f"""
        SELECT DISTINCT ON (
            hadm_id::bigint,
            DATE(charttime::timestamp),
            itemid::int
        )
            hadm_id::bigint                    AS hadm_id,
            DATE(charttime::timestamp)         AS lab_date,
            charttime::timestamp               AS draw_time,
            itemid::int                        AS itemid,
            CAST(valuenum AS DOUBLE PRECISION) AS value
        FROM {schema_hosp}.labevents
        WHERE hadm_id IS NOT NULL
          AND valuenum IS NOT NULL
          AND itemid::int IN ({itemid_csv})
          {hadm_filt}
        ORDER BY
            hadm_id::bigint,
            DATE(charttime::timestamp),
            itemid::int,
            charttime::timestamp
    """
    raw = pd.read_sql(sql, conn)
    raw["hadm_id"]  = raw["hadm_id"].astype("int64")
    raw["lab_date"] = pd.to_datetime(raw["lab_date"])
    log.info("  Raw lab rows (first draw per hadm/date/itemid): %d", len(raw))

    # Intraday lab complexity: count distinct draw-hour buckets per (hadm_id, date)
    # Uses only the 13 panel itemids; simple GROUP BY, no LATERAL.
    panel_itemid_csv = ",".join(
        str(i) for i, col in ITEMID_TO_COL.items() if col in PANEL_COLS
    )
    complexity_sql = f"""
        SELECT
            hadm_id::bigint            AS hadm_id,
            DATE(charttime::timestamp) AS lab_date,
            COUNT(DISTINCT DATE_TRUNC('hour', charttime::timestamp)) AS n_draw_hours
        FROM {schema_hosp}.labevents
        WHERE hadm_id IS NOT NULL
          AND valuenum IS NOT NULL
          AND itemid::int IN ({panel_itemid_csv})
          {hadm_filt}
        GROUP BY 1, 2
    """
    draw_hours = pd.read_sql(complexity_sql, conn)
    draw_hours["hadm_id"]  = draw_hours["hadm_id"].astype("int64")
    draw_hours["lab_date"] = pd.to_datetime(draw_hours["lab_date"])

    # first_lab_charttime = earliest draw across ALL lab items on that day
    first_ct = (
        raw.groupby(["hadm_id", "lab_date"])["draw_time"]
        .min()
        .reset_index()
        .rename(columns={"draw_time": "first_lab_charttime"})
    )

    # n_labs_today = distinct lab items drawn that day (all itemids)
    n_labs = (
        raw.groupby(["hadm_id", "lab_date"])["itemid"]
        .nunique()
        .reset_index()
        .rename(columns={"itemid": "n_labs_today"})
    )

    # Pivot to wide
    raw["col_name"] = raw["itemid"].map(ITEMID_TO_COL)
    wide = raw.pivot_table(
        index=["hadm_id", "lab_date"],
        columns="col_name",
        values="value",
        aggfunc="first",
    ).reset_index()
    wide.columns.name = None

    # NLR (neutrophils / lymphocytes)
    if "neutrophils" in wide.columns and "lymphocytes" in wide.columns:
        wide["nlr"] = (
            wide["neutrophils"] / wide["lymphocytes"].replace(0, np.nan)
        ).clip(upper=50)
        wide.drop(columns=["neutrophils", "lymphocytes"], inplace=True)
    else:
        wide["nlr"] = np.nan

    # lactate_elevated flag
    lactate_threshold = cfg.get("lactate_threshold", 2.0)
    if "lactate_raw" in wide.columns:
        wide["lactate_elevated"] = (wide["lactate_raw"] > lactate_threshold).astype(int)
        wide.drop(columns=["lactate_raw"], inplace=True)
    else:
        wide["lactate_elevated"] = np.nan

    # Merge metadata
    wide = wide.merge(first_ct, on=["hadm_id", "lab_date"], how="left")
    wide = wide.merge(n_labs,   on=["hadm_id", "lab_date"], how="left")
    wide["n_labs_today"] = wide["n_labs_today"].fillna(0).astype(int)

    # Partial SOFA from state_T labs
    wide["partial_sofa"] = compute_partial_sofa(wide)

    # Merge onto spine
    spine["calendar_date_dt"] = pd.to_datetime(spine["calendar_date"])
    wide["lab_date"]          = pd.to_datetime(wide["lab_date"])
    spine = spine.merge(
        wide,
        left_on=["hadm_id", "calendar_date_dt"],
        right_on=["hadm_id", "lab_date"],
        how="left",
    )
    spine.drop(columns=["lab_date", "calendar_date_dt"], inplace=True)

    # PANEL FILTER: drop rows missing any of the 13 core labs
    before = len(spine)
    spine = spine.dropna(subset=PANEL_COLS).reset_index(drop=True)
    log.info(
        "  Panel days kept: %d / %d (%.1f%%)",
        len(spine), before, 100 * len(spine) / before if before else 0,
    )
    log.info("  first_lab_charttime coverage: %d non-null",
             spine["first_lab_charttime"].notna().sum())

    # Log intraday lab complexity for the kept panel days
    spine_lab_dates = pd.DataFrame({
        "hadm_id": spine["hadm_id"].astype("int64"),
        "lab_date": pd.to_datetime(spine["calendar_date"]),
    })
    lab_complexity = spine_lab_dates.merge(draw_hours, on=["hadm_id", "lab_date"], how="left")
    multi_draw = int((lab_complexity["n_draw_hours"] > 1).sum())
    log.info(
        "  Intraday lab complexity: %d / %d panel days (%.1f%%) have core labs drawn at >1 distinct hour",
        multi_draw, len(spine), 100 * multi_draw / len(spine) if len(spine) else 0,
    )
    return spine


# ===========================================================================
# Step 4: LOCATION STATE (unchanged from V2)
# ===========================================================================

def step4_location(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 4: Location state")
    schema_hosp = cfg["schemas"]["hosp"]

    sql = f"""
        SELECT hadm_id, careunit, intime, outtime
        FROM {schema_hosp}.transfers
        WHERE hadm_id IS NOT NULL AND eventtype != 'discharge'
    """
    transfers = pd.read_sql(sql, conn)
    transfers["indate"]  = pd.to_datetime(transfers["intime"]).dt.normalize()
    transfers["outdate"] = pd.to_datetime(
        transfers["outtime"].fillna(transfers["intime"] + pd.Timedelta(days=1))
    ).dt.normalize()

    transfer_groups = {
        hadm_id: grp.sort_values("intime").reset_index(drop=True)
        for hadm_id, grp in transfers.groupby("hadm_id")
    }

    careunit_groups, is_icu_flags, days_in_unit = [], [], []
    for _, row in spine.iterrows():
        hadm_id = row["hadm_id"]
        day_d   = pd.Timestamp(row["calendar_date"]).normalize()
        grp     = transfer_groups.get(hadm_id)
        if grp is None:
            careunit_groups.append("other"); is_icu_flags.append(0); days_in_unit.append(0)
            continue
        mask    = (grp["indate"] <= day_d) & (day_d < grp["outdate"])
        matched = grp[mask]
        if matched.empty:
            before = grp[grp["indate"] <= day_d]
            if before.empty:
                careunit_groups.append("other"); is_icu_flags.append(0); days_in_unit.append(0)
                continue
            best = before.iloc[-1]
        else:
            best = matched.iloc[-1]
        grp_name = careunit_to_group(best["careunit"])
        careunit_groups.append(grp_name)
        is_icu_flags.append(int(grp_name == "icu"))
        days_in_unit.append((day_d - best["indate"]).days)

    spine = spine.copy()
    spine["careunit_group"]       = careunit_groups
    spine["is_icu"]               = is_icu_flags
    spine["days_in_current_unit"] = days_in_unit
    log.info("  Location state assigned")
    return spine


# ===========================================================================
# Step 5: SERVICE STATE (unchanged from V2)
# ===========================================================================

def step5_service(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 5: Service state")
    schema_hosp = cfg["schemas"]["hosp"]

    sql = f"""
        SELECT hadm_id, curr_service, transfertime
        FROM {schema_hosp}.services
    """
    services = pd.read_sql(sql, conn)
    services["svc_date"] = pd.to_datetime(services["transfertime"]).dt.normalize()
    services = services.sort_values(["hadm_id", "transfertime"])

    svc_groups = {
        hadm_id: grp.reset_index(drop=True)
        for hadm_id, grp in services.groupby("hadm_id")
    }

    svc_results = []
    for _, row in spine.iterrows():
        hadm_id = row["hadm_id"]
        day_d   = pd.Timestamp(row["calendar_date"]).normalize()
        grp     = svc_groups.get(hadm_id)
        if grp is None:
            svc_results.append("other"); continue
        before = grp[grp["svc_date"] <= day_d]
        svc_results.append(
            service_to_group(before.iloc[-1]["curr_service"])
            if not before.empty else "other"
        )

    spine = spine.copy()
    spine["curr_service_group"] = svc_results
    log.info("  Service state assigned")
    return spine


# ===========================================================================
# Step 6: INFECTION STATE (unchanged from V2)
# ===========================================================================

def step6_infection(conn: Any, cfg: dict[str, Any], spine: pd.DataFrame) -> pd.DataFrame:
    log.info("Step 6: Infection state")
    schema_hosp = cfg["schemas"]["hosp"]
    hadm_filt   = _hadm_filter(spine)

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
    micro["hadm_id"]      = micro["hadm_id"].astype("int64")
    micro["culture_date"] = pd.to_datetime(micro["culture_date"])
    micro["culture_ordered_today"] = 1

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


# ===========================================================================
# Step 7: ACTIONS — with causal ordering filter
# ===========================================================================

def step7_actions(
    conn: Any,
    cfg: dict[str, Any],
    spine: pd.DataFrame,
    ckpt_dir: "Path | None" = None,
    resume: bool = False,
) -> pd.DataFrame:
    """Drug flags, IV route, eMAR confirmation, discharge meds, transitions.

    KEY V3 CHANGE:
    Any row where a new start of one of the 5 ATE drugs occurred BEFORE
    first_lab_charttime on that day is dropped entirely (Option A).
    Pre-existing drugs (started before this calendar day) are always valid.
    """
    log.info("Step 7: Actions")
    schema_hosp = cfg["schemas"]["hosp"]

    # Try to resume from step7a checkpoint (skips ~35-min rx loop)
    _ckpt7a_spine = _load_ckpt(ckpt_dir, "step7a") if (resume and ckpt_dir) else None

    # ------------------------------------------------------------------
    # 7a: Prescriptions  (SQL always runs — rx is also needed for 7b/7c)
    # ------------------------------------------------------------------
    log.info("  7a: Prescriptions")
    hadm_filt = _hadm_filter(spine if _ckpt7a_spine is None else _ckpt7a_spine)
    rx_sql = f"""
        SELECT hadm_id, drug, starttime, stoptime, dose_val_rx, dose_unit_rx, route
        FROM {schema_hosp}.prescriptions
        WHERE hadm_id IS NOT NULL
          AND drug_type = 'MAIN'
          AND starttime IS NOT NULL
          {hadm_filt}
    """
    rx = pd.read_sql(rx_sql, conn)
    rx["hadm_id"]    = rx["hadm_id"].astype("int64")
    rx["start_date"] = pd.to_datetime(rx["starttime"]).dt.normalize()
    rx["stop_date"]  = pd.to_datetime(rx["stoptime"]).dt.normalize()
    rx["stop_date"]  = rx["stop_date"].fillna(rx["start_date"] + pd.Timedelta(days=1))
    log.info("  Prescription rows: %d", len(rx))

    compiled_patterns = {
        cls: [re.compile(p, re.IGNORECASE) for p in patterns]
        for cls, patterns in DRUG_CLASSES.items()
    }

    def classify_drug(drug_name):
        if not drug_name or not isinstance(drug_name, str):
            return []
        return [cls for cls, pats in compiled_patterns.items()
                if any(p.search(drug_name) for p in pats)]

    rx["drug_classes"] = rx["drug"].apply(classify_drug)
    rx["route_norm"]   = rx["route"].fillna("").str.strip().str.upper()

    if _ckpt7a_spine is not None:
        log.info("  7a: Resumed from step7a checkpoint — skipping rx loop")
        spine = _ckpt7a_spine
    else:
        # Build per-class key sets (memory-efficient; enables vectorised isin() lookups)
        active_by_class: dict[str, set] = {cls: set() for cls in DRUG_CLASSES}
        iv_by_class:     dict[str, set] = {cls: set() for cls in ATE_DRUGS}
        ab_started_keys: set = set()
        ab_stopped_keys: set = set()
        new_ate_start:   dict[tuple, pd.Timestamp] = {}

        n_rx = len(rx)
        for row_num, (_, r) in enumerate(rx.iterrows()):
            if row_num % 500_000 == 0:
                log.info("  rx loop: %d / %d rows (%.0f%%)", row_num, n_rx, 100 * row_num / n_rx)
            if not r["drug_classes"]:
                continue
            hadm_id = r["hadm_id"]
            is_iv   = r["route_norm"] in IV_ROUTE_KEYWORDS
            dates   = pd.date_range(r["start_date"], r["stop_date"],
                                    freq="D", inclusive="left")
            for d in dates:
                key = (hadm_id, d.date())
                for cls in r["drug_classes"]:
                    active_by_class[cls].add(key)
                if is_iv:
                    for cls in r["drug_classes"]:
                        if cls in ATE_DRUGS:
                            iv_by_class[cls].add(key)

            ate_classes = [c for c in r["drug_classes"] if c in ATE_DRUGS]
            if ate_classes:
                start_key = (hadm_id, r["start_date"].date())
                start_ts  = pd.Timestamp(r["starttime"])
                if start_key not in new_ate_start or start_ts < new_ate_start[start_key]:
                    new_ate_start[start_key] = start_ts

            if "antibiotic" in r["drug_classes"]:
                ab_started_keys.add((hadm_id, r["start_date"].date()))
                ab_stopped_keys.add((hadm_id, r["stop_date"].date()))

        # Vectorised lookups — avoids spine.apply() which OOMs on large spines
        spine_key = pd.Series(
            list(zip(spine["hadm_id"].tolist(), spine["calendar_date"].tolist())),
            index=spine.index, dtype=object,
        )

        for cls in DRUG_CLASSES:
            spine[f"{cls}_active"] = spine_key.isin(active_by_class[cls]).astype("int8")
        del active_by_class

        for cls in ATE_DRUGS:
            spine[f"{cls}_route_iv"] = spine_key.isin(iv_by_class[cls]).astype("int8")
        del iv_by_class

        spine["antibiotic_started"] = spine_key.isin(ab_started_keys).astype("int8")
        spine["antibiotic_stopped"] = spine_key.isin(ab_stopped_keys).astype("int8")
        del ab_started_keys, ab_stopped_keys

        drug_class_cols = [f"{cls}_active" for cls in DRUG_CLASSES]
        spine["n_active_drug_classes"] = spine[drug_class_cols].sum(axis=1)

        # KEY V3 CHANGE: drop rows where new ATE drug start before morning lab
        spine["_new_ate_start_ts"] = spine_key.map(new_ate_start)
        before = len(spine)
        drop_mask = (
            spine["_new_ate_start_ts"].notna()
            & (spine["_new_ate_start_ts"] <= spine["first_lab_charttime"])
        )
        spine = spine[~drop_mask].copy()
        spine.drop(columns=["_new_ate_start_ts"], inplace=True)
        del new_ate_start
        log.info(
            "  Action filter: dropped %d rows (new ATE start before morning lab), kept %d",
            before - len(spine), len(spine),
        )

        # Intraday drug complexity — vectorised, no iterrows
        ate_mask = rx["drug_classes"].apply(
            lambda x: any(c in ATE_DRUGS for c in x)
        )
        rx_ate = rx.loc[ate_mask, ["hadm_id", "starttime", "start_date"]].copy()
        rx_ate["start_date_key"] = rx_ate["start_date"].dt.date
        rx_ate["starttime_ts"]   = pd.to_datetime(rx_ate["starttime"])

        flct_df = spine[["hadm_id", "calendar_date", "first_lab_charttime"]].copy()
        flct_df["start_date_key"] = flct_df["calendar_date"]

        rx_ate_merged = rx_ate.merge(
            flct_df[["hadm_id", "start_date_key", "first_lab_charttime"]],
            on=["hadm_id", "start_date_key"], how="inner",
        )
        rx_ate_merged = rx_ate_merged[
            rx_ate_merged["starttime_ts"] > rx_ate_merged["first_lab_charttime"]
        ].copy()
        rx_ate_merged["start_hour"] = rx_ate_merged["starttime_ts"].dt.floor("h")

        n_hrs = (
            rx_ate_merged.groupby(["hadm_id", "start_date_key"])["start_hour"]
            .nunique()
            .reset_index()
            .rename(columns={"start_hour": "_n_ate_hrs", "start_date_key": "calendar_date"})
        )
        del rx_ate, rx_ate_merged

        spine = spine.merge(n_hrs, on=["hadm_id", "calendar_date"], how="left")
        spine["_n_ate_hrs"] = spine["_n_ate_hrs"].fillna(0).astype(int)
        del n_hrs

        any_act   = int((spine["_n_ate_hrs"] >= 1).sum())
        multi_act = int((spine["_n_ate_hrs"] > 1).sum())
        log.info(
            "  Intraday drug complexity: %d / %d kept rows (%.1f%%) have a new ATE start after morning lab; "
            "%d (%.1f%%) have >1 distinct ATE start hour",
            any_act, len(spine), 100 * any_act / len(spine) if len(spine) else 0,
            multi_act, 100 * multi_act / len(spine) if len(spine) else 0,
        )
        spine.drop(columns=["_n_ate_hrs"], inplace=True)

        if ckpt_dir:
            _save_ckpt(spine, ckpt_dir, "step7a")

    # ------------------------------------------------------------------
    # 7b: Discharge medication flags (unchanged from V2)
    # ------------------------------------------------------------------
    log.info("  7b: Discharge medication flags")
    disch_times = (
        spine[["hadm_id", "dischtime"]].drop_duplicates("hadm_id").copy()
    )
    disch_times["disch_date_norm"] = pd.to_datetime(
        disch_times["dischtime"]
    ).dt.normalize()

    rx_disch = rx.merge(disch_times[["hadm_id", "disch_date_norm"]],
                        on="hadm_id", how="left")
    rx_disch["active_at_disch"] = (
        (rx_disch["start_date"] <= rx_disch["disch_date_norm"]) &
        (rx_disch["stop_date"]  >  rx_disch["disch_date_norm"])
    )
    rx_disch = rx_disch[rx_disch["active_at_disch"]]

    # Vectorised: explode drug_classes → groupby → set per class (no iterrows)
    rx_disch_matched = rx_disch[rx_disch["drug_classes"].apply(len) > 0].copy()
    rx_disch_exploded = rx_disch_matched[["hadm_id", "drug_classes"]].explode("drug_classes")
    disch_by_class: dict[str, set] = {cls: set() for cls in DRUG_CLASSES}
    for cls, grp in rx_disch_exploded.groupby("drug_classes"):
        if cls in disch_by_class:
            disch_by_class[cls] = set(grp["hadm_id"].astype(int).tolist())
    del rx_disch_matched, rx_disch_exploded

    for cls in DRUG_CLASSES:
        spine[f"{cls}_active_at_discharge"] = (
            spine["hadm_id"].isin(disch_by_class[cls]).astype("int8")
        )
    del disch_by_class, rx  # Free 16.8M-row rx DataFrame — no longer needed after 7b

    # ------------------------------------------------------------------
    # 7c: eMAR confirmation (unchanged from V2)
    # ------------------------------------------------------------------
    log.info("  7c: eMAR confirmation")
    if _table_exists(conn, schema_hosp, "emar"):
        hadm_filt_e = _hadm_filter(spine)
        # Pre-filter medications in SQL using regex to avoid pulling ~20M rows
        all_drug_keywords = [
            kw for patterns in DRUG_CLASSES.values() for kw in patterns
        ]
        drug_regex = "|".join(all_drug_keywords)  # case-insensitive via ~*
        emar_sql = f"""
            SELECT hadm_id, DATE(charttime) AS admin_date, medication
            FROM {schema_hosp}.emar
            WHERE hadm_id IS NOT NULL
              AND event_txt IN (
                  'Administered', 'Delayed Administered',
                  'Administered Bolus from IV Drip',
                  'Administered in Other Location', 'Partial Administered'
              )
              AND medication ~* '{drug_regex}'
              {hadm_filt_e}
        """
        emar = pd.read_sql(emar_sql, conn)
        log.info("  eMAR rows (drug-filtered): %d", len(emar))
        emar["hadm_id"]    = emar["hadm_id"].astype("int64")
        emar["admin_date"] = pd.to_datetime(emar["admin_date"]).dt.date
        emar["drug_classes"] = emar["medication"].apply(classify_drug)

        # Vectorised: explode → groupby → set per class (no iterrows)
        emar_matched = emar[emar["drug_classes"].apply(len) > 0].copy()
        emar_exploded = emar_matched[["hadm_id", "admin_date", "drug_classes"]].explode("drug_classes")
        emar_by_class: dict[str, set] = {cls: set() for cls in DRUG_CLASSES}
        for cls, grp in emar_exploded.groupby("drug_classes"):
            if cls in emar_by_class:
                emar_by_class[cls] = set(zip(
                    grp["hadm_id"].tolist(), grp["admin_date"].tolist()
                ))
        del emar, emar_matched, emar_exploded

        spine_key_emar = pd.Series(
            list(zip(spine["hadm_id"].tolist(), spine["calendar_date"].tolist())),
            index=spine.index, dtype=object,
        )
        for cls in DRUG_CLASSES:
            spine[f"{cls}_emar_confirmed"] = (
                spine_key_emar.isin(emar_by_class[cls]).astype("int8")
            )
        del emar_by_class, spine_key_emar
        log.info("  eMAR confirmation flags added")
    else:
        log.warning("  emar table not found — skipping eMAR flags")
        for cls in DRUG_CLASSES:
            spine[f"{cls}_emar_confirmed"] = np.nan

    # ------------------------------------------------------------------
    # 7d: Care intensity transitions (unchanged from V2)
    # ------------------------------------------------------------------
    log.info("  7d: Care intensity transitions")
    xfer_sql = f"""
        SELECT hadm_id, careunit, eventtype, intime, outtime
        FROM {schema_hosp}.transfers
        WHERE hadm_id IS NOT NULL
    """
    xfers = pd.read_sql(xfer_sql, conn)
    xfers["indate"] = pd.to_datetime(xfers["intime"]).dt.normalize()

    discharge_events = (
        xfers[xfers["eventtype"] == "discharge"]
        .groupby(["hadm_id", "indate"]).size().reset_index().rename(columns={0: "_c"})
    )
    discharge_set = set(zip(discharge_events["hadm_id"], discharge_events["indate"]))

    non_discharge = xfers[xfers["eventtype"] != "discharge"].sort_values(
        ["hadm_id", "intime"]
    )
    icu_esc_set:  set = set()
    icu_step_set: set = set()

    for hadm_id, grp in non_discharge.groupby("hadm_id"):
        grp = grp.reset_index(drop=True)
        for i in range(1, len(grp)):
            prev_icu = careunit_to_group(grp.iloc[i - 1]["careunit"]) == "icu"
            curr_icu = careunit_to_group(grp.iloc[i]["careunit"])     == "icu"
            day_key  = (hadm_id, grp.iloc[i]["indate"])
            if curr_icu and not prev_icu:
                icu_esc_set.add(day_key)
            elif not curr_icu and prev_icu:
                icu_step_set.add(day_key)

    cal_ts = pd.to_datetime(spine["calendar_date"]).dt.normalize()
    spine_key_ts = pd.Series(
        list(zip(spine["hadm_id"].tolist(), cal_ts.tolist())),
        index=spine.index, dtype=object,
    )
    spine["icu_escalation"] = spine_key_ts.isin(icu_esc_set).astype("int8")
    spine["icu_stepdown"]   = spine_key_ts.isin(icu_step_set).astype("int8")
    spine["discharged"]     = spine_key_ts.isin(discharge_set).astype("int8")
    del spine_key_ts

    log.info("  Actions assigned")
    return spine


# ===========================================================================
# Step 8: TRIPLET CONSTRUCTION
# ===========================================================================

def step8_triplets(spine: pd.DataFrame) -> pd.DataFrame:
    """Join each valid panel day with its next calendar day to form triplets.

    Both day T and day T+1 must be valid panel days (already filtered by
    steps 3 and 7). The next_* columns carry the outcome state.
    Lab deltas (next_X - X) are pre-computed for convenience.
    """
    log.info("Step 8: Triplet construction")

    # Columns to carry forward as next_state
    next_lab_cols = [c for c in LAB_COLS if c in spine.columns]
    next_extra    = ["nlr", "lactate_elevated", "partial_sofa",
                     "n_labs_today", "first_lab_charttime",
                     "careunit_group", "is_icu",
                     "positive_culture_cumulative",
                     "blood_culture_positive_cumulative"]
    next_state_cols = list(dict.fromkeys(
        next_lab_cols + [c for c in next_extra if c in spine.columns]
    ))

    # Build next-day lookup
    next_df = spine[["hadm_id", "calendar_date"] + next_state_cols].copy()
    next_df = next_df.rename(columns={c: f"next_{c}" for c in next_state_cols})
    next_df["calendar_date"] = pd.to_datetime(
        next_df["calendar_date"]
    ) - pd.Timedelta(days=1)
    next_df["calendar_date"] = next_df["calendar_date"].dt.date

    spine["calendar_date_dt"] = pd.to_datetime(spine["calendar_date"])
    next_df["calendar_date"]  = pd.to_datetime(next_df["calendar_date"])

    triplets = spine.merge(
        next_df,
        left_on=["hadm_id", "calendar_date_dt"],
        right_on=["hadm_id", "calendar_date"],
        how="inner",
        suffixes=("", "_next_drop"),
    )
    # Drop duplicate calendar_date column from right side
    drop_cols = [c for c in triplets.columns if c.endswith("_next_drop")]
    triplets.drop(columns=drop_cols + ["calendar_date_dt"], inplace=True)

    log.info(
        "  Triplets (consecutive valid pairs): %d from %d panel days",
        len(triplets), len(spine),
    )

    # Lab deltas: next_X - X
    for col in next_lab_cols:
        next_col = f"next_{col}"
        if next_col in triplets.columns:
            triplets[f"{col}_delta"] = triplets[next_col] - triplets[col]

    # Partial SOFA delta
    if "next_partial_sofa" in triplets.columns and "partial_sofa" in triplets.columns:
        triplets["partial_sofa_delta"] = (
            triplets["next_partial_sofa"] - triplets["partial_sofa"]
        )

    return triplets.reset_index(drop=True)


# ===========================================================================
# Step 9: LABEL + SPLIT + OUTPUT
# ===========================================================================

def step9_label_split_output(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame,
    charlson_source: str, sample_only: bool = False,
) -> pd.DataFrame:
    log.info("Step 9: Labels, split, output")
    schema_hosp = cfg["schemas"]["hosp"]

    # Readmission label
    readmit_sql = f"""
        SELECT a1.hadm_id,
            CASE WHEN MIN(a2.admittime::timestamp)
                      <= a1.dischtime::timestamp + INTERVAL '30 days'
                 THEN 1 ELSE 0 END AS readmit_30d
        FROM {schema_hosp}.admissions a1
        LEFT JOIN {schema_hosp}.admissions a2
            ON  a1.subject_id = a2.subject_id
            AND a2.admittime::timestamp > a1.dischtime::timestamp
        GROUP BY a1.hadm_id, a1.dischtime
    """
    readmit = pd.read_sql(readmit_sql, conn)
    readmit["hadm_id"] = readmit["hadm_id"].astype("int64")
    spine = spine.merge(readmit, on="hadm_id", how="left")
    spine["readmit_30d"] = spine["readmit_30d"].fillna(0).astype(int)

    # Train/valid/test split
    split_cfg = cfg["split"]
    spine["patient_id"] = spine["subject_id"]
    spine = assign_subject_splits(
        spine,
        train=split_cfg["train"],
        valid=split_cfg["valid"],
        test=split_cfg["test"],
        seed=split_cfg["seed"],
    )
    spine.drop(columns=["patient_id"], inplace=True)

    build_ts = datetime.now(timezone.utc).isoformat()
    spine["schema_version"]      = SCHEMA_VERSION
    spine["build_timestamp_utc"] = build_ts

    # Column ordering
    action_cols      = [f"{d}_active" for d in DRUG_CLASSES.keys()]
    iv_cols          = [f"{d}_route_iv" for d in ATE_DRUGS]
    emar_cols        = [f"{d}_emar_confirmed" for d in list(DRUG_CLASSES.keys())]
    discharge_cols   = [f"{d}_active_at_discharge" for d in list(DRUG_CLASSES.keys())]
    state_lab_cols   = PANEL_COLS
    extra_lab_cols   = ["inr", "bilirubin", "ptt", "albumin", "lactate_elevated",
                        "nlr", "crp", "troponin_t", "troponin_i", "bnp", "alt", "ast"]
    delta_cols       = [f"{c}_delta" for c in LAB_COLS if f"{c}_delta" in spine.columns]
    next_lab_cols    = [f"next_{c}" for c in state_lab_cols + extra_lab_cols if f"next_{c}" in spine.columns]
    next_ctx_cols    = ["next_careunit_group", "next_is_icu",
                        "next_positive_culture_cumulative",
                        "next_blood_culture_positive_cumulative",
                        "next_partial_sofa", "next_n_labs_today",
                        "next_first_lab_charttime"]

    desired_order = (
        ["hadm_id", "subject_id", "day_of_stay", "calendar_date", "is_last_day"]
        + ["admittime", "dischtime", "hospital_expire_flag"]
        + ["age_at_admit", "gender", "admission_type", "admission_location",
           "discharge_location", "insurance", "marital_status", "race", "language", "dod"]
        + ["drg_severity", "drg_mortality", "charlson_score"]
        + ["first_lab_charttime"]
        + ["careunit_group", "is_icu", "days_in_current_unit", "curr_service_group"]
        + state_lab_cols
        + extra_lab_cols
        + ["partial_sofa", "n_labs_today"]
        + ["culture_ordered_today", "positive_culture_cumulative",
           "blood_culture_positive_cumulative"]
        + action_cols
        + iv_cols
        + emar_cols
        + ["antibiotic_started", "antibiotic_stopped"]
        + discharge_cols
        + ["icu_escalation", "icu_stepdown", "discharged", "n_active_drug_classes"]
        + next_lab_cols
        + next_ctx_cols
        + delta_cols
        + ["partial_sofa_delta"]
        + ["readmit_30d", "split", "schema_version", "build_timestamp_utc"]
    )
    present_cols = [c for c in desired_order if c in spine.columns]
    extra_cols   = [c for c in spine.columns if c not in desired_order]
    if extra_cols:
        log.warning("  Unexpected extra columns (dropped): %s", extra_cols)
    spine = spine[present_cols]

    # Write
    project_root  = Path(cfg.get("_project_root", "."))
    out_dir       = project_root / cfg["output"]["dir"]
    out_path      = out_dir / cfg["output"]["filename"]
    sample_path   = out_dir / cfg["output"]["sample_filename"]
    manifest_path = project_root / "reports" / "hosp_daily" / "build_manifest_v3.json"

    if sample_only:
        log.info("  Writing sample: %s", sample_path)
        write_csv(spine, sample_path)
        sample_df = spine
    else:
        log.info("  Writing full dataset: %s", out_path)
        write_csv(spine, out_path)
        sample_cfg = cfg["sample"]
        n_subjects = spine["subject_id"].nunique()
        fraction   = sample_cfg["n_episodes"] / n_subjects if n_subjects > 0 else 1.0
        sample_df  = subject_level_sample(spine, fraction=fraction, seed=sample_cfg["seed"])
        log.info("  Sample: %d rows (%d episodes)",
                 len(sample_df), sample_df["hadm_id"].nunique())
        write_csv(sample_df, sample_path)

    manifest = {
        "schema_version":       SCHEMA_VERSION,
        "build_timestamp_utc":  build_ts,
        "row_count":            len(spine),
        "column_count":         len(spine.columns),
        "columns":              list(spine.columns),
        "hadm_id_count":        int(spine["hadm_id"].nunique()),
        "subject_id_count":     int(spine["subject_id"].nunique()),
        "split_counts":         spine["split"].value_counts().to_dict(),
        "sample_row_count":     len(sample_df),
        "sample_episode_count": int(sample_df["hadm_id"].nunique()),
        "charlson_source":      charlson_source,
        "sample_only_build":    sample_only,
        "output_path":          str(out_path) if not sample_only else None,
        "sample_path":          str(sample_path),
        "panel_cols":           PANEL_COLS,
        "action_cols":          action_cols,
    }
    write_json(manifest, manifest_path)
    log.info("  Manifest written: %s", manifest_path)
    return spine


# ===========================================================================
# Pipeline orchestrator
# ===========================================================================

ALL_STEPS = [1, 2, 3, 4, 5, 6, 7, 8, 9]


def _sample_spine_early(
    spine: pd.DataFrame, n_episodes: int, seed: int
) -> pd.DataFrame:
    n_subjects = spine["subject_id"].nunique()
    if n_subjects == 0:
        return spine
    n_hadm = spine["hadm_id"].nunique()
    eps_per_subj = n_hadm / n_subjects
    target   = int(n_episodes / eps_per_subj) + 1
    fraction = min(target / n_subjects, 1.0)
    sampled  = subject_level_sample(spine, fraction=fraction, seed=seed)
    log.info(
        "  Early sample: %d subjects, %d admissions, %d rows",
        sampled["subject_id"].nunique(),
        sampled["hadm_id"].nunique(),
        len(sampled),
    )
    return sampled


# ===========================================================================
# CHECKPOINT HELPERS
# ===========================================================================

def _ckpt_dir(cfg: dict, sample_only: bool) -> Path:
    root = Path(cfg.get("_project_root", "."))
    n    = cfg.get("sample", {}).get("n_episodes", 500)
    tag  = f"sample_{n}" if sample_only else "full"
    d    = root / "data" / "interim" / "checkpoints" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ckpt(spine: pd.DataFrame, d: Path, name: str) -> None:
    path = d / f"{name}.parquet"
    spine.to_parquet(path, index=False)
    log.info("  [ckpt] Saved %s — %d rows, %d cols → %s", name, len(spine), len(spine.columns), path)


def _load_ckpt(d: Path, name: str) -> "pd.DataFrame | None":
    path = d / f"{name}.parquet"
    if path.exists():
        spine = pd.read_parquet(path)
        log.info("  [ckpt] Loaded %s — %d rows, %d cols ← %s", name, len(spine), len(spine.columns), path)
        return spine
    return None


def _save_meta(d: Path, data: dict) -> None:
    path = d / "meta.json"
    existing = {}
    if path.exists():
        with open(path) as f:
            existing = json.load(f)
    existing.update(data)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)


def _load_meta(d: Path) -> dict:
    path = d / "meta.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def run_pipeline_v3(
    cfg: dict[str, Any],
    steps: list[int] | None = None,
    dry_run: bool = False,
    sample_only: bool = False,
    resume: bool = False,
) -> pd.DataFrame | None:
    """Run the V3 clean triplet build pipeline.

    Args:
        cfg:         Config dict (from YAML).
        steps:       Optional subset of steps (default: all 9).
        dry_run:     Run Step 1 only, log counts, return early.
        sample_only: Down-sample after Step 1 so heavy SQL steps are fast.
    """
    steps = steps or ALL_STEPS
    conn  = _get_conn(cfg)
    log.info(
        "Connected to PostgreSQL %s:%s/%s",
        cfg["db"]["host"], cfg["db"]["port"], cfg["db"]["name"],
    )

    cd = _ckpt_dir(cfg, sample_only)  # checkpoint directory for this run
    charlson_source = "unknown"

    try:
        spine = None

        # ── Step 1: Spine ──────────────────────────────────────────────────
        if 1 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step1")) is not None:
                spine = ckpt
                log.info("Step 1: Resumed from checkpoint")
            else:
                spine = step1_spine(conn, cfg)
                if dry_run:
                    log.info(
                        "Dry-run: %d rows, %d admissions, %d subjects",
                        len(spine), spine["hadm_id"].nunique(),
                        spine["subject_id"].nunique(),
                    )
                    return spine
                if sample_only:
                    sample_cfg = cfg["sample"]
                    spine = _sample_spine_early(
                        spine,
                        n_episodes=sample_cfg["n_episodes"],
                        seed=sample_cfg["seed"],
                    )
                _save_ckpt(spine, cd, "step1")

        # ── Step 2: Static features ────────────────────────────────────────
        if 2 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step2")) is not None:
                spine = ckpt
                charlson_source = _load_meta(cd).get("charlson_source", "unknown")
                log.info("Step 2: Resumed from checkpoint")
            else:
                spine, charlson_source = step2_static(conn, cfg, spine)
                _save_ckpt(spine, cd, "step2")
                _save_meta(cd, {"charlson_source": charlson_source})

        # ── Step 3: Labs ───────────────────────────────────────────────────
        if 3 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step3")) is not None:
                spine = ckpt
                log.info("Step 3: Resumed from checkpoint")
            else:
                spine = step3_labs(conn, cfg, spine)
                _save_ckpt(spine, cd, "step3")

        # ── Step 4: Location state ─────────────────────────────────────────
        if 4 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step4")) is not None:
                spine = ckpt
                log.info("Step 4: Resumed from checkpoint")
            else:
                spine = step4_location(conn, cfg, spine)
                _save_ckpt(spine, cd, "step4")

        # ── Step 5: Service state ──────────────────────────────────────────
        if 5 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step5")) is not None:
                spine = ckpt
                log.info("Step 5: Resumed from checkpoint")
            else:
                spine = step5_service(conn, cfg, spine)
                _save_ckpt(spine, cd, "step5")

        # ── Step 6: Infection state ────────────────────────────────────────
        if 6 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step6")) is not None:
                spine = ckpt
                log.info("Step 6: Resumed from checkpoint")
            else:
                spine = step6_infection(conn, cfg, spine)
                _save_ckpt(spine, cd, "step6")

        # ── Step 7: Actions ────────────────────────────────────────────────
        if 7 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step7")) is not None:
                spine = ckpt
                log.info("Step 7: Resumed from checkpoint")
            else:
                spine = step7_actions(conn, cfg, spine, ckpt_dir=cd, resume=resume)
                _save_ckpt(spine, cd, "step7")

        # ── Step 8: Triplets ───────────────────────────────────────────────
        if 8 in steps:
            spine = step8_triplets(spine)

        # ── Step 9: Labels, split, output ─────────────────────────────────
        if 9 in steps:
            spine = step9_label_split_output(
                conn, cfg, spine, charlson_source,
                sample_only=sample_only,
            )

        log.info("Pipeline V3 complete.")
        return spine
    finally:
        conn.close()
