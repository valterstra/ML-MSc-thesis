"""Hospital-stay daily transition dataset v2 -- extended pipeline.

V2 additions over v1 (build.py):
  States:
    - Additional labs: CRP, troponin T/I, BNP, PTT, ALT, AST
    - Lab deltas: day-over-day change for all continuous labs
    - Partial SOFA score (renal + hepatic + coagulation, from labs)
    - Partial SOFA delta
    - Drug route context: {drug_class}_route_iv per ATE drug
    - Extended demographics: insurance, race, marital_status, language
    - Date of death (dod) from patients table

  Actions / confirmation:
    - eMAR administration confirmation: {drug_class}_emar_confirmed
      (actual administration recorded, not just prescription order)

  Post-discharge features (admission-level, on every row like readmit_30d):
    - discharge_location (where patient went after discharge)
    - {drug_class}_active_at_discharge (prescription open at dischtime)

  Intermediate reward signals (columns available for RL reward shaping):
    - icu_escalation          (already in v1)
    - hospital_expire_flag    (already in v1)
    - day_of_stay             (LOS signal, already in v1)
    - lab deltas              (new -- deterioration signal)
    - partial_sofa / partial_sofa_delta (new -- organ failure signal)

Schema version: hosp_daily_v2
"""

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

SCHEMA_VERSION = "hosp_daily_v2"

# ---------------------------------------------------------------------------
# itemid -> column name (v1 preserved + v2 additions)
# ---------------------------------------------------------------------------
ITEMID_TO_COL: dict[int, str] = {
    # V1 labs (unchanged)
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
    # V2 additions
    50889: "crp",          # C-Reactive Protein (inflammation marker)
    51003: "troponin_t",   # Troponin T (cardiac injury)
    51002: "troponin_i",   # Troponin I (cardiac injury, alternative assay)
    50963: "bnp",          # BNP/NT-proBNP (heart failure severity)
    51275: "ptt",          # PTT (coagulation, anticoagulant monitoring)
    50861: "alt",          # ALT (liver function / hepatotoxicity)
    50878: "ast",          # AST (liver function)
}

# ---------------------------------------------------------------------------
# Forward-fill tiers (v1 preserved + v2 added)
# ---------------------------------------------------------------------------
TIER1_COLS = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "glucose", "hemoglobin", "wbc", "platelets", "magnesium", "calcium",
    "phosphate",
]
TIER2_COLS = ["inr", "bilirubin", "nlr"]
TIER3_ALBUMIN = ["albumin"]

# V2 tiers
TIER_V2_FAST    = ["crp", "ptt"]                          # 2-day fill
TIER_V2_CARDIAC = ["troponin_t", "troponin_i", "bnp"]     # 3-day fill
TIER_V2_LFT     = ["alt", "ast"]                          # 3-day fill

# All continuous lab columns to compute day-over-day deltas for
LAB_DELTA_COLS = (
    TIER1_COLS + TIER2_COLS + TIER3_ALBUMIN
    + TIER_V2_FAST + TIER_V2_CARDIAC + TIER_V2_LFT
)

# Drug classes that get IV route context flag (opioid excluded from RL space)
ATE_DRUGS = ["antibiotic", "anticoagulant", "diuretic", "steroid", "insulin"]

# Route values treated as intravenous
IV_ROUTE_KEYWORDS = {
    "IV", "IV DRIP", "IV BOLUS", "IVPB", "IV PUSH",
    "IV CONT", "IVPB DRIP", "IV LOCK", "IV INFUSION",
}

# eMAR event_txt values that confirm actual administration
EMAR_ADMINISTERED = {
    "Administered",
    "Delayed Administered",
    "Administered Bolus from IV Drip",
    "Administered in Other Location",
    "Partial Administered",
}

# ---------------------------------------------------------------------------
# Partial SOFA score (3 lab-computable components)
# ---------------------------------------------------------------------------

def compute_partial_sofa(df: pd.DataFrame) -> pd.Series:
    """Compute 3-component lab-based SOFA (renal + hepatic + coagulation).

    Returns a float Series. NaN when all three components are unmeasured.
    Where a component is NaN (lab not available), it contributes 0 to the sum
    but only if at least one other component is available (min_count=1).

    Renal    (creatinine mg/dL): 0/<1.2, 1/1.2-2.0, 2/2.0-3.5, 3/3.5-5.0, 4/>=5.0
    Hepatic  (bilirubin  mg/dL): 0/<1.2, 1/1.2-2.0, 2/2.0-6.0, 3/6.0-12.0, 4/>=12.0
    Coagulation (platelets K/uL): 0/>=150, 1/100-149, 2/50-99, 3/20-49, 4/<20
    """
    parts: list[pd.Series] = []

    if "creatinine" in df.columns:
        cr = df["creatinine"]
        renal = np.select(
            [cr.isna(), cr < 1.2, cr < 2.0, cr < 3.5, cr < 5.0],
            [np.nan,    0,        1,        2,        3],
            default=4,
        )
        parts.append(pd.Series(renal, index=df.index, dtype="float64"))

    if "bilirubin" in df.columns:
        bili = df["bilirubin"]
        hepatic = np.select(
            [bili.isna(), bili < 1.2, bili < 2.0, bili < 6.0, bili < 12.0],
            [np.nan,      0,          1,          2,           3],
            default=4,
        )
        parts.append(pd.Series(hepatic, index=df.index, dtype="float64"))

    if "platelets" in df.columns:
        plt = df["platelets"]
        coag = np.select(
            [plt.isna(), plt >= 150, plt >= 100, plt >= 50, plt >= 20],
            [np.nan,     0,          1,          2,         3],
            default=4,
        )
        parts.append(pd.Series(coag, index=df.index, dtype="float64"))

    if not parts:
        return pd.Series(np.nan, index=df.index, dtype="float64")

    stacked = pd.concat(parts, axis=1)
    # min_count=1: returns NaN only when ALL components are NaN
    return stacked.sum(axis=1, min_count=1)


# ---------------------------------------------------------------------------
# ICU unit set + careunit grouping (unchanged from v1)
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
# Service grouping (unchanged from v1)
# ---------------------------------------------------------------------------
SVC_MAP: dict[str, set[str]] = {
    "medicine":   {"MED", "CMED", "NMED", "OMED", "GU"},
    "surgery":    {"SURG", "NSURG", "CSURG", "VSURG", "TSURG", "PSURG",
                   "ORTHO", "ENT", "EYE"},
    "icu_svc":    {"MICU", "SICU", "TSICU", "CSRU", "CCU"},
    "psychiatry": {"PSYCH"},
    "obstetrics": {"OBS", "GYN"},
    "trauma":     {"TRAUM"},
}
_SVC_LOOKUP: dict[str, str] = {
    code: group for group, codes in SVC_MAP.items() for code in codes
}


def service_to_group(svc: str | None) -> str:
    if svc is None:
        return "other"
    return _SVC_LOOKUP.get(svc, "other")


# ---------------------------------------------------------------------------
# Charlson ICD-10 fallback (unchanged from v1)
# ---------------------------------------------------------------------------
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

_CHARLSON_LOOKUP: dict[str, tuple[str, int]] = {
    pfx: (cat, w)
    for prefixes, cat, w in _CHARLSON_PREFIXES
    for pfx in prefixes
}


def _compute_charlson_from_icd(conn: Any, schema_hosp: str) -> pd.DataFrame:
    """Fallback: compute Charlson score from diagnoses_icd."""
    log.info("  Computing Charlson score from diagnoses_icd (fallback)")
    sql = f"SELECT hadm_id, icd_code, icd_version FROM {schema_hosp}.diagnoses_icd"
    diag = pd.read_sql(sql, conn)
    diag = diag[diag["icd_version"] == 10].copy()
    diag["icd_code"] = diag["icd_code"].str.strip()

    records: list[dict[str, Any]] = []
    for hadm_id, grp in diag.groupby("hadm_id"):
        found: dict[str, int] = {}
        for code in grp["icd_code"]:
            for length in range(len(code), 2, -1):
                prefix = code[:length]
                if prefix in _CHARLSON_LOOKUP:
                    cat, weight = _CHARLSON_LOOKUP[prefix]
                    if cat not in found or weight > found[cat]:
                        found[cat] = weight
                    break
        records.append({"hadm_id": hadm_id, "charlson_score": sum(found.values())})

    return pd.DataFrame(records) if records else pd.DataFrame(
        columns=["hadm_id", "charlson_score"]
    )


# ---------------------------------------------------------------------------
# DB helpers (unchanged from v1)
# ---------------------------------------------------------------------------

def _hadm_filter(spine: pd.DataFrame, alias: str = "") -> str:
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


# ===========================================================================
# Step 1: SPINE (v2)
# ===========================================================================

def step1_spine(conn, cfg):
    log.info("Step 1: Building spine (v2)")
    schema_hosp = cfg["schemas"]["hosp"]
    min_age = cfg["cohort"]["min_age"]

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
    log.info("  Admissions loaded: %d", len(adm))

    adm["admit_date"] = pd.to_datetime(adm["admittime"]).dt.normalize()
    adm["disch_date"] = pd.to_datetime(adm["dischtime"]).dt.normalize()
    adm = adm[adm["disch_date"] > adm["admit_date"]].copy()
    log.info("  After removing same-day stays: %d admissions", len(adm))

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
# Step 2: STATIC FEATURES (v2)
# ===========================================================================

def step2_static(conn, cfg, spine):
    log.info("Step 2: Static features (v2)")
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


# ===========================================================================
# Step 3: LOCATION STATE (unchanged from v1)
# ===========================================================================

def step3_location(conn, cfg, spine):
    log.info("Step 3: Location state")
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
            if not before.empty:
                best = before.iloc[-1]
            else:
                careunit_groups.append("other"); is_icu_flags.append(0); days_in_unit.append(0)
                continue
        else:
            best = matched.iloc[-1]
        unit     = best["careunit"]
        grp_name = careunit_to_group(unit)
        careunit_groups.append(grp_name)
        is_icu_flags.append(int(grp_name == "icu"))
        days_in_unit.append((day_d - best["indate"]).days)

    spine["careunit_group"]       = careunit_groups
    spine["is_icu"]               = is_icu_flags
    spine["days_in_current_unit"] = days_in_unit
    log.info("  Location state assigned")
    return spine


# ===========================================================================
# Step 4: SERVICE STATE (unchanged from v1)
# ===========================================================================

def step4_service(conn, cfg, spine):
    log.info("Step 4: Service state")
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
            svc_results.append("other")
            continue
        before = grp[grp["svc_date"] <= day_d]
        svc_results.append(
            service_to_group(before.iloc[-1]["curr_service"])
            if not before.empty else "other"
        )

    spine["curr_service_group"] = svc_results
    log.info("  Service state assigned")
    return spine


# ===========================================================================
# Step 5: LAB STATE (v2 -- extended labs + deltas + partial SOFA)
# ===========================================================================

def step5_labs(conn, cfg, spine):
    """Query labevents (v2 itemids), pivot, forward-fill, compute deltas + SOFA."""
    log.info("Step 5: Lab state (v2)")
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
    labs_raw["itemid"]  = labs_raw["itemid"].astype("int64")
    log.info("  Lab rows (aggregated): %d", len(labs_raw))

    labs_raw["col_name"] = labs_raw["itemid"].map(ITEMID_TO_COL)

    # Count distinct lab types per (hadm_id, lab_date)
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

    # Compute NLR (neutrophils / lymphocytes)
    if "neutrophils" in labs_wide.columns and "lymphocytes" in labs_wide.columns:
        labs_wide["nlr"] = (
            labs_wide["neutrophils"] / labs_wide["lymphocytes"].replace(0, np.nan)
        ).clip(upper=50)
        labs_wide.drop(columns=["neutrophils", "lymphocytes"], inplace=True)
    else:
        labs_wide["nlr"] = np.nan

    # Compute lactate_elevated flag
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
    labs_wide["lab_date"]     = pd.to_datetime(labs_wide["lab_date"])
    spine = spine.merge(
        labs_wide,
        left_on=["hadm_id", "calendar_date_dt"],
        right_on=["hadm_id", "lab_date"],
        how="left",
    )
    spine.drop(columns=["lab_date", "calendar_date_dt"], inplace=True)

    # -------------------------------------------------------------------
    # Forward-fill within each hadm_id (v1 tiers preserved + v2 added)
    # -------------------------------------------------------------------
    fill_limits = cfg.get("lab_fill_limits", {})
    tier1_limit   = fill_limits.get("tier1",   2)
    tier2_limit   = fill_limits.get("tier2",   3)
    albumin_limit = fill_limits.get("albumin", 5)
    v2_fast_limit    = fill_limits.get("v2_fast",    2)
    v2_cardiac_limit = fill_limits.get("v2_cardiac", 3)
    v2_lft_limit     = fill_limits.get("v2_lft",     3)

    spine.sort_values(["hadm_id", "day_of_stay"], inplace=True)

    for cols, limit in [
        (TIER1_COLS,      tier1_limit),
        (TIER2_COLS,      tier2_limit),
        (TIER3_ALBUMIN,   albumin_limit),
        (TIER_V2_FAST,    v2_fast_limit),
        (TIER_V2_CARDIAC, v2_cardiac_limit),
        (TIER_V2_LFT,     v2_lft_limit),
    ]:
        present = [c for c in cols if c in spine.columns]
        if present:
            spine[present] = spine.groupby("hadm_id")[present].ffill(limit=limit)

    # lactate_elevated: no forward-fill, NaN -> 0
    if "lactate_elevated" in spine.columns:
        spine["lactate_elevated"] = spine["lactate_elevated"].fillna(0).astype(int)

    # n_labs_today: NaN -> 0
    if "n_labs_today" in spine.columns:
        spine["n_labs_today"] = spine["n_labs_today"].fillna(0).astype(int)

    # -------------------------------------------------------------------
    # Lab deltas (day-over-day change within each admission)
    # Computed on forward-filled values so that silent days get delta=0.
    # -------------------------------------------------------------------
    log.info("  Computing lab deltas")
    for col in LAB_DELTA_COLS:
        if col in spine.columns:
            spine[f"{col}_delta"] = spine.groupby("hadm_id")[col].diff()

    # -------------------------------------------------------------------
    # Partial SOFA score (3-component: renal + hepatic + coagulation)
    # -------------------------------------------------------------------
    log.info("  Computing partial SOFA score")
    spine["partial_sofa"] = compute_partial_sofa(spine)
    spine["partial_sofa_delta"] = spine.groupby("hadm_id")["partial_sofa"].diff()

    log.info("  Lab state merged, forward-filled, deltas and SOFA computed")
    return spine


# ===========================================================================
# Step 6: INFECTION STATE (unchanged from v1)
# ===========================================================================

def step6_infection(conn, cfg, spine):
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
# Step 7: ACTIONS (v2 -- drugs + route context + eMAR confirmation
#                        + discharge medication flags + care transitions)
# ===========================================================================

def step7_actions(conn, cfg, spine):
    """Drug flags, route context, eMAR confirmation, discharge meds, transitions."""
    log.info("Step 7: Actions (v2)")
    schema_hosp = cfg["schemas"]["hosp"]

    # ------------------------------------------------------------------
    # 7a: Prescriptions -- binary active flags + route IV context
    # ------------------------------------------------------------------
    log.info("  7a: Prescriptions")
    hadm_filt = _hadm_filter(spine)
    rx_sql = f"""
        SELECT hadm_id, drug, starttime, stoptime,
               dose_val_rx, dose_unit_rx, route
        FROM {schema_hosp}.prescriptions
        WHERE hadm_id IS NOT NULL
          AND drug_type = 'MAIN'
          AND starttime IS NOT NULL
          {hadm_filt}
    """
    rx = pd.read_sql(rx_sql, conn)
    log.info("  Prescription rows: %d", len(rx))

    rx["start_date"] = pd.to_datetime(rx["starttime"]).dt.normalize()
    rx["stop_date"]  = pd.to_datetime(rx["stoptime"]).dt.normalize()
    rx["stop_date"]  = rx["stop_date"].fillna(rx["start_date"] + pd.Timedelta(days=1))

    compiled_patterns = {
        cls: [re.compile(p, re.IGNORECASE) for p in patterns]
        for cls, patterns in DRUG_CLASSES.items()
    }

    def classify_drug(drug_name):
        if not drug_name or not isinstance(drug_name, str):
            return []
        classes = []
        for cls, patterns in compiled_patterns.items():
            if any(p.search(drug_name) for p in patterns):
                classes.append(cls)
        return classes

    rx["drug_classes"] = rx["drug"].apply(classify_drug)

    # Normalise route to uppercase stripped string
    rx["route_norm"] = rx["route"].fillna("").str.strip().str.upper()

    # Build lookup dicts:
    #   active_classes:  (hadm_id, date) -> set of active drug classes
    #   iv_classes:      (hadm_id, date) -> set of drug classes given IV
    #   ab_started/stopped: (hadm_id, date) -> 1
    active_classes: dict[tuple, set] = {}
    iv_classes:     dict[tuple, set] = {}
    ab_started: dict[tuple, int] = {}
    ab_stopped: dict[tuple, int] = {}

    for _, r in rx.iterrows():
        if not r["drug_classes"]:
            continue
        hadm_id   = r["hadm_id"]
        is_iv     = r["route_norm"] in IV_ROUTE_KEYWORDS
        dates     = pd.date_range(r["start_date"], r["stop_date"],
                                  freq="D", inclusive="left")
        for d in dates:
            key = (hadm_id, d.date())
            active_classes.setdefault(key, set()).update(r["drug_classes"])
            if is_iv:
                iv_classes.setdefault(key, set()).update(r["drug_classes"])

        if "antibiotic" in r["drug_classes"]:
            ab_started[(hadm_id, r["start_date"].date())] = 1
            ab_stopped[(hadm_id, r["stop_date"].date())]  = 1

    # Map binary active flags
    for cls in DRUG_CLASSES:
        col = f"{cls}_active"
        spine[col] = spine.apply(
            lambda row: int(cls in active_classes.get(
                (row["hadm_id"], row["calendar_date"]), set()
            )),
            axis=1,
        )

    # Map IV route context flags (ATE drugs only)
    for cls in ATE_DRUGS:
        col = f"{cls}_route_iv"
        spine[col] = spine.apply(
            lambda row: int(cls in iv_classes.get(
                (row["hadm_id"], row["calendar_date"]), set()
            )),
            axis=1,
        )

    spine["antibiotic_started"] = spine.apply(
        lambda row: ab_started.get((row["hadm_id"], row["calendar_date"]), 0), axis=1
    )
    spine["antibiotic_stopped"] = spine.apply(
        lambda row: ab_stopped.get((row["hadm_id"], row["calendar_date"]), 0), axis=1
    )

    drug_class_cols = [f"{cls}_active" for cls in DRUG_CLASSES]
    spine["n_active_drug_classes"] = spine[drug_class_cols].sum(axis=1)

    # ------------------------------------------------------------------
    # 7b: Discharge medication flags
    #   Which drug classes had an active inpatient prescription at dischtime?
    #   Stored on every row (admission-level, like readmit_30d).
    #   Note: this is a proxy for take-home medications, not a confirmed fill.
    # ------------------------------------------------------------------
    log.info("  7b: Discharge medication flags")
    disch_times = (
        spine[["hadm_id", "dischtime"]]
        .drop_duplicates("hadm_id")
        .copy()
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

    disch_drug_map: dict[tuple, int] = {}
    for _, r in rx_disch.iterrows():
        for cls in r["drug_classes"]:
            disch_drug_map[(int(r["hadm_id"]), cls)] = 1

    for cls in DRUG_CLASSES:
        col = f"{cls}_active_at_discharge"
        spine[col] = spine["hadm_id"].apply(
            lambda h: disch_drug_map.get((int(h), cls), 0)
        )

    # ------------------------------------------------------------------
    # 7c: eMAR confirmation flags
    #   For each (hadm_id, calendar_date, drug_class), was at least one
    #   administration actually recorded in eMAR?
    #   Falls back gracefully if emar table does not exist.
    # ------------------------------------------------------------------
    log.info("  7c: eMAR confirmation")
    if _table_exists(conn, schema_hosp, "emar"):
        hadm_filt_e = _hadm_filter(spine)
        emar_sql = f"""
            SELECT hadm_id, DATE(charttime) AS admin_date, medication
            FROM {schema_hosp}.emar
            WHERE hadm_id IS NOT NULL
              AND event_txt IN (
                  'Administered',
                  'Delayed Administered',
                  'Administered Bolus from IV Drip',
                  'Administered in Other Location',
                  'Partial Administered'
              )
              {hadm_filt_e}
        """
        emar = pd.read_sql(emar_sql, conn)
        emar["hadm_id"]    = emar["hadm_id"].astype("int64")
        emar["admin_date"] = pd.to_datetime(emar["admin_date"]).dt.date
        log.info("  eMAR administered rows: %d", len(emar))

        emar["drug_classes"] = emar["medication"].apply(classify_drug)

        emar_confirmed: dict[tuple, set] = {}
        for _, r in emar.iterrows():
            if not r["drug_classes"]:
                continue
            key = (r["hadm_id"], r["admin_date"])
            emar_confirmed.setdefault(key, set()).update(r["drug_classes"])

        for cls in DRUG_CLASSES:
            col = f"{cls}_emar_confirmed"
            spine[col] = spine.apply(
                lambda row: int(cls in emar_confirmed.get(
                    (row["hadm_id"], row["calendar_date"]), set()
                )),
                axis=1,
            )
        log.info("  eMAR confirmation flags added")
    else:
        log.warning("  emar table not found -- skipping eMAR confirmation flags")
        for cls in DRUG_CLASSES:
            spine[f"{cls}_emar_confirmed"] = np.nan

    # ------------------------------------------------------------------
    # 7d: Care intensity transitions from transfers (unchanged from v1)
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

    log.info("  Actions (v2) assigned")
    return spine


# ===========================================================================
# Step 8: LABEL + SPLIT + OUTPUT (v2)
# ===========================================================================

def step8_label_split_output(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame, charlson_source: str,
    sample_only: bool = False,
) -> pd.DataFrame:
    """Readmit_30d label, train/valid/test split, write CSV + manifest."""
    log.info("Step 8: Label, split, output (v2)")
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

    # --- Column ordering (v2 -- all new columns included) ---
    desired_order = [
        # Episode identifiers
        "hadm_id", "subject_id", "day_of_stay", "calendar_date", "is_last_day",
        # Episode frame (static)
        "admittime", "dischtime", "hospital_expire_flag",
        "age_at_admit", "gender",
        "admission_type", "admission_location", "discharge_location",
        "insurance", "marital_status", "race", "language", "dod",
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
        # Labs V2
        "crp", "ptt", "troponin_t", "troponin_i", "bnp", "alt", "ast",
        # Lab meta
        "n_labs_today",
        # Lab deltas (Tier 1)
        "creatinine_delta", "bun_delta", "sodium_delta", "potassium_delta",
        "bicarbonate_delta", "anion_gap_delta", "glucose_delta", "hemoglobin_delta",
        "wbc_delta", "platelets_delta", "magnesium_delta", "calcium_delta",
        "phosphate_delta",
        # Lab deltas (Tier 2)
        "inr_delta", "bilirubin_delta", "nlr_delta",
        # Lab deltas (Tier 3)
        "albumin_delta",
        # Lab deltas (V2)
        "crp_delta", "ptt_delta",
        "troponin_t_delta", "troponin_i_delta", "bnp_delta",
        "alt_delta", "ast_delta",
        # Partial SOFA
        "partial_sofa", "partial_sofa_delta",
        # Infection
        "culture_ordered_today", "positive_culture_cumulative",
        "blood_culture_positive_cumulative",
        # Actions -- binary active flags (all 6 classes)
        "antibiotic_active", "anticoagulant_active", "diuretic_active",
        "steroid_active", "insulin_active", "opioid_active",
        # Actions -- IV route context (ATE drugs only)
        "antibiotic_route_iv", "anticoagulant_route_iv", "diuretic_route_iv",
        "steroid_route_iv", "insulin_route_iv",
        # Actions -- eMAR confirmation (all 6 classes)
        "antibiotic_emar_confirmed", "anticoagulant_emar_confirmed",
        "diuretic_emar_confirmed", "steroid_emar_confirmed",
        "insulin_emar_confirmed", "opioid_emar_confirmed",
        # Antibiotic start/stop events
        "antibiotic_started", "antibiotic_stopped",
        # Discharge medications (all 6 classes -- admission-level)
        "antibiotic_active_at_discharge", "anticoagulant_active_at_discharge",
        "diuretic_active_at_discharge", "steroid_active_at_discharge",
        "insulin_active_at_discharge", "opioid_active_at_discharge",
        # Care transitions
        "icu_escalation", "icu_stepdown", "discharged",
        "n_active_drug_classes",
        # Label + split + metadata
        "readmit_30d", "split", "schema_version", "build_timestamp_utc",
    ]
    present_cols = [c for c in desired_order if c in spine.columns]
    extra_cols   = [c for c in spine.columns if c not in desired_order]
    if extra_cols:
        log.warning("  Unexpected extra columns (will be dropped): %s", extra_cols)
    spine = spine[present_cols]

    # --- Write outputs ---
    project_root = Path(cfg.get("_project_root", "."))
    out_dir      = project_root / cfg["output"]["dir"]
    out_path     = out_dir / cfg["output"]["filename"]
    sample_path  = out_dir / cfg["output"]["sample_filename"]
    manifest_path = project_root / "reports" / "hosp_daily" / "build_manifest_v2.json"

    if sample_only:
        log.info("  Writing sample dataset: %s", sample_path)
        write_csv(spine, sample_path)
        sample_df = spine
    else:
        log.info("  Writing full dataset: %s", out_path)
        write_csv(spine, out_path)

        sample_cfg = cfg["sample"]
        n_subjects = spine["subject_id"].nunique()
        fraction   = sample_cfg["n_episodes"] / n_subjects if n_subjects > 0 else 1.0
        sample_df  = subject_level_sample(spine, fraction=fraction, seed=sample_cfg["seed"])
        log.info(
            "  Sample: %d rows (%d episodes)",
            len(sample_df),
            sample_df["hadm_id"].nunique(),
        )
        write_csv(sample_df, sample_path)

    # Manifest
    manifest = {
        "schema_version":      SCHEMA_VERSION,
        "build_timestamp_utc": build_ts,
        "row_count":           len(spine),
        "column_count":        len(spine.columns),
        "columns":             list(spine.columns),
        "hadm_id_count":       int(spine["hadm_id"].nunique()),
        "subject_id_count":    int(spine["subject_id"].nunique()),
        "split_counts":        spine["split"].value_counts().to_dict(),
        "sample_row_count":    len(sample_df),
        "sample_episode_count": int(sample_df["hadm_id"].nunique()),
        "charlson_source":     charlson_source,
        "sample_only_build":   sample_only,
        "output_path":         str(out_path) if not sample_only else None,
        "sample_path":         str(sample_path),
    }
    write_json(manifest, manifest_path)
    log.info("  Manifest written: %s", manifest_path)

    return spine


# ===========================================================================
# Pipeline orchestrator (v2)
# ===========================================================================

ALL_STEPS = [1, 2, 3, 4, 5, 6, 7, 8]


def _sample_spine_early(
    spine: pd.DataFrame, n_episodes: int, seed: int
) -> pd.DataFrame:
    """Down-sample spine to ~n_episodes admissions (by subject) for fast builds."""
    n_subjects = spine["subject_id"].nunique()
    if n_subjects == 0:
        return spine
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


def run_pipeline_v2(
    cfg: dict[str, Any],
    steps: list[int] | None = None,
    dry_run: bool = False,
    sample_only: bool = False,
) -> pd.DataFrame | None:
    """Run the full (or partial) v2 build pipeline.

    Args:
        cfg:         Config dict (from YAML).
        steps:       Optional subset of steps to run (default: all 8).
        dry_run:     Run Step 1 only, log counts, return early.
        sample_only: Down-sample to ~cfg[sample][n_episodes] right after
                     Step 1 so all subsequent steps run much faster.
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

        log.info("Pipeline v2 complete.")
        return spine
    finally:
        conn.close()
