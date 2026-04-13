"""Admission-level 30-day readmission dataset — V4.

One row per hospital admission (~545k after exclusions).
Goal: causal discovery of large discharge-time actions that reduce 30-day readmission.

Exclusions applied in Step 1:
  - hospital_expire_flag = 1  (in-hospital death)
  - obstetrics/gynecology service  (OBS, GYN — distinct readmission mechanisms)
  - post-discharge death within 30 days  (competing risk)

Feature groups (~85 columns total):
  Admission context:   age, sex, LOS, admit_type, weekday, ED dwell, insurance,
                       language, marital_status, race
  Severity:            DRG severity/mortality, Charlson score (ICD-9 + ICD-10)
                       first_service, discharge_service
  Chronic flags:       18 conditions, ICD-9 and ICD-10 dual mapping
  ICU / transfers:     had_icu, icu_days, n_transfers, n_services, discharge_location
  Labs at discharge:   last value within 48h before dischtime (11 labs)
                       + n_abnormal_labs
  Microbiology:        positive_culture, blood_culture_positive, resistant_organism
  Drug exposure:       5 in-hospital drug class flags (any prescription during stay)
  Consult actions:     8 consult/discharge-planning flags (from POE)
  Discharge meds:      9 drug class flags at discharge + n_dc_medications

Outcome:
  readmit_30d:         1 if any readmission within 30 days of discharge, else 0
  split:               train / valid / test (70/15/15 by subject_id)

Schema version: readmission_v4
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

from careai.io.write_outputs import write_csv, write_json
from careai.transitions.split import assign_subject_splits

log = logging.getLogger(__name__)

SCHEMA_VERSION = "readmission_v4"

# ---------------------------------------------------------------------------
# Lab itemids (11 labs pulled at discharge)
# ---------------------------------------------------------------------------
DISCHARGE_LAB_ITEMIDS: dict[int, str] = {
    50912: "creatinine",
    51006: "bun",
    50983: "sodium",
    50971: "potassium",
    50882: "bicarbonate",
    51222: "hemoglobin",
    51301: "wbc",
    51265: "platelets",
    50868: "anion_gap",
    50893: "calcium",
    50862: "albumin",
}

DISCHARGE_LAB_COLS: list[str] = list(DISCHARGE_LAB_ITEMIDS.values())

# Normal ranges for abnormal-lab count (inclusive normal interval)
# Values outside these ranges count as abnormal.
LAB_NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "creatinine":  (0.5, 1.2),
    "bun":         (7.0, 20.0),
    "sodium":      (136.0, 145.0),
    "potassium":   (3.5, 5.0),
    "bicarbonate": (22.0, 28.0),
    "hemoglobin":  (11.5, 17.5),
    "wbc":         (4.5, 11.0),
    "platelets":   (150.0, 400.0),
    "anion_gap":   (3.0, 12.0),
    "calcium":     (8.4, 10.2),
    "albumin":     (3.5, 5.0),
}

# ---------------------------------------------------------------------------
# Drug regex patterns — in-hospital exposure (prescriptions table)
# ---------------------------------------------------------------------------
_RX_PATTERNS: dict[str, re.Pattern] = {
    "antibiotic": re.compile(
        r"amoxicillin|ampicillin|azithromycin|aztreonam|cefazolin|cefepime|"
        r"ceftriaxone|ceftazidime|cefuroxime|cephalexin|ciprofloxacin|"
        r"clindamycin|daptomycin|doxycycline|ertapenem|erythromycin|"
        r"fluconazole|gentamicin|imipenem|levofloxacin|linezolid|"
        r"meropenem|metronidazole|minocycline|moxifloxacin|nafcillin|"
        r"nitrofurantoin|oxacillin|penicillin|pip.?tazo|piperacillin|"
        r"rifampin|sulfamethoxazole|tetracycline|tobramycin|trimethoprim|"
        r"vancomycin|voriconazole|micafungin|caspofungin|anidulafungin|"
        r"fluconazole|itraconazole|posaconazole",
        re.IGNORECASE,
    ),
    "anticoagulant": re.compile(
        r"heparin|enoxaparin|dalteparin|fondaparinux|argatroban|bivalirudin|"
        r"warfarin|rivaroxaban|apixaban|dabigatran|edoxaban|lovenox",
        re.IGNORECASE,
    ),
    "diuretic": re.compile(
        r"furosemide|torsemide|bumetanide|ethacrynic|spironolactone|"
        r"eplerenone|hydrochlorothiazide|chlorthalidone|metolazone|"
        r"acetazolamide|mannitol|amiloride|triamterene",
        re.IGNORECASE,
    ),
    "steroid": re.compile(
        r"methylprednisolone|prednisolone|prednisone|dexamethasone|"
        r"hydrocortisone|fludrocortisone|budesonide|triamcinolone|"
        r"betamethasone|cortisone",
        re.IGNORECASE,
    ),
    "insulin": re.compile(r"insulin", re.IGNORECASE),
}

# ---------------------------------------------------------------------------
# Drug regex patterns — discharge medications (pharmacy table)
# Discharge meds use same first 5 + 4 new classes
# ---------------------------------------------------------------------------
_DC_MED_PATTERNS: dict[str, re.Pattern] = {
    **_RX_PATTERNS,  # reuse antibiotic, anticoagulant, diuretic, steroid, insulin
    "antihypertensive": re.compile(
        r"lisinopril|enalapril|captopril|ramipril|benazepril|quinapril|"
        r"perindopril|fosinopril|trandolapril|losartan|valsartan|irbesartan|"
        r"olmesartan|candesartan|telmisartan|amlodipine|nifedipine|diltiazem|"
        r"verapamil|felodipine|metoprolol|atenolol|bisoprolol|carvedilol|"
        r"labetalol|propranolol|nebivolol|hydralazine|clonidine|minoxidil|"
        r"doxazosin|terazosin|prazosin",
        re.IGNORECASE,
    ),
    "statin": re.compile(
        r"atorvastatin|simvastatin|rosuvastatin|pravastatin|lovastatin|"
        r"fluvastatin|pitavastatin|cerivastatin",
        re.IGNORECASE,
    ),
    "antiplatelet": re.compile(
        r"aspirin|clopidogrel|ticagrelor|prasugrel|ticlopidine|dipyridamole|"
        r"cilostazol|vorapaxar",
        re.IGNORECASE,
    ),
    "opiate": re.compile(
        r"oxycodone|hydrocodone|codeine|morphine|hydromorphone|tramadol|"
        r"fentanyl|methadone|oxymorphone|tapentadol|buprenorphine|"
        r"meperidine|nalbuphine|butorphanol",
        re.IGNORECASE,
    ),
}

DC_MED_COLS: list[str] = [
    "dc_med_antibiotic", "dc_med_anticoagulant", "dc_med_diuretic",
    "dc_med_steroid", "dc_med_insulin", "dc_med_antihypertensive",
    "dc_med_statin", "dc_med_antiplatelet", "dc_med_opiate",
]

# ---------------------------------------------------------------------------
# ICD-10 Charlson mappings (from V3, unchanged)
# ---------------------------------------------------------------------------
_ICD10_CHARLSON: list[tuple[list[str], str, int]] = [
    (["I21", "I22", "I25.2"],                                          "mi",               1),
    (["I50", "I11.0", "I13.0", "I13.2", "I42"],                       "chf",              1),
    (["I70", "I71", "I73.1", "I73.8", "I73.9", "I77.1"],              "pvd",              1),
    (["I60", "I61", "I62", "I63", "I64", "I65", "I66", "G45", "G46"],"cvd",              1),
    (["F00", "F01", "F02", "F03", "F05.1", "G30", "G31.1"],           "dementia",         1),
    (["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
      "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"],       "copd",             1),
    (["M05", "M06", "M32", "M33", "M34", "M35.3"],                    "rheumatic",        1),
    (["K25", "K26", "K27", "K28"],                                     "pud",              1),
    (["K70.0", "K70.1", "K70.2", "K70.3", "K73", "K74"],              "mild_liver",       1),
    (["E10.0", "E10.1", "E10.9", "E11.0", "E11.1", "E11.9",
      "E13.0", "E13.1", "E13.9"],                                      "dm_uncomp",        1),
    (["E10.2", "E10.3", "E10.4", "E10.5", "E11.2", "E11.3",
      "E11.4", "E11.5", "E13.2", "E13.3", "E13.4", "E13.5"],         "dm_comp",          2),
    (["G81", "G82", "G04.1"],                                          "hemiplegia",       2),
    (["N03", "N05", "N18", "N19", "N25.0"],                            "renal",            2),
    (["C0", "C1", "C2", "C3", "C4", "C5", "C6",
      "C70", "C71", "C72", "C73", "C74", "C75", "C76",
      "C81", "C82", "C83", "C84", "C85", "C88",
      "C90", "C91", "C92", "C93", "C94", "C95", "C96"],               "cancer",           2),
    (["K72", "K76.6", "K76.7"],                                        "severe_liver",     3),
    (["C77", "C78", "C79", "C80"],                                     "metastatic",       6),
    (["B20", "B21", "B22", "B24"],                                     "aids",             6),
]
_ICD10_CHARLSON_LOOKUP: dict[str, tuple[str, int]] = {
    pfx: (cat, w) for prefixes, cat, w in _ICD10_CHARLSON for pfx in prefixes
}

# ICD-9 Charlson prefix mappings (MIMIC stores without decimal, e.g. "4280" for 428.0)
_ICD9_CHARLSON: list[tuple[list[str], str, int]] = [
    (["410", "412"],                                                    "mi",               1),
    (["428", "40201", "40211", "40291", "40401", "40403",
      "40411", "40413", "40491", "40493"],                             "chf",              1),
    (["440", "441", "4431", "4432", "4433", "4434", "4435",
      "4436", "4437", "4438", "4439", "7854", "V434"],                 "pvd",              1),
    (["430", "431", "432", "433", "434", "435", "436", "437", "438"], "cvd",              1),
    (["290", "2941", "3312"],                                          "dementia",         1),
    (["490", "491", "492", "493", "494", "495", "496",
      "500", "501", "502", "503", "504", "505"],                       "copd",             1),
    (["7100", "7101", "7104", "7140", "7141", "7142", "7148", "725"], "rheumatic",        1),
    (["531", "532", "533", "534"],                                     "pud",              1),
    (["5712", "5714", "5715", "5716", "5718", "5728"],                 "mild_liver",       1),
    (["2500", "2501", "2502", "2503", "2508", "2509"],                 "dm_uncomp",        1),
    (["2504", "2505", "2506", "2507"],                                 "dm_comp",          2),
    (["342", "343", "3441", "3442", "3443", "3444",
      "3445", "3446", "3449"],                                         "hemiplegia",       2),
    (["582", "5830", "5831", "5832", "5833", "5834",
      "5835", "5836", "5837", "585", "586", "5880",
      "V420", "V451", "V56"],                                          "renal",            2),
    (["14", "15", "16", "17", "174", "175", "176",
      "179", "18", "190", "191", "192", "193", "194",
      "195", "200", "201", "202", "203", "204",
      "205", "206", "207", "208"],                                     "cancer",           2),
    (["5722", "5723", "5724", "5725", "5726", "5727", "5728"],        "severe_liver",     3),
    (["196", "197", "198", "199"],                                     "metastatic",       6),
    (["042", "043", "044"],                                            "aids",             6),
]
_ICD9_CHARLSON_LOOKUP: dict[str, tuple[str, int]] = {
    pfx: (cat, w) for prefixes, cat, w in _ICD9_CHARLSON for pfx in prefixes
}

# ---------------------------------------------------------------------------
# ICD-9 and ICD-10 chronic disease flag prefixes (18 conditions)
# ---------------------------------------------------------------------------
# Structure: {condition_name: {"icd9": [pfx...], "icd10": [pfx...]}}
CHRONIC_CONDITIONS: dict[str, dict[str, list[str]]] = {
    "depression": {
        "icd9":  ["2962", "2963", "2965", "3004", "311"],
        "icd10": ["F32", "F33", "F341"],
    },
    "drug_use": {
        "icd9":  ["292", "304", "3052", "3053", "3054", "3055",
                  "3056", "3057", "3058", "3059"],
        "icd10": ["F11", "F12", "F13", "F14", "F15", "F16", "F18", "F19"],
    },
    "alcohol": {
        "icd9":  ["291", "303", "3050"],
        "icd10": ["F10"],
    },
    "obesity": {
        "icd9":  ["2780", "2781", "V8530", "V8531", "V8532",
                  "V8533", "V8534", "V8535", "V8536", "V8537",
                  "V8538", "V8539", "V854"],
        "icd10": ["E66"],
    },
    "malnutrition": {
        "icd9":  ["260", "261", "262", "263", "2699"],
        "icd10": ["E40", "E41", "E42", "E43", "E44", "E46"],
    },
    "liver_disease": {
        "icd9":  ["571", "572", "573"],
        "icd10": ["K70", "K71", "K72", "K73", "K74", "K75", "K76", "K77"],
    },
    "chf": {
        "icd9":  ["428", "40201", "40211", "40291",
                  "40401", "40403", "40411", "40413",
                  "40491", "40493"],
        "icd10": ["I50", "I110", "I130", "I132", "I42"],
    },
    "copd": {
        "icd9":  ["490", "491", "492", "493", "494", "495", "496",
                  "500", "501", "502", "503", "504", "505"],
        "icd10": ["J40", "J41", "J42", "J43", "J44", "J45", "J46", "J47",
                  "J60", "J61", "J62", "J63", "J64", "J65", "J66", "J67"],
    },
    "diabetes": {
        "icd9":  ["250"],
        "icd10": ["E10", "E11", "E12", "E13", "E14"],
    },
    "ckd": {
        "icd9":  ["585", "586", "5880", "V420", "V451", "V56"],
        "icd10": ["N18", "N19"],
    },
    "cancer": {
        "icd9":  ["14", "15", "16", "17", "174", "175", "176", "179",
                  "18", "190", "191", "192", "193", "194", "195",
                  "200", "201", "202", "203", "204", "205", "206", "207", "208"],
        "icd10": ["C0", "C1", "C2", "C3", "C4", "C5", "C6",
                  "C70", "C71", "C72", "C73", "C74", "C75", "C76",
                  "C81", "C82", "C83", "C84", "C85", "C88",
                  "C90", "C91", "C92", "C93", "C94", "C95", "C96",
                  "C77", "C78", "C79", "C80"],
    },
    "dementia": {
        "icd9":  ["290", "2941", "3312"],
        "icd10": ["F00", "F01", "F02", "F03", "G30", "G311"],
    },
    "pvd": {
        "icd9":  ["440", "441", "4431", "4432", "4433", "4434",
                  "4435", "4436", "4437", "4438", "4439"],
        "icd10": ["I70", "I71", "I73", "I74"],
    },
    "atrial_fibrillation": {
        "icd9":  ["42731"],
        "icd10": ["I48"],
    },
    "hypertension": {
        "icd9":  ["401", "402", "403", "404", "405"],
        "icd10": ["I10", "I11", "I12", "I13", "I15"],
    },
    "schizophrenia": {
        "icd9":  ["295"],
        "icd10": ["F20", "F21", "F22", "F23", "F24", "F25", "F28", "F29"],
    },
    "sleep_apnea": {
        "icd9":  ["32723", "78051", "78053", "78057"],
        "icd10": ["G473"],
    },
    "hypothyroidism": {
        "icd9":  ["244"],
        "icd10": ["E03", "E890"],
    },
}

CHRONIC_FLAG_COLS: list[str] = [f"flag_{c}" for c in CHRONIC_CONDITIONS]

# ---------------------------------------------------------------------------
# Consult subtypes from POE table (order_type='Consults', match on order_subtype)
# Values confirmed from the actual DB: exact strings used in MIMIC-IV POE table.
# ---------------------------------------------------------------------------
CONSULT_SUBTYPES: dict[str, list[str]] = {
    "consult_pt":                 ["Physical Therapy"],
    "consult_ot":                 ["Occupational Therapy"],
    "consult_social_work":        ["Social Work"],
    "consult_followup":           ["Discharge Followup Appointment"],
    "consult_palliative":         ["Palliative Care", "Palliative Care/Ethics Support"],
    "consult_diabetes_education": ["Diabetes Consult"],
    "consult_speech":             ["Speech/Swallowing"],
    "consult_addiction":          ["Addiction"],
}

CONSULT_COLS: list[str] = list(CONSULT_SUBTYPES.keys())

# Keep CONSULT_PATTERNS for any fallback regex use — not used in main pipeline
CONSULT_PATTERNS: dict[str, re.Pattern] = {
    col: re.compile("|".join(re.escape(s) for s in subtypes), re.IGNORECASE)
    for col, subtypes in CONSULT_SUBTYPES.items()
}

# Discharge location groupings
_DC_LOC_HOME     = {"HOME", "HOME HEALTH CARE"}
_DC_LOC_SNF      = {"SKILLED NURSING FACILITY"}
_DC_LOC_REHAB    = {"REHAB", "REHABILITATION"}
_DC_LOC_HOSPICE  = {"HOSPICE", "HOSPICE - HOME", "HOSPICE - MEDICAL FACILITY"}
_DC_LOC_LTAC     = {"LONG TERM CARE HOSPITAL"}
_DC_LOC_PSYCH    = {"PSYCHIATRIC FACILITY", "PSYCH FACILITY - PARTIAL HOSPITALIZATION"}
_DC_LOC_OTHER_FACILITY = {
    "CHRONIC/LONG TERM ACUTE CARE", "OTHER FACILITY",
    "ASSISTED LIVING", "INTERMEDIATE CARE FACILITY (ICF/DD)",
}


def _simplify_discharge_location(loc: str | None) -> str:
    if not loc:
        return "other"
    loc_upper = str(loc).strip().upper()
    if loc_upper in _DC_LOC_HOME:
        return "home"
    if loc_upper in _DC_LOC_SNF:
        return "snf"
    if any(loc_upper.startswith(r) for r in ("REHAB",)):
        return "rehab"
    if any(loc_upper.startswith(h) for h in ("HOSPICE",)):
        return "hospice"
    if loc_upper in _DC_LOC_LTAC:
        return "ltac"
    if loc_upper in _DC_LOC_PSYCH or "PSYCH" in loc_upper:
        return "psych"
    if "DIED" in loc_upper or "EXPIRED" in loc_upper:
        return "died"
    if any(loc_upper.startswith(f) for f in ("ANOTHER", "OTHER", "TRANSFER")):
        return "other_facility"
    if "HOME" in loc_upper:
        return "home"
    if "SKILLED" in loc_upper or "SNF" in loc_upper:
        return "snf"
    return "other"


# ---------------------------------------------------------------------------
# DB helpers
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
    """Return ' AND alias.hadm_id IN (...)' clause for small cohorts, else empty."""
    hadm_ids = spine["hadm_id"].unique()
    if len(hadm_ids) > 50_000:
        return ""
    prefix = f"{alias}." if alias else ""
    csv = ",".join(str(int(h)) for h in hadm_ids)
    return f" AND {prefix}hadm_id::bigint IN ({csv})"


def _subject_filter(spine: pd.DataFrame, alias: str = "") -> str:
    """Return ' AND alias.subject_id IN (...)' for small cohorts."""
    sids = spine["subject_id"].unique()
    if len(sids) > 50_000:
        return ""
    prefix = f"{alias}." if alias else ""
    csv = ",".join(str(int(s)) for s in sids)
    return f" AND {prefix}subject_id::bigint IN ({csv})"


# ===========================================================================
# CHECKPOINT HELPERS
# ===========================================================================

def _ckpt_dir(cfg: dict, sample_only: bool) -> Path:
    root = Path(cfg.get("_project_root", "."))
    n    = cfg.get("sample", {}).get("n_episodes", 500)
    tag  = f"sample_{n}" if sample_only else "full"
    d    = root / "data" / "interim" / "checkpoints_v4" / tag
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_ckpt(df: pd.DataFrame, d: Path, name: str) -> None:
    path = d / f"{name}.parquet"
    df.to_parquet(path, index=False)
    log.info(
        "  [ckpt] Saved %s -- %d rows, %d cols -> %s",
        name, len(df), len(df.columns), path,
    )


def _load_ckpt(d: Path, name: str) -> "pd.DataFrame | None":
    path = d / f"{name}.parquet"
    if path.exists():
        df = pd.read_parquet(path)
        # Ensure id columns are int64 (parquet can round-trip them as object)
        for col in ("hadm_id", "subject_id"):
            if col in df.columns:
                df[col] = df[col].astype("int64")
        log.info(
            "  [ckpt] Loaded %s -- %d rows, %d cols <- %s",
            name, len(df), len(df.columns), path,
        )
        return df
    return None


def _save_meta(d: Path, data: dict) -> None:
    path = d / "meta.json"
    existing: dict = {}
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


# ===========================================================================
# Step 1: SPINE
# Build one row per hadm_id with admission context.
# Apply all cohort exclusions here.
# ===========================================================================

def step1_spine(conn: Any, cfg: dict[str, Any]) -> pd.DataFrame:
    """Build admission-level spine.  One row per hadm_id.

    Exclusions:
      - hospital_expire_flag = 1
      - OBS/GYN service (obstetrics/gynecology)
      - post-discharge death within 30 days of dischtime
    """
    log.info("Step 1: Building spine")
    schema_hosp = cfg["schemas"]["hosp"]
    exclude_obs = cfg["cohort"].get("exclude_obstetrics", True)
    exclude_dc_death_days = cfg["cohort"].get("exclude_postdischarge_death_days", 30)

    # --- base admissions + patients join ---
    sql_base = f"""
        SELECT
            a.hadm_id,
            a.subject_id,
            a.admittime,
            a.dischtime,
            a.deathtime,
            a.hospital_expire_flag,
            a.admission_type,
            a.admission_location,
            a.discharge_location,
            a.insurance,
            a.language,
            a.marital_status,
            a.race,
            a.edregtime,
            a.edouttime,
            p.gender,
            p.anchor_age,
            p.anchor_year,
            p.dod
        FROM {schema_hosp}.admissions a
        JOIN {schema_hosp}.patients   p ON p.subject_id::bigint = a.subject_id::bigint
        WHERE a.hospital_expire_flag::integer = 0
    """
    log.info("  Querying admissions + patients ...")
    spine = pd.read_sql(sql_base, conn)
    for col in ("hadm_id", "subject_id"):
        if col in spine.columns:
            spine[col] = spine[col].astype("int64")
    log.info("  Raw rows (excl. hospital deaths): %d", len(spine))

    # --- convert timestamps ---
    for col in ["admittime", "dischtime", "deathtime", "edregtime", "edouttime"]:
        if col in spine.columns:
            spine[col] = pd.to_datetime(spine[col], utc=False, errors="coerce")
    spine["dod"] = pd.to_datetime(spine["dod"], errors="coerce")

    # --- exclude post-discharge death within N days ---
    if exclude_dc_death_days is not None and exclude_dc_death_days > 0:
        dc_death_mask = (
            spine["dod"].notna()
            & (spine["dod"] - spine["dischtime"])
            .dt.total_seconds()
            .div(86400)
            .between(0, exclude_dc_death_days, inclusive="both")
        )
        n_before = len(spine)
        spine = spine[~dc_death_mask].copy()
        log.info(
            "  Excluded %d admissions: post-discharge death within %d days",
            n_before - len(spine), exclude_dc_death_days,
        )

    # --- exclude obstetrics via services table ---
    if exclude_obs:
        sql_obs = f"""
            SELECT DISTINCT hadm_id
            FROM {schema_hosp}.services
            WHERE curr_service IN ('OBS', 'GYN')
        """
        obs_hadm = pd.read_sql(sql_obs, conn)["hadm_id"].astype(int)
        obs_set  = set(obs_hadm)
        n_before = len(spine)
        spine    = spine[~spine["hadm_id"].isin(obs_set)].copy()
        log.info(
            "  Excluded %d admissions: OBS/GYN service",
            n_before - len(spine),
        )

    # --- compute derived admission features ---
    # Age at admission: anchor_age refers to anchor_year, adjust by admittime year
    spine["age_at_admit"] = (
        spine["anchor_age"]
        + (spine["admittime"].dt.year - spine["anchor_year"])
    ).clip(lower=0).astype(float)

    # LOS in days
    spine["los_days"] = (
        (spine["dischtime"] - spine["admittime"])
        .dt.total_seconds()
        .div(86400)
        .round(3)
    )

    # Admit weekday (0=Monday, 6=Sunday)
    spine["admit_weekday"] = spine["admittime"].dt.dayofweek.astype("Int8")
    spine["is_weekend_admit"] = spine["admit_weekday"].isin([5, 6]).astype("Int8")

    # ED dwell in hours (null if not via ED)
    spine["ed_dwell_hours"] = (
        (spine["edouttime"] - spine["edregtime"])
        .dt.total_seconds()
        .div(3600)
        .round(2)
    )
    # Negative or implausible values → null
    spine.loc[spine["ed_dwell_hours"] < 0, "ed_dwell_hours"] = np.nan
    spine.loc[spine["ed_dwell_hours"] > 72, "ed_dwell_hours"] = np.nan

    # Binary: came through ED
    spine["via_ed"] = spine["ed_dwell_hours"].notna().astype("Int8")

    # --- simplify categorical variables ---
    # Insurance: 3 categories
    def _simplify_insurance(s: str | None) -> str:
        if not s:
            return "other"
        s = str(s).upper()
        if "MEDICARE" in s:
            return "medicare"
        if "MEDICAID" in s:
            return "medicaid"
        return "other"

    spine["insurance_cat"] = spine["insurance"].apply(_simplify_insurance)

    # Language: binary English vs other
    spine["english_only"] = (
        spine["language"].str.upper().str.strip() == "ENGLISH"
    ).astype("Int8")

    # Marital status: partnered vs not
    spine["is_partnered"] = (
        spine["marital_status"]
        .str.upper()
        .str.strip()
        .isin({"MARRIED", "PARTNERED"})
    ).astype("Int8")

    # Race: 5 categories
    def _simplify_race(r: str | None) -> str:
        if not r:
            return "other"
        r = str(r).upper()
        if "WHITE" in r:
            return "white"
        if "BLACK" in r or "AFRICAN" in r:
            return "black"
        if "HISPANIC" in r or "LATIN" in r:
            return "hispanic"
        if "ASIAN" in r:
            return "asian"
        return "other"

    spine["race_cat"] = spine["race"].apply(_simplify_race)

    # Discharge location simplified
    spine["dc_location"] = spine["discharge_location"].apply(
        _simplify_discharge_location
    )

    # Admission type: clean up
    spine["admission_type"] = spine["admission_type"].str.strip().fillna("UNKNOWN")

    # Observation status flag (CMS outpatient observation — shorter stays, fewer DRGs,
    # different care pathway than true inpatients; ~43% of MIMIC admissions)
    spine["is_observation"] = (
        spine["admission_type"].str.upper().str.contains("OBSERVATION", na=False)
    ).astype("Int8")

    log.info(
        "  Spine complete: %d admissions, %d subjects",
        spine["hadm_id"].nunique(), spine["subject_id"].nunique(),
    )

    # Drop raw columns no longer needed (keep dod for readmit_30d in step 10)
    spine = spine.drop(
        columns=["deathtime", "edregtime", "edouttime",
                 "anchor_age", "anchor_year",
                 "insurance", "language", "marital_status",
                 "race", "discharge_location"],
        errors="ignore",
    )

    return spine


# ===========================================================================
# Step 2: SEVERITY — DRG + Charlson
# ===========================================================================

def _compute_charlson_icd(conn: Any, schema_hosp: str,
                          hadm_ids: list[int]) -> tuple[pd.DataFrame, str]:
    """Compute Charlson score from diagnoses_icd (ICD-9 and ICD-10)."""
    log.info("  Computing Charlson from diagnoses_icd (ICD-9 + ICD-10 fallback)")
    filter_sql = ""
    if len(hadm_ids) <= 50_000:
        csv = ",".join(str(h) for h in hadm_ids)
        filter_sql = f" WHERE hadm_id::bigint IN ({csv})"
    sql = (
        f"SELECT hadm_id, icd_code, icd_version "
        f"FROM {schema_hosp}.diagnoses_icd{filter_sql}"
    )
    diag = pd.read_sql(sql, conn)
    diag["icd_code"] = diag["icd_code"].str.strip()

    records: list[dict] = []
    for hadm_id, grp in diag.groupby("hadm_id"):
        found: dict[str, int] = {}
        for _, row in grp.iterrows():
            code    = row["icd_code"]
            version = int(row["icd_version"])
            lookup  = _ICD10_CHARLSON_LOOKUP if version == 10 else _ICD9_CHARLSON_LOOKUP
            for length in range(len(code), 2, -1):
                pfx = code[:length]
                if pfx in lookup:
                    cat, weight = lookup[pfx]
                    if cat not in found or weight > found[cat]:
                        found[cat] = weight
                    break
        records.append({"hadm_id": hadm_id, "charlson_score": sum(found.values())})

    df = pd.DataFrame(records) if records else pd.DataFrame(
        columns=["hadm_id", "charlson_score"]
    )
    return df, "diagnoses_icd"


def step2_severity(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> tuple[pd.DataFrame, str]:
    """Add DRG severity/mortality + Charlson score."""
    log.info("Step 2: Severity (DRG + Charlson)")
    schema_hosp = cfg["schemas"]["hosp"]
    schema_derived = cfg["schemas"]["derived"]
    hadm_ids = spine["hadm_id"].tolist()
    hadm_filter = _hadm_filter(spine)

    # --- DRG severity and mortality ---
    sql_drg = f"""
        SELECT
            hadm_id,
            MAX(CASE WHEN drg_type = 'APR' THEN drg_severity  END) AS drg_severity,
            MAX(CASE WHEN drg_type = 'APR' THEN drg_mortality END) AS drg_mortality
        FROM {schema_hosp}.drgcodes
        WHERE 1=1{hadm_filter}
        GROUP BY hadm_id
    """
    log.info("  Querying DRG codes ...")
    drg = pd.read_sql(sql_drg, conn)
    log.info("  DRG rows: %d (hadm coverage: %d)", len(drg), drg["hadm_id"].nunique())

    # --- Services (first and last) ---
    sql_svc = f"""
        SELECT
            hadm_id,
            FIRST_VALUE(curr_service) OVER (
                PARTITION BY hadm_id ORDER BY transfertime ASC
            ) AS first_service,
            LAST_VALUE(curr_service) OVER (
                PARTITION BY hadm_id ORDER BY transfertime ASC
                ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
            ) AS discharge_service
        FROM {schema_hosp}.services
        WHERE 1=1{hadm_filter}
    """
    log.info("  Querying services ...")
    svc_raw = pd.read_sql(sql_svc, conn)
    svc = svc_raw.drop_duplicates(subset=["hadm_id"])
    log.info("  Services hadm coverage: %d", svc["hadm_id"].nunique())

    # --- Charlson: try derived table first ---
    charlson_source = "unknown"
    charlson_df     = None
    if _table_exists(conn, schema_derived, "charlson"):
        log.info("  Using derived.charlson ...")
        sql_ch = f"""
            SELECT hadm_id, charlson_comorbidity_index AS charlson_score
            FROM {schema_derived}.charlson
            WHERE 1=1{hadm_filter}
        """
        try:
            charlson_df = pd.read_sql(sql_ch, conn)
            charlson_source = "derived.charlson"
            log.info("  Charlson from derived: %d rows", len(charlson_df))
        except Exception as e:
            log.warning("  derived.charlson failed (%s), falling back", e)
            charlson_df = None

    if charlson_df is None or len(charlson_df) == 0:
        charlson_df, charlson_source = _compute_charlson_icd(
            conn, schema_hosp, hadm_ids
        )

    # --- merge everything ---
    spine = spine.merge(drg, on="hadm_id", how="left")
    spine = spine.merge(svc[["hadm_id", "first_service", "discharge_service"]],
                        on="hadm_id", how="left")
    spine = spine.merge(charlson_df, on="hadm_id", how="left")

    # Fill missing charlson with 0 (no documented comorbidities)
    spine["charlson_score"] = spine["charlson_score"].fillna(0).astype(float)

    for col in ["drg_severity", "drg_mortality"]:
        spine[col] = pd.to_numeric(spine[col], errors="coerce")

    # Binary flag: whether APR-DRG was assigned (missing for observation stays / no APR code)
    spine["drg_available"] = spine["drg_severity"].notna().astype("Int8")

    # Impute missing DRG scores with 0 (minimum severity — conservative)
    spine["drg_severity"]  = spine["drg_severity"].fillna(0).astype(float)
    spine["drg_mortality"] = spine["drg_mortality"].fillna(0).astype(float)

    log.info(
        "  DRG available: %.1f%% | Charlson source: %s",
        100 * spine["drg_available"].mean(), charlson_source,
    )
    return spine, charlson_source


# ===========================================================================
# Step 3: CHRONIC DISEASE FLAGS (ICD-9 + ICD-10 dual mapping)
# ===========================================================================

def step3_chronic_flags(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Add 18 binary chronic disease flags using both ICD-9 and ICD-10."""
    log.info("Step 3: Chronic disease flags (18 conditions)")
    schema_hosp = cfg["schemas"]["hosp"]
    hadm_filter = _hadm_filter(spine)

    sql = (
        f"SELECT hadm_id, icd_code, icd_version "
        f"FROM {schema_hosp}.diagnoses_icd "
        f"WHERE 1=1{hadm_filter}"
    )
    log.info("  Querying diagnoses_icd ...")
    diag = pd.read_sql(sql, conn)
    diag["icd_code"] = diag["icd_code"].str.strip()
    diag["icd_version"] = diag["icd_version"].astype(int)
    log.info("  Diagnoses rows: %d, hadm coverage: %d",
             len(diag), diag["hadm_id"].nunique())

    # Build per-hadm_id prefix sets for fast lookups
    # Group by hadm_id, then check each code against condition prefixes
    log.info("  Computing flags (groupby) ...")

    # For each condition, build a vectorised check:
    #   for icd9 rows: code starts with any icd9 prefix
    #   for icd10 rows: code starts with any icd10 prefix
    flag_rows: list[dict] = []

    for hadm_id, grp in diag.groupby("hadm_id"):
        row: dict[str, Any] = {"hadm_id": hadm_id}
        codes9  = set(grp.loc[grp["icd_version"] == 9,  "icd_code"])
        codes10 = set(grp.loc[grp["icd_version"] == 10, "icd_code"])

        for cond, maps in CHRONIC_CONDITIONS.items():
            hit = False
            for pfx in maps["icd9"]:
                if any(c.startswith(pfx) for c in codes9):
                    hit = True
                    break
            if not hit:
                for pfx in maps["icd10"]:
                    if any(c.startswith(pfx) for c in codes10):
                        hit = True
                        break
            row[f"flag_{cond}"] = int(hit)
        flag_rows.append(row)

    flags_df = pd.DataFrame(flag_rows)
    log.info("  Flags computed for %d hadm_ids", len(flags_df))

    spine = spine.merge(flags_df, on="hadm_id", how="left")
    for col in CHRONIC_FLAG_COLS:
        spine[col] = spine[col].fillna(0).astype("Int8")

    for cond in CHRONIC_CONDITIONS:
        col = f"flag_{cond}"
        pct = 100 * spine[col].mean()
        log.info("    %s: %.1f%%", col, pct)

    return spine


# ===========================================================================
# Step 4: ICU AND TRANSFERS
# ===========================================================================

def step4_icu_transfers(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Add ICU stay flags and transfer counts."""
    log.info("Step 4: ICU stays and transfers")
    schema_icu  = cfg["schemas"]["icu"]
    schema_hosp = cfg["schemas"]["hosp"]
    hadm_filter = _hadm_filter(spine)

    # --- ICU stays ---
    sql_icu = f"""
        SELECT
            hadm_id,
            COUNT(*)              AS n_icu_stays,
            SUM(los)              AS icu_days
        FROM {schema_icu}.icustays
        WHERE 1=1{hadm_filter}
        GROUP BY hadm_id
    """
    log.info("  Querying icustays ...")
    icu = pd.read_sql(sql_icu, conn)
    icu["had_icu"] = 1

    # --- Transfers: count distinct careunits visited ---
    sql_tr = f"""
        SELECT
            hadm_id,
            COUNT(*)                      AS n_transfers,
            COUNT(DISTINCT careunit)       AS n_distinct_careunits
        FROM {schema_hosp}.transfers
        WHERE 1=1{hadm_filter}
          AND eventtype NOT IN ('admit', 'discharge')
        GROUP BY hadm_id
    """
    log.info("  Querying transfers ...")
    transfers = pd.read_sql(sql_tr, conn)

    # --- Services count ---
    sql_svc = f"""
        SELECT hadm_id, COUNT(DISTINCT curr_service) AS n_distinct_services
        FROM {schema_hosp}.services
        WHERE 1=1{hadm_filter}
        GROUP BY hadm_id
    """
    svc_counts = pd.read_sql(sql_svc, conn)

    spine = spine.merge(
        icu[["hadm_id", "had_icu", "icu_days", "n_icu_stays"]],
        on="hadm_id", how="left",
    )
    spine["had_icu"]     = spine["had_icu"].fillna(0).astype("Int8")
    spine["icu_days"]    = spine["icu_days"].fillna(0).astype(float)
    spine["n_icu_stays"] = spine["n_icu_stays"].fillna(0).astype("Int8")

    spine = spine.merge(transfers, on="hadm_id", how="left")
    spine["n_transfers"]         = spine["n_transfers"].fillna(0).astype("Int8")
    spine["n_distinct_careunits"]= spine["n_distinct_careunits"].fillna(0).astype("Int8")

    spine = spine.merge(svc_counts, on="hadm_id", how="left")
    spine["n_distinct_services"] = spine["n_distinct_services"].fillna(0).astype("Int8")

    log.info(
        "  ICU coverage: %.1f%% | median icu_days: %.1f | median transfers: %.1f",
        100 * spine["had_icu"].mean(),
        spine["icu_days"].median(),
        spine["n_transfers"].median(),
    )
    return spine


# ===========================================================================
# Step 5: LABS AT DISCHARGE
# Last value within 48h before dischtime for 11 key labs.
# ===========================================================================

def step5_discharge_labs(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Pull last lab values within discharge window."""
    log.info("Step 5: Discharge labs (last value within 48h before dischtime)")
    schema_hosp = cfg["schemas"]["hosp"]
    window_h    = cfg.get("labs_window_hours", 48)
    itemid_csv  = ",".join(str(i) for i in DISCHARGE_LAB_ITEMIDS)
    subject_filter = _subject_filter(spine)

    # Query all relevant labs for cohort subjects (itemid-filtered, no time filter yet)
    sql = f"""
        SELECT subject_id::bigint   AS subject_id,
               hadm_id::bigint      AS hadm_id,
               charttime,
               itemid::integer      AS itemid,
               valuenum::float      AS valuenum
        FROM {schema_hosp}.labevents
        WHERE itemid::integer IN ({itemid_csv})
          AND valuenum IS NOT NULL{subject_filter}
    """
    log.info("  Querying labevents (itemids: %s) ...", itemid_csv[:60])
    labs = pd.read_sql(sql, conn)
    # Ensure numeric types (DB may return as object depending on driver)
    for col in ("subject_id", "hadm_id", "itemid", "valuenum"):
        if col in labs.columns:
            labs[col] = pd.to_numeric(labs[col], errors="coerce")
    labs["charttime"] = pd.to_datetime(labs["charttime"], errors="coerce")
    log.info("  Lab rows loaded: %d", len(labs))

    # Join with spine on subject_id + time window
    # Use hadm_id where available, fall back to subject+time range
    labs_with_hadm = labs.dropna(subset=["hadm_id"]).copy()
    labs_no_hadm   = labs[labs["hadm_id"].isna()].copy()

    # For hadm_id-linked rows: direct join
    merge_cols = spine[["hadm_id", "dischtime"]].copy()
    labs_linked = labs_with_hadm.merge(merge_cols, on="hadm_id", how="inner")

    # For null-hadm_id rows: join on subject_id then filter by admittime/dischtime
    if len(labs_no_hadm) > 0:
        merge_cols2 = spine[["subject_id", "hadm_id", "admittime", "dischtime"]].copy()
        labs_fallback = labs_no_hadm.drop(columns=["hadm_id"]).merge(
            merge_cols2, on="subject_id", how="inner"
        )
        labs_fallback = labs_fallback[
            (labs_fallback["charttime"] >= labs_fallback["admittime"])
            & (labs_fallback["charttime"] <= labs_fallback["dischtime"])
        ]
        labs_linked = pd.concat(
            [labs_linked, labs_fallback[labs_linked.columns]], ignore_index=True
        )

    # Filter to discharge window (last N hours)
    window_td = pd.Timedelta(hours=window_h)
    labs_linked = labs_linked[
        (labs_linked["charttime"] >= labs_linked["dischtime"] - window_td)
        & (labs_linked["charttime"] <= labs_linked["dischtime"])
    ].copy()
    log.info("  Labs within discharge window: %d", len(labs_linked))

    # Get last value per (hadm_id, itemid)
    labs_linked = labs_linked.sort_values("charttime")
    last_labs = (
        labs_linked.groupby(["hadm_id", "itemid"])["valuenum"]
        .last()
        .reset_index()
    )
    last_labs["col_name"] = last_labs["itemid"].map(DISCHARGE_LAB_ITEMIDS)
    last_labs = last_labs.dropna(subset=["col_name"])

    # Pivot to wide format
    pivoted = last_labs.pivot_table(
        index="hadm_id", columns="col_name", values="valuenum", aggfunc="last"
    ).reset_index()
    pivoted.columns.name = None
    # Add dc_ prefix to distinguish from in-stay labs
    rename_map = {c: f"dc_lab_{c}" for c in DISCHARGE_LAB_COLS if c in pivoted.columns}
    pivoted = pivoted.rename(columns=rename_map)
    # Ensure all lab columns are float (pivot may leave object dtype)
    for col in [f"dc_lab_{c}" for c in DISCHARGE_LAB_COLS]:
        if col in pivoted.columns:
            pivoted[col] = pd.to_numeric(pivoted[col], errors="coerce")

    spine = spine.merge(pivoted, on="hadm_id", how="left")

    # Count abnormal labs
    n_abnormal = pd.Series(0.0, index=spine.index)
    for lab_col, (lo, hi) in LAB_NORMAL_RANGES.items():
        col = f"dc_lab_{lab_col}"
        if col in spine.columns:
            abnormal = (
                spine[col].notna()
                & ((spine[col] < lo) | (spine[col] > hi))
            ).astype(float)
            n_abnormal += abnormal
    spine["n_abnormal_dc_labs"] = n_abnormal.astype("Int16")

    # Coverage log
    for col in [f"dc_lab_{c}" for c in DISCHARGE_LAB_COLS]:
        if col in spine.columns:
            cov = 100 * spine[col].notna().mean()
            log.info("    %s coverage: %.1f%%", col, cov)

    return spine


# ===========================================================================
# Step 6: MICROBIOLOGY
# ===========================================================================

def step6_microbiology(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Add microbiology flags: positive_culture, blood_culture_positive,
    resistant_organism."""
    log.info("Step 6: Microbiology")
    schema_hosp = cfg["schemas"]["hosp"]
    hadm_filter = _hadm_filter(spine)

    sql = f"""
        SELECT
            m.hadm_id::bigint AS hadm_id,
            COUNT(*)                                        AS n_cultures,
            COUNT(m.org_name)                               AS n_positive_cultures,
            SUM(CASE WHEN UPPER(m.spec_type_desc) LIKE '%BLOOD%'
                     AND m.org_name IS NOT NULL THEN 1 ELSE 0 END) AS n_blood_positive,
            SUM(CASE WHEN m.interpretation = 'R' THEN 1 ELSE 0 END) AS n_resistant
        FROM {schema_hosp}.microbiologyevents m
        WHERE m.hadm_id IS NOT NULL{hadm_filter}
        GROUP BY m.hadm_id
    """
    log.info("  Querying microbiologyevents ...")
    micro = pd.read_sql(sql, conn)
    micro["positive_culture"]      = (micro["n_positive_cultures"] > 0).astype("Int8")
    micro["blood_culture_positive"]= (micro["n_blood_positive"] > 0).astype("Int8")
    micro["resistant_organism"]    = (micro["n_resistant"] > 0).astype("Int8")

    spine = spine.merge(
        micro[["hadm_id", "positive_culture",
               "blood_culture_positive", "resistant_organism",
               "n_cultures"]],
        on="hadm_id", how="left",
    )
    for col in ["positive_culture", "blood_culture_positive", "resistant_organism"]:
        spine[col] = spine[col].fillna(0).astype("Int8")
    spine["n_cultures"] = spine["n_cultures"].fillna(0).astype("Int16")

    log.info(
        "  positive_culture: %.1f%% | blood_pos: %.1f%% | resistant: %.1f%%",
        100 * spine["positive_culture"].mean(),
        100 * spine["blood_culture_positive"].mean(),
        100 * spine["resistant_organism"].mean(),
    )
    return spine


# ===========================================================================
# Step 7: IN-HOSPITAL DRUG EXPOSURE (prescriptions, any time during stay)
# ===========================================================================

def step7_drug_exposure(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Add binary flags for 5 drug classes prescribed any time during admission."""
    log.info("Step 7: In-hospital drug exposure (prescriptions)")
    schema_hosp = cfg["schemas"]["hosp"]
    hadm_filter = _hadm_filter(spine)

    sql = f"""
        SELECT hadm_id::bigint AS hadm_id, drug
        FROM {schema_hosp}.prescriptions
        WHERE drug IS NOT NULL{hadm_filter}
    """
    log.info("  Querying prescriptions ...")
    rx = pd.read_sql(sql, conn)
    log.info("  Prescription rows: %d", len(rx))

    # For each drug class, flag the hadm_id if any matching drug name found
    for drug_class, pattern in _RX_PATTERNS.items():
        matched_mask  = rx["drug"].str.contains(pattern, na=False)
        matched_hadms = set(rx.loc[matched_mask, "hadm_id"])
        col = f"rx_{drug_class}"
        spine[col] = spine["hadm_id"].isin(matched_hadms).astype("Int8")
        log.info(
            "    %s: %.1f%% of admissions", col, 100 * spine[col].mean()
        )

    return spine


# ===========================================================================
# Step 8: CONSULTS AND DISCHARGE PLANNING ACTIONS (POE table)
# ===========================================================================

def step8_consults(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Add binary flags for 8 discharge-planning consult types from POE."""
    log.info("Step 8: Consults / discharge-planning actions (POE)")
    schema_hosp = cfg["schemas"]["hosp"]
    hadm_filter = _hadm_filter(spine)

    # Pull consult orders — order_type='Consults', service type in order_subtype
    sql = f"""
        SELECT hadm_id, order_subtype
        FROM {schema_hosp}.poe
        WHERE order_type = 'Consults'
          AND hadm_id IS NOT NULL{hadm_filter}
    """
    log.info("  Querying POE (consults) ...")
    poe = pd.read_sql(sql, conn)
    log.info("  POE consult rows: %d", len(poe))

    for col, subtypes in CONSULT_SUBTYPES.items():
        matched_mask  = poe["order_subtype"].isin(subtypes)
        matched_hadms = set(poe.loc[matched_mask, "hadm_id"])
        spine[col] = spine["hadm_id"].isin(matched_hadms).astype("Int8")
        log.info("    %s: %.1f%%", col, 100 * spine[col].mean())

    return spine


# ===========================================================================
# Step 9: DISCHARGE MEDICATIONS (pharmacy table)
# Rows with status indicating discharged with medication.
# ===========================================================================

def step9_discharge_meds(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame
) -> pd.DataFrame:
    """Add discharge medication flags from the pharmacy table."""
    log.info("Step 9: Discharge medications (pharmacy table)")
    schema_hosp = cfg["schemas"]["hosp"]

    # Join pharmacy (status = 'Discontinued via patient discharge') with prescriptions
    # on pharmacy_id to filter to ORAL routes only.
    # This distinguishes take-home medications from hospital-only IV medications
    # (e.g., heparin DVT prophylaxis is IV and would otherwise inflate anticoagulant rate).
    # Routes kept: PO, PO/NG, ORAL, SL — all routes a patient can self-administer at home.
    hadm_filter_ph = _hadm_filter(spine, alias="ph")
    sql = f"""
        SELECT DISTINCT ph.hadm_id::bigint AS hadm_id, ph.medication
        FROM {schema_hosp}.pharmacy ph
        JOIN {schema_hosp}.prescriptions pr
          ON pr.pharmacy_id::bigint = ph.pharmacy_id::bigint
        WHERE ph.hadm_id IS NOT NULL
          AND ph.medication IS NOT NULL
          AND LOWER(ph.status) LIKE '%discharg%'
          AND pr.route IN ('PO', 'PO/NG', 'ORAL', 'SL', 'PO OR NG'){hadm_filter_ph}
    """
    log.info("  Querying pharmacy discharge meds (oral route only) ...")
    dc_meds = pd.read_sql(sql, conn)
    log.info("  Discharge med rows: %d", len(dc_meds))

    # Count unique medications per hadm_id
    med_count = (
        dc_meds.groupby("hadm_id")["medication"]
        .nunique()
        .reset_index()
        .rename(columns={"medication": "n_dc_medications"})
    )
    spine = spine.merge(med_count, on="hadm_id", how="left")
    spine["n_dc_medications"] = spine["n_dc_medications"].fillna(0).astype("Int16")

    # Per-class flags
    dc_drug_key_map = {
        "antibiotic":       "dc_med_antibiotic",
        "anticoagulant":    "dc_med_anticoagulant",
        "diuretic":         "dc_med_diuretic",
        "steroid":          "dc_med_steroid",
        "insulin":          "dc_med_insulin",
        "antihypertensive": "dc_med_antihypertensive",
        "statin":           "dc_med_statin",
        "antiplatelet":     "dc_med_antiplatelet",
        "opiate":           "dc_med_opiate",
    }

    for drug_class, col in dc_drug_key_map.items():
        pattern       = _DC_MED_PATTERNS[drug_class]
        matched_mask  = dc_meds["medication"].str.contains(pattern, na=False)
        matched_hadms = set(dc_meds.loc[matched_mask, "hadm_id"])
        spine[col]    = spine["hadm_id"].isin(matched_hadms).astype("Int8")
        log.info("    %s: %.1f%%", col, 100 * spine[col].mean())

    log.info(
        "  median n_dc_medications: %.0f | max: %d",
        spine["n_dc_medications"].median(), spine["n_dc_medications"].max(),
    )
    return spine


# ===========================================================================
# Step 10: READMIT_30D OUTCOME + SPLIT + OUTPUT
# ===========================================================================

def step10_outcome_split_output(
    conn: Any, cfg: dict[str, Any], spine: pd.DataFrame,
    sample_only: bool, charlson_source: str,
) -> pd.DataFrame:
    """Compute 30-day readmission outcome, assign splits, write output."""
    log.info("Step 10: Readmit_30d outcome + split + output")
    schema_hosp = cfg["schemas"]["hosp"]

    # --- Compute readmit_30d ---
    # For each admission: is there another admission for the same subject
    # with admittime within 30 days after this dischtime?
    # We already have all cohort admissions in spine.
    # Build a lookup: subject_id -> sorted list of (admittime, hadm_id)

    log.info("  Computing 30-day readmission from cohort admissions ...")

    # Self-join on subject_id
    readmit_input = spine[["hadm_id", "subject_id", "dischtime", "admittime"]].copy()
    # Create future admissions lookup
    future = readmit_input.rename(columns={
        "hadm_id": "future_hadm_id",
        "admittime": "future_admittime",
        "dischtime": "future_dischtime",
    })
    merged = readmit_input.merge(
        future[["subject_id", "future_hadm_id", "future_admittime"]],
        on="subject_id", how="left",
    )
    # A future admission is valid if:
    # 1. It's a different hadm_id (not same admission)
    # 2. future_admittime is within (0, 30] days after dischtime
    delta = (merged["future_admittime"] - merged["dischtime"]).dt.total_seconds() / 86400
    valid_readmit = (
        (merged["future_hadm_id"] != merged["hadm_id"])
        & (delta > 0)
        & (delta <= 30)
    )
    readmit_hadms = set(merged.loc[valid_readmit, "hadm_id"])
    spine["readmit_30d"] = spine["hadm_id"].isin(readmit_hadms).astype("Int8")

    readmit_rate = 100 * spine["readmit_30d"].mean()
    log.info("  30-day readmission rate: %.1f%%", readmit_rate)

    # --- Check for last admissions (no 30-day follow-up window possible) ---
    # Flag admissions where we cannot verify readmission (near end of data)
    # This is informational only — we do NOT exclude them since MIMIC data
    # is longitudinal and covers the full observation period.
    # (In production, you'd censor admissions too close to data end date.)

    # --- Train/valid/test split by subject_id ---
    split_cfg = cfg["split"]
    log.info(
        "  Splitting: train=%.0f%% valid=%.0f%% test=%.0f%%",
        100 * split_cfg["train"], 100 * split_cfg["valid"], 100 * split_cfg["test"],
    )
    # assign_subject_splits requires a 'patient_id' column
    spine["patient_id"] = spine["subject_id"]
    spine = assign_subject_splits(
        spine,
        train=split_cfg["train"],
        valid=split_cfg["valid"],
        test=split_cfg["test"],
        seed=split_cfg["seed"],
    )
    spine = spine.drop(columns=["patient_id"])

    split_counts = spine["split"].value_counts().to_dict()
    log.info("  Split counts: %s", split_counts)

    # --- Output ---
    root = Path(cfg.get("_project_root", "."))
    out_dir = root / cfg["output"]["dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    if sample_only:
        out_path = out_dir / cfg["output"]["sample_filename"]
    else:
        out_path = out_dir / cfg["output"]["filename"]

    write_csv(spine, out_path)
    log.info("  Output written: %s (%d rows, %d cols)", out_path, len(spine), len(spine.columns))

    # --- Manifest ---
    manifest = {
        "schema_version":    SCHEMA_VERSION,
        "build_timestamp":   datetime.now(timezone.utc).isoformat(),
        "row_count":         len(spine),
        "hadm_count":        int(spine["hadm_id"].nunique()),
        "subject_count":     int(spine["subject_id"].nunique()),
        "readmit_30d_rate":  round(readmit_rate, 3),
        "split_counts":      {str(k): int(v) for k, v in split_counts.items()},
        "charlson_source":   charlson_source,
        "sample_only":       sample_only,
        "output_path":       str(out_path),
        "feature_count":     len(spine.columns) - 3,  # exclude hadm_id/subject_id/split
    }
    manifest_path = out_path.with_suffix(".manifest.json")
    write_json(manifest, manifest_path)
    log.info("  Manifest: %s", manifest_path)

    return spine


# ===========================================================================
# Pipeline orchestrator
# ===========================================================================

ALL_STEPS = list(range(1, 11))  # 1 through 10


def run_pipeline_v4(
    cfg: dict[str, Any],
    steps: list[int] | None = None,
    dry_run: bool = False,
    sample_only: bool = False,
    resume: bool = False,
) -> pd.DataFrame | None:
    """Run the V4 admission-level readmission dataset pipeline.

    Args:
        cfg:         Config dict (from YAML).
        steps:       Optional subset of steps (default: all 10).
        dry_run:     Run Step 1 only, log counts, return early.
        sample_only: Down-sample to ~N episodes after Step 1.
        resume:      Skip steps with existing checkpoint files.
    """
    steps = steps or ALL_STEPS
    conn  = _get_conn(cfg)
    log.info(
        "Connected to PostgreSQL %s:%s/%s",
        cfg["db"]["host"], cfg["db"]["port"], cfg["db"]["name"],
    )

    cd             = _ckpt_dir(cfg, sample_only)
    charlson_source = "unknown"

    try:
        spine: pd.DataFrame | None = None

        # ── Step 1: Spine ──────────────────────────────────────────────────
        if 1 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step1")) is not None:
                spine = ckpt
                log.info("Step 1: Resumed from checkpoint")
            else:
                spine = step1_spine(conn, cfg)
                if dry_run:
                    log.info(
                        "Dry-run complete: %d admissions, %d subjects",
                        spine["hadm_id"].nunique(), spine["subject_id"].nunique(),
                    )
                    return spine
                if sample_only:
                    sample_cfg = cfg["sample"]
                    from careai.transitions.sampling import subject_level_sample
                    n         = sample_cfg["n_episodes"]
                    n_subj    = spine["subject_id"].nunique()
                    n_hadm    = spine["hadm_id"].nunique()
                    frac      = min((n / n_hadm) * (n_subj / n_hadm) + 0.01, 1.0)
                    frac      = max(frac, n / n_subj)
                    spine = subject_level_sample(spine, fraction=frac, seed=sample_cfg["seed"])
                    # Trim to target
                    if spine["hadm_id"].nunique() > n * 1.2:
                        keep_subj = (
                            spine["subject_id"]
                            .drop_duplicates()
                            .sample(n=min(n, spine["subject_id"].nunique()),
                                    random_state=sample_cfg["seed"])
                        )
                        spine = spine[spine["subject_id"].isin(keep_subj)]
                    log.info(
                        "  Sample: %d subjects, %d admissions",
                        spine["subject_id"].nunique(), spine["hadm_id"].nunique(),
                    )
                _save_ckpt(spine, cd, "step1")

        # ── Step 2: Severity ───────────────────────────────────────────────
        if 2 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step2")) is not None:
                spine = ckpt
                charlson_source = _load_meta(cd).get("charlson_source", "unknown")
                log.info("Step 2: Resumed from checkpoint")
            else:
                spine, charlson_source = step2_severity(conn, cfg, spine)
                _save_ckpt(spine, cd, "step2")
                _save_meta(cd, {"charlson_source": charlson_source})

        # ── Step 3: Chronic flags ──────────────────────────────────────────
        if 3 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step3")) is not None:
                spine = ckpt
                log.info("Step 3: Resumed from checkpoint")
            else:
                spine = step3_chronic_flags(conn, cfg, spine)
                _save_ckpt(spine, cd, "step3")

        # ── Step 4: ICU + transfers ────────────────────────────────────────
        if 4 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step4")) is not None:
                spine = ckpt
                log.info("Step 4: Resumed from checkpoint")
            else:
                spine = step4_icu_transfers(conn, cfg, spine)
                _save_ckpt(spine, cd, "step4")

        # ── Step 5: Discharge labs ─────────────────────────────────────────
        if 5 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step5")) is not None:
                spine = ckpt
                log.info("Step 5: Resumed from checkpoint")
            else:
                spine = step5_discharge_labs(conn, cfg, spine)
                _save_ckpt(spine, cd, "step5")

        # ── Step 6: Microbiology ───────────────────────────────────────────
        if 6 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step6")) is not None:
                spine = ckpt
                log.info("Step 6: Resumed from checkpoint")
            else:
                spine = step6_microbiology(conn, cfg, spine)
                _save_ckpt(spine, cd, "step6")

        # ── Step 7: Drug exposure ──────────────────────────────────────────
        if 7 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step7")) is not None:
                spine = ckpt
                log.info("Step 7: Resumed from checkpoint")
            else:
                spine = step7_drug_exposure(conn, cfg, spine)
                _save_ckpt(spine, cd, "step7")

        # ── Step 8: Consults ───────────────────────────────────────────────
        if 8 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step8")) is not None:
                spine = ckpt
                log.info("Step 8: Resumed from checkpoint")
            else:
                spine = step8_consults(conn, cfg, spine)
                _save_ckpt(spine, cd, "step8")

        # ── Step 9: Discharge meds ─────────────────────────────────────────
        if 9 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step9")) is not None:
                spine = ckpt
                log.info("Step 9: Resumed from checkpoint")
            else:
                spine = step9_discharge_meds(conn, cfg, spine)
                _save_ckpt(spine, cd, "step9")

        # ── Step 10: Outcome + split + output ──────────────────────────────
        if 10 in steps:
            if resume and (ckpt := _load_ckpt(cd, "step10")) is not None:
                spine = ckpt
                log.info("Step 10: Resumed from checkpoint")
            else:
                spine = step10_outcome_split_output(
                    conn, cfg, spine,
                    sample_only=sample_only,
                    charlson_source=charlson_source,
                )
                _save_ckpt(spine, cd, "step10")

        log.info(
            "Pipeline complete. Final: %d rows, %d cols",
            len(spine), len(spine.columns),
        )
        return spine

    finally:
        conn.close()
        log.info("DB connection closed.")
