"""
Post-processing: add per-lab measured indicator columns to hosp_daily CSV.

For each lab column, adds ``{col}_measured``:
  1 = the lab was actually drawn on this (hadm_id, calendar_date)
  0 = value is forward-filled from a prior day, or lab was never measured

This is computed from a lightweight query against labevents (no values, only
presence of (hadm_id, date, itemid) tuples) and then merged onto the existing
CSV.  No existing columns are modified.

Usage
-----
    python scripts/hosp_daily/add_missingness_flags.py --config configs/hosp_daily.yaml

Optional overrides::

    --input  path/to/other.csv   (default: sample5k from config)
    --output path/to/out.csv     (default: overwrites input)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import warnings
from pathlib import Path

import pandas as pd
import psycopg2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# itemid → final column name  (direct 1-to-1 mappings)
# ---------------------------------------------------------------------------
DIRECT_ITEMID_TO_COL: dict[int, str] = {
    50912: "creatinine",  51006: "bun",        50983: "sodium",
    50971: "potassium",   50882: "bicarbonate", 50868: "anion_gap",
    50931: "glucose",     51222: "hemoglobin",  51301: "wbc",
    51265: "platelets",   50960: "magnesium",   50893: "calcium",
    50970: "phosphate",   51237: "inr",         50885: "bilirubin",
    50862: "albumin",
}

# nlr_measured = 1 only when BOTH lymphocytes AND neutrophils were drawn today
NLR_ITEMIDS: frozenset[int] = frozenset({51244, 51256})

# lactate_elevated_measured = 1 when lactate_raw (50813) was drawn today
LACTATE_ITEMID: int = 50813

ALL_ITEMIDS: list[int] = (
    list(DIRECT_ITEMID_TO_COL) + list(NLR_ITEMIDS) + [LACTATE_ITEMID]
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _db_connect(cfg: dict) -> psycopg2.extensions.connection:
    db = cfg["db"]
    return psycopg2.connect(
        host=db["host"],
        port=db["port"],
        dbname=db["name"],
        user=os.environ[db["user_env"]],
        password=os.environ[db["password_env"]],
    )


def _query_lab_dates(conn, hosp: str, hadm_ids: list[int]) -> pd.DataFrame:
    """Return (hadm_id, lab_date, itemid) for every measured lab in ALL_ITEMIDS."""
    itemid_list = ",".join(str(i) for i in ALL_ITEMIDS)
    hadm_list   = ",".join(str(i) for i in hadm_ids)
    sql = f"""
        SELECT hadm_id::bigint                       AS hadm_id,
               DATE(CAST(charttime AS TIMESTAMP))    AS lab_date,
               itemid::integer                       AS itemid
        FROM   {hosp}.labevents
        WHERE  hadm_id::bigint  IN ({hadm_list})
          AND  itemid::integer  IN ({itemid_list})
          AND  valuenum IS NOT NULL
        GROUP  BY 1, 2, 3
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        df = pd.read_sql(sql, conn)
    df["hadm_id"] = df["hadm_id"].astype("int64")
    df["itemid"]  = df["itemid"].astype(int)
    df["lab_date"] = pd.to_datetime(df["lab_date"]).dt.normalize()
    return df


def _left_merge_flag(
    df: pd.DataFrame,
    hits: pd.DataFrame,
    flag_col: str,
) -> pd.DataFrame:
    """
    Left-merge a (hadm_id, lab_date) indicator onto df via calendar_date.
    Returns df with a new int8 column ``flag_col`` (1 = hit, 0 = miss).
    """
    df = df.merge(
        hits[["hadm_id", "lab_date", flag_col]],
        left_on=["hadm_id", "calendar_date"],
        right_on=["hadm_id", "lab_date"],
        how="left",
    )
    df[flag_col] = df[flag_col].fillna(0).astype("int8")
    df = df.drop(columns=["lab_date"], errors="ignore")
    return df


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add per-lab measured indicator columns to hosp_daily CSV."
    )
    parser.add_argument("--config", required=True, help="Path to YAML config.")
    parser.add_argument("--input",  default=None,
                        help="Input CSV (default: sample5k from config).")
    parser.add_argument("--output", default=None,
                        help="Output CSV (default: overwrites input).")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )

    cfg = yaml.safe_load(Path(args.config).read_text())
    out_dir = PROJECT_ROOT / cfg["output"]["dir"]

    in_path  = Path(args.input)  if args.input  else out_dir / cfg["output"]["sample_filename"]
    out_path = Path(args.output) if args.output else in_path

    # ------------------------------------------------------------------
    # 1. Load existing CSV
    # ------------------------------------------------------------------
    log.info("Reading %s ...", in_path)
    df = pd.read_csv(in_path, parse_dates=["calendar_date"])
    df["calendar_date"] = pd.to_datetime(df["calendar_date"]).dt.normalize()
    df["hadm_id"] = df["hadm_id"].astype("int64")
    log.info("  %d rows, %d columns", len(df), len(df.columns))

    # Guard: skip if flags already present
    if "creatinine_measured" in df.columns:
        log.warning("Measured flags already present — nothing to do.")
        return

    hadm_ids = sorted(df["hadm_id"].unique().tolist())
    log.info("  %d unique admissions", len(hadm_ids))

    # ------------------------------------------------------------------
    # 2. Query measurement dates from labevents
    # ------------------------------------------------------------------
    log.info("Connecting to database ...")
    conn = _db_connect(cfg)
    hosp = cfg["schemas"]["hosp"]

    log.info("Querying %s.labevents for measurement presence ...", hosp)
    lab_dates = _query_lab_dates(conn, hosp, hadm_ids)
    conn.close()
    log.info("  %d (hadm_id, date, itemid) tuples retrieved", len(lab_dates))

    # ------------------------------------------------------------------
    # 3. Build measured flags
    # ------------------------------------------------------------------
    log.info("Building measured flags ...")

    # 3a. Direct columns (one itemid → one column)
    for itemid, col in DIRECT_ITEMID_TO_COL.items():
        flag = f"{col}_measured"
        hits = (
            lab_dates[lab_dates["itemid"] == itemid][["hadm_id", "lab_date"]]
            .drop_duplicates()
            .assign(**{flag: 1})
        )
        df = _left_merge_flag(df, hits, flag)
        log.info("  %-32s  %4.1f%% of days measured", flag, df[flag].mean() * 100)

    # 3b. nlr_measured — both lymphocytes (51244) AND neutrophils (51256) drawn
    nlr_hits = (
        lab_dates[lab_dates["itemid"].isin(NLR_ITEMIDS)]
        .groupby(["hadm_id", "lab_date"])["itemid"]
        .nunique()
        .reset_index()
        .query("itemid == 2")          # both itemids present on this day
        [["hadm_id", "lab_date"]]
        .assign(nlr_measured=1)
    )
    df = _left_merge_flag(df, nlr_hits, "nlr_measured")
    log.info("  %-32s  %4.1f%% of days measured", "nlr_measured", df["nlr_measured"].mean() * 100)

    # 3c. lactate_elevated_measured — lactate_raw (50813) drawn today
    lac_hits = (
        lab_dates[lab_dates["itemid"] == LACTATE_ITEMID][["hadm_id", "lab_date"]]
        .drop_duplicates()
        .assign(lactate_elevated_measured=1)
    )
    df = _left_merge_flag(df, lac_hits, "lactate_elevated_measured")
    log.info(
        "  %-32s  %4.1f%% of days measured",
        "lactate_elevated_measured",
        df["lactate_elevated_measured"].mean() * 100,
    )

    # ------------------------------------------------------------------
    # 4. Write output
    # ------------------------------------------------------------------
    new_cols = [c for c in df.columns if c.endswith("_measured")]
    log.info("Added %d measured-flag columns.", len(new_cols))
    log.info("Writing %s (%d columns total) ...", out_path, len(df.columns))
    df.to_csv(out_path, index=False)
    log.info("Done.")


if __name__ == "__main__":
    main()
