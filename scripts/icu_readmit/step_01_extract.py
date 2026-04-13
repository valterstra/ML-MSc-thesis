"""
Step 01 -- Extract raw data from PostgreSQL (local MIMIC-IV).

Adapted from scripts/sepsis/step_01_extract.py.  All query logic lives in
src/careai/icu_readmit/queries.py.

Key differences from sepsis step_01:
  - charlson_flags temp table (not elixhauser_flags) is created before demog() runs
  - QUERY_ORDER: removes abx/culture/microbio; adds drugs_mv (9 binary drug classes)
  - demog() returns extended column list (race, insurance, charlson, prior_ed, etc.)
  - Output directory: data/interim/icu_readmit/raw_data/

Usage:
    source ../.venv/Scripts/activate
    PGUSER=postgres PGPASSWORD=<pw> python scripts/icu_readmit/step_01_extract.py
    PGUSER=postgres PGPASSWORD=<pw> python scripts/icu_readmit/step_01_extract.py --skip-existing
    PGUSER=postgres PGPASSWORD=<pw> python scripts/icu_readmit/step_01_extract.py --smoke-test
"""

import os
import sys
import argparse
import logging
import pandas as pd
import psycopg2

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit import queries
from careai.icu_readmit.columns import STAY_ID_OPTIONAL_DTYPE_SPEC

# ---------------------------------------------------------------------------
# Canonical output column lists
# ---------------------------------------------------------------------------
RAW_DATA_COLUMNS = {
    "demog": [
        "icustayid", "hadm_id", "subject_id",
        "admittime", "dischtime", "intime", "outtime", "los", "adm_order",
        "age", "gender", "dod", "morta_hosp", "morta_90",
        "discharge_location", "admission_type", "admission_location",
        "insurance", "language", "marital_status", "race",
        "prior_ed_visits_6m",
        "charlson_score",
        "cc_mi", "cc_chf", "cc_pvd", "cc_cvd", "cc_dementia", "cc_copd",
        "cc_rheum", "cc_pud", "cc_mild_liver", "cc_dm_no_cc", "cc_dm_cc",
        "cc_paralysis", "cc_renal", "cc_malign", "cc_sev_liver",
        "cc_metastatic", "cc_hiv", "cc_hemiplegia",
        "drg_severity", "drg_mortality",
        "readmit_30d",
    ],
    "ce":          ["icustayid", "charttime", "itemid", "valuenum"],
    "labs_ce":     ["icustayid", "charttime", "itemid", "valuenum"],
    "labs_le":     ["icustayid", "charttime", "itemid", "valuenum"],
    "mechvent":    ["icustayid", "charttime", "mechvent", "extubated", "selfextubated"],
    "mechvent_pe": ["subject_id", "hadm_id", "icustayid", "starttime",
                    "endtime", "mechvent", "extubated", "selfextubated", "itemid", "value"],
    "preadm_fluid":["icustayid", "input_preadm"],
    "fluid_mv":    ["icustayid", "starttime", "endtime", "itemid", "amount", "rate", "tev"],
    "vaso_mv":     ["icustayid", "itemid", "starttime", "endtime", "ratestd"],
    "preadm_uo":   ["icustayid", "charttime", "itemid", "value", "datediff_minutes"],
    "uo":          ["icustayid", "charttime", "itemid", "value"],
    "drugs_mv":    ["icustayid", "starttime", "endtime", "itemid", "amount", "rate"],
}

# stay_id range for chartevents batching (MIMIC-IV: stay_ids in ~30M-40M range)
CE_ID_MIN  = 30_000_000
CE_ID_MAX  = 40_000_000
CE_ID_STEP = 1_000_000

SMOKE_LIMIT = 5_000


def get_connection():
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ.get("PGDATABASE", "mimic"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
    )


def run_query(conn, sql, columns, dtype_spec=None):
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    if dtype_spec:
        for col, dtype in dtype_spec.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
    return df


def create_charlson_temp_table(conn):
    """
    Materialise charlson_flags as a session-local temp table so that demog()
    can join it.  Analogous to create_elixhauser_temp_table in the sepsis pipeline.
    """
    logging.info("Creating charlson_flags temp table ...")
    charlson_sql = queries.charlson()
    create_sql = f"""
        CREATE TEMP TABLE charlson_flags AS
        {charlson_sql}
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS charlson_flags")
        cur.execute(create_sql)
    conn.commit()
    logging.info("charlson_flags temp table created.")


def load_data(conn, file_name, query_fn, output_dir,
              skip_if_present=False, smoke_test=False):
    """
    Run query_fn, save result to output_dir/<file_name>.csv.
    For 'ce': runs in stay_id batches, saves ce<min><max>.csv files.
    """
    columns = RAW_DATA_COLUMNS[file_name]

    if file_name == "ce":
        id_min  = CE_ID_MIN
        id_max  = CE_ID_MAX if not smoke_test else CE_ID_MIN + CE_ID_STEP
        id_step = CE_ID_STEP
        for i in range(id_min, id_max, id_step):
            batch_min = i
            batch_max = i + id_step
            out_path = os.path.join(output_dir, f"ce{batch_min}{batch_max}.csv")
            if skip_if_present and os.path.exists(out_path):
                logging.info(f"  ce {batch_min}-{batch_max}: exists, skipping")
                continue
            logging.info(f"  ce {batch_min}-{batch_max} ...")
            sql = query_fn(batch_min, batch_max)
            if smoke_test:
                sql = sql.rstrip().rstrip(";") + f" LIMIT {SMOKE_LIMIT}"
            df = run_query(conn, sql, columns, STAY_ID_OPTIONAL_DTYPE_SPEC)
            df.to_csv(out_path, index=False)
            logging.info(f"    -> {len(df)} rows -> {out_path}")
        return

    out_path = os.path.join(output_dir, f"{file_name}.csv")
    if skip_if_present and os.path.exists(out_path):
        logging.info(f"  {file_name}: exists, skipping")
        return

    logging.info(f"  {file_name} ...")
    sql = query_fn()
    if smoke_test:
        sql = sql.rstrip().rstrip(";") + f" LIMIT {SMOKE_LIMIT}"
    df = run_query(conn, sql, columns, STAY_ID_OPTIONAL_DTYPE_SPEC)
    df.to_csv(out_path, index=False)
    logging.info(f"    -> {len(df)} rows -> {out_path}")


# Order matches sepsis pipeline except:
#   - abx / culture / microbio removed (not needed for general ICU cohort)
#   - drugs_mv added (9 binary drug class inputevents)
#   - demog last (requires charlson_flags temp table)
QUERY_ORDER = [
    ("ce",           queries.ce),           # batched chartevents
    ("labs_ce",      queries.labs_ce),
    ("labs_le",      queries.labs_le),
    ("mechvent",     queries.mechvent),
    ("mechvent_pe",  queries.mechvent_pe),
    ("preadm_fluid", queries.preadm_fluid),
    ("fluid_mv",     queries.fluid_mv),
    ("vaso_mv",      queries.vaso_mv),
    ("preadm_uo",    queries.preadm_uo),
    ("uo",           queries.uo),
    ("drugs_mv",     queries.drugs_mv),     # NEW: 9 binary drug classes
    ("demog",        queries.demog),        # last: needs charlson_flags
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV data from PostgreSQL for the ICU readmission pipeline."
    )
    parser.add_argument(
        "--out", dest="output_dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "interim", "icu_readmit", "raw_data"
        ),
        help="Directory for output CSV files (default: data/interim/icu_readmit/raw_data/)",
    )
    parser.add_argument(
        "--skip-existing", dest="skip_existing", action="store_true", default=False,
        help="Skip files that already exist on disk.",
    )
    parser.add_argument(
        "--smoke-test", dest="smoke_test", action="store_true", default=False,
        help="Limit every query to 5000 rows for a quick end-to-end smoke test.",
    )
    parser.add_argument(
        "--only", dest="only", type=str, default=None,
        help="Run only this query (e.g. --only demog).",
    )
    args = parser.parse_args()

    log_level = logging.DEBUG if args.smoke_test else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
    )

    output_dir = os.path.normpath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Output directory: {output_dir}")
    if args.smoke_test:
        logging.info("SMOKE TEST MODE: queries limited to %d rows", SMOKE_LIMIT)

    conn = get_connection()
    try:
        # charlson_flags must exist before demog() runs
        create_charlson_temp_table(conn)

        for file_name, query_fn in QUERY_ORDER:
            if args.only and file_name != args.only:
                continue
            load_data(conn, file_name, query_fn, output_dir,
                      skip_if_present=args.skip_existing,
                      smoke_test=args.smoke_test)
    finally:
        conn.close()

    logging.info("Step 01 complete.")


if __name__ == "__main__":
    main()
