"""
Step 01 — Extract raw data from PostgreSQL (local MIMIC-IV).

Faithful adaptation of ai_clinician/data_extraction/extract.py, replacing
BigQuery client with psycopg2.  All query logic lives in
src/careai/sepsis/queries.py (unchanged from the BigQuery originals aside
from SQL-dialect translation).

Key design decisions (matching AI-Clinician):
  - Elixhauser flags are computed first and held in a PostgreSQL temp table
    for the duration of the connection, then joined inside demog().
  - chartevents (ce) are extracted in stay_id batches of 1e6 each to stay
    within memory limits (MIMIC-IV stay_ids cluster around 3e7-4e7).
  - Every result is renamed to the canonical column list from RAW_DATA_COLUMNS
    (matching ai_clinician/preprocessing/columns.py) before saving.
  - --skip-existing skips already-written CSV files (safe to re-run).

Usage:
    source ../.venv/Scripts/activate
    PGUSER=postgres PGPASSWORD=<pw> python scripts/sepsis/step_01_extract.py
    PGUSER=postgres PGPASSWORD=<pw> python scripts/sepsis/step_01_extract.py --skip-existing
    PGUSER=postgres PGPASSWORD=<pw> python scripts/sepsis/step_01_extract.py --smoke-test
"""

import os
import sys
import argparse
import logging
import pandas as pd
import psycopg2

# Ensure src/ is importable when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis import queries
from careai.sepsis.columns import STAY_ID_OPTIONAL_DTYPE_SPEC

# ---------------------------------------------------------------------------
# Canonical output column lists (mirrors AI-Clinician RAW_DATA_COLUMNS)
# ---------------------------------------------------------------------------
RAW_DATA_COLUMNS = {
    "abx":         ["hadm_id", "icustayid", "startdate", "enddate", "gsn", "ndc",
                    "dose_val_rx", "dose_unit_rx", "route"],
    "culture":     ["subject_id", "hadm_id", "icustayid", "charttime", "itemid"],
    "microbio":    ["subject_id", "hadm_id", "icustayid", "charttime", "chartdate",
                    "org_itemid", "spec_itemid", "ab_itemid", "interpretation"],
    "demog":       ["subject_id", "hadm_id", "icustayid", "admittime", "dischtime",
                    "adm_order", "unit", "intime", "outtime", "los",
                    "age", "dob", "dod", "expire_flag", "gender",
                    "morta_hosp", "morta_90", "elixhauser", "readmit_30d"],
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
}

# stay_id range for chartevents batching (MIMIC-IV: stay_ids in ~30M-40M range)
CE_ID_MIN = 30_000_000
CE_ID_MAX = 40_000_000
CE_ID_STEP = 1_000_000

# Smoke-test limit (rows read via LIMIT in SQL)
SMOKE_LIMIT = 5_000


def get_connection():
    """Open a psycopg2 connection using env vars for credentials."""
    return psycopg2.connect(
        host=os.environ.get("PGHOST", "localhost"),
        port=int(os.environ.get("PGPORT", 5432)),
        dbname=os.environ.get("PGDATABASE", "mimic"),
        user=os.environ.get("PGUSER", "postgres"),
        password=os.environ.get("PGPASSWORD", ""),
    )


def run_query(conn, sql, columns, dtype_spec=None):
    """Execute SQL and return a DataFrame with canonical column names."""
    with conn.cursor() as cur:
        cur.execute(sql)
        rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=columns)
    if dtype_spec:
        for col, dtype in dtype_spec.items():
            if col in df.columns:
                df[col] = df[col].astype(dtype)
    return df


def create_elixhauser_temp_table(conn):
    """
    Compute Elixhauser flags and store them in a session-local temp table.
    This mirrors AI-Clinician's generate_elixhauser_if_needed(), but for
    PostgreSQL: we simply run the CTE-based SQL from queries.elixhauser()
    and materialise the result as a TEMP TABLE.
    """
    logging.info("Creating elixhauser_flags temp table ...")
    elix_sql = queries.elixhauser()
    create_sql = f"""
        CREATE TEMP TABLE elixhauser_flags AS
        {elix_sql}
    """
    with conn.cursor() as cur:
        cur.execute("DROP TABLE IF EXISTS elixhauser_flags")
        cur.execute(create_sql)
    conn.commit()
    logging.info("elixhauser_flags temp table created.")


def load_data(conn, file_name, query_fn, output_dir, skip_if_present=False,
              smoke_test=False):
    """
    Run query_fn, save result to output_dir/<file_name>.csv.
    For 'ce' (chartevents): runs in stay_id batches, saves multiple files
    named ce<min><max>.csv (matching AI-Clinician naming).
    """
    columns = RAW_DATA_COLUMNS[file_name]

    if file_name == "ce":
        id_min = CE_ID_MIN
        id_max = CE_ID_MAX if not smoke_test else CE_ID_MIN + CE_ID_STEP
        id_step = CE_ID_STEP
        for i in range(id_min, id_max, id_step):
            batch_min = i
            batch_max = i + id_step
            out_path = os.path.join(output_dir, f"ce{batch_min}{batch_max}.csv")
            if skip_if_present and os.path.exists(out_path):
                logging.info(f"  ce {batch_min}-{batch_max}: file exists, skipping")
                continue
            logging.info(f"  ce {batch_min}-{batch_max} ...")
            sql = query_fn(batch_min, batch_max)
            if smoke_test:
                sql = sql.rstrip().rstrip(";") + f" LIMIT {SMOKE_LIMIT}"
            df = run_query(conn, sql, columns, STAY_ID_OPTIONAL_DTYPE_SPEC)
            df.to_csv(out_path, index=False)
            logging.info(f"    -> {len(df)} rows saved to {out_path}")
        return

    out_path = os.path.join(output_dir, f"{file_name}.csv")
    if skip_if_present and os.path.exists(out_path):
        logging.info(f"  {file_name}: file exists, skipping")
        return

    logging.info(f"  {file_name} ...")
    sql = query_fn()
    if smoke_test:
        sql = sql.rstrip().rstrip(";") + f" LIMIT {SMOKE_LIMIT}"
    df = run_query(conn, sql, columns, STAY_ID_OPTIONAL_DTYPE_SPEC)
    df.to_csv(out_path, index=False)
    logging.info(f"    -> {len(df)} rows saved to {out_path}")


# Ordered list matching AI-Clinician extract.py (demog last, needs elix table)
QUERY_ORDER = [
    ("abx",          queries.abx),
    ("culture",      queries.culture),
    ("microbio",     queries.microbio),
    ("ce",           queries.ce),
    ("labs_ce",      queries.labs_ce),
    ("labs_le",      queries.labs_le),
    ("mechvent",     queries.mechvent),
    ("mechvent_pe",  queries.mechvent_pe),
    ("preadm_fluid", queries.preadm_fluid),
    ("fluid_mv",     queries.fluid_mv),
    ("vaso_mv",      queries.vaso_mv),
    ("preadm_uo",    queries.preadm_uo),
    ("uo",           queries.uo),
    ("demog",        queries.demog),   # last: needs elixhauser_flags temp table
]


def main():
    parser = argparse.ArgumentParser(
        description="Extract MIMIC-IV data from PostgreSQL for the sepsis pipeline."
    )
    parser.add_argument(
        "--out", dest="output_dir",
        default=os.path.join(
            os.path.dirname(__file__), "..", "..", "data", "interim", "sepsis", "raw_data"
        ),
        help="Directory for output CSV files (default: data/interim/sepsis/raw_data/)",
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
        # Elixhauser temp table must exist before demog() runs
        create_elixhauser_temp_table(conn)

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
