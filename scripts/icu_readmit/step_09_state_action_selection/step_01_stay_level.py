"""
Step 01 -- Build stay-level table for causal discovery.

PURPOSE
-------
Collapse ICUdataset.csv (4-hour blocs, ~1.5M rows) to one row per ICU stay
(~61,771 rows) for use in the state->readmission causal discovery track.

Two types of variables are extracted:

  State variables  -- last-bloc values of all time-varying measurements
                      (vitals, labs, derived scores, ventilation, I/O).
                      Prefixed with 'last_' to make the representation explicit.
                      "Last bloc" = the patient's physiological state at discharge.

  Static variables -- values that do not change per bloc (demographics,
                      comorbidities). Taken from any bloc (all identical).
                      Names unchanged (no prefix).

Also included:
  readmit_30d  -- 30-day readmission label (the causal target)
  num_blocs    -- stay length in 4-hour blocs (proxy for ICU LOS)

ACTION COLUMNS ARE EXCLUDED intentionally.
This table is for the state -> readmission causal question only.
Drug actions are not included at this stage.

This is step 1 of the active step-09 state/action selection track
(scripts/icu_readmit/step_09_state_action_selection/).

OUTPUTS
-------
  data/processed/icu_readmit/step_09_state_action_selection/stay_level.parquet
  reports/icu_readmit/step_09_state_action_selection/stay_level_summary.json

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_09_state_action_selection/step_01_stay_level.py
    python scripts/icu_readmit/step_09_state_action_selection/step_01_stay_level.py --smoke
    python scripts/icu_readmit/step_09_state_action_selection/step_01_stay_level.py \\
        --input data/processed/icu_readmit/ICUdataset.csv \\
        --out-dir data/processed/icu_readmit/step_09_state_action_selection
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_BLOC, C_READMIT_30D,
    # Static / demographic
    C_GENDER, C_AGE, C_WEIGHT,
    C_RACE, C_INSURANCE, C_MARITAL_STATUS,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_CHARLSON, C_RE_ADMISSION, C_PRIOR_ED_VISITS,
    C_DRG_SEVERITY, C_DRG_MORTALITY, C_DISCHARGE_DISPOSITION,
    CHARLSON_FLAG_COLS,
    # Time-varying: vitals / chart
    CHART_FIELD_NAMES,
    # Time-varying: labs
    LAB_FIELD_NAMES,
    # Time-varying: ventilation
    VENT_FIELD_NAMES,
    # Time-varying: derived scores
    COMPUTED_FIELD_NAMES,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column group definitions
# ---------------------------------------------------------------------------

STATIC_COLS = [
    C_GENDER, C_AGE, C_WEIGHT,
    C_RACE, C_INSURANCE, C_MARITAL_STATUS,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_CHARLSON, C_RE_ADMISSION, C_PRIOR_ED_VISITS,
    C_DRG_SEVERITY, C_DRG_MORTALITY, C_DISCHARGE_DISPOSITION,
] + CHARLSON_FLAG_COLS

# Time-varying state columns -- these get 'last_' prefix in output
# Include vitals, labs, derived scores, ventilation.
# Also include fluid I/O columns present in the dataset (renamed in step_08).
STATE_COLS_BASE = (
    CHART_FIELD_NAMES
    + LAB_FIELD_NAMES
    + list(VENT_FIELD_NAMES)
    + COMPUTED_FIELD_NAMES
    + [
        'input_4hourly_tev',   # renamed from input_step in step_08
        'output_4hourly',      # renamed from output_step in step_08
        'input_total',
        'output_total',
        'cumulated_balance',
    ]
)

# Deduplicate while preserving order
_seen: set = set()
STATE_COLS_BASE_DEDUP: list[str] = []
for _c in STATE_COLS_BASE:
    if _c not in _seen:
        _seen.add(_c)
        STATE_COLS_BASE_DEDUP.append(_c)
STATE_COLS_BASE = STATE_COLS_BASE_DEDUP


def build_stay_level(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse bloc-level DataFrame to one row per stay.

    Static columns  -> .first()  (same value in every bloc)
    State columns   -> .last()   (discharge-state value, last observed bloc)
    readmit_30d     -> .first()  (broadcast label, same in every bloc)
    num_blocs       -> .count()  on bloc column
    """
    df = df.sort_values([C_ICUSTAYID, C_BLOC])

    # Filter to columns that actually exist in this dataset
    present_static = [c for c in STATIC_COLS if c in df.columns]
    present_state  = [c for c in STATE_COLS_BASE if c in df.columns]
    missing_static = [c for c in STATIC_COLS if c not in df.columns]
    missing_state  = [c for c in STATE_COLS_BASE if c not in df.columns]

    if missing_static:
        log.warning("Static columns not found in dataset (skipped): %s", missing_static)
    if missing_state:
        log.warning("State columns not found in dataset (skipped): %s", missing_state)

    grp = df.groupby(C_ICUSTAYID)

    log.info("Extracting static columns (first bloc)...")
    static_df = grp[present_static + [C_READMIT_30D]].first()

    log.info("Extracting state columns (last bloc)...")
    state_df = grp[present_state].last()
    # Rename with 'last_' prefix
    state_df.columns = ['last_' + c for c in state_df.columns]

    log.info("Counting blocs per stay...")
    num_blocs = grp[C_BLOC].count().rename('num_blocs')

    stay_level = pd.concat([static_df, state_df, num_blocs], axis=1).reset_index()
    return stay_level


def summarise(stay_df: pd.DataFrame) -> dict:
    n_stays = len(stay_df)
    readmit_rate = float(stay_df[C_READMIT_30D].mean())
    n_state_cols = sum(1 for c in stay_df.columns if c.startswith('last_'))
    n_static_cols = len([c for c in STATIC_COLS if c in stay_df.columns])

    # Coverage: fraction non-NaN per last_ column
    state_cols = [c for c in stay_df.columns if c.startswith('last_')]
    coverage = (stay_df[state_cols].notna().mean() * 100).round(1).to_dict()
    low_coverage = {k: v for k, v in coverage.items() if v < 50.0}

    return {
        'n_stays':           n_stays,
        'readmit_rate_pct':  round(readmit_rate * 100, 2),
        'n_state_cols':      n_state_cols,
        'n_static_cols':     n_static_cols,
        'total_cols':        len(stay_df.columns),
        'num_blocs_median':  float(stay_df['num_blocs'].median()),
        'num_blocs_p25':     float(stay_df['num_blocs'].quantile(0.25)),
        'num_blocs_p75':     float(stay_df['num_blocs'].quantile(0.75)),
        'low_coverage_state_cols': low_coverage,    # < 50% non-NaN
    }


def main(args: argparse.Namespace) -> None:
    t0 = time.time()

    input_path = Path(args.input)
    out_dir    = Path(args.out_dir)
    report_dir = Path(args.report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading %s ...", input_path)
    df = pd.read_csv(input_path, low_memory=False)
    log.info("Loaded: %d rows, %d columns", len(df), len(df.columns))

    if args.smoke:
        # Take first 5000 unique stays
        keep_stays = df[C_ICUSTAYID].unique()[:5000]
        df = df[df[C_ICUSTAYID].isin(keep_stays)].copy()
        log.info("Smoke test: restricted to %d stays (%d rows)", len(keep_stays), len(df))

    log.info("Building stay-level table...")
    stay_df = build_stay_level(df)
    log.info("Stay-level table: %d stays, %d columns", len(stay_df), len(stay_df.columns))

    # Output
    out_path = out_dir / 'stay_level.parquet'
    stay_df.to_parquet(out_path, index=False)
    log.info("Saved: %s", out_path)

    # Summary report
    summary = summarise(stay_df)
    summary['input_file'] = str(input_path)
    summary['output_file'] = str(out_path)
    summary['runtime_s'] = round(time.time() - t0, 1)

    report_path = report_dir / 'stay_level_summary.json'
    with open(report_path, 'w') as f:
        json.dump(summary, f, indent=2)
    log.info("Report saved: %s", report_path)

    log.info("Done in %.1fs", time.time() - t0)
    log.info("  Stays:              %d", summary['n_stays'])
    log.info("  Readmission rate:   %.1f%%", summary['readmit_rate_pct'])
    log.info("  State cols (last_): %d", summary['n_state_cols'])
    log.info("  Static cols:        %d", summary['n_static_cols'])
    if summary['low_coverage_state_cols']:
        log.info("  Low-coverage state cols (<50%% non-NaN): %d",
                 len(summary['low_coverage_state_cols']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        '--input', default=str(
            PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'ICUdataset.csv'),
        help='Path to ICUdataset.csv (step 08 output)')
    parser.add_argument(
        '--out-dir', default=str(
            PROJECT_ROOT / 'data' / 'processed' / 'icu_readmit' / 'step_09_state_action_selection'),
        help='Directory for stay_level.parquet output')
    parser.add_argument(
        '--report-dir', default=str(
            PROJECT_ROOT / 'reports' / 'icu_readmit' / 'step_09_state_action_selection'),
        help='Directory for summary JSON report')
    parser.add_argument(
        '--smoke', action='store_true',
        help='Smoke test: use first 5000 stays only')
    args = parser.parse_args()
    main(args)
