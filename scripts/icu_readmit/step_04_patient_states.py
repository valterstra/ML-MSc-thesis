"""
Step 04 -- Build patient states (unbinned).

Adapted from scripts/sepsis/step_04_patient_states.py.

Key differences from sepsis step_04:
  - Anchor: ICU intime (not sepsis onset). Observes entire ICU stay from
    intime to outtime (no fixed window).
  - Input: icu_cohort.csv (not sepsis_onset.csv)
  - No age-18 minimum (user confirmed: include all ages)
  - Output qstime equivalent: icu_stay_bounds.csv
    (columns: icustayid, intime, outtime, first_timestep, last_timestep, dischtime)
  - CHART_FIELD_NAMES / LAB_FIELD_NAMES from icu_readmit.columns (extended set)

Performance: vectorized pivot approach (replaces per-patient iterrows loop).
  Original: ~2.5 hours for 62k patients.
  Vectorized: ~10-20 minutes.

  Core idea: instead of iterating patient-by-patient with iterrows(), we:
    1. Pre-filter all events to cohort stay_ids + ICU time windows (bulk merge)
    2. Map itemids to field names (map())
    3. pivot_table(index=[icustayid,charttime], columns=field, values=valuenum)
    4. Outer-merge ce_wide + labs_wide + mechvent

  --resume still supported: if patient_states.csv already exists, stays
  already written are skipped before the pivot.

Inputs:
  data/interim/icu_readmit/raw_data/    -- ce*.csv
  data/interim/icu_readmit/intermediates/
    labs_ce.csv, labs_le.csv, mechvent.csv, mechvent_pe.csv,
    demog.csv, icu_cohort.csv

Outputs:
  <output_dir>/patient_states.csv
  <output_dir>/icu_stay_bounds.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_04_patient_states.py \\
        data/interim/icu_readmit/intermediates/patient_states \\
        2>&1 | python -c "import sys; d=sys.stdin.read(); open('logs/step_04_icu_readmit_full.log','w').write(d); print(d)"

    # Resume after crash
    python scripts/icu_readmit/step_04_patient_states.py \\
        data/interim/icu_readmit/intermediates/patient_states --resume
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_CHARTTIME, C_STARTTIME, C_TIMESTEP, C_BLOC,
    C_ITEMID, C_VALUENUM,
    C_INTIME, C_OUTTIME, C_DISCHTIME,
    C_FIRST_TIMESTEP, C_LAST_TIMESTEP,
    C_MECHVENT, C_EXTUBATED,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES,
    REF_VITALS, REF_LABS,
)
from careai.icu_readmit.utils import load_csv, load_intermediate_or_raw_csv

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
BOUNDS_FILE = 'icu_stay_bounds.csv'


def build_patient_states_vectorized(ce_dfs, cohort, labU, MV, MV_procedure, out_dir, resume):
    """
    Vectorized replacement for the per-patient iterrows loop.

    Steps:
      1. Filter cohort to stays with valid time windows
      2. Bulk-filter ce/labs/mechvent to cohort stays + windows
      3. Map itemids -> field names
      4. pivot_table to wide format
      5. Outer-merge ce_wide + labs_wide + mechvent
      6. Add bloc numbers, build icu_stay_bounds
    """
    states_path = os.path.join(out_dir, 'patient_states.csv')
    bounds_path = os.path.join(out_dir, BOUNDS_FILE)

    # --- Resume: find already-processed stays ---
    done_ids = set()
    if resume and os.path.exists(states_path):
        try:
            partial = pd.read_csv(states_path, usecols=[C_ICUSTAYID])
            done_ids = set(partial[C_ICUSTAYID].dropna().astype(int).unique())
            logging.info("Resume: %d stays already processed, skipping them", len(done_ids))
        except Exception as e:
            logging.warning("Could not read partial output for resume: %s", e)

    # --- Valid cohort windows ---
    cohort_work = cohort.dropna(subset=[C_ICUSTAYID, C_INTIME, C_OUTTIME]).copy()
    cohort_work = cohort_work[cohort_work[C_OUTTIME] > cohort_work[C_INTIME]]
    if done_ids:
        cohort_work = cohort_work[~cohort_work[C_ICUSTAYID].isin(done_ids)]
    cohort_ids = set(cohort_work[C_ICUSTAYID].astype(int))
    logging.info("%d stays to process (%d already done)", len(cohort_ids), len(done_ids))

    windows = cohort_work[[C_ICUSTAYID, C_INTIME, C_OUTTIME, C_DISCHTIME]].copy()
    windows[C_ICUSTAYID] = windows[C_ICUSTAYID].astype(int)

    # --- itemid -> field name maps ---
    # CE: REF_VITALS has len(REF_VITALS)=38 groups -> itemids 1..38
    ce_iid_map = {i + 1: CHART_FIELD_NAMES[i] for i in range(len(REF_VITALS))}
    # Labs: REF_LABS has 45 groups -> itemids 1..45
    lab_iid_map = {i + 1: LAB_FIELD_NAMES[i] for i in range(len(REF_LABS))}

    # ---------------------------------------------------------------
    # 1. Chartevents
    # ---------------------------------------------------------------
    logging.info("Processing chartevents ...")
    ce_parts = []
    for df in ce_dfs:
        sub = df[df[C_ICUSTAYID].isin(cohort_ids)][[C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM]]
        ce_parts.append(sub)
    all_ce = pd.concat(ce_parts, ignore_index=True)
    logging.info("  %d ce rows for cohort stays (before window filter)", len(all_ce))

    all_ce = all_ce.merge(windows[[C_ICUSTAYID, C_INTIME, C_OUTTIME]], on=C_ICUSTAYID, how='inner')
    all_ce = all_ce[
        (all_ce[C_CHARTTIME] >= all_ce[C_INTIME]) &
        (all_ce[C_CHARTTIME] <= all_ce[C_OUTTIME])
    ][[C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM]]
    logging.info("  %d ce rows after window filter", len(all_ce))

    all_ce[C_ITEMID] = pd.to_numeric(all_ce[C_ITEMID], errors='coerce')
    all_ce['_field'] = all_ce[C_ITEMID].map(ce_iid_map)
    all_ce = all_ce.dropna(subset=['_field', C_VALUENUM])

    logging.info("  Pivoting ce to wide format ...")
    ce_wide = all_ce.pivot_table(
        index=[C_ICUSTAYID, C_CHARTTIME],
        columns='_field',
        values=C_VALUENUM,
        aggfunc='first',
    )
    ce_wide.columns.name = None
    ce_wide = ce_wide.reset_index()
    logging.info("  ce_wide: %d rows x %d cols", len(ce_wide), len(ce_wide.columns))

    # ---------------------------------------------------------------
    # 2. Labs
    # ---------------------------------------------------------------
    logging.info("Processing labs ...")
    labs = labU[labU[C_ICUSTAYID].isin(cohort_ids)][[C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM]].copy()
    labs = labs.merge(windows[[C_ICUSTAYID, C_INTIME, C_OUTTIME]], on=C_ICUSTAYID, how='inner')
    labs = labs[
        (labs[C_CHARTTIME] >= labs[C_INTIME]) &
        (labs[C_CHARTTIME] <= labs[C_OUTTIME])
    ][[C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM]]
    logging.info("  %d lab rows after window filter", len(labs))

    labs[C_ITEMID] = pd.to_numeric(labs[C_ITEMID], errors='coerce')
    labs['_field'] = labs[C_ITEMID].map(lab_iid_map)
    labs = labs.dropna(subset=['_field', C_VALUENUM])

    logging.info("  Pivoting labs to wide format ...")
    labs_wide = labs.pivot_table(
        index=[C_ICUSTAYID, C_CHARTTIME],
        columns='_field',
        values=C_VALUENUM,
        aggfunc='first',
    )
    labs_wide.columns.name = None
    labs_wide = labs_wide.reset_index()
    logging.info("  labs_wide: %d rows x %d cols", len(labs_wide), len(labs_wide.columns))

    # ---------------------------------------------------------------
    # 3. Mechvent
    # ---------------------------------------------------------------
    logging.info("Processing mechvent ...")
    mv = MV[MV[C_ICUSTAYID].isin(cohort_ids)][[C_ICUSTAYID, C_CHARTTIME, C_MECHVENT, C_EXTUBATED]].copy()
    mv = mv.merge(windows[[C_ICUSTAYID, C_INTIME, C_OUTTIME]], on=C_ICUSTAYID, how='inner')
    mv = mv[
        (mv[C_CHARTTIME] >= mv[C_INTIME]) &
        (mv[C_CHARTTIME] <= mv[C_OUTTIME])
    ][[C_ICUSTAYID, C_CHARTTIME, C_MECHVENT, C_EXTUBATED]]
    logging.info("  %d mechvent rows", len(mv))

    from_pe = 0
    if MV_procedure is not None:
        pe = MV_procedure[MV_procedure[C_ICUSTAYID].isin(cohort_ids)].copy()
        pe = pe.merge(windows[[C_ICUSTAYID, C_INTIME, C_OUTTIME]], on=C_ICUSTAYID, how='inner')
        pe = pe[
            (pe[C_STARTTIME] >= pe[C_INTIME]) &
            (pe[C_STARTTIME] <= pe[C_OUTTIME])
        ]
        pe_agg = pe.groupby([C_ICUSTAYID, C_STARTTIME]).agg(
            **{C_MECHVENT: (C_MECHVENT, 'max'), C_EXTUBATED: (C_EXTUBATED, 'max')}
        ).reset_index().rename(columns={C_STARTTIME: C_CHARTTIME})
        from_pe = len(pe_agg)
        # Combine: prefer regular mechvent over pe (drop pe entries where mv already has data)
        existing_keys = set(zip(mv[C_ICUSTAYID], mv[C_CHARTTIME]))
        pe_new = pe_agg[~pe_agg.apply(
            lambda r: (r[C_ICUSTAYID], r[C_CHARTTIME]) in existing_keys, axis=1
        )]
        mv = pd.concat([mv, pe_new[[C_ICUSTAYID, C_CHARTTIME, C_MECHVENT, C_EXTUBATED]]], ignore_index=True)
        logging.info("  Added %d mechvent_pe rows", from_pe)

    # ---------------------------------------------------------------
    # 4. Merge all wide tables
    # ---------------------------------------------------------------
    logging.info("Merging ce + labs + mechvent ...")
    combined = ce_wide.merge(labs_wide, on=[C_ICUSTAYID, C_CHARTTIME], how='outer')
    combined = combined.merge(mv,        on=[C_ICUSTAYID, C_CHARTTIME], how='outer')
    logging.info("  Combined: %d rows x %d cols", len(combined), len(combined.columns))

    # Drop rows with no icustayid (shouldn't happen but safety check)
    combined = combined.dropna(subset=[C_ICUSTAYID])
    combined[C_ICUSTAYID] = combined[C_ICUSTAYID].astype(int)

    # ---------------------------------------------------------------
    # 5. Sort, add timestep + bloc columns
    # ---------------------------------------------------------------
    combined = combined.sort_values([C_ICUSTAYID, C_CHARTTIME]).reset_index(drop=True)
    combined[C_TIMESTEP] = combined[C_CHARTTIME]
    combined[C_BLOC]     = combined.groupby(C_ICUSTAYID).cumcount()

    # Reorder: bloc, icustayid, timestep first
    front = [C_BLOC, C_ICUSTAYID, C_TIMESTEP]
    rest  = [c for c in combined.columns if c not in front + [C_CHARTTIME]]
    combined = combined[front + rest]

    # ---------------------------------------------------------------
    # 6. Build icu_stay_bounds
    # ---------------------------------------------------------------
    ts_bounds = combined.groupby(C_ICUSTAYID)[C_TIMESTEP].agg(
        first_timestep='min', last_timestep='max'
    ).reset_index()
    ts_bounds.columns = [C_ICUSTAYID, C_FIRST_TIMESTEP, C_LAST_TIMESTEP]
    bounds = ts_bounds.merge(windows[[C_ICUSTAYID, C_INTIME, C_OUTTIME, C_DISCHTIME]], on=C_ICUSTAYID, how='left')
    bounds = bounds[[C_ICUSTAYID, C_INTIME, C_OUTTIME, C_FIRST_TIMESTEP, C_LAST_TIMESTEP, C_DISCHTIME]]

    # ---------------------------------------------------------------
    # 7. If resuming, append to existing files; otherwise write fresh
    # ---------------------------------------------------------------
    mode   = 'a' if done_ids else 'w'
    header = not bool(done_ids)
    combined.to_csv(states_path, mode=mode, index=False, header=header, float_format='%g')
    bounds.to_csv(bounds_path,   mode=mode, index=False, header=header, float_format='%g')
    logging.info("Written %d rows -> %s", len(combined), states_path)
    logging.info("Written %d bounds rows -> %s", len(bounds), bounds_path)
    logging.info("mechvent_pe rows added: %d", from_pe)

    return len(combined)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Generates patient_states.csv and icu_stay_bounds.csv (vectorized, unbinned).'))
    parser.add_argument('output_dir', type=str,
                        help='Output directory (e.g. data/interim/icu_readmit/intermediates/patient_states)')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Data parent directory (default: data/interim/icu_readmit/)')
    parser.add_argument('--head', dest='head', type=int, default=None,
                        help='Limit to first N cohort rows (smoke test)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Skip stays already present in the partial output file')
    parser.add_argument('--log', dest='log_file', type=str, default=None,
                        help='Log file (default: logs/step_04_icu_readmit.log)')

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(REPO_DIR, 'data', 'interim', 'icu_readmit')
    out_dir  = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_04_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.info("Step 04 started (vectorized). output_dir=%s resume=%s", out_dir, args.resume)

    raw_dir = os.path.join(data_dir, 'raw_data')
    interm  = os.path.join(data_dir, 'intermediates')

    logging.info("Reading chartevents ...")
    ce_files = sorted(set(
        [p for p in os.listdir(raw_dir) if p.startswith("ce") and p.endswith(".csv")] +
        [p for p in os.listdir(interm)  if p.startswith("ce") and p.endswith(".csv")]
    ))
    logging.info("  Found %d chartevents files", len(ce_files))
    ce_dfs = [load_intermediate_or_raw_csv(data_dir, p) for p in ce_files]

    logging.info("Reading cohort ...")
    cohort = load_csv(os.path.join(interm, 'icu_cohort.csv'), null_icustayid=True)
    logging.info("  %d ICU stays in cohort", len(cohort))

    logging.info("Reading labs ...")
    labU = pd.concat([
        load_intermediate_or_raw_csv(data_dir, 'labs_ce.csv'),
        load_intermediate_or_raw_csv(data_dir, 'labs_le.csv'),
    ], ignore_index=True)
    logging.info("  %d lab rows total", len(labU))

    logging.info("Reading mechvent ...")
    MV = load_intermediate_or_raw_csv(data_dir, 'mechvent.csv')
    try:
        MV_procedure = load_intermediate_or_raw_csv(data_dir, 'mechvent_pe.csv')
        logging.info("  mechvent_pe loaded: %d rows", len(MV_procedure))
    except FileNotFoundError:
        MV_procedure = None
        logging.info("  mechvent_pe not found, skipping")

    if args.head:
        cohort = cohort.head(args.head)
        logging.info("Head mode: processing first %d cohort records", len(cohort))

    total = build_patient_states_vectorized(
        ce_dfs, cohort, labU, MV, MV_procedure,
        out_dir, args.resume,
    )

    # Verify column coverage
    states_path = os.path.join(out_dir, 'patient_states.csv')
    logging.info("Verifying column coverage ...")
    df_check = pd.read_csv(states_path, nrows=0)
    missing = [c for c in CHART_FIELD_NAMES + LAB_FIELD_NAMES if c not in df_check.columns]
    for col in missing:
        logging.warning("Column '%s' absent from output (no data for this feature)", col)

    logging.info("Step 04 complete. Total rows written: %d", total)
