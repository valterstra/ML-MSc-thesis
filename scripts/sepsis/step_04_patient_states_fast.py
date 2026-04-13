"""
Step 04 — Build patient states (unbinned). OPTIMIZED VERSION.

Same logic as step_04_patient_states.py but pre-groups all DataFrames by
icustayid before the patient loop. This converts per-patient full-table scans
(O(n_patients * n_rows)) into O(1) dictionary lookups, giving ~20-50x speedup.

All other logic is identical to the original.
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis.columns import (
    C_ICUSTAYID, C_CHARTTIME, C_STARTTIME, C_TIMESTEP, C_BLOC,
    C_ITEMID, C_VALUENUM, C_AGE, C_DISCHTIME,
    C_ONSET_TIME, C_FIRST_TIMESTEP, C_LAST_TIMESTEP,
    C_MECHVENT, C_EXTUBATED,
    CHART_FIELD_NAMES, LAB_FIELD_NAMES,
)
from careai.sepsis.utils import load_csv, load_intermediate_or_raw_csv

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CHECKPOINT_EVERY = 200

# Fixed column order for patient_states.csv — prevents checkpoint column-scrambling.
# pd.DataFrame(list_of_dicts) orders columns by first key appearance, which varies
# across checkpoint batches. Appending with header=False then writes data in a different
# column order than the header. Reindexing to this fixed list before writing fixes that.
_STATES_COLUMNS = (
    [C_BLOC, C_ICUSTAYID, C_TIMESTEP]
    + CHART_FIELD_NAMES
    + LAB_FIELD_NAMES
    + [C_MECHVENT, C_EXTUBATED]
)
_QSTIME_COLUMNS = [C_ICUSTAYID, C_ONSET_TIME, C_FIRST_TIMESTEP, C_LAST_TIMESTEP, C_DISCHTIME]

_EMPTY_CE  = pd.DataFrame(columns=[C_ICUSTAYID, C_CHARTTIME, C_ITEMID, C_VALUENUM])
_EMPTY_MV  = pd.DataFrame(columns=[C_ICUSTAYID, C_CHARTTIME, C_MECHVENT, C_EXTUBATED])
_EMPTY_MVE = pd.DataFrame(columns=[C_ICUSTAYID, C_STARTTIME, C_MECHVENT, C_EXTUBATED])


class ChartEventsFast:
    """
    Pre-groups each CE batch DataFrame by icustayid so fetch() is O(1).
    """
    def __init__(self, ce_dfs, stay_id_col=C_ICUSTAYID):
        self.stay_id_col = stay_id_col
        self.ranges  = [(df[stay_id_col].min(), df[stay_id_col].max()) for df in ce_dfs]
        self.groups  = [df.groupby(stay_id_col) for df in ce_dfs]

    def fetch(self, stay_id):
        results = []
        for grp, (min_id, max_id) in zip(self.groups, self.ranges):
            if min_id <= stay_id <= max_id:
                try:
                    results.append(grp.get_group(stay_id))
                except KeyError:
                    pass
        if results:
            return pd.concat(results)
        return _EMPTY_CE.copy()


def time_window(df, col, center_time, lower_window, upper_window):
    return df[(df[col] >= center_time - lower_window) & (df[col] <= center_time + upper_window)]


def flush_checkpoint(combined_data, infection_times, out_dir, first_flush):
    if not combined_data:
        return
    mode   = 'w' if first_flush else 'a'
    header = first_flush
    states_path = os.path.join(out_dir, 'patient_states.csv')
    qstime_path = os.path.join(out_dir, 'qstime.csv')
    pd.DataFrame(combined_data).reindex(columns=_STATES_COLUMNS).to_csv(
        states_path, mode=mode, index=False, header=header, float_format='%g')
    pd.DataFrame(infection_times).reindex(columns=_QSTIME_COLUMNS).to_csv(
        qstime_path, mode=mode, index=False, header=header, float_format='%g')
    logging.info("Checkpoint: flushed %d rows to %s", len(combined_data), states_path)


def build_patient_states(chart_events, onset_data, demog, labU, MV, MV_procedure,
                          winb4, winaft, out_dir, checkpoint_every, resume):
    states_path = os.path.join(out_dir, 'patient_states.csv')
    qstime_path = os.path.join(out_dir, 'qstime.csv')

    # Resume: skip stays already in output
    done_ids = set()
    if resume and os.path.exists(states_path):
        try:
            partial  = pd.read_csv(states_path, usecols=[C_ICUSTAYID])
            done_ids = set(partial[C_ICUSTAYID].dropna().astype(int).unique())
            logging.info("Resume: %d stays already processed, skipping them", len(done_ids))
        except Exception as e:
            logging.warning("Could not read partial output for resume: %s", e)

    remaining = onset_data[~onset_data[C_ICUSTAYID].isin(done_ids)]
    logging.info("%d patients to process (%d already done)", len(remaining), len(done_ids))

    # Pre-group by icustayid for O(1) per-patient lookups
    logging.info("Pre-grouping labs by icustayid...")
    lab_groups = labU.groupby(C_ICUSTAYID)
    logging.info("Pre-grouping mechvent by icustayid...")
    mv_groups  = MV.groupby(C_ICUSTAYID)
    mv_pe_groups = MV_procedure.groupby(C_ICUSTAYID) if MV_procedure is not None else None
    logging.info("Pre-grouping demog by icustayid...")
    demog_groups = demog.groupby(C_ICUSTAYID)
    logging.info("Pre-grouping done. Starting patient loop.")

    combined_data      = []
    infection_times    = []
    from_pe            = 0
    total_rows_written = 0
    first_flush        = not bool(done_ids)

    for idx, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining),
                                        desc='Building patient states')):
        qst       = row[C_ONSET_TIME]
        icustayid = int(row[C_ICUSTAYID])
        assert qst > 0

        try:
            d1 = demog_groups.get_group(icustayid)[[C_AGE, C_DISCHTIME]].values.tolist()
        except KeyError:
            d1 = []
        if not d1 or d1[0][0] < 18:
            continue

        bounds = ((winb4 + 4) * 3600, (winaft + 4) * 3600)

        try:
            lab_df = lab_groups.get_group(icustayid)
        except KeyError:
            lab_df = _EMPTY_CE

        try:
            mv_df = mv_groups.get_group(icustayid)
        except KeyError:
            mv_df = _EMPTY_MV

        mv_pe_df = None
        if mv_pe_groups is not None:
            try:
                mv_pe_df = mv_pe_groups.get_group(icustayid)
            except KeyError:
                mv_pe_df = _EMPTY_MVE

        temp  = time_window(chart_events.fetch(icustayid), C_CHARTTIME, qst, *bounds)
        temp2 = time_window(lab_df,   C_CHARTTIME, qst, *bounds)
        temp3 = time_window(mv_df,    C_CHARTTIME, qst, *bounds)
        temp4 = None
        if mv_pe_df is not None:
            temp4 = time_window(mv_pe_df, C_STARTTIME, qst, *bounds)

        timesteps = sorted(pd.unique(pd.concat(
            [temp[C_CHARTTIME], temp2[C_CHARTTIME], temp3[C_CHARTTIME]] +
            ([temp4[C_STARTTIME]] if temp4 is not None else []),
            ignore_index=True
        )))
        if len(timesteps) == 0:
            continue

        # Pre-group by timestamp and extract numpy arrays for fast inner loop
        ce_by_ts = {}
        if len(temp) > 0:
            ce_ids = temp[C_ITEMID].values
            ce_vals = temp[C_VALUENUM].values
            ce_times = temp[C_CHARTTIME].values
            for ts_val in pd.unique(ce_times):
                mask = ce_times == ts_val
                ce_by_ts[ts_val] = (ce_ids[mask], ce_vals[mask])

        lab_by_ts = {}
        if len(temp2) > 0:
            lab_ids = temp2[C_ITEMID].values
            lab_vals = temp2[C_VALUENUM].values
            lab_times = temp2[C_CHARTTIME].values
            for ts_val in pd.unique(lab_times):
                mask = lab_times == ts_val
                lab_by_ts[ts_val] = (lab_ids[mask], lab_vals[mask])

        mv_by_ts = {}
        if len(temp3) > 0:
            mv_mechvent = temp3[C_MECHVENT].values
            mv_extubated = temp3[C_EXTUBATED].values
            mv_times = temp3[C_CHARTTIME].values
            for ts_val in pd.unique(mv_times):
                mask = mv_times == ts_val
                mv_by_ts[ts_val] = (mv_mechvent[mask][0], mv_extubated[mask][0])

        pe_by_ts = {}
        if temp4 is not None and len(temp4) > 0:
            pe_mechvent = temp4[C_MECHVENT].values
            pe_extubated = temp4[C_EXTUBATED].values
            pe_times = temp4[C_STARTTIME].values
            for ts_val in pd.unique(pe_times):
                mask = pe_times == ts_val
                pe_by_ts[ts_val] = (pe_mechvent[mask].any(), pe_extubated[mask].any())

        n_chart = len(CHART_FIELD_NAMES)
        n_lab = len(LAB_FIELD_NAMES)

        for i, timestep in enumerate(timesteps):
            item = {C_BLOC: i, C_ICUSTAYID: icustayid, C_TIMESTEP: timestep}

            ce_data = ce_by_ts.get(timestep)
            if ce_data is not None:
                ids, vals = ce_data
                for j in range(len(ids)):
                    iid = int(ids[j])
                    if 0 < iid <= n_chart:
                        item[CHART_FIELD_NAMES[iid - 1]] = vals[j]

            lab_data = lab_by_ts.get(timestep)
            if lab_data is not None:
                ids, vals = lab_data
                for j in range(len(ids)):
                    iid = int(ids[j])
                    if 0 < iid <= n_lab:
                        item[LAB_FIELD_NAMES[iid - 1]] = vals[j]

            mv_data = mv_by_ts.get(timestep)
            if mv_data is not None:
                item[C_MECHVENT] = mv_data[0]
                item[C_EXTUBATED] = mv_data[1]

            if C_MECHVENT not in item:
                pe_data = pe_by_ts.get(timestep)
                if pe_data is not None:
                    item[C_MECHVENT] = int(pe_data[0])
                    item[C_EXTUBATED] = int(pe_data[1])
                    from_pe += 1

            combined_data.append(item)

        infection_times.append({
            C_ICUSTAYID:      icustayid,
            C_ONSET_TIME:     qst,
            C_FIRST_TIMESTEP: timesteps[0],
            C_LAST_TIMESTEP:  timesteps[-1],
            C_DISCHTIME:      d1[0][1],
        })

        if (idx + 1) % checkpoint_every == 0:
            flush_checkpoint(combined_data, infection_times, out_dir, first_flush)
            total_rows_written += len(combined_data)
            combined_data   = []
            infection_times = []
            first_flush     = False

    if combined_data:
        flush_checkpoint(combined_data, infection_times, out_dir, first_flush)
        total_rows_written += len(combined_data)

    logging.info("Got %d items from procedure events", from_pe)
    return total_rows_written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Generates patient_states.csv and qstime.csv (unbinned, no imputation). FAST version.'))
    parser.add_argument('output_dir', type=str)
    parser.add_argument('--data', dest='data_dir', type=str, default=None)
    parser.add_argument('--window-before', dest='window_before', type=int, default=49)
    parser.add_argument('--window-after',  dest='window_after',  type=int, default=25)
    parser.add_argument('--head', dest='head', type=int, default=None)
    parser.add_argument('--filter-stays', dest='filter_stays_path', type=str, default=None)
    parser.add_argument('--checkpoint-every', dest='checkpoint_every', type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args     = parser.parse_args()
    data_dir = args.data_dir or os.path.join(REPO_DIR, 'data', 'interim', 'sepsis')
    out_dir  = args.output_dir
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_04_patient_states.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 04 (fast) started. output_dir=%s resume=%s", out_dir, args.resume)

    raw_dir = os.path.join(data_dir, 'raw_data')
    interm  = os.path.join(data_dir, 'intermediates')

    logging.info("Reading chartevents...")
    ce_paths = (
        [p for p in os.listdir(raw_dir) if p.startswith("ce") and p.endswith(".csv")] +
        [p for p in os.listdir(interm)  if p.startswith("ce") and p.endswith(".csv")]
    )
    logging.info("  Found %d chartevents files", len(ce_paths))
    chart_events = ChartEventsFast([load_intermediate_or_raw_csv(data_dir, p) for p in ce_paths])

    logging.info("Reading onset data...")
    onset_data = load_csv(os.path.join(interm, 'sepsis_onset.csv'))
    logging.info("  %d sepsis onset records", len(onset_data))

    logging.info("Reading demog...")
    demog = load_intermediate_or_raw_csv(data_dir, 'demog.csv')

    logging.info("Reading labs...")
    labU = pd.concat([
        load_intermediate_or_raw_csv(data_dir, 'labs_ce.csv'),
        load_intermediate_or_raw_csv(data_dir, 'labs_le.csv'),
    ], ignore_index=True)
    logging.info("  %d lab rows total", len(labU))

    logging.info("Reading mechvent...")
    MV = load_intermediate_or_raw_csv(data_dir, 'mechvent.csv')
    try:
        MV_procedure = load_intermediate_or_raw_csv(data_dir, 'mechvent_pe.csv')
        logging.info("  mechvent_pe loaded: %d rows", len(MV_procedure))
    except FileNotFoundError:
        MV_procedure = None
        logging.info("  mechvent_pe not found, skipping")

    if args.filter_stays_path:
        allowed    = load_csv(args.filter_stays_path)[C_ICUSTAYID]
        old_n      = len(onset_data)
        onset_data = onset_data[onset_data[C_ICUSTAYID].isin(allowed)]
        logging.info("Filtered onset_data from %d to %d stays", old_n, len(onset_data))

    if args.head:
        onset_data = onset_data.head(args.head)
        logging.info("Head mode: processing first %d onset records", len(onset_data))

    total = build_patient_states(
        chart_events, onset_data, demog, labU, MV, MV_procedure,
        args.window_before, args.window_after,
        out_dir, args.checkpoint_every, args.resume,
    )

    states_path = os.path.join(out_dir, 'patient_states.csv')
    logging.info("Verifying column coverage in %s ...", states_path)
    df_check = pd.read_csv(states_path, nrows=0)
    missing_cols = [col for col in CHART_FIELD_NAMES + LAB_FIELD_NAMES
                    if col not in df_check.columns]
    if missing_cols:
        logging.warning("%d columns absent from output (CareVue-only items not in MIMIC-IV): %s",
                        len(missing_cols), missing_cols)
        logging.info("Adding missing columns as NaN so downstream steps do not crash...")
        df_full = pd.read_csv(states_path, dtype={C_ICUSTAYID: 'Int64'})
        for col in missing_cols:
            df_full[col] = pd.NA
        df_full.to_csv(states_path, index=False, float_format='%g')
        logging.info("Missing columns added. File updated: %s", states_path)

    logging.info("Step 04 complete. Total rows written: %d", total)
