"""
Step 04 — Build patient states (unbinned).

Faithful adaptation of ai_clinician/preprocessing/03_build_patient_states.py.
Changes: import paths only (ai_clinician.* -> careai.sepsis.*).

Checkpointing: every --checkpoint-every patients, partial results are flushed
to patient_states.csv (append) and qstime.csv (append). If the run is
interrupted, restart with --resume to skip already-processed stay IDs.

Inputs:
  data/interim/sepsis/raw_data/ and intermediates/
    ce*.csv, labs_ce.csv, labs_le.csv, mechvent.csv, mechvent_pe.csv,
    demog.csv, sepsis_onset.csv
Outputs:
  <output_dir>/patient_states.csv
  <output_dir>/qstime.csv

Usage:
    source ../.venv/Scripts/activate
    # Full run (background, log to file)
    python scripts/sepsis/step_04_patient_states.py \\
        data/interim/sepsis/intermediates/patient_states \\
        > logs/step_04_patient_states.log 2>&1 &

    # Resume after crash
    python scripts/sepsis/step_04_patient_states.py \\
        data/interim/sepsis/intermediates/patient_states --resume
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


class ChartEvents:
    """
    Manages a set of chart event tables; retrieves all events for a given stay.
    Identical to AI-Clinician 03_build_patient_states.ChartEvents.
    """
    def __init__(self, ce_dfs, stay_id_col=C_ICUSTAYID):
        self.dfs = ce_dfs
        self.ranges = [(df[stay_id_col].min(), df[stay_id_col].max()) for df in ce_dfs]
        self.stay_id_col = stay_id_col

    def fetch(self, stay_id):
        results = []
        for df, (min_id, max_id) in zip(self.dfs, self.ranges):
            if stay_id >= min_id and stay_id <= max_id:
                results.append(df[df[self.stay_id_col] == stay_id])
        if results:
            return pd.concat(results)
        return pd.DataFrame(columns=[self.stay_id_col, C_CHARTTIME, C_ITEMID, C_VALUENUM])


def time_window(df, col, center_time, lower_window, upper_window):
    return df[(df[col] >= center_time - lower_window) & (df[col] <= center_time + upper_window)]


def flush_checkpoint(combined_data, infection_times, out_dir, first_flush):
    """Append accumulated rows to output files."""
    if not combined_data:
        return
    mode = 'w' if first_flush else 'a'
    header = first_flush

    states_path = os.path.join(out_dir, 'patient_states.csv')
    qstime_path = os.path.join(out_dir, 'qstime.csv')

    pd.DataFrame(combined_data).to_csv(
        states_path, mode=mode, index=False, header=header, float_format='%g')
    pd.DataFrame(infection_times).to_csv(
        qstime_path, mode=mode, index=False, header=header, float_format='%g')

    logging.info("Checkpoint: flushed %d rows to %s", len(combined_data), states_path)


def build_patient_states(chart_events, onset_data, demog, labU, MV, MV_procedure,
                          winb4, winaft, out_dir, checkpoint_every, resume):
    """
    Process each sepsis patient and build one row per unique timestamp.
    Writes results incrementally every checkpoint_every patients.
    Returns total rows written and patients processed.
    """
    states_path = os.path.join(out_dir, 'patient_states.csv')
    qstime_path = os.path.join(out_dir, 'qstime.csv')

    # Resume: skip stays already in output
    done_ids = set()
    if resume and os.path.exists(states_path):
        try:
            partial = pd.read_csv(states_path, usecols=[C_ICUSTAYID])
            done_ids = set(partial[C_ICUSTAYID].dropna().astype(int).unique())
            logging.info("Resume: %d stays already processed, skipping them", len(done_ids))
        except Exception as e:
            logging.warning("Could not read partial output for resume: %s", e)

    remaining = onset_data[~onset_data[C_ICUSTAYID].isin(done_ids)]
    logging.info("%d patients to process (%d already done)", len(remaining), len(done_ids))

    combined_data   = []
    infection_times = []
    from_pe          = 0
    total_rows_written = 0
    first_flush = not bool(done_ids)   # write header only on a clean start

    for idx, (_, row) in enumerate(tqdm(remaining.iterrows(), total=len(remaining),
                                        desc='Building patient states')):
        qst = row[C_ONSET_TIME]
        icustayid = int(row[C_ICUSTAYID])
        assert qst > 0

        d1 = demog.loc[demog[C_ICUSTAYID] == icustayid, [C_AGE, C_DISCHTIME]].values.tolist()
        if not d1 or d1[0][0] < 18:
            continue

        bounds = ((winb4 + 4) * 3600, (winaft + 4) * 3600)

        temp  = time_window(chart_events.fetch(icustayid), C_CHARTTIME, qst, *bounds)
        temp2 = time_window(labU[labU[C_ICUSTAYID] == icustayid],       C_CHARTTIME, qst, *bounds)
        temp3 = time_window(MV[MV[C_ICUSTAYID] == icustayid],           C_CHARTTIME, qst, *bounds)
        temp4 = None
        if MV_procedure is not None:
            temp4 = time_window(MV_procedure[MV_procedure[C_ICUSTAYID] == icustayid],
                                C_STARTTIME, qst, *bounds)

        timesteps = sorted(pd.unique(pd.concat(
            [temp[C_CHARTTIME], temp2[C_CHARTTIME], temp3[C_CHARTTIME]] +
            ([temp4[C_STARTTIME]] if temp4 is not None else []),
            ignore_index=True
        )))
        if len(timesteps) == 0:
            continue

        for i, timestep in enumerate(timesteps):
            item = {C_BLOC: i, C_ICUSTAYID: icustayid, C_TIMESTEP: timestep}

            for _, event in temp[temp[C_CHARTTIME] == timestep].iterrows():
                if event[C_ITEMID] <= 0 or event[C_ITEMID] > len(CHART_FIELD_NAMES):
                    continue
                item[CHART_FIELD_NAMES[event[C_ITEMID] - 1]] = event[C_VALUENUM]

            for _, event in temp2[temp2[C_CHARTTIME] == timestep].iterrows():
                if event[C_ITEMID] <= 0 or event[C_ITEMID] > len(LAB_FIELD_NAMES):
                    continue
                item[LAB_FIELD_NAMES[event[C_ITEMID] - 1]] = event[C_VALUENUM]

            matching_mv = (temp3[C_CHARTTIME] == timestep)
            if matching_mv.sum() > 0:
                event = temp3[matching_mv].iloc[0]
                item[C_MECHVENT]  = event[C_MECHVENT]
                item[C_EXTUBATED] = event[C_EXTUBATED]

            if (temp4 is not None and C_MECHVENT not in item and
                    (temp4[C_STARTTIME] == timestep).sum() > 0):
                events = temp4[temp4[C_STARTTIME] == timestep]
                item[C_MECHVENT]  = events[C_MECHVENT].any().astype(int)
                item[C_EXTUBATED] = events[C_EXTUBATED].any().astype(int)
                from_pe += 1

            combined_data.append(item)

        infection_times.append({
            C_ICUSTAYID:      icustayid,
            C_ONSET_TIME:     qst,
            C_FIRST_TIMESTEP: timesteps[0],
            C_LAST_TIMESTEP:  timesteps[-1],
            C_DISCHTIME:      d1[0][1],
        })

        # Checkpoint flush
        if (idx + 1) % checkpoint_every == 0:
            flush_checkpoint(combined_data, infection_times, out_dir, first_flush)
            total_rows_written += len(combined_data)
            combined_data   = []
            infection_times = []
            first_flush = False

    # Final flush
    if combined_data:
        flush_checkpoint(combined_data, infection_times, out_dir, first_flush)
        total_rows_written += len(combined_data)

    logging.info("Got %d items from procedure events", from_pe)
    return total_rows_written


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Generates patient_states.csv and qstime.csv (unbinned, no imputation).'))
    parser.add_argument('output_dir', type=str,
                        help='Output directory (e.g. data/interim/sepsis/intermediates/patient_states)')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Data parent directory (default: data/interim/sepsis/)')
    parser.add_argument('--window-before', dest='window_before', type=int, default=49)
    parser.add_argument('--window-after',  dest='window_after',  type=int, default=25)
    parser.add_argument('--head', dest='head', type=int, default=None,
                        help='Limit to first N rows of onset data (smoke test)')
    parser.add_argument('--filter-stays', dest='filter_stays_path', type=str, default=None)
    parser.add_argument('--checkpoint-every', dest='checkpoint_every', type=int,
                        default=DEFAULT_CHECKPOINT_EVERY,
                        help='Flush output every N patients (default 200)')
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Skip stays already present in the partial output file')
    parser.add_argument('--log', dest='log_file', type=str, default=None,
                        help='Log file (default: logs/step_04_patient_states.log)')

    args = parser.parse_args()
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

    logging.info("Step 04 started. output_dir=%s resume=%s checkpoint_every=%d",
                 out_dir, args.resume, args.checkpoint_every)

    raw_dir = os.path.join(data_dir, 'raw_data')
    interm  = os.path.join(data_dir, 'intermediates')

    logging.info("Reading chartevents...")
    ce_paths = (
        [p for p in os.listdir(raw_dir) if p.startswith("ce") and p.endswith(".csv")] +
        [p for p in os.listdir(interm)  if p.startswith("ce") and p.endswith(".csv")]
    )
    logging.info("  Found %d chartevents files", len(ce_paths))
    chart_events = ChartEvents([load_intermediate_or_raw_csv(data_dir, p) for p in ce_paths])

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
        allowed = load_csv(args.filter_stays_path)[C_ICUSTAYID]
        old_n = len(onset_data)
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

    # Ensure all expected columns exist in the final output
    states_path = os.path.join(out_dir, 'patient_states.csv')
    logging.info("Verifying column coverage in %s ...", states_path)
    df_check = pd.read_csv(states_path, nrows=0)
    for col in CHART_FIELD_NAMES + LAB_FIELD_NAMES:
        if col not in df_check.columns:
            logging.warning("Column '%s' absent from output (no data points for this feature)", col)

    logging.info("Step 04 complete. Total rows written: %d", total)
