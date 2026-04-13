"""
Step 06 -- Build binned states and actions (4-hour time windows).

Adapted from scripts/sepsis/step_06_states_actions.py.

Key differences from sepsis step_06:
  - Time window: iterate from ICU intime to outtime (full stay),
    not a fixed window around sepsis onset.
  - Input stay-bounds: icu_stay_bounds.csv (not qstime.csv).
  - Binary drug class flags (9) added per bloc from drugs_mv.csv.
    For each 4-hour bloc [t0, t1], any active drug in the class = 1.
  - Demographics: charlson_score + 18 Charlson flags (not elixhauser).
    Additional static confounders: race, insurance, marital_status,
    admission_type, admission_location, prior_ed_visits_6m, drg_severity, drg_mortality.
  - Outcome: readmit_30d broadcast to every bloc (terminal label).
  - No inputCV / vasoCV (MIMIC-IV only; same as sepsis pipeline behaviour).

Checkpointing: every --checkpoint-every patients, results are flushed.
Use --resume to skip stays already in the output file.

Inputs:
  patient_states_imputed.csv, icu_stay_bounds.csv, demog.csv,
  fluid_mv.csv, vaso_mv.csv, preadm_fluid.csv, uo.csv, drugs_mv.csv

Output:
  data/interim/icu_readmit/intermediates/states_actions.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_06_states_actions.py \\
        data/interim/icu_readmit/intermediates/patient_states/patient_states_imputed.csv \\
        data/interim/icu_readmit/intermediates/patient_states/icu_stay_bounds.csv \\
        data/interim/icu_readmit/intermediates/states_actions.csv \\
        > logs/step_06_icu_readmit.log 2>&1 &

    # Resume after crash:
    python scripts/icu_readmit/step_06_states_actions.py ... --resume
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.columns import (
    C_ICUSTAYID, C_CHARTTIME, C_TIMESTEP, C_BLOC, C_BIN_INDEX,
    C_STARTTIME, C_ENDTIME, C_RATE, C_TEV, C_NORM_INFUSION_RATE,
    C_RATESTD, C_INPUT_PREADM, C_VALUE, C_ITEMID,
    C_INPUT_TOTAL, C_INPUT_STEP, C_OUTPUT_TOTAL, C_OUTPUT_STEP, C_CUMULATED_BALANCE,
    C_MEDIAN_DOSE_VASO, C_MAX_DOSE_VASO,
    C_GENDER, C_AGE, C_WEIGHT, C_RE_ADMISSION,
    C_RACE, C_INSURANCE, C_MARITAL_STATUS,
    C_ADMISSION_TYPE, C_ADMISSION_LOC,
    C_CHARLSON, C_PRIOR_ED_VISITS, C_DRG_SEVERITY, C_DRG_MORTALITY,
    C_DISCHARGE_DISPOSITION,
    C_DIED_IN_HOSP, C_DIED_WITHIN_48H_OF_OUT_TIME,
    C_MORTA_90, C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH,
    C_ADM_ORDER, C_MORTA_HOSP, C_DOD, C_OUTTIME,
    C_INTIME, C_LAST_TIMESTEP, C_DISCHTIME,
    C_READMIT_30D,
    SAH_FIELD_NAMES, DEMOGRAPHICS_FIELD_NAMES, IO_FIELD_NAMES,
    BINARY_ACTION_COLS, CHARLSON_FLAG_COLS,
)
from careai.icu_readmit.queries import DRUG_ACTION_ITEMIDS
from careai.icu_readmit.utils import load_csv, load_intermediate_or_raw_csv

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CHECKPOINT_EVERY = 200

# Fixed column order for checkpoint CSV writes
# dict.fromkeys preserves insertion order and removes duplicates
# (C_MECHVENT appears in both SAH_FIELD_NAMES and BINARY_ACTION_COLS)
_OUTPUT_COLUMNS = list(dict.fromkeys(
    [C_BLOC, C_ICUSTAYID, C_TIMESTEP]
    + DEMOGRAPHICS_FIELD_NAMES
    + SAH_FIELD_NAMES
    + IO_FIELD_NAMES
    + BINARY_ACTION_COLS
    + [C_READMIT_30D]
))


def flush_checkpoint(combined_data, out_path, first_flush):
    if not combined_data:
        return
    mode   = 'w' if first_flush else 'a'
    header = first_flush
    pd.DataFrame(combined_data).reindex(columns=_OUTPUT_COLUMNS).to_csv(
        out_path, mode=mode, index=False, header=header, float_format='%g')
    logging.info("Checkpoint: flushed %d rows to %s", len(combined_data), out_path)


def build_states_and_actions(df, stay_bounds, inputMV, inputpreadm,
                              vasoMV, demog, drugs_mv, UOpreadm, UO,
                              timestep_resolution,
                              out_path, checkpoint_every, resume,
                              head=None, allowed_stays=None):
    """
    Process each ICU stay: bin states into 4-hour blocs, compute I/O totals,
    vasopressor doses, and binary drug class flags.

    Key change vs sepsis: window is [intime, outtime] (full ICU stay) per patient.
    """
    icustayidlist = np.unique(df[C_ICUSTAYID])
    icustayidlist = sorted(icustayidlist[~pd.isna(icustayidlist)])
    if allowed_stays is not None:
        allowed_stays = set(allowed_stays)
        old_count = len(icustayidlist)
        icustayidlist = [i for i in icustayidlist if i in allowed_stays]
        logging.info("Filtered from %d to %d ICU stay ids", old_count, len(icustayidlist))
    else:
        logging.info("%d ICU stay IDs", len(icustayidlist))

    if head:
        icustayidlist = icustayidlist[:head]

    # Resume: skip stays already in output
    done_ids = set()
    if resume and os.path.exists(out_path):
        try:
            partial = pd.read_csv(out_path, usecols=[C_ICUSTAYID])
            done_ids = set(partial[C_ICUSTAYID].dropna().astype(int).unique())
            logging.info("Resume: %d stays already processed", len(done_ids))
        except Exception as e:
            logging.warning("Could not read partial output for resume: %s", e)
    icustayidlist = [i for i in icustayidlist if i not in done_ids]
    logging.info("%d stays remaining to process", len(icustayidlist))

    combined_data = []
    total_rows    = 0
    first_flush   = not bool(done_ids)

    for idx, icustayid in enumerate(tqdm(icustayidlist, desc='Building states and actions')):
        temp = df.loc[df[C_ICUSTAYID] == icustayid, :]
        if len(temp) == 0:
            continue

        # ICU stay time bounds
        if icustayid not in stay_bounds.index:
            continue
        bounds_row = stay_bounds.loc[icustayid]
        intime   = bounds_row[C_INTIME]
        outtime  = bounds_row[C_OUTTIME]
        dischtime = bounds_row[C_DISCHTIME]
        last_ts  = bounds_row[C_LAST_TIMESTEP]

        if pd.isna(intime) or pd.isna(outtime) or outtime <= intime:
            continue

        # ICU LOS in hours — determines number of blocs
        icu_los_hours = (outtime - intime) / 3600.0
        beg = intime

        # ---- Fluid inputs ----
        input_ = inputMV.loc[inputMV[C_ICUSTAYID] == icustayid, :]
        startt = input_[C_STARTTIME]
        endt   = input_[C_ENDTIME]
        rate   = input_[C_NORM_INFUSION_RATE]

        pread  = inputpreadm.loc[inputpreadm[C_ICUSTAYID] == icustayid, C_INPUT_PREADM]
        totvol = pread.sum() if not pread.empty else 0

        # Pre-ICU fluid volume (before intime)
        t0, t1 = 0, beg
        infu = np.nansum(
            rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 +
            rate * (endt - t0)    * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 +
            rate * (t1 - startt)  * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 +
            rate * (t1 - t0)      * ((endt >= t1) & (startt <= t0)) / 3600
        )
        bolusMV = np.nansum(
            input_.loc[pd.isna(input_[C_RATE]) &
                       (input_[C_STARTTIME] >= t0) & (input_[C_STARTTIME] <= t1), C_TEV])
        totvol = np.nansum([totvol, infu, bolusMV])

        # ---- Vasopressors ----
        vaso1  = vasoMV.loc[vasoMV[C_ICUSTAYID] == icustayid, :]
        startv = vaso1[C_STARTTIME]
        endv   = vaso1[C_ENDTIME]
        ratev  = vaso1[C_RATESTD]

        # ---- Demographics (broadcast to every bloc) ----
        demog_rows = demog[demog[C_ICUSTAYID] == icustayid]
        if len(demog_rows) == 0:
            continue
        dr = demog_rows.iloc[0]

        dem = {
            C_GENDER:          dr[C_GENDER],
            C_AGE:             dr[C_AGE],
            # Weight may be in demog or from chartevents (use demog if present)
            C_RE_ADMISSION:    int(dr[C_ADM_ORDER] > 1),
            C_RACE:            dr.get(C_RACE, None),
            C_INSURANCE:       dr.get(C_INSURANCE, None),
            C_MARITAL_STATUS:  dr.get(C_MARITAL_STATUS, None),
            C_ADMISSION_TYPE:  dr.get(C_ADMISSION_TYPE, None),
            C_ADMISSION_LOC:   dr.get(C_ADMISSION_LOC, None),
            C_CHARLSON:        dr.get(C_CHARLSON, 0),
            C_PRIOR_ED_VISITS: dr.get(C_PRIOR_ED_VISITS, 0),
            C_DRG_SEVERITY:    dr.get(C_DRG_SEVERITY, 0),
            C_DRG_MORTALITY:   dr.get(C_DRG_MORTALITY, 0),
            C_DISCHARGE_DISPOSITION: dr.get(C_DISCHARGE_DISPOSITION, None),
            C_DIED_IN_HOSP:    dr[C_MORTA_HOSP],
            # All cohort patients survived hospital (excluded in step_03), so 0
            C_DIED_WITHIN_48H_OF_OUT_TIME: 0,
            C_MORTA_90:        dr.get(C_MORTA_90, 0),
            C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH:
                (dischtime - last_ts) / 3600 if not pd.isna(dischtime) and not pd.isna(last_ts) else 0,
            C_READMIT_30D:     dr.get(C_READMIT_30D, 0),
        }
        # Charlson component flags
        for flag in CHARLSON_FLAG_COLS:
            dem[flag] = dr.get(flag, 0)

        # ---- Urine output ----
        output = UO.loc[UO[C_ICUSTAYID] == icustayid, :]
        pread  = UOpreadm.loc[UOpreadm[C_ICUSTAYID] == icustayid, C_VALUE]
        UOtot  = pread.sum() if not pread.empty else 0
        # Add pre-ICU urine output
        UOtot += np.nansum(output.loc[(output[C_CHARTTIME] >= 0) & (output[C_CHARTTIME] <= beg), C_VALUE])

        # ---- Drug actions (drugs_mv for this stay) ----
        drug_stay = drugs_mv.loc[drugs_mv[C_ICUSTAYID] == icustayid, :]

        # ---- Iterate over 4-hour blocs ----
        for j in np.arange(0, icu_los_hours, timestep_resolution):
            t0 = beg + 3600 * j
            t1 = beg + 3600 * (j + timestep_resolution)
            value = temp.loc[(temp[C_TIMESTEP] >= t0) & (temp[C_TIMESTEP] <= t1), :]
            if len(value) == 0:
                continue

            item = {
                C_BLOC:      (j / timestep_resolution) + 1,
                C_ICUSTAYID: icustayid,
                C_TIMESTEP:  int(t0),
            }
            item.update(dem)

            # State features: mean within bloc
            item.update({col: value[col].mean(skipna=True)
                         for col in SAH_FIELD_NAMES if col in value.columns})

            # Vasopressor dose
            v = (((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv <= t1)) |
                 ((startv >= t0) & (startv <= t1)) | ((startv <= t0) & (endv >= t1)))
            if not ratev.loc[v].empty:
                all_vs = ratev.loc[v].values
                all_vs = all_vs[~np.isnan(all_vs)]
                if all_vs.size > 0:
                    item[C_MEDIAN_DOSE_VASO] = np.nanmedian(all_vs)
                    item[C_MAX_DOSE_VASO]    = np.nanmax(all_vs)

            # IV fluid (infusions + boluses)
            infu = np.nansum(
                rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 +
                rate * (endt - t0)    * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 +
                rate * (t1 - startt)  * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 +
                rate * (t1 - t0)      * ((endt >= t1) & (startt <= t0)) / 3600
            )
            bolusMV = np.nansum(
                input_.loc[pd.isna(input_[C_RATE]) &
                           (input_[C_STARTTIME] >= t0) & (input_[C_STARTTIME] <= t1), C_TEV])
            totvol = np.nansum([totvol, infu, bolusMV])
            item[C_INPUT_TOTAL] = totvol
            item[C_INPUT_STEP]  = infu + bolusMV

            # Urine output
            UOnow = np.nansum(output.loc[(output[C_CHARTTIME] >= t0) & (output[C_CHARTTIME] <= t1), C_VALUE])
            UOtot = np.nansum([UOtot, UOnow])
            item[C_OUTPUT_TOTAL]      = UOtot
            item[C_OUTPUT_STEP]       = UOnow
            item[C_CUMULATED_BALANCE] = totvol - UOtot

            # Binary drug class flags (NEW vs sepsis)
            # A drug class is active if any event overlaps with bloc [t0, t1]
            for drug_class, itemids in DRUG_ACTION_ITEMIDS.items():
                d = drug_stay[drug_stay[C_ITEMID].isin(itemids)]
                active = (
                    ((d[C_STARTTIME] < t1) & (d[C_ENDTIME] > t0)) |
                    ((d[C_STARTTIME] >= t0) & (d[C_STARTTIME] < t1))
                )
                item[drug_class] = int(active.any())

            combined_data.append(item)

        if (idx + 1) % checkpoint_every == 0:
            flush_checkpoint(combined_data, out_path, first_flush)
            total_rows   += len(combined_data)
            combined_data = []
            first_flush   = False

    if combined_data:
        flush_checkpoint(combined_data, out_path, first_flush)
        total_rows += len(combined_data)

    return total_rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Bins ICU patient states into 4-hour blocs and adds I/O, vasopressor, '
        'and binary drug class actions.'))
    parser.add_argument('input',      type=str, help='patient_states_imputed.csv')
    parser.add_argument('stay_bounds',type=str, help='icu_stay_bounds.csv')
    parser.add_argument('output',     type=str, help='Output states_actions.csv path')
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Data parent directory (default: data/interim/icu_readmit/)')
    parser.add_argument('--resolution', dest='resolution', type=float, default=4.0,
                        help='Time bloc resolution in hours (default: 4.0)')
    parser.add_argument('--head', dest='head', type=int, default=None)
    parser.add_argument('--filter-stays', dest='filter_stays_path', type=str, default=None)
    parser.add_argument('--checkpoint-every', dest='checkpoint_every', type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument('--resume', dest='resume', action='store_true', default=False)
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(REPO_DIR, 'data', 'interim', 'icu_readmit')

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_06_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logging.info("Step 06 started. output=%s resume=%s", args.output, args.resume)

    logging.info("Reading states...")
    df = load_csv(args.input)
    logging.info("  %d rows loaded", len(df))

    logging.info("Reading stay bounds...")
    stay_bounds = load_csv(args.stay_bounds, null_icustayid=True).set_index(C_ICUSTAYID, drop=True)
    logging.info("  %d stay records", len(stay_bounds))

    logging.info("Reading data files...")
    interm = os.path.join(data_dir, 'intermediates')
    demog       = load_intermediate_or_raw_csv(data_dir, 'demog.csv')
    inputpreadm = load_intermediate_or_raw_csv(data_dir, 'preadm_fluid.csv')
    inputMV     = load_intermediate_or_raw_csv(data_dir, 'fluid_mv.csv')
    vasoMV      = load_intermediate_or_raw_csv(data_dir, 'vaso_mv.csv')
    UO          = load_intermediate_or_raw_csv(data_dir, 'uo.csv')
    UOpreadm    = load_intermediate_or_raw_csv(data_dir, 'preadm_uo.csv')
    drugs_mv    = load_intermediate_or_raw_csv(data_dir, 'drugs_mv.csv')
    logging.info("  drugs_mv: %d rows", len(drugs_mv))

    allowed_stays = None
    if args.filter_stays_path:
        allowed_stays = load_csv(args.filter_stays_path)[C_ICUSTAYID]

    total = build_states_and_actions(
        df, stay_bounds, inputMV, inputpreadm, vasoMV, demog, drugs_mv,
        UOpreadm, UO,
        args.resolution,
        args.output, args.checkpoint_every, args.resume,
        head=args.head, allowed_stays=allowed_stays,
    )

    # Verify output columns
    df_check = pd.read_csv(args.output, nrows=0)
    expected = DEMOGRAPHICS_FIELD_NAMES + SAH_FIELD_NAMES + IO_FIELD_NAMES + BINARY_ACTION_COLS
    for col in expected:
        if col not in df_check.columns:
            logging.warning("Expected column '%s' absent from output", col)

    logging.info("Step 06 complete. Total rows written: %d", total)
