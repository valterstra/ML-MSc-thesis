"""
Step 06 — Build binned states and actions (4-hour time windows). FAST VERSION.

Same logic as step_06_states_actions.py but pre-groups all DataFrames by
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
    C_ICUSTAYID, C_CHARTTIME, C_TIMESTEP, C_BLOC, C_BIN_INDEX,
    C_STARTTIME, C_ENDTIME, C_RATE, C_TEV, C_NORM_INFUSION_RATE,
    C_RATESTD, C_INPUT_PREADM, C_VALUE,
    C_INPUT_TOTAL, C_INPUT_STEP, C_OUTPUT_TOTAL, C_OUTPUT_STEP, C_CUMULATED_BALANCE,
    C_MEDIAN_DOSE_VASO, C_MAX_DOSE_VASO,
    C_GENDER, C_AGE, C_ELIXHAUSER, C_RE_ADMISSION, C_DIED_IN_HOSP,
    C_DIED_WITHIN_48H_OF_OUT_TIME, C_MORTA_90, C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH,
    C_ADM_ORDER, C_MORTA_HOSP, C_DOD, C_OUTTIME,
    C_LAST_TIMESTEP, C_DISCHTIME,
    SAH_FIELD_NAMES, DEMOGRAPHICS_FIELD_NAMES, IO_FIELD_NAMES, COMPUTED_FIELD_NAMES,
)
from careai.sepsis.utils import load_csv, load_intermediate_or_raw_csv

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))
DEFAULT_CHECKPOINT_EVERY = 200

_EMPTY_FLUID = pd.DataFrame(columns=[C_ICUSTAYID, C_STARTTIME, C_ENDTIME,
                                      C_RATE, C_NORM_INFUSION_RATE, C_TEV])
_EMPTY_VASO  = pd.DataFrame(columns=[C_ICUSTAYID, C_STARTTIME, C_ENDTIME, C_RATESTD])
_EMPTY_UO    = pd.DataFrame(columns=[C_ICUSTAYID, C_CHARTTIME, C_VALUE])
_EMPTY_PREADM_FLUID = pd.DataFrame(columns=[C_ICUSTAYID, C_INPUT_PREADM])
_EMPTY_PREADM_UO    = pd.DataFrame(columns=[C_ICUSTAYID, C_VALUE])

# Fixed column order for checkpoint CSV writes — prevents column-scrambling across batches.
_OUTPUT_COLUMNS = (
    [C_BLOC, C_ICUSTAYID, C_TIMESTEP]
    + DEMOGRAPHICS_FIELD_NAMES
    + SAH_FIELD_NAMES
    + IO_FIELD_NAMES
    + COMPUTED_FIELD_NAMES
)


def flush_checkpoint(combined_data, out_path, first_flush):
    if not combined_data:
        return
    mode   = 'w' if first_flush else 'a'
    header = first_flush
    pd.DataFrame(combined_data).reindex(columns=_OUTPUT_COLUMNS).to_csv(
        out_path, mode=mode, index=False, header=header, float_format='%g')
    logging.info("Checkpoint: flushed %d rows to %s", len(combined_data), out_path)


def safe_get(groups, icustayid, empty):
    try:
        return groups.get_group(icustayid)
    except KeyError:
        return empty


def build_states_and_actions(df, qstime, inputMV, inputCV, inputpreadm,
                              vasoMV, vasoCV, demog, UOpreadm, UO,
                              timestep_resolution, winb4, winaft,
                              out_path, checkpoint_every, resume,
                              head=None, allowed_stays=None):
    icustayidlist = np.unique(df[C_ICUSTAYID])
    icustayidlist = sorted(icustayidlist[~pd.isna(icustayidlist)])
    if allowed_stays is not None:
        old_count = len(icustayidlist)
        allowed_stays = set(allowed_stays)
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

    # Pre-group all DataFrames by icustayid for O(1) per-patient lookups
    logging.info("Pre-grouping patient states by icustayid...")
    df_groups = df.groupby(C_ICUSTAYID)

    logging.info("Pre-grouping fluid_mv by icustayid...")
    inputMV_groups = inputMV.groupby(C_ICUSTAYID)

    logging.info("Pre-grouping preadm_fluid by icustayid...")
    inputpreadm_groups = inputpreadm.groupby(C_ICUSTAYID)

    logging.info("Pre-grouping vaso_mv by icustayid...")
    vasoMV_groups = vasoMV.groupby(C_ICUSTAYID)

    logging.info("Pre-grouping demog by icustayid...")
    demog_groups = demog.groupby(C_ICUSTAYID)

    logging.info("Pre-grouping UO by icustayid...")
    UO_groups = UO.groupby(C_ICUSTAYID)

    logging.info("Pre-grouping preadm_uo by icustayid...")
    UOpreadm_groups = UOpreadm.groupby(C_ICUSTAYID)

    inputCV_groups = inputCV.groupby(C_ICUSTAYID) if inputCV is not None else None
    vasoCV_groups  = vasoCV.groupby(C_ICUSTAYID)  if vasoCV  is not None else None

    logging.info("Pre-grouping done. Starting patient loop.")

    combined_data = []
    total_rows    = 0
    first_flush   = not bool(done_ids)
    total_duration = (winb4 + 3) + (winaft + 3)

    for idx, icustayid in enumerate(tqdm(icustayidlist, desc='Building states and actions')):
        temp   = safe_get(df_groups, icustayid, pd.DataFrame())
        if temp.empty:
            continue
        beg    = temp[C_TIMESTEP].iloc[0]

        input_ = safe_get(inputMV_groups, icustayid, _EMPTY_FLUID)
        input2 = (safe_get(inputCV_groups, icustayid, _EMPTY_FLUID)
                  if inputCV_groups is not None else None)
        startt = input_[C_STARTTIME]
        endt   = input_[C_ENDTIME]
        rate   = input_[C_NORM_INFUSION_RATE]

        pread_fluid = safe_get(inputpreadm_groups, icustayid, _EMPTY_PREADM_FLUID)
        pread  = pread_fluid[C_INPUT_PREADM]
        totvol = pread.sum() if not pread.empty else 0

        t0, t1 = 0, beg
        infu = np.nansum(
            rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 +
            rate * (endt - t0)    * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 +
            rate * (t1 - startt)  * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 +
            rate * (t1 - t0)      * ((endt >= t1) & (startt <= t0)) / 3600
        )
        bolusMV = np.nansum(
            input_.loc[pd.isna(input_[C_RATE]) & (input_[C_STARTTIME] >= t0) & (input_[C_STARTTIME] <= t1), C_TEV])
        bolus   = bolusMV + (np.nansum(
            input2.loc[(input2[C_CHARTTIME] >= t0) & (input2[C_CHARTTIME] <= t1), C_TEV])
            if input2 is not None else 0)
        totvol = np.nansum([totvol, infu, bolus])

        vaso1  = safe_get(vasoMV_groups, icustayid, _EMPTY_VASO)
        vaso2  = (safe_get(vasoCV_groups, icustayid, _EMPTY_VASO)
                  if vasoCV_groups is not None else None)
        startv = vaso1[C_STARTTIME]
        endv   = vaso1[C_ENDTIME]
        ratev  = vaso1[C_RATESTD]

        try:
            demog_grp = demog_groups.get_group(icustayid)
        except KeyError:
            continue  # patient not in demog, skip
        demog_row = demog_grp.iloc[0]
        dem = {
            C_GENDER:         demog_row[C_GENDER],
            C_AGE:            demog_row[C_AGE],
            C_ELIXHAUSER:     demog_row[C_ELIXHAUSER],
            C_RE_ADMISSION:   demog_row[C_ADM_ORDER] > 1,
            C_DIED_IN_HOSP:   demog_row[C_MORTA_HOSP],
            C_DIED_WITHIN_48H_OF_OUT_TIME:
                abs(demog_row[C_DOD] - demog_row[C_OUTTIME]) < (24 * 3600 * 2),
            C_MORTA_90:       demog_row[C_MORTA_90],
            C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH:
                (qstime.loc[icustayid, C_DISCHTIME] - qstime.loc[icustayid, C_LAST_TIMESTEP]) / 3600,
        }

        output  = safe_get(UO_groups, icustayid, _EMPTY_UO)
        pread_uo = safe_get(UOpreadm_groups, icustayid, _EMPTY_PREADM_UO)
        pread   = pread_uo[C_VALUE]
        UOtot   = pread.sum() if not pread.empty else 0
        UOnow   = np.sum(output.loc[(output[C_CHARTTIME] >= t0) & (output[C_CHARTTIME] <= t1), C_VALUE])
        UOtot  += UOnow

        for j in np.arange(0, total_duration, timestep_resolution):
            t0    = 3600 * j + beg
            t1    = 3600 * (j + timestep_resolution) + beg
            value = temp.loc[(temp[C_TIMESTEP] >= t0) & (temp[C_TIMESTEP] <= t1), :]
            if len(value) == 0:
                continue

            item = {
                C_BLOC:      (j / timestep_resolution) + 1,
                C_ICUSTAYID: icustayid,
                C_TIMESTEP:  int(3600 * j + beg),
            }
            item.update(dem)
            item.update({col: value[col].mean(skipna=True) for col in SAH_FIELD_NAMES})

            v = (((endv >= t0) & (endv <= t1)) | ((startv >= t0) & (endv <= t1)) |
                 ((startv >= t0) & (startv <= t1)) | ((startv <= t0) & (endv >= t1)))
            v2 = (vaso2.loc[(vaso2[C_CHARTTIME] >= t0) & (vaso2[C_CHARTTIME] <= t1), C_RATESTD]
                  if vaso2 is not None else pd.Series([], dtype=np.float64))
            if not ratev.loc[v].empty or not v2.empty:
                all_vs = np.concatenate([ratev.loc[v].values, v2.values])
                all_vs = all_vs[~np.isnan(all_vs)]
                if all_vs.size > 0:
                    item[C_MEDIAN_DOSE_VASO] = np.nanmedian(all_vs)
                    item[C_MAX_DOSE_VASO]    = np.nanmax(all_vs)

            infu = np.nansum(
                rate * (endt - startt) * ((endt <= t1) & (startt >= t0)) / 3600 +
                rate * (endt - t0)    * ((startt <= t0) & (endt <= t1) & (endt >= t0)) / 3600 +
                rate * (t1 - startt)  * ((startt >= t0) & (endt >= t1) & (startt <= t1)) / 3600 +
                rate * (t1 - t0)      * ((endt >= t1) & (startt <= t0)) / 3600
            )
            bolusMV = np.nansum(
                input_.loc[pd.isna(input_[C_RATE]) & (input_[C_STARTTIME] >= t0) & (input_[C_STARTTIME] <= t1), C_TEV])
            bolus = bolusMV + (np.nansum(
                input2.loc[(input2[C_CHARTTIME] >= t0) & (input2[C_CHARTTIME] <= t1), C_TEV])
                if input2 is not None else 0)
            totvol = np.nansum([totvol, infu, bolus])
            item[C_INPUT_TOTAL]       = totvol
            item[C_INPUT_STEP]        = infu + bolus
            UOnow = np.nansum(output.loc[(output[C_CHARTTIME] >= t0) & (output[C_CHARTTIME] <= t1), C_VALUE])
            UOtot = np.nansum([UOtot, UOnow])
            item[C_OUTPUT_TOTAL]      = UOtot
            item[C_OUTPUT_STEP]       = UOnow
            item[C_CUMULATED_BALANCE] = totvol - UOtot
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
        'Bins patient states into time windows and adds I/O and vasopressor data. FAST version.'))
    parser.add_argument('input',   type=str, help='Patient states (imputed) CSV')
    parser.add_argument('qstime',  type=str, help='qstime.csv')
    parser.add_argument('output',  type=str, help='Output states_actions CSV path')
    parser.add_argument('--data', dest='data_dir', type=str, default=None)
    parser.add_argument('--resolution', dest='resolution', type=float, default=4.0)
    parser.add_argument('--window-before', dest='window_before', type=int, default=49)
    parser.add_argument('--window-after',  dest='window_after',  type=int, default=25)
    parser.add_argument('--head', dest='head', type=int, default=None)
    parser.add_argument('--filter-stays', dest='filter_stays_path', type=str, default=None)
    parser.add_argument('--checkpoint-every', dest='checkpoint_every', type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument('--resume', dest='resume', action='store_true', default=False,
                        help='Skip stays already present in the output file')
    parser.add_argument('--log', dest='log_file', type=str, default=None)

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(REPO_DIR, 'data', 'interim', 'sepsis')

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_06_states_actions.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    logging.info("Step 06 (fast) started. output=%s resume=%s", args.output, args.resume)

    logging.info("Reading states...")
    df = load_csv(args.input)
    logging.info("  %d rows loaded", len(df))
    qstime = load_csv(args.qstime).set_index(C_ICUSTAYID, drop=True)

    logging.info("Reading data files...")
    demog       = load_intermediate_or_raw_csv(data_dir, 'demog.csv')
    inputpreadm = load_intermediate_or_raw_csv(data_dir, 'preadm_fluid.csv')
    inputMV     = load_intermediate_or_raw_csv(data_dir, 'fluid_mv.csv')
    vasoMV      = load_intermediate_or_raw_csv(data_dir, 'vaso_mv.csv')
    UO          = load_intermediate_or_raw_csv(data_dir, 'uo.csv')
    UOpreadm    = load_intermediate_or_raw_csv(data_dir, 'preadm_uo.csv')
    logging.info("  preadm_uo loaded: %d rows", len(UOpreadm))
    try:
        inputCV = load_intermediate_or_raw_csv(data_dir, 'fluid_cv.csv')
        logging.info("  fluid_cv loaded: %d rows", len(inputCV))
    except FileNotFoundError:
        inputCV = None
        logging.info("  fluid_cv not found (expected for MIMIC-IV)")
    try:
        vasoCV = load_intermediate_or_raw_csv(data_dir, 'vaso_cv.csv')
        logging.info("  vaso_cv loaded: %d rows", len(vasoCV))
    except FileNotFoundError:
        vasoCV = None
        logging.info("  vaso_cv not found (expected for MIMIC-IV)")

    allowed_stays = None
    if args.filter_stays_path:
        allowed_stays = load_csv(args.filter_stays_path)[C_ICUSTAYID]

    total = build_states_and_actions(
        df, qstime, inputMV, inputCV, inputpreadm, vasoMV, vasoCV, demog, UOpreadm, UO,
        args.resolution, args.window_before, args.window_after,
        args.output, args.checkpoint_every, args.resume,
        head=args.head, allowed_stays=allowed_stays,
    )

    # Ensure expected columns present
    df_check = pd.read_csv(args.output, nrows=0)
    expected = DEMOGRAPHICS_FIELD_NAMES + SAH_FIELD_NAMES + IO_FIELD_NAMES + COMPUTED_FIELD_NAMES
    for col in expected:
        if col not in df_check.columns:
            logging.warning("Expected column '%s' absent from output", col)

    logging.info("Step 06 complete. Total rows written: %d", total)
