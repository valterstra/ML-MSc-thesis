"""
Step 06 — Build binned states and actions (4-hour time windows). VECTORIZED VERSION.

Eliminates the per-bin Python loop by:
  1. Assigning each state row a bin index then calling groupby.mean() once
     instead of 18 x 65 individual .mean() calls.
  2. Computing fluid infusion, UO, and vasopressor per-bin values with numpy
     broadcasting ([n_bins, n_records] matrix) instead of 18 separate boolean
     masks.
  3. Using np.cumsum for the running totvol / UOtot accumulators.

All logic and output columns are identical to the original.
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
    C_ICUSTAYID, C_CHARTTIME, C_TIMESTEP, C_BLOC,
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


def safe_get(groups, key, empty):
    try:
        return groups.get_group(key)
    except KeyError:
        return empty


def fluid_per_bin(startt, endt, rate, t0_bins, t1_bins):
    """
    Compute infusion volume per bin using numpy broadcasting.
    startt/endt/rate: 1-D arrays of shape [n_fluid]
    t0_bins/t1_bins:  1-D arrays of shape [n_bins]
    Returns: 1-D array of shape [n_bins]
    """
    if len(startt) == 0:
        return np.zeros(len(t0_bins))
    st = startt[np.newaxis, :]   # [1, n_fluid]
    et = endt[np.newaxis, :]
    rt = rate[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]  # [n_bins, 1]
    t1 = t1_bins[:, np.newaxis]
    with np.errstate(invalid='ignore'):
        mat = (
            rt * (et - st) * ((et <= t1) & (st >= t0)) / 3600 +
            rt * (et - t0) * ((st <= t0) & (et <= t1) & (et >= t0)) / 3600 +
            rt * (t1 - st) * ((st >= t0) & (et >= t1) & (st <= t1)) / 3600 +
            rt * (t1 - t0) * ((et >= t1) & (st <= t0)) / 3600
        )
    return np.nansum(mat, axis=1)


def bolus_per_bin(times, tevs, t0_bins, t1_bins):
    """Sum of bolus TEV values in each bin."""
    if len(times) == 0:
        return np.zeros(len(t0_bins))
    bt = times[np.newaxis, :]    # [1, n_bolus]
    bv = tevs[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]  # [n_bins, 1]
    t1 = t1_bins[:, np.newaxis]
    mask = (bt >= t0) & (bt <= t1)
    return np.nansum(np.where(mask, bv, 0.0), axis=1)


def uo_per_bin(times, values, t0_bins, t1_bins):
    """Sum of UO values in each bin."""
    if len(times) == 0:
        return np.zeros(len(t0_bins))
    ut = times[np.newaxis, :]
    uv = values[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]
    t1 = t1_bins[:, np.newaxis]
    mask = (ut >= t0) & (ut <= t1)
    return np.nansum(np.where(mask, uv, 0.0), axis=1)


def vaso_per_bin(startv, endv, ratev, t0_bins, t1_bins):
    """
    Compute (median_dose, max_dose) per bin for vasopressors.
    Uses broadcasting for the active-interval mask, then a small loop
    for median/max (unavoidable with variable-length masked arrays).
    """
    n_bins = len(t0_bins)
    median_out = np.full(n_bins, np.nan)
    max_out    = np.full(n_bins, np.nan)
    if len(startv) == 0:
        return median_out, max_out
    sv = startv[np.newaxis, :]   # [1, n_vaso]
    ev = endv[np.newaxis, :]
    rv = ratev[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]  # [n_bins, 1]
    t1 = t1_bins[:, np.newaxis]
    active = (((ev >= t0) & (ev <= t1)) | ((sv >= t0) & (ev <= t1)) |
              ((sv >= t0) & (sv <= t1)) | ((sv <= t0) & (ev >= t1)))  # [n_bins, n_vaso]
    for b in range(n_bins):
        rates = rv[0, active[b]]
        rates = rates[~np.isnan(rates)]
        if rates.size > 0:
            median_out[b] = np.nanmedian(rates)
            max_out[b]    = np.nanmax(rates)
    return median_out, max_out


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

    # Pre-group all DataFrames by icustayid
    logging.info("Pre-grouping all DataFrames by icustayid...")
    df_groups          = df.groupby(C_ICUSTAYID)
    inputMV_groups     = inputMV.groupby(C_ICUSTAYID)
    inputpreadm_groups = inputpreadm.groupby(C_ICUSTAYID)
    vasoMV_groups      = vasoMV.groupby(C_ICUSTAYID)
    demog_groups       = demog.groupby(C_ICUSTAYID)
    UO_groups          = UO.groupby(C_ICUSTAYID)
    UOpreadm_groups    = UOpreadm.groupby(C_ICUSTAYID)
    inputCV_groups     = inputCV.groupby(C_ICUSTAYID) if inputCV is not None else None
    vasoCV_groups      = vasoCV.groupby(C_ICUSTAYID)  if vasoCV  is not None else None
    logging.info("Pre-grouping done. Starting patient loop.")

    # Precompute bin time arrays (same for every patient, offsets added per-patient)
    j_vals         = np.arange(0, (winb4 + 3) + (winaft + 3), timestep_resolution)
    n_bins         = len(j_vals)
    bin_offsets_t0 = 3600.0 * j_vals          # relative to beg
    bin_offsets_t1 = bin_offsets_t0 + 3600.0 * timestep_resolution
    bloc_numbers   = (j_vals / timestep_resolution) + 1

    combined_data = []
    total_rows    = 0
    first_flush   = not bool(done_ids)

    for idx, icustayid in enumerate(tqdm(icustayidlist, desc='Building states and actions')):
        temp = safe_get(df_groups, icustayid, pd.DataFrame())
        if temp.empty:
            continue
        beg = float(temp[C_TIMESTEP].iloc[0])

        t0_bins = bin_offsets_t0 + beg
        t1_bins = bin_offsets_t1 + beg

        # ── Fluid ──────────────────────────────────────────────────────────
        input_ = safe_get(inputMV_groups, icustayid, _EMPTY_FLUID)
        startt = input_[C_STARTTIME].values.astype(float)
        endt   = input_[C_ENDTIME].values.astype(float)
        rate   = input_[C_NORM_INFUSION_RATE].values.astype(float)

        pread_fluid = safe_get(inputpreadm_groups, icustayid, _EMPTY_PREADM_FLUID)
        totvol_0    = float(pread_fluid[C_INPUT_PREADM].sum()) if not pread_fluid.empty else 0.0

        # Pre-beg fluid (t0=0, t1=beg)
        totvol_0 += float(fluid_per_bin(startt, endt, rate,
                                         np.array([0.0]), np.array([beg]))[0])
        bolus_mask = np.isnan(rate)
        if bolus_mask.any():
            totvol_0 += float(bolus_per_bin(
                input_.loc[bolus_mask, C_STARTTIME].values.astype(float),
                input_.loc[bolus_mask, C_TEV].values.astype(float),
                np.array([0.0]), np.array([beg]))[0])

        # Per-bin fluid
        infu_bins  = fluid_per_bin(startt[~bolus_mask], endt[~bolus_mask],
                                    rate[~bolus_mask], t0_bins, t1_bins)
        if bolus_mask.any():
            bolus_bins = bolus_per_bin(
                input_.loc[bolus_mask, C_STARTTIME].values.astype(float),
                input_.loc[bolus_mask, C_TEV].values.astype(float),
                t0_bins, t1_bins)
        else:
            bolus_bins = np.zeros(n_bins)

        if inputCV_groups is not None:
            input2 = safe_get(inputCV_groups, icustayid, _EMPTY_FLUID)
            cv_bolus_bins = bolus_per_bin(
                input2[C_CHARTTIME].values.astype(float),
                input2[C_TEV].values.astype(float),
                t0_bins, t1_bins)
        else:
            cv_bolus_bins = np.zeros(n_bins)

        input_step_bins = infu_bins + bolus_bins + cv_bolus_bins  # per-bin I/O step

        # ── UO ─────────────────────────────────────────────────────────────
        output   = safe_get(UO_groups, icustayid, _EMPTY_UO)
        pread_uo = safe_get(UOpreadm_groups, icustayid, _EMPTY_PREADM_UO)
        UOtot_0  = float(pread_uo[C_VALUE].sum()) if not pread_uo.empty else 0.0
        UOtot_0 += float(uo_per_bin(
            output[C_CHARTTIME].values.astype(float),
            output[C_VALUE].values.astype(float),
            np.array([0.0]), np.array([beg]))[0])

        uo_bins = uo_per_bin(
            output[C_CHARTTIME].values.astype(float),
            output[C_VALUE].values.astype(float),
            t0_bins, t1_bins)

        # ── Vaso ───────────────────────────────────────────────────────────
        vaso1 = safe_get(vasoMV_groups, icustayid, _EMPTY_VASO)
        vaso_med, vaso_max = vaso_per_bin(
            vaso1[C_STARTTIME].values.astype(float),
            vaso1[C_ENDTIME].values.astype(float),
            vaso1[C_RATESTD].values.astype(float),
            t0_bins, t1_bins)

        if vasoCV_groups is not None:
            vaso2 = safe_get(vasoCV_groups, icustayid, _EMPTY_VASO)
            if not vaso2.empty:
                cv_med, cv_max = vaso_per_bin(
                    vaso2[C_CHARTTIME].values.astype(float),
                    vaso2[C_CHARTTIME].values.astype(float),  # cv has charttime only
                    vaso2[C_RATESTD].values.astype(float),
                    t0_bins, t1_bins)
                # Merge: combine MV and CV rates per bin
                for b in range(n_bins):
                    all_rates = [r for r in [vaso_med[b], cv_med[b]] if not np.isnan(r)]
                    if all_rates:
                        vaso_med[b] = np.nanmedian(all_rates)
                        vaso_max[b] = max(r for r in [vaso_max[b], cv_max[b]]
                                          if not np.isnan(r))

        # ── Demog ──────────────────────────────────────────────────────────
        try:
            demog_row = demog_groups.get_group(icustayid).iloc[0]
        except KeyError:
            continue
        dem = {
            C_GENDER:       demog_row[C_GENDER],
            C_AGE:          demog_row[C_AGE],
            C_ELIXHAUSER:   demog_row[C_ELIXHAUSER],
            C_RE_ADMISSION: demog_row[C_ADM_ORDER] > 1,
            C_DIED_IN_HOSP: demog_row[C_MORTA_HOSP],
            C_DIED_WITHIN_48H_OF_OUT_TIME:
                abs(demog_row[C_DOD] - demog_row[C_OUTTIME]) < (24 * 3600 * 2),
            C_MORTA_90:     demog_row[C_MORTA_90],
            C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH:
                (qstime.loc[icustayid, C_DISCHTIME] - qstime.loc[icustayid, C_LAST_TIMESTEP]) / 3600,
        }

        # ── State means per bin (one groupby instead of 18 × 65 .mean() calls) ──
        ts = temp[C_TIMESTEP].values.astype(float)
        bin_idx = np.floor((ts - beg) / (3600.0 * timestep_resolution)).astype(int)
        valid   = (bin_idx >= 0) & (bin_idx < n_bins)
        if not valid.any():
            continue
        temp_v  = temp.iloc[valid]
        bidx_v  = bin_idx[valid]
        t_aug   = temp_v.assign(_bin=bidx_v)
        state_means = t_aug.groupby('_bin')[SAH_FIELD_NAMES].mean()
        occupied = state_means.index.values  # bin indices with data

        if len(occupied) == 0:
            continue

        # ── Cumulative totals over occupied bins only ──────────────────────
        input_step_occ = input_step_bins[occupied]
        input_total    = totvol_0 + np.cumsum(input_step_occ)
        uo_step_occ    = uo_bins[occupied]
        output_total   = UOtot_0 + np.cumsum(uo_step_occ)
        cum_balance    = input_total - output_total

        # ── Assemble output rows ───────────────────────────────────────────
        for k, b in enumerate(occupied):
            item = {
                C_BLOC:      float(bloc_numbers[b]),
                C_ICUSTAYID: icustayid,
                C_TIMESTEP:  int(t0_bins[b]),
            }
            item.update(dem)
            item.update(state_means.loc[b].to_dict())

            if not np.isnan(vaso_med[b]):
                item[C_MEDIAN_DOSE_VASO] = vaso_med[b]
                item[C_MAX_DOSE_VASO]    = vaso_max[b]

            item[C_INPUT_STEP]        = float(input_step_occ[k])
            item[C_INPUT_TOTAL]       = float(input_total[k])
            item[C_OUTPUT_STEP]       = float(uo_step_occ[k])
            item[C_OUTPUT_TOTAL]      = float(output_total[k])
            item[C_CUMULATED_BALANCE] = float(cum_balance[k])
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
        'Bins patient states into time windows and adds I/O and vasopressor data. VECTORIZED.'))
    parser.add_argument('input',   type=str)
    parser.add_argument('qstime',  type=str)
    parser.add_argument('output',  type=str)
    parser.add_argument('--data',            dest='data_dir',        type=str,   default=None)
    parser.add_argument('--resolution',      dest='resolution',      type=float, default=4.0)
    parser.add_argument('--window-before',   dest='window_before',   type=int,   default=49)
    parser.add_argument('--window-after',    dest='window_after',    type=int,   default=25)
    parser.add_argument('--head',            dest='head',            type=int,   default=None)
    parser.add_argument('--filter-stays',    dest='filter_stays_path', type=str, default=None)
    parser.add_argument('--checkpoint-every',dest='checkpoint_every',type=int,   default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument('--resume',          dest='resume',          action='store_true', default=False)
    parser.add_argument('--log',             dest='log_file',        type=str,   default=None)

    args     = parser.parse_args()
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
    logging.info("Step 06 (vectorized) started. output=%s resume=%s", args.output, args.resume)

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
    except FileNotFoundError:
        inputCV = None
        logging.info("  fluid_cv not found (expected for MIMIC-IV)")
    try:
        vasoCV = load_intermediate_or_raw_csv(data_dir, 'vaso_cv.csv')
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

    df_check = pd.read_csv(args.output, nrows=0)
    expected = DEMOGRAPHICS_FIELD_NAMES + SAH_FIELD_NAMES + IO_FIELD_NAMES + COMPUTED_FIELD_NAMES
    for col in expected:
        if col not in df_check.columns:
            logging.warning("Expected column '%s' absent from output", col)

    logging.info("Step 06 complete. Total rows written: %d", total)
