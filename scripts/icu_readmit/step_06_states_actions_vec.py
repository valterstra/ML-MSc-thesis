"""
Step 06 -- Build binned states and actions (4-hour time windows). VECTORIZED VERSION.

Port of scripts/sepsis/step_06_states_actions_vec.py to the ICU readmission pipeline.

Speed improvements vs the per-stay per-bin loop:
  1. Pre-group all DataFrames by icustayid once -- eliminates repeated df[df[id]==x] scans.
  2. Per-stay state aggregation: assign bin_index to each row, call groupby.mean() once
     instead of one .mean() call per 4-hour bloc.
  3. Fluid / UO / vasopressor / drug flags: numpy broadcasting [n_bins x n_events]
     instead of per-bloc boolean masks in a Python loop.
  4. np.cumsum for running input_total / output_total accumulators.

Output is mathematically identical to step_06_states_actions.py.

Key ICU readmit differences vs sepsis vectorized version:
  - Variable n_bins per patient (full ICU stay, not fixed onset window).
  - beg = intime from stay_bounds (not first chartevent timestamp).
  - 9 binary drug class flags added via drug_per_bin() function.
  - Extended demographics: Charlson + 18 flags, race, insurance, etc.
  - Outcome: readmit_30d (not mortality).
  - No CareVue tables (MIMIC-IV only).

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_06_states_actions_vec.py \\
        data/interim/icu_readmit/intermediates/patient_states/patient_states_imputed.csv \\
        data/interim/icu_readmit/intermediates/patient_states/icu_stay_bounds.csv \\
        data/interim/icu_readmit/intermediates/states_actions.csv \\
        --log logs/step_06_icu_readmit_full.log
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

_EMPTY_FLUID = pd.DataFrame(columns=[C_ICUSTAYID, C_STARTTIME, C_ENDTIME,
                                      C_RATE, C_NORM_INFUSION_RATE, C_TEV])
_EMPTY_VASO  = pd.DataFrame(columns=[C_ICUSTAYID, C_STARTTIME, C_ENDTIME, C_RATESTD])
_EMPTY_UO    = pd.DataFrame(columns=[C_ICUSTAYID, C_CHARTTIME, C_VALUE])
_EMPTY_PREADM_FLUID = pd.DataFrame(columns=[C_ICUSTAYID, C_INPUT_PREADM])
_EMPTY_PREADM_UO    = pd.DataFrame(columns=[C_ICUSTAYID, C_VALUE])
_EMPTY_DRUGS = pd.DataFrame(columns=[C_ICUSTAYID, C_STARTTIME, C_ENDTIME, C_ITEMID])

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


def safe_get(groups, key, empty):
    try:
        return groups.get_group(key)
    except KeyError:
        return empty


# ---------------------------------------------------------------------------
# Vectorized per-bin computation functions (numpy broadcasting)
# ---------------------------------------------------------------------------

def fluid_per_bin(startt, endt, rate, t0_bins, t1_bins):
    """Infusion volume per bin. startt/endt/rate: [n_fluid], t0/t1: [n_bins]."""
    if len(startt) == 0:
        return np.zeros(len(t0_bins))
    st = startt[np.newaxis, :]
    et = endt[np.newaxis, :]
    rt = rate[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]
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
    """Sum of bolus volumes in each bin."""
    if len(times) == 0:
        return np.zeros(len(t0_bins))
    bt = times[np.newaxis, :]
    bv = tevs[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]
    t1 = t1_bins[:, np.newaxis]
    mask = (bt >= t0) & (bt <= t1)
    return np.nansum(np.where(mask, bv, 0.0), axis=1)


def uo_per_bin(times, values, t0_bins, t1_bins):
    """Sum of urine output in each bin."""
    if len(times) == 0:
        return np.zeros(len(t0_bins))
    ut = times[np.newaxis, :]
    uv = values[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]
    t1 = t1_bins[:, np.newaxis]
    mask = (ut >= t0) & (ut <= t1)
    return np.nansum(np.where(mask, uv, 0.0), axis=1)


def vaso_per_bin(startv, endv, ratev, t0_bins, t1_bins):
    """(median_dose, max_dose) per bin for vasopressors."""
    n_bins     = len(t0_bins)
    median_out = np.full(n_bins, np.nan)
    max_out    = np.full(n_bins, np.nan)
    if len(startv) == 0:
        return median_out, max_out
    sv = startv[np.newaxis, :]
    ev = endv[np.newaxis, :]
    rv = ratev[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]
    t1 = t1_bins[:, np.newaxis]
    active = (((ev >= t0) & (ev <= t1)) | ((sv >= t0) & (ev <= t1)) |
              ((sv >= t0) & (sv <= t1)) | ((sv <= t0) & (ev >= t1)))
    for b in range(n_bins):
        rates = rv[0, active[b]]
        rates = rates[~np.isnan(rates)]
        if rates.size > 0:
            median_out[b] = np.nanmedian(rates)
            max_out[b]    = np.nanmax(rates)
    return median_out, max_out


def drug_per_bin(startt, endt, t0_bins, t1_bins):
    """
    Binary (0/1) per bin: 1 if any drug event overlaps with the bin window.
    Matches the overlap logic in the original step_06_states_actions.py.
    startt/endt: [n_drug_events], t0/t1: [n_bins]
    Returns: int array [n_bins]
    """
    if len(startt) == 0:
        return np.zeros(len(t0_bins), dtype=int)
    st = startt[np.newaxis, :]   # [1, n_drug]
    et = endt[np.newaxis, :]
    t0 = t0_bins[:, np.newaxis]  # [n_bins, 1]
    t1 = t1_bins[:, np.newaxis]
    active = ((st < t1) & (et > t0)) | ((st >= t0) & (st < t1))
    return active.any(axis=1).astype(int)


# ---------------------------------------------------------------------------
# Main build function
# ---------------------------------------------------------------------------

def build_states_and_actions(df, stay_bounds, inputMV, inputpreadm,
                              vasoMV, demog, drugs_mv, UOpreadm, UO,
                              timestep_resolution,
                              out_path, checkpoint_every, resume,
                              head=None, allowed_stays=None):

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

    done_ids = set()
    if resume and os.path.exists(out_path):
        try:
            partial  = pd.read_csv(out_path, usecols=[C_ICUSTAYID])
            done_ids = set(partial[C_ICUSTAYID].dropna().astype(int).unique())
            logging.info("Resume: %d stays already processed", len(done_ids))
        except Exception as e:
            logging.warning("Could not read partial output for resume: %s", e)
    icustayidlist = [i for i in icustayidlist if i not in done_ids]
    logging.info("%d stays remaining to process", len(icustayidlist))

    # Pre-group everything by icustayid once
    logging.info("Pre-grouping all DataFrames by icustayid ...")
    df_groups          = df.groupby(C_ICUSTAYID)
    inputMV_groups     = inputMV.groupby(C_ICUSTAYID)
    inputpreadm_groups = inputpreadm.groupby(C_ICUSTAYID)
    vasoMV_groups      = vasoMV.groupby(C_ICUSTAYID)
    demog_groups       = demog.groupby(C_ICUSTAYID)
    UO_groups          = UO.groupby(C_ICUSTAYID)
    UOpreadm_groups    = UOpreadm.groupby(C_ICUSTAYID)
    drugs_groups       = drugs_mv.groupby(C_ICUSTAYID)
    logging.info("Pre-grouping done. Starting patient loop.")

    combined_data = []
    total_rows    = 0
    first_flush   = not bool(done_ids)

    for idx, icustayid in enumerate(tqdm(icustayidlist, desc='Building states and actions')):
        temp = safe_get(df_groups, icustayid, pd.DataFrame())
        if temp.empty:
            continue

        # ICU stay time bounds (variable per patient)
        if icustayid not in stay_bounds.index:
            continue
        bounds_row = stay_bounds.loc[icustayid]
        intime    = float(bounds_row[C_INTIME])
        outtime   = float(bounds_row[C_OUTTIME])
        dischtime = bounds_row[C_DISCHTIME]
        last_ts   = bounds_row[C_LAST_TIMESTEP]

        if pd.isna(intime) or pd.isna(outtime) or outtime <= intime:
            continue

        icu_los_hours = (outtime - intime) / 3600.0

        # Per-patient bin arrays (variable length)
        j_vals       = np.arange(0, icu_los_hours, timestep_resolution)
        n_bins       = len(j_vals)
        t0_bins      = intime + 3600.0 * j_vals
        t1_bins      = t0_bins + 3600.0 * timestep_resolution
        bloc_numbers = (j_vals / timestep_resolution) + 1

        # ── Fluid ──────────────────────────────────────────────────────
        input_     = safe_get(inputMV_groups, icustayid, _EMPTY_FLUID)
        startt     = input_[C_STARTTIME].values.astype(float)
        endt       = input_[C_ENDTIME].values.astype(float)
        rate       = input_[C_NORM_INFUSION_RATE].values.astype(float)
        bolus_mask = np.isnan(rate)

        pread_fluid = safe_get(inputpreadm_groups, icustayid, _EMPTY_PREADM_FLUID)
        totvol_0    = float(pread_fluid[C_INPUT_PREADM].sum()) if not pread_fluid.empty else 0.0

        # Pre-ICU fluid (t0=0, t1=intime)
        if (~bolus_mask).any():
            totvol_0 += float(fluid_per_bin(
                startt[~bolus_mask], endt[~bolus_mask], rate[~bolus_mask],
                np.array([0.0]), np.array([intime]))[0])
        if bolus_mask.any():
            totvol_0 += float(bolus_per_bin(
                input_.loc[bolus_mask, C_STARTTIME].values.astype(float),
                input_.loc[bolus_mask, C_TEV].values.astype(float),
                np.array([0.0]), np.array([intime]))[0])

        # Per-bin fluid
        infu_bins = fluid_per_bin(
            startt[~bolus_mask], endt[~bolus_mask], rate[~bolus_mask],
            t0_bins, t1_bins) if (~bolus_mask).any() else np.zeros(n_bins)
        bolus_bins = bolus_per_bin(
            input_.loc[bolus_mask, C_STARTTIME].values.astype(float),
            input_.loc[bolus_mask, C_TEV].values.astype(float),
            t0_bins, t1_bins) if bolus_mask.any() else np.zeros(n_bins)
        input_step_bins = infu_bins + bolus_bins

        # ── UO ─────────────────────────────────────────────────────────
        output   = safe_get(UO_groups, icustayid, _EMPTY_UO)
        pread_uo = safe_get(UOpreadm_groups, icustayid, _EMPTY_PREADM_UO)
        UOtot_0  = float(pread_uo[C_VALUE].sum()) if not pread_uo.empty else 0.0
        UOtot_0 += float(uo_per_bin(
            output[C_CHARTTIME].values.astype(float),
            output[C_VALUE].values.astype(float),
            np.array([0.0]), np.array([intime]))[0])
        uo_bins = uo_per_bin(
            output[C_CHARTTIME].values.astype(float),
            output[C_VALUE].values.astype(float),
            t0_bins, t1_bins)

        # ── Vasopressors ────────────────────────────────────────────────
        vaso1    = safe_get(vasoMV_groups, icustayid, _EMPTY_VASO)
        vaso_med, vaso_max = vaso_per_bin(
            vaso1[C_STARTTIME].values.astype(float),
            vaso1[C_ENDTIME].values.astype(float),
            vaso1[C_RATESTD].values.astype(float),
            t0_bins, t1_bins)

        # ── Drug class flags ────────────────────────────────────────────
        drug_stay  = safe_get(drugs_groups, icustayid, _EMPTY_DRUGS)
        drug_flags = {}
        for drug_class, itemids in DRUG_ACTION_ITEMIDS.items():
            d = drug_stay[drug_stay[C_ITEMID].isin(itemids)]
            drug_flags[drug_class] = drug_per_bin(
                d[C_STARTTIME].values.astype(float),
                d[C_ENDTIME].values.astype(float),
                t0_bins, t1_bins)

        # ── Demographics (broadcast to every bloc) ──────────────────────
        try:
            dr = demog_groups.get_group(icustayid).iloc[0]
        except KeyError:
            continue

        dem = {
            C_GENDER:          dr[C_GENDER],
            C_AGE:             dr[C_AGE],
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
            C_DIED_WITHIN_48H_OF_OUT_TIME: 0,
            C_MORTA_90:        dr.get(C_MORTA_90, 0),
            C_DELAY_END_OF_RECORD_AND_DISCHARGE_OR_DEATH:
                (float(dischtime) - float(last_ts)) / 3600
                if not pd.isna(dischtime) and not pd.isna(last_ts) else 0,
            C_READMIT_30D:     dr.get(C_READMIT_30D, 0),
        }
        for flag in CHARLSON_FLAG_COLS:
            dem[flag] = dr.get(flag, 0)

        # ── State means per bin (vectorized groupby instead of per-bloc loop) ──
        ts      = temp[C_TIMESTEP].values.astype(float)
        bin_idx = np.floor((ts - intime) / (3600.0 * timestep_resolution)).astype(int)
        valid   = (bin_idx >= 0) & (bin_idx < n_bins)
        if not valid.any():
            continue
        temp_v  = temp.iloc[valid]
        bidx_v  = bin_idx[valid]
        t_aug   = temp_v.assign(_bin=bidx_v)
        sah_cols_present = [c for c in SAH_FIELD_NAMES if c in temp_v.columns]
        state_means = t_aug.groupby('_bin')[sah_cols_present].mean()
        occupied    = state_means.index.values

        if len(occupied) == 0:
            continue

        # ── Cumulative totals (np.cumsum — no Python accumulator loop) ──
        input_step_occ = input_step_bins[occupied]
        input_total    = totvol_0 + np.cumsum(input_step_occ)
        uo_step_occ    = uo_bins[occupied]
        output_total   = UOtot_0 + np.cumsum(uo_step_occ)
        cum_balance    = input_total - output_total

        # ── Assemble output rows ────────────────────────────────────────
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

            for drug_class in DRUG_ACTION_ITEMIDS:
                item[drug_class] = int(drug_flags[drug_class][b])

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
        'and binary drug class actions. VECTORIZED VERSION.'))
    parser.add_argument('input',       type=str, help='patient_states_imputed.csv')
    parser.add_argument('stay_bounds', type=str, help='icu_stay_bounds.csv')
    parser.add_argument('output',      type=str, help='Output states_actions.csv path')
    parser.add_argument('--data',            dest='data_dir',        type=str,   default=None)
    parser.add_argument('--resolution',      dest='resolution',      type=float, default=4.0)
    parser.add_argument('--head',            dest='head',            type=int,   default=None)
    parser.add_argument('--filter-stays',    dest='filter_stays_path', type=str, default=None)
    parser.add_argument('--checkpoint-every',dest='checkpoint_every', type=int,
                        default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument('--resume',          dest='resume',          action='store_true',
                        default=False)
    parser.add_argument('--log',             dest='log_file',        type=str,   default=None)

    args     = parser.parse_args()
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
    logging.info("Step 06 (vectorized) started. output=%s resume=%s", args.output, args.resume)

    logging.info("Reading states ...")
    df = load_csv(args.input)
    logging.info("  %d rows loaded", len(df))

    logging.info("Reading stay bounds ...")
    stay_bounds = load_csv(args.stay_bounds, null_icustayid=True).set_index(C_ICUSTAYID,
                                                                             drop=True)
    logging.info("  %d stay records", len(stay_bounds))

    logging.info("Reading data files ...")
    interm      = os.path.join(data_dir, 'intermediates')
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

    logging.info("Step 06 (vectorized) complete. Total rows written: %d", total)
