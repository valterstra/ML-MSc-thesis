"""
Step 02 — Preprocess raw data.

Faithful adaptation of ai_clinician/preprocessing/01_preprocess_raw_data.py.
Changes: import paths only (ai_clinician.* -> careai.sepsis.*).

Tasks:
  - Simplify itemids in ce/labs files using REF_VITALS/REF_LABS converters
  - Combine microbio + culture -> bacterio.csv
  - Impute missing icustay_ids in bacterio and abx from demog
  - Compute normalized rate of infusion for fluid_mv
  - Remove rows with null icustayid from mechvent, vaso_mv

Inputs:  data/interim/sepsis/raw_data/
Outputs: data/interim/sepsis/intermediates/

Usage:
    source ../.venv/Scripts/activate
    python scripts/sepsis/step_02_preprocess.py 2>&1 | tee logs/step_02_preprocess.log
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis.utils import load_csv, load_intermediate_or_raw_csv
from careai.sepsis.columns import (
    C_ITEMID, C_ICUSTAYID, C_CHARTTIME, C_CHARTDATE,
    C_SUBJECT_ID, C_HADM_ID, C_STARTDATE, C_MORTA_90, C_MORTA_HOSP,
    C_ELIXHAUSER, C_TEV, C_RATE, C_AMOUNT, C_NORM_INFUSION_RATE,
    REF_VITALS, REF_LABS,
)
from careai.sepsis.imputation import impute_icustay_ids

tqdm.pandas()

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Preprocesses chart and lab events to save space and deduplicate '
        'itemids. Combines microbio and culture -> bacterio. Normalises '
        'fluid infusion rates.'))
    parser.add_argument('--in', dest='input_dir', type=str, default=None,
                        help='Raw data directory (default: data/interim/sepsis/raw_data/)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Intermediates directory (default: data/interim/sepsis/intermediates/)')
    parser.add_argument('--no-events', dest='no_simplify_events', action='store_true',
                        help="Skip simplifying ce and labs itemids")
    parser.add_argument('--no-bacterio', dest='no_bacterio', action='store_true',
                        help="Skip producing bacterio.csv")
    parser.add_argument('--log', dest='log_file', type=str, default=None,
                        help='Log file path (default: logs/step_02_preprocess.log)')

    args = parser.parse_args()
    in_dir  = args.input_dir  or os.path.join(REPO_DIR, 'data', 'interim', 'sepsis', 'raw_data')
    out_dir = args.output_dir or os.path.join(REPO_DIR, 'data', 'interim', 'sepsis', 'intermediates')
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_02_preprocess.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 02 started. in=%s out=%s", in_dir, out_dir)

    if not args.no_simplify_events:
        paths = [p for p in os.listdir(in_dir)
                 if (p.startswith("labs") or p.startswith("ce")) and p.endswith(".csv")]
        logging.info("Simplifying itemids for %d files", len(paths))
        for file_name in tqdm(paths, desc='Simplifying itemids'):
            ref_vals = REF_LABS if file_name.startswith("labs") else REF_VITALS
            df = pd.read_csv(os.path.join(in_dir, file_name), dtype={C_ITEMID: int})
            converter = {x: i + 1 for i, ids in enumerate(ref_vals) for x in ids}
            df[C_ITEMID].replace(converter, inplace=True)
            df.to_csv(os.path.join(out_dir, file_name), index=False)
            logging.info("  %s -> %d rows", file_name, len(df))

    demog = load_csv(os.path.join(in_dir, 'demog.csv'), null_icustayid=True)
    abx   = load_csv(os.path.join(in_dir, 'abx.csv'),   null_icustayid=True)

    if not args.no_bacterio:
        logging.info("Generating bacterio")
        culture  = load_csv(os.path.join(in_dir, 'culture.csv'),  null_icustayid=True)
        microbio = load_csv(os.path.join(in_dir, 'microbio.csv'), null_icustayid=True)

        ii = microbio[C_CHARTTIME].isnull()
        microbio.loc[ii, C_CHARTTIME] = microbio.loc[ii, C_CHARTDATE]
        microbio.loc[:, C_CHARTDATE] = 0

        cols = [C_SUBJECT_ID, C_HADM_ID, C_ICUSTAYID, C_CHARTTIME]
        bacterio = pd.concat([microbio[cols], culture[cols]], ignore_index=True)

        n_null = pd.isna(bacterio[C_ICUSTAYID]).sum()
        logging.info("Bacterio: %d/%d null ICU stay IDs, imputing...", n_null, len(bacterio))
        bacterio.loc[pd.isna(bacterio[C_ICUSTAYID]), C_ICUSTAYID] = impute_icustay_ids(
            demog, bacterio[pd.isna(bacterio[C_ICUSTAYID])])
        logging.info("After imputation: %d nulls remain", pd.isna(bacterio[C_ICUSTAYID]).sum())

        bacterio = bacterio[~pd.isna(bacterio[C_ICUSTAYID])]
        bacterio.to_csv(os.path.join(out_dir, "bacterio.csv"), index=False)
        logging.info("bacterio.csv written: %d rows", len(bacterio))

    logging.info("Trimming unusable abx entries (no startdate or icustay_id)")
    abx = abx[(~pd.isna(abx[C_STARTDATE])) & (~pd.isna(abx[C_ICUSTAYID]))]
    n_null = pd.isna(abx[C_ICUSTAYID]).sum()
    logging.info("Abx: %d/%d nulls, imputing...", n_null, len(abx))
    abx.loc[pd.isna(abx[C_ICUSTAYID]), C_ICUSTAYID] = impute_icustay_ids(
        demog, abx[pd.isna(abx[C_ICUSTAYID])])
    logging.info("After imputation: %d nulls remain", pd.isna(abx[C_ICUSTAYID]).sum())
    abx.to_csv(os.path.join(out_dir, "abx.csv"), index=False)
    logging.info("abx.csv written: %d rows", len(abx))

    logging.info("Correcting NaNs [demog]")
    demog.loc[pd.isna(demog[C_MORTA_90]),   C_MORTA_90]   = 0
    demog.loc[pd.isna(demog[C_MORTA_HOSP]), C_MORTA_HOSP] = 0
    demog.loc[pd.isna(demog[C_ELIXHAUSER]), C_ELIXHAUSER] = 0
    demog.to_csv(os.path.join(out_dir, "demog.csv"), index=False)
    logging.info("demog.csv written: %d rows", len(demog))

    logging.info("Computing normalized rate of infusion [fluid_mv]")
    inputMV = load_csv(os.path.join(in_dir, 'fluid_mv.csv'))
    inputMV.loc[:, C_NORM_INFUSION_RATE] = inputMV[C_TEV] * inputMV[C_RATE] / inputMV[C_AMOUNT]
    inputMV.to_csv(os.path.join(out_dir, "fluid_mv.csv"), index=False)
    logging.info("fluid_mv.csv written: %d rows", len(inputMV))

    logging.info("Correcting NaNs [mechvent]")
    mechvent = load_csv(os.path.join(in_dir, 'mechvent.csv'), null_icustayid=True)
    mechvent = mechvent[~pd.isna(mechvent[C_ICUSTAYID])]
    mechvent.to_csv(os.path.join(out_dir, "mechvent.csv"), index=False)
    logging.info("mechvent.csv written: %d rows", len(mechvent))

    logging.info("Correcting NaNs [vaso_mv]")
    vaso_mv = load_csv(os.path.join(in_dir, 'vaso_mv.csv'), null_icustayid=True)
    vaso_mv = vaso_mv[~pd.isna(vaso_mv[C_ICUSTAYID])]
    vaso_mv.to_csv(os.path.join(out_dir, "vaso_mv.csv"), index=False)
    logging.info("vaso_mv.csv written: %d rows", len(vaso_mv))

    for fname in ('vaso_cv.csv', 'fluid_cv.csv'):
        try:
            df_cv = load_csv(os.path.join(in_dir, fname), null_icustayid=True)
            logging.info("Correcting NaNs [%s]", fname)
            df_cv = df_cv[~pd.isna(df_cv[C_ICUSTAYID])]
            df_cv.to_csv(os.path.join(out_dir, fname), index=False)
            logging.info("%s written: %d rows", fname, len(df_cv))
        except FileNotFoundError:
            logging.info("No %s found, skipping (MIMIC-IV: expected)", fname)

    logging.info("Step 02 complete.")
