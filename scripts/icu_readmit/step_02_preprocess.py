"""
Step 02 -- Preprocess raw data.

Adapted from scripts/sepsis/step_02_preprocess.py.

Tasks:
  - Simplify itemids in ce/labs files using REF_VITALS/REF_LABS converters
  - Copy demog.csv to intermediates, filling NaN in morta_hosp/morta_90/charlson flags
  - Compute normalized rate of infusion for fluid_mv
  - Remove rows with null icustayid from mechvent, vaso_mv, drugs_mv

Key differences from sepsis step_02:
  - No abx / culture / microbio / bacterio processing
  - demog: replaces elixhauser NaN fill with charlson_score + 18 cc_* flags NaN fill
  - drugs_mv: new file — copy with null-icustayid filter (same as vaso_mv treatment)

Inputs:  data/interim/icu_readmit/raw_data/
Outputs: data/interim/icu_readmit/intermediates/

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_02_preprocess.py 2>&1 | tee logs/step_02_icu_readmit.log
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
import argparse
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.icu_readmit.utils import load_csv
from careai.icu_readmit.columns import (
    C_ITEMID, C_ICUSTAYID,
    C_MORTA_HOSP, C_MORTA_90,
    C_CHARLSON, CHARLSON_FLAG_COLS,
    C_TEV, C_RATE, C_AMOUNT, C_NORM_INFUSION_RATE,
    REF_VITALS, REF_LABS,
)

tqdm.pandas()

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw ICU readmit data: simplify itemids, normalise fluid rates."
    )
    parser.add_argument('--in', dest='input_dir', type=str, default=None,
                        help='Raw data directory (default: data/interim/icu_readmit/raw_data/)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Intermediates directory (default: data/interim/icu_readmit/intermediates/)')
    parser.add_argument('--no-events', dest='no_simplify_events', action='store_true',
                        help="Skip simplifying ce and labs itemids")
    parser.add_argument('--log', dest='log_file', type=str, default=None,
                        help='Log file path (default: logs/step_02_icu_readmit.log)')
    args = parser.parse_args()

    in_dir  = args.input_dir  or os.path.join(REPO_DIR, 'data', 'interim', 'icu_readmit', 'raw_data')
    out_dir = args.output_dir or os.path.join(REPO_DIR, 'data', 'interim', 'icu_readmit', 'intermediates')
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_02_icu_readmit.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )
    logging.info("Step 02 started. in=%s out=%s", in_dir, out_dir)

    # ---- 1. Simplify itemids in ce / labs files --------------------------------
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

    # ---- 2. demog: fill NaN, write to intermediates ---------------------------
    logging.info("Processing demog.csv")
    demog = load_csv(os.path.join(in_dir, 'demog.csv'), null_icustayid=True)
    demog.loc[pd.isna(demog[C_MORTA_90]),   C_MORTA_90]   = 0
    demog.loc[pd.isna(demog[C_MORTA_HOSP]), C_MORTA_HOSP] = 0
    demog.loc[pd.isna(demog[C_CHARLSON]),   C_CHARLSON]   = 0
    for flag in CHARLSON_FLAG_COLS:
        if flag in demog.columns:
            demog.loc[pd.isna(demog[flag]), flag] = 0
    demog.to_csv(os.path.join(out_dir, "demog.csv"), index=False)
    logging.info("demog.csv written: %d rows", len(demog))

    # ---- 3. fluid_mv: compute normalized infusion rate -----------------------
    logging.info("Computing normalized rate of infusion [fluid_mv]")
    fluid_mv = load_csv(os.path.join(in_dir, 'fluid_mv.csv'))
    fluid_mv.loc[:, C_NORM_INFUSION_RATE] = (
        fluid_mv[C_TEV] * fluid_mv[C_RATE] / fluid_mv[C_AMOUNT]
    )
    fluid_mv.to_csv(os.path.join(out_dir, "fluid_mv.csv"), index=False)
    logging.info("fluid_mv.csv written: %d rows", len(fluid_mv))

    # ---- 4. mechvent: drop null icustayid rows --------------------------------
    logging.info("Filtering mechvent.csv")
    mechvent = load_csv(os.path.join(in_dir, 'mechvent.csv'), null_icustayid=True)
    mechvent = mechvent[~pd.isna(mechvent[C_ICUSTAYID])]
    mechvent.to_csv(os.path.join(out_dir, "mechvent.csv"), index=False)
    logging.info("mechvent.csv written: %d rows", len(mechvent))

    # ---- 5. vaso_mv: drop null icustayid rows ---------------------------------
    logging.info("Filtering vaso_mv.csv")
    vaso_mv = load_csv(os.path.join(in_dir, 'vaso_mv.csv'), null_icustayid=True)
    vaso_mv = vaso_mv[~pd.isna(vaso_mv[C_ICUSTAYID])]
    vaso_mv.to_csv(os.path.join(out_dir, "vaso_mv.csv"), index=False)
    logging.info("vaso_mv.csv written: %d rows", len(vaso_mv))

    # ---- 6. drugs_mv: drop null icustayid rows (new vs sepsis) ---------------
    logging.info("Filtering drugs_mv.csv")
    drugs_mv = load_csv(os.path.join(in_dir, 'drugs_mv.csv'), null_icustayid=True)
    drugs_mv = drugs_mv[~pd.isna(drugs_mv[C_ICUSTAYID])]
    drugs_mv.to_csv(os.path.join(out_dir, "drugs_mv.csv"), index=False)
    logging.info("drugs_mv.csv written: %d rows", len(drugs_mv))

    # ---- 7. Pass-through files (copy raw -> intermediates if not processed) --
    pass_through = ['mechvent_pe.csv', 'preadm_fluid.csv', 'preadm_uo.csv', 'uo.csv']
    for fname in pass_through:
        src = os.path.join(in_dir, fname)
        dst = os.path.join(out_dir, fname)
        if os.path.exists(src) and not os.path.exists(dst):
            df = pd.read_csv(src)
            df.to_csv(dst, index=False)
            logging.info("%s copied: %d rows", fname, len(df))

    logging.info("Step 02 complete.")


if __name__ == "__main__":
    main()
