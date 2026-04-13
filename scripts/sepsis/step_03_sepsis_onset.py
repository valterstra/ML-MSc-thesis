"""
Step 03 — Calculate sepsis onset times.

Faithful adaptation of ai_clinician/preprocessing/02_calculate_sepsis_onset.py.
Changes: import paths only (ai_clinician.* -> careai.sepsis.*).

Saves output immediately on completion (single-pass, fast enough to not need checkpoints).

Inputs:
  data/interim/sepsis/intermediates/abx.csv      (from step 02)
  data/interim/sepsis/intermediates/bacterio.csv (from step 02)
Outputs:
  data/interim/sepsis/intermediates/sepsis_onset.csv

Usage:
    source ../.venv/Scripts/activate
    python scripts/sepsis/step_03_sepsis_onset.py 2>&1 | tee logs/step_03_sepsis_onset.log
"""

import pandas as pd
import numpy as np
import os
import sys
import argparse
import logging
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from careai.sepsis.columns import C_ICUSTAYID
from careai.sepsis.utils import load_csv, load_intermediate_or_raw_csv
from careai.sepsis.derived_features import calculate_onset

REPO_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=(
        'Calculates the presumed time of sepsis onset for each patient and '
        'generates sepsis_onset.csv.'))
    parser.add_argument('--data', dest='data_dir', type=str, default=None,
                        help='Data parent directory (default: data/interim/sepsis/)')
    parser.add_argument('--out', dest='output_dir', type=str, default=None,
                        help='Output directory (default: data/interim/sepsis/intermediates/)')
    parser.add_argument('--log', dest='log_file', type=str, default=None,
                        help='Log file path (default: logs/step_03_sepsis_onset.log)')

    args = parser.parse_args()
    data_dir = args.data_dir or os.path.join(REPO_DIR, 'data', 'interim', 'sepsis')
    out_dir  = args.output_dir or os.path.join(data_dir, 'intermediates')
    os.makedirs(out_dir, exist_ok=True)

    log_file = args.log_file or os.path.join(REPO_DIR, 'logs', 'step_03_sepsis_onset.log')
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
        datefmt='%H:%M:%S',
        handlers=[logging.StreamHandler(), logging.FileHandler(log_file)],
    )

    logging.info("Step 03 started.")
    abx      = load_intermediate_or_raw_csv(data_dir, "abx.csv")
    bacterio = load_csv(os.path.join(data_dir, "intermediates", "bacterio.csv"))

    unique_stays = abx[C_ICUSTAYID].unique()
    logging.info("Computing sepsis onset for %d unique ICU stays", len(unique_stays))

    onset_data = pd.DataFrame([
        onset for onset in (
            calculate_onset(abx, bacterio, stay_id)
            for stay_id in tqdm(unique_stays, desc='Sepsis onset')
        )
        if onset is not None
    ])

    out_path = os.path.join(out_dir, "sepsis_onset.csv")
    onset_data.to_csv(out_path, index=False)
    logging.info("Step 03 complete. %d sepsis onset records -> %s", len(onset_data), out_path)
