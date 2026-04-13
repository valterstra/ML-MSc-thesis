"""Step 09 -- Preprocessing for V3 hospital RL pipeline.

Converts hosp_daily_v3_triplets.csv into normalized (s, a, r, s', done)
transition datasets for offline DDQN training.

Usage:
  # Smoke test (2k rows)
  python scripts/sepsis_v3/step_09_preprocess.py --smoke

  # Full run (background with log)
  python scripts/sepsis_v3/step_09_preprocess.py > logs/step_09_preprocess_v3.log 2>&1
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from careai.sepsis_v3.preprocessing import run_preprocessing

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv",
        default=str(ROOT / "data/processed/hosp_daily_v3_triplets.csv"),
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "data/processed/sepsis_v3"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on first 2k rows only for a quick sanity check",
    )
    args = parser.parse_args()

    log.info("=== Step 09: V3 RL Preprocessing%s ===", " [SMOKE]" if args.smoke else "")
    log.info("Input : %s", args.csv)
    log.info("Output: %s", args.out_dir)

    run_preprocessing(
        csv_path=args.csv,
        out_dir=args.out_dir if not args.smoke else args.out_dir + "_smoke",
        smoke=args.smoke,
    )
    log.info("=== Step 09 complete ===")


if __name__ == "__main__":
    main()
