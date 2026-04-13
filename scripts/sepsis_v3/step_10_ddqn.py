"""Step 10 -- Offline Dueling DDQN training for V3 RL pipeline.

Trains on preprocessed transition data from Step 09.

Usage:
  # Smoke test (2k steps)
  python scripts/sepsis_v3/step_10_ddqn.py --smoke

  # Full run (background with log)
  python scripts/sepsis_v3/step_10_ddqn.py > logs/step_10_ddqn_v3.log 2>&1
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from careai.sepsis_v3.ddqn import OfflineDDQN

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
        "--data-dir",
        default=str(ROOT / "data/processed/sepsis_v3"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(ROOT / "models/sepsis_v3/ddqn"),
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=50_000,
        help="Number of gradient steps (default: 50000)",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run on smoke data for 500 steps only",
    )
    args = parser.parse_args()

    if args.smoke:
        data_dir  = args.data_dir + "_smoke"
        model_dir = args.model_dir + "_smoke"
        n_steps   = 500
    else:
        data_dir  = args.data_dir
        model_dir = args.model_dir
        n_steps   = args.n_steps

    data_dir  = Path(data_dir)
    model_dir = Path(model_dir)

    train_csv = data_dir / "rl_train.csv"
    val_csv   = data_dir / "rl_val.csv"

    log.info("=== Step 10: V3 DDQN%s ===", " [SMOKE]" if args.smoke else "")
    log.info("Data   : %s", data_dir)
    log.info("Model  : %s", model_dir)
    log.info("Steps  : %d", n_steps)

    agent = OfflineDDQN(device="cpu")
    agent.fit(
        train_csv=train_csv,
        val_csv=val_csv,
        n_steps=n_steps,
        log_every=max(100, n_steps // 10),
    )
    agent.save(model_dir)

    log.info("=== Step 10 complete ===")


if __name__ == "__main__":
    main()
