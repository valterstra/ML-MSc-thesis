"""Step 11 -- Off-policy evaluation of V3 DDQN policy.

Evaluates the trained DDQN against the clinician behavior policy on the
held-out test set. Reports agreement rates, Q-value comparison, DR estimate,
and readmission outcome analysis.

Usage:
  # Smoke test
  python scripts/sepsis_v3/step_11_evaluate.py --smoke

  # Full run (background with log)
  python scripts/sepsis_v3/step_11_evaluate.py > logs/step_11_evaluate_v3.log 2>&1
"""

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from careai.sepsis_v3.ddqn import OfflineDDQN
from careai.sepsis_v3.evaluate import evaluate_policy

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
        "--report-dir",
        default=str(ROOT / "reports/sepsis_v3"),
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Use smoke data and model",
    )
    args = parser.parse_args()

    if args.smoke:
        data_dir   = args.data_dir + "_smoke"
        model_dir  = args.model_dir + "_smoke"
        report_dir = args.report_dir + "_smoke"
    else:
        data_dir   = args.data_dir
        model_dir  = args.model_dir
        report_dir = args.report_dir

    data_dir   = Path(data_dir)
    model_dir  = Path(model_dir)
    report_dir = Path(report_dir)

    log.info("=== Step 11: V3 Evaluation%s ===", " [SMOKE]" if args.smoke else "")
    log.info("Data   : %s", data_dir)
    log.info("Model  : %s", model_dir)
    log.info("Report : %s", report_dir)

    log.info("Loading DDQN model...")
    agent = OfflineDDQN.load(model_dir, device="cpu")

    results = evaluate_policy(
        test_csv=data_dir / "rl_test.csv",
        train_csv=data_dir / "rl_train.csv",
        model=agent,
        report_dir=report_dir,
    )

    log.info("--- Results Summary ---")
    log.info("Full agreement rate : %.2f%%", 100 * results["agreement"]["full_agreement"])
    log.info("Mean Q (DDQN)       : %.4f", results["q_stats"]["mean_q_ddqn"])
    log.info("Mean Q (clinician)  : %.4f", results["q_stats"]["mean_q_clinician"])
    log.info("DR estimate         : %.4f +/- %.4f",
             results["dr_estimate"]["dr_mean"], results["dr_estimate"]["dr_std"])
    log.info("=== Step 11 complete ===")


if __name__ == "__main__":
    main()
