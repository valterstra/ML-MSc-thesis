"""
Full RL training pipeline runner (steps 11 + 12).

Run this to kick off the complete training. Steps 09 and 10 are already done.
This script handles the Windows torch DLL issue by importing torch first.

Usage:
    python scripts/sepsis/run_full_rl.py
    python scripts/sepsis/run_full_rl.py --skip-autoencoder   (saves ~45 min)
"""
import torch  # Must import first on Windows to avoid DLL loading issues
import sys
import os
import time
import logging

# Setup master log
LOG_FILE = "logs/run_full_rl.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)

logging.info("=" * 60)
logging.info("SEPSIS RL FULL TRAINING PIPELINE")
logging.info("=" * 60)
logging.info("Steps 09-10: already complete")
logging.info("Steps 11-12: starting now")
logging.info("PyTorch: %s (CUDA: %s)", torch.__version__, torch.cuda.is_available())
logging.info("Master log: %s", LOG_FILE)
logging.info("Step 11 log: logs/step_11_continuous_rl.log")
logging.info("Step 12 log: logs/step_12_evaluate.log")
logging.info("=" * 60)

t0 = time.time()

# Parse args
skip_ae = "--skip-autoencoder" in sys.argv

# ── Step 11: Continuous RL ────────────────────────────────────────────
logging.info("")
logging.info(">>> STEP 11: Continuous RL (DQN + SARSA + Autoencoder)")
logging.info("    Expected time: ~1-2 hours on CPU")
if skip_ae:
    logging.info("    Autoencoder branch: SKIPPED")
logging.info("")

t11 = time.time()

step11_args = ["step_11", "--log", "logs/step_11_continuous_rl.log"]
if skip_ae:
    step11_args.append("--skip-autoencoder")
sys.argv = step11_args

# Reset logging for step 11 (it sets up its own handlers)
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

exec(open("scripts/sepsis/step_11_continuous_rl.py").read())

dt11 = time.time() - t11

# Restore master logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)

logging.info("")
logging.info(">>> STEP 11 COMPLETE: %.1f sec (%.1f min)", dt11, dt11 / 60)
logging.info("")

# ── Step 12: Evaluation ──────────────────────────────────────────────
logging.info(">>> STEP 12: Offline Policy Evaluation")
logging.info("    Expected time: ~20-30 min on CPU")
logging.info("")

t12 = time.time()

sys.argv = ["step_12", "--log", "logs/step_12_evaluate.log"]

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)

exec(open("scripts/sepsis/step_12_evaluate.py").read())

dt12 = time.time() - t12

# Restore master logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="a", encoding="utf-8"),
        logging.StreamHandler(),
    ],
    force=True,
)

logging.info("")
logging.info(">>> STEP 12 COMPLETE: %.1f sec (%.1f min)", dt12, dt12 / 60)

# ── Summary ───────────────────────────────────────────────────────────
dt_total = time.time() - t0
logging.info("")
logging.info("=" * 60)
logging.info("ALL DONE!")
logging.info("  Step 11 (Continuous RL): %.1f min", dt11 / 60)
logging.info("  Step 12 (Evaluation):    %.1f min", dt12 / 60)
logging.info("  Total:                   %.1f min", dt_total / 60)
logging.info("")
logging.info("Results:")
logging.info("  Models:  models/sepsis_rl/continuous/, models/sepsis_rl/eval/")
logging.info("  Reports: reports/sepsis_rl/evaluation_results.json")
logging.info("  Logs:    logs/step_11_continuous_rl.log, logs/step_12_evaluate.log")
logging.info("=" * 60)
