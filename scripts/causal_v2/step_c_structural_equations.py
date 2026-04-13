"""Step C: Fit structural equations on the PC-discovered causal DAG.

Usage:
    python scripts/causal_v2/step_c_structural_equations.py \
        --csv data/processed/hosp_daily_v2_transitions.csv \
        --parent-sets reports/causal_v2/step_b_300k/parent_sets.csv \
        --model-dir models/causal_v2/structural_equations \
        [--report-dir reports/causal_v2/step_c]
"""

import argparse
import logging
import sys
import time
from pathlib import Path


def setup_logging(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("step_c")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s %(levelname)s: %(message)s",
                            datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--parent-sets", required=True)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--report-dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    report_dir = Path(args.report_dir) if args.report_dir else model_dir
    log = setup_logging(report_dir / "run.log")

    log.info("=== Step C: Structural equation fitting ===")
    log.info("CSV:         %s", args.csv)
    log.info("Parent sets: %s", args.parent_sets)
    log.info("Model dir:   %s", model_dir)

    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from careai.causal_v2.structural_equations import (
        load_parent_sets, prepare_training_data,
        train_all_equations, save_models,
    )

    # Stage 1: Load parent sets
    log.info("[1/4] Loading parent sets (Option A: strip next_* parents)...")
    parent_sets = load_parent_sets(args.parent_sets)
    log.info("  %d targets loaded", len(parent_sets))

    # Stage 2: Load and prepare data
    log.info("[2/4] Loading dataset...")
    t0 = time.time()
    import pandas as pd
    df = pd.read_csv(args.csv, low_memory=False)
    log.info("  Loaded %d rows in %.1fs", len(df), time.time() - t0)

    log.info("[3/4] Preparing training data (transitions, train split)...")
    t0 = time.time()
    train_df = prepare_training_data(df, split="train")
    log.info("  Ready in %.1fs", time.time() - t0)

    # Stage 3: Train all equations
    log.info("[4/4] Training structural equations (1 LightGBM per target)...")
    t0 = time.time()
    models, all_metrics = train_all_equations(train_df, parent_sets)
    log.info("  Training complete in %.1fs", time.time() - t0)

    # Save
    save_models(models, all_metrics, parent_sets, model_dir)

    import pandas as pd
    metrics_df = pd.DataFrame(all_metrics)
    log.info("=== DONE ===")
    log.info("Training metrics summary:")
    for _, row in metrics_df.iterrows():
        r2 = row.get("train_r2", row.get("train_auc", "N/A"))
        log.info("  %s: r2/auc=%.3f  parents=%s",
                 row["target"], float(r2) if r2 != "N/A" else 0,
                 row.get("parents", []))


if __name__ == "__main__":
    main()
