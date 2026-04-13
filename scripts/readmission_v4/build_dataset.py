"""CLI entry point for building the admission-level readmission dataset v4.

V4: One row per hospital admission.
    Goal: causal discovery of large discharge-time actions that reduce 30-day readmission.
    Exclusions: in-hospital deaths, OBS/GYN, post-discharge death within 30 days.
    ~545k admissions, ~85 features + readmit_30d outcome.

Usage:
    # Smoke test on ~500 admissions
    python scripts/readmission_v4/build_dataset.py \\
        --config configs/readmission_v4.yaml --sample-only --n-episodes 500

    # Full build
    python scripts/readmission_v4/build_dataset.py \\
        --config configs/readmission_v4.yaml

    # Dry-run (Step 1 only, log counts, exit)
    python scripts/readmission_v4/build_dataset.py \\
        --config configs/readmission_v4.yaml --dry-run

    # Resume from latest checkpoint (skips completed steps)
    python scripts/readmission_v4/build_dataset.py \\
        --config configs/readmission_v4.yaml --resume

    # Run specific steps only
    python scripts/readmission_v4/build_dataset.py \\
        --config configs/readmission_v4.yaml --steps 5,6,7
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.readmission_v4.build import run_pipeline_v4, ALL_STEPS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build admission-level 30-day readmission dataset v4."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file (e.g. configs/readmission_v4.yaml)",
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated subset of steps to run, e.g. '1,2,3'",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run Step 1 only, print admission counts, then exit",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Build only a small sample (default ~500 episodes). "
             "Use --n-episodes to override.",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Override sample size for --sample-only, e.g. --n-episodes 500",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from the latest checkpoint (skips already-completed steps).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    config_path = Path(args.config)
    if not config_path.exists():
        print(f"ERROR: Config file not found: {config_path}", file=sys.stderr)
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg["_project_root"] = str(PROJECT_ROOT)

    steps = None
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    if args.n_episodes is not None:
        cfg["sample"]["n_episodes"] = args.n_episodes

    run_pipeline_v4(
        cfg,
        steps=steps,
        dry_run=args.dry_run,
        sample_only=args.sample_only,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
