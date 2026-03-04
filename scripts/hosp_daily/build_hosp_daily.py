"""CLI entry point for building the hospital-stay daily transition dataset.

Usage:
    python scripts/hosp_daily/build_hosp_daily.py --config configs/hosp_daily.yaml
    python scripts/hosp_daily/build_hosp_daily.py --config configs/hosp_daily.yaml --steps 1,2,3
    python scripts/hosp_daily/build_hosp_daily.py --config configs/hosp_daily.yaml --dry-run
    python scripts/hosp_daily/build_hosp_daily.py --config configs/hosp_daily.yaml --sample-only
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# Ensure the project src is importable
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.hosp_daily.build import run_pipeline  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hospital-stay daily transition dataset."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file (e.g. configs/hosp_daily.yaml)",
    )
    parser.add_argument(
        "--steps", type=str, default=None,
        help="Comma-separated subset of steps to run, e.g. '1,2,3'",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Run Step 1 only, print row counts, then exit",
    )
    parser.add_argument(
        "--sample-only", action="store_true",
        help="Build only ~5k-episode sample (much faster — samples subjects "
             "early so heavy SQL steps query less data)",
    )
    parser.add_argument(
        "--n-episodes", type=int, default=None,
        help="Override sample size, e.g. --n-episodes 50 for a tiny test run",
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

    # Attach project root so output paths resolve correctly
    cfg["_project_root"] = str(PROJECT_ROOT)

    steps = None
    if args.steps:
        steps = [int(s.strip()) for s in args.steps.split(",")]

    if args.n_episodes is not None:
        cfg["sample"]["n_episodes"] = args.n_episodes

    run_pipeline(cfg, steps=steps, dry_run=args.dry_run,
                 sample_only=args.sample_only)


if __name__ == "__main__":
    main()
