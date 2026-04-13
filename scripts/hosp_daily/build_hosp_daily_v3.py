"""CLI entry point for building the hospital-stay clean triplet dataset v3.

V3: first-draw labs, no forward-fill, causal ordering filter, triplet output.
    Core state (13 labs + 3 context): all must be measured for row inclusion.
    Actions (5): antibiotic, anticoagulant, diuretic, steroid, insulin.
    All V2 extras kept: eMAR, IV route, discharge meds, DRG, demographics, etc.
    Output: ~1.02M triplet rows (state_T, action_T, next_state_T+1).

Usage:
    # Smoke test on ~500 episodes
    python scripts/hosp_daily/build_hosp_daily_v3.py \\
        --config configs/hosp_daily_v3.yaml --sample-only --n-episodes 500

    # Full build
    python scripts/hosp_daily/build_hosp_daily_v3.py \\
        --config configs/hosp_daily_v3.yaml

    # Dry-run (Step 1 only, log counts, exit)
    python scripts/hosp_daily/build_hosp_daily_v3.py \\
        --config configs/hosp_daily_v3.yaml --dry-run

    # Re-run specific steps only (e.g. redo triplets + output)
    python scripts/hosp_daily/build_hosp_daily_v3.py \\
        --config configs/hosp_daily_v3.yaml --steps 8,9
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.hosp_daily.build_v3 import run_pipeline_v3, ALL_STEPS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build hospital-stay clean triplet dataset v3."
    )
    parser.add_argument(
        "--config", required=True, type=str,
        help="Path to YAML config file (e.g. configs/hosp_daily_v3.yaml)",
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

    run_pipeline_v3(
        cfg,
        steps=steps,
        dry_run=args.dry_run,
        sample_only=args.sample_only,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
