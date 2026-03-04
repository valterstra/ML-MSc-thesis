#!/usr/bin/env python
"""Load trained daily transition model, run rollouts, compare to real data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from careai.sim_daily.data import prepare_daily_data
from careai.sim_daily.env import DailySimEnv
from careai.sim_daily.evaluate import rollout_comparison, run_rollouts
from careai.sim_daily.transition import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate daily simulator via rollouts")
    parser.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions_sample5k.csv"),
    )
    parser.add_argument(
        "--model-dir",
        default=str(_PROJECT_ROOT / "models" / "sim_daily"),
    )
    parser.add_argument(
        "--report-dir",
        default=str(_PROJECT_ROOT / "reports" / "sim_daily"),
    )
    parser.add_argument("--n-rollouts", type=int, default=500)
    parser.add_argument("--max-days", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # 1. Load data (for initial states + real comparison)
    print(f"Loading data from {args.csv} ...")
    data = prepare_daily_data(args.csv)

    # 2. Load trained model
    print(f"Loading model from {args.model_dir} ...")
    model = load_model(args.model_dir)

    # 3. Create environment
    env = DailySimEnv(model, data.initial_states, max_days=args.max_days)

    # 4. Run rollouts
    print(f"Running {args.n_rollouts} rollouts (max {args.max_days} days, seed={args.seed}) ...")
    sim_traj = run_rollouts(env, n_rollouts=args.n_rollouts, max_days=args.max_days, seed=args.seed)
    print(f"  Simulated rows: {len(sim_traj):,}")

    # 5. Compare distributions
    print("\nRollout vs real distribution comparison:")
    comparison = rollout_comparison(sim_traj, data.raw)
    for col, m in comparison.items():
        if m.get("ks_stat") is not None:
            print(f"  {col:40s}  KS={m['ks_stat']:.4f}  p={m['ks_pval']:.6f}"
                  f"  sim_mean={m['sim_mean']:.2f}  real_mean={m['real_mean']:.2f}")
        else:
            print(f"  {col:40s}  (insufficient data)")

    # 6. Save outputs
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    comparison_path = report_dir / "rollout_comparison.json"
    comparison_path.write_text(json.dumps(comparison, indent=2))

    traj_path = report_dir / "simulated_trajectories.csv"
    sim_traj.to_csv(traj_path, index=False)

    print(f"\nComparison saved to {comparison_path}")
    print(f"Trajectories saved to {traj_path}")


if __name__ == "__main__":
    main()
