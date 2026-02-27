from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.sim_hourly.data import prepare_hourly_data
from careai.sim_hourly.dynamics import fit_dynamics_model
from careai.sim_hourly.policies import build_policies
from careai.sim_hourly.readmission import fit_readmission_model
from careai.sim_hourly.reporting import build_summary, summary_markdown
from careai.sim_hourly.rollout import rollout_policies


def main() -> int:
    parser = argparse.ArgumentParser(description="Run hourly simulator v1 with multi-step rollouts.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sim_hourly_v1.yaml"))
    parser.add_argument("--transitions-input", default=None)
    parser.add_argument("--episode-input", default=None)
    parser.add_argument("--n-rollouts", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg: dict[str, Any] = load_yaml(config_path)
    if args.n_rollouts is not None:
        cfg["simulation"]["n_rollouts_per_policy"] = int(args.n_rollouts)
    if args.max_steps is not None:
        cfg["simulation"]["max_steps"] = int(args.max_steps)
    if args.seed is not None:
        cfg["simulation"]["seed"] = int(args.seed)

    transitions_path = resolve_from_config(config_path, cfg["input"]["transitions_path"])
    episode_path = resolve_from_config(config_path, cfg["input"]["episode_table_path"])
    if args.transitions_input:
        transitions_path = Path(args.transitions_input).resolve()
    if args.episode_input:
        episode_path = Path(args.episode_input).resolve()

    out_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg["output"]["prefix"])

    transitions = pd.read_csv(transitions_path)
    episodes = pd.read_csv(episode_path)

    hourly = prepare_hourly_data(transitions, cfg)
    dynamics = fit_dynamics_model(hourly.train, cfg, hourly.state_cols, hourly.action_cols)
    readmit = fit_readmission_model(episodes, cfg)
    observed_actions = hourly.train[hourly.action_cols].dropna().to_numpy(dtype=float)
    policies = build_policies(observed_actions=observed_actions)
    wanted = set(cfg["actions"]["policies"])
    policies = [p for p in policies if p.name in wanted]
    outputs = rollout_policies(
        start_states=hourly.eval_starts,
        dynamics=dynamics,
        readmit=readmit,
        policies=policies,
        max_steps=int(cfg["simulation"]["max_steps"]),
        n_rollouts=int(cfg["simulation"]["n_rollouts_per_policy"]),
        done_threshold=float(cfg["simulation"]["done_threshold"]),
        seed=int(cfg["simulation"]["seed"]),
    )

    summary = build_summary(cfg, outputs.policy_metrics, len(hourly.train), len(hourly.eval_starts))
    summary["config_path"] = str(config_path)
    summary["transitions_path"] = str(transitions_path)
    summary["episode_path"] = str(episode_path)

    summary_json = out_dir / f"{prefix}_summary.json"
    summary_md = out_dir / f"{prefix}_summary.md"
    policy_csv = out_dir / f"{prefix}_policy_metrics.csv"
    traj_csv = out_dir / f"{prefix}_trajectories.csv"
    summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary_md.write_text(summary_markdown(summary), encoding="utf-8")
    outputs.policy_metrics.to_csv(policy_csv, index=False)
    outputs.trajectories.to_csv(traj_csv, index=False)

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {policy_csv}")
    print(f"Wrote: {traj_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

