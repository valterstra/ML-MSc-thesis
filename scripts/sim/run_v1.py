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
from careai.sim.data_prep import prepare_sim_data
from careai.sim.env_bandit_v1 import BanditEnvV1
from careai.sim.outcome_model import fit_outcome_model
from careai.sim.policies import AlwaysHighPolicy, AlwaysLowPolicy, ObservedPolicy, RiskThresholdPolicy
from careai.sim.reporting import summary_markdown, summary_payload
from careai.sim.runner import run_policies
from careai.sim.weights import build_sim_training_weights


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one-step simulator (Sim v1).")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sim_v1.yaml"))
    parser.add_argument("--input", default=None, help="Optional override for transitions CSV path.")
    parser.add_argument("--n-episodes", type=int, default=None, help="Override simulation.n_episodes")
    parser.add_argument("--seed", type=int, default=None, help="Override simulation.seed")
    parser.add_argument("--high-cost", type=float, default=None, help="Override reward.cost_high")
    parser.add_argument("--threshold", type=float, default=None, help="Override policies.risk_threshold")
    parser.add_argument(
        "--weighting-mode",
        type=str,
        default=None,
        choices=["none", "ipw_stabilized"],
        help="Override model.weighting.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg: dict[str, Any] = load_yaml(config_path)
    if args.n_episodes is not None:
        cfg["simulation"]["n_episodes"] = int(args.n_episodes)
    if args.seed is not None:
        cfg["simulation"]["seed"] = int(args.seed)
    if args.high_cost is not None:
        cfg["reward"]["cost_high"] = float(args.high_cost)
    if args.threshold is not None:
        cfg["policies"]["risk_threshold"] = float(args.threshold)
    cfg.setdefault("model", {})
    if args.weighting_mode is not None:
        cfg["model"]["weighting"] = str(args.weighting_mode)
    weighting_mode = str(cfg["model"].get("weighting", "none"))

    in_path = resolve_from_config(config_path, cfg["input"]["transitions_path"])
    if args.input:
        in_path = Path(args.input).resolve()
    out_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg["output"]["prefix"])

    df = pd.read_csv(in_path)
    prep = prepare_sim_data(df, cfg)
    sample_weight = None
    weighting_diag = {"mode": "none"}
    train_weights_df = None
    if weighting_mode == "ipw_stabilized":
        wres = build_sim_training_weights(prep.train_df, cfg)
        sample_weight = wres.sample_weight
        weighting_diag = dict(wres.diagnostics)
        train_weights_df = prep.train_df[["a_t"]].copy()
        train_weights_df["sample_weight"] = sample_weight
        train_weights_df["propensity_high"] = wres.propensity
    model = fit_outcome_model(prep.train_df, cfg, sample_weight=sample_weight)

    env = BanditEnvV1(
        eval_df=prep.eval_df,
        outcome_model=model,
        cfg=cfg,
        seed=int(cfg["simulation"]["seed"]),
    )
    policies = [
        AlwaysLowPolicy(),
        AlwaysHighPolicy(),
        ObservedPolicy(),
        RiskThresholdPolicy(threshold=float(cfg["policies"]["risk_threshold"])),
    ]
    outputs = run_policies(env, policies=policies, n_episodes=int(cfg["simulation"]["n_episodes"]))

    payload = summary_payload(cfg, outputs.metrics_df, weighting=weighting_diag)
    payload["config_path"] = str(config_path)
    payload["input_path"] = str(in_path)
    payload["n_train_rows"] = int(len(prep.train_df))
    payload["n_eval_rows"] = int(len(prep.eval_df))

    summary_json = out_dir / f"{prefix}_summary.json"
    summary_md = out_dir / f"{prefix}_summary.md"
    metrics_csv = out_dir / f"{prefix}_policy_metrics.csv"
    episodes_csv = out_dir / f"{prefix}_episodes.csv"
    train_weights_csv = out_dir / f"{prefix}_train_weights.csv"

    summary_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_md.write_text(summary_markdown(payload), encoding="utf-8")
    outputs.metrics_df.to_csv(metrics_csv, index=False)
    outputs.episodes_df.to_csv(episodes_csv, index=False)
    if train_weights_df is not None:
        train_weights_df.to_csv(train_weights_csv, index=False)

    print(json.dumps(payload, indent=2))
    print(f"Wrote: {summary_json}")
    print(f"Wrote: {summary_md}")
    print(f"Wrote: {metrics_csv}")
    print(f"Wrote: {episodes_csv}")
    if train_weights_df is not None:
        print(f"Wrote: {train_weights_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


