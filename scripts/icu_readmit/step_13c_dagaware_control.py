"""Step 13c -- DAG-aware control layer: planner baseline + DDQN policy."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.control.ddqn import DQNConfig, greedy_policy_from_model, train_ddqn  # noqa: E402
from careai.icu_readmit.caresim.control.evaluation import (  # noqa: E402
    evaluate_policies,
    load_seed_episodes,
    make_planner_policy,
    make_random_policy,
    make_repeat_last_policy,
)
from careai.icu_readmit.caresim.control.planner import PlannerConfig  # noqa: E402
from careai.icu_readmit.caresim.readmit import LightGBMReadmitModel  # noqa: E402
from careai.icu_readmit.caresim.simulator import CareSimEnvironment  # noqa: E402
from careai.icu_readmit.caresim.severity import load_severity_model  # noqa: E402
from careai.icu_readmit.dagaware.ensemble import DAGAwareEnsemble  # noqa: E402
from careai.icu_readmit.rl.bandit import (  # noqa: E402
    BanditConfig,
    greedy_policy_from_values,
    load_bandit_model,
    save_bandit_artifacts,
    train_bandit,
)
from careai.icu_readmit.rl.networks import DuelingDQN  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 13c: DAG-aware control layer")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p):
        p.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
        p.add_argument("--model-dir", default="models/icu_readmit/dagaware_selected_causal")
        p.add_argument("--control-model-dir", default="models/icu_readmit/dagaware_control_selected_causal")
        p.add_argument("--report-dir", default="reports/icu_readmit/dagaware_control_selected_causal")
        p.add_argument("--history-len", type=int, default=5)
        p.add_argument("--observation-window", type=int, default=5)
        p.add_argument("--rollout-steps", type=int, default=5)
        p.add_argument("--planner-horizon", type=int, default=3)
        p.add_argument("--gamma", type=float, default=0.99)
        p.add_argument("--uncertainty-penalty", type=float, default=0.25)
        p.add_argument("--uncertainty-threshold", type=float, default=1.0)
        p.add_argument("--device", default=None)
        p.add_argument("--use-severity-reward", action="store_true")
        p.add_argument("--severity-mode", choices=["surrogate", "handcrafted"], default="handcrafted")
        p.add_argument("--severity-model-dir", default="models/icu_readmit/severity_selected")
        p.add_argument("--use-terminal-readmit-reward", action="store_true")
        p.add_argument("--terminal-model-dir", default="models/icu_readmit/terminal_readmit_selected")
        p.add_argument("--terminal-reward-scale", type=float, default=15.0)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--eval-progress-every", type=int, default=10)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--log", default="logs/step_13c_dagaware_control.log")

    planner_p = sub.add_parser("planner", help="Run planner-only evaluation")
    add_common(planner_p)
    planner_p.add_argument("--episodes-per-split", type=int, default=100)

    ddqn_p = sub.add_parser("train-ddqn", help="Train DDQN against DAG-aware simulator")
    add_common(ddqn_p)
    ddqn_p.add_argument("--train-episodes", type=int, default=1000)
    ddqn_p.add_argument("--train-steps", type=int, default=20000)
    ddqn_p.add_argument("--batch-size", type=int, default=64)
    ddqn_p.add_argument("--lr", type=float, default=1e-4)
    ddqn_p.add_argument("--target-sync-every", type=int, default=250)
    ddqn_p.add_argument("--warmup-steps", type=int, default=500)
    ddqn_p.add_argument("--replay-capacity", type=int, default=20000)
    ddqn_p.add_argument("--ddqn-log-every", type=int, default=100)
    ddqn_p.add_argument("--epsilon-start", type=float, default=1.0)
    ddqn_p.add_argument("--epsilon-end", type=float, default=0.10)
    ddqn_p.add_argument("--epsilon-decay-steps", type=int, default=20000)

    bandit_p = sub.add_parser("train-bandit", help="Train simple multi-armed bandit against DAG-aware simulator")
    add_common(bandit_p)
    bandit_p.add_argument("--train-episodes", type=int, default=1000)
    bandit_p.add_argument("--train-steps", type=int, default=5000)
    bandit_p.add_argument("--bandit-log-every", type=int, default=100)
    bandit_p.add_argument("--epsilon-start", type=float, default=1.0)
    bandit_p.add_argument("--epsilon-end", type=float, default=0.05)
    bandit_p.add_argument("--epsilon-decay-steps", type=int, default=5000)

    eval_p = sub.add_parser("eval", help="Evaluate planner and DDQN policies")
    add_common(eval_p)
    eval_p.add_argument("--episodes-per-split", type=int, default=100)
    eval_p.add_argument("--ddqn-path", default="models/icu_readmit/dagaware_control_selected_causal/ddqn_model.pt")
    eval_p.add_argument("--bandit-path", default="models/icu_readmit/dagaware_control_selected_causal/bandit_model.json")

    smoke_p = sub.add_parser("smoke", help="Run a compact planner + DDQN smoke pass")
    add_common(smoke_p)
    smoke_p.add_argument("--episodes-per-split", type=int, default=12)

    return parser


def setup_logging(log_path: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def planner_config_from_args(args) -> PlannerConfig:
    return PlannerConfig(
        horizon=args.planner_horizon,
        gamma=args.gamma,
        uncertainty_penalty=args.uncertainty_penalty,
        max_steps=args.rollout_steps,
        uncertainty_threshold=args.uncertainty_threshold,
    )


def ddqn_config_from_args(args) -> DQNConfig:
    train_steps = getattr(args, "train_steps", 10000)
    batch_size = getattr(args, "batch_size", 64)
    warmup_steps = getattr(args, "warmup_steps", 500)
    replay_capacity = getattr(args, "replay_capacity", 20000)
    target_sync_every = getattr(args, "target_sync_every", 250)
    log_every = getattr(args, "ddqn_log_every", 100)
    lr = getattr(args, "lr", 1e-4)
    epsilon_start = getattr(args, "epsilon_start", 1.0)
    epsilon_end = getattr(args, "epsilon_end", 0.10)
    epsilon_decay_steps = getattr(args, "epsilon_decay_steps", 20000)
    if args.smoke:
        train_steps = min(train_steps, 200)
        batch_size = min(batch_size, 32)
        warmup_steps = min(warmup_steps, 32)
        replay_capacity = min(replay_capacity, 1000)
        target_sync_every = min(target_sync_every, 50)
        log_every = min(log_every, 20)
        epsilon_decay_steps = min(epsilon_decay_steps, 500)
    return DQNConfig(
        observation_window=args.observation_window,
        rollout_steps=args.rollout_steps,
        gamma=args.gamma,
        lr=lr,
        batch_size=batch_size,
        replay_capacity=replay_capacity,
        warmup_steps=warmup_steps,
        train_steps=train_steps,
        target_sync_every=target_sync_every,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        uncertainty_penalty=args.uncertainty_penalty,
        uncertainty_threshold=args.uncertainty_threshold,
        log_every=log_every,
        seed=args.seed,
    )


def save_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def build_policy_diagnostics(traces):
    terminal_rows = traces[traces["step"] == 0].copy()
    step_rows = traces[traces["step"] > 0].copy()
    diagnostics = {"pairwise": {}}

    if not terminal_rows.empty:
        pivot = terminal_rows.pivot(index="stay_id", columns="policy", values="discounted_return")
        policy_names = list(pivot.columns)
        for i, left in enumerate(policy_names):
            for right in policy_names[i + 1:]:
                diff = (pivot[left] - pivot[right]).dropna()
                diagnostics["pairwise"][f"{left}_minus_{right}"] = {
                    "mean_diff": float(diff.mean()) if len(diff) else 0.0,
                    "win_rate": float((diff > 0).mean()) if len(diff) else 0.0,
                    "n_common": int(len(diff)),
                }

    policy_action_stats = {}
    for policy_name, policy_steps in step_rows.groupby("policy"):
        action_counts = policy_steps["action_id"].value_counts().sort_index()
        policy_action_stats[policy_name] = {
            "unique_actions": int(action_counts.shape[0]),
            "top_actions": [
                {"action_id": int(action_id), "count": int(count)}
                for action_id, count in action_counts.head(5).items()
            ],
        }
    diagnostics["actions"] = policy_action_stats
    return diagnostics


def save_ddqn_artifacts(model: DuelingDQN, metrics: dict, config: DQNConfig, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "ddqn_model.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "config": config.__dict__,
        "obs_dim": model.fc1.in_features,
        "n_actions": model.adv_out.out_features,
    }, model_path)
    save_json(out_dir / "ddqn_train_metrics.json", metrics)
    save_json(out_dir / "ddqn_train_config.json", config.__dict__)
    return model_path


def load_ddqn_model(model_path: str, device: torch.device) -> DuelingDQN:
    ckpt = torch.load(model_path, map_location=device)
    model = DuelingDQN(n_input=int(ckpt["obs_dim"]), n_actions=int(ckpt["n_actions"]), hidden=128).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def bandit_config_from_args(args) -> BanditConfig:
    train_steps = getattr(args, "train_steps", 5000)
    log_every = getattr(args, "bandit_log_every", 100)
    epsilon_start = getattr(args, "epsilon_start", 1.0)
    epsilon_end = getattr(args, "epsilon_end", 0.05)
    epsilon_decay_steps = getattr(args, "epsilon_decay_steps", 5000)
    if args.smoke:
        train_steps = min(train_steps, 200)
        log_every = min(log_every, 20)
        epsilon_decay_steps = min(epsilon_decay_steps, 500)
    return BanditConfig(
        rollout_steps=args.rollout_steps,
        train_steps=train_steps,
        epsilon_start=epsilon_start,
        epsilon_end=epsilon_end,
        epsilon_decay_steps=epsilon_decay_steps,
        uncertainty_penalty=args.uncertainty_penalty,
        log_every=log_every,
        seed=args.seed,
    )


def run_evaluation(
    args,
    ensemble: DAGAwareEnsemble,
    device: torch.device,
    ddqn_model: DuelingDQN | None = None,
    bandit_values: np.ndarray | None = None,
    severity_model = None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
) -> dict:
    planner_cfg = planner_config_from_args(args)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    summary_bundle = {
        "meta": {
            "data": args.data,
            "model_dir": args.model_dir,
            "history_len": args.history_len,
            "observation_window": args.observation_window,
            "rollout_steps": args.rollout_steps,
            "planner_horizon": args.planner_horizon,
            "uncertainty_penalty": args.uncertainty_penalty,
            "uncertainty_threshold": args.uncertainty_threshold,
            "use_severity_reward": bool(args.use_severity_reward),
            "severity_mode": args.severity_mode if args.use_severity_reward else None,
            "severity_model_dir": args.severity_model_dir if args.use_severity_reward else None,
            "use_terminal_readmit_reward": bool(args.use_terminal_readmit_reward),
            "terminal_model_dir": args.terminal_model_dir if args.use_terminal_readmit_reward else None,
            "terminal_reward_scale": args.terminal_reward_scale if args.use_terminal_readmit_reward else None,
        }
    }

    for split, split_seed in [("val", args.seed), ("test", args.seed + 1)]:
        episodes = load_seed_episodes(
            data_path=args.data,
            split=split,
            history_len=args.history_len,
            max_episodes=args.episodes_per_split,
            seed=split_seed,
        )
        logging.info("Evaluating split=%s on %d seed episodes", split, len(episodes))
        policies = {
            "planner": make_planner_policy(ensemble, planner_cfg, device),
            "repeat_last": make_repeat_last_policy(),
            "random": make_random_policy(np.random.default_rng(split_seed)),
        }
        if bandit_values is not None:
            policies["bandit"] = greedy_policy_from_values(bandit_values)
        if ddqn_model is not None:
            policies["ddqn"] = greedy_policy_from_model(ddqn_model, device)

        split_summary, traces = evaluate_policies(
            ensemble=ensemble,
            episodes=episodes,
            policies=policies,
            planner_config=planner_cfg,
            rollout_steps=args.rollout_steps,
            observation_window=args.observation_window,
            uncertainty_threshold=args.uncertainty_threshold,
            uncertainty_penalty=args.uncertainty_penalty,
            device=device,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=args.terminal_reward_scale,
            progress_every=args.eval_progress_every,
        )
        summary_bundle[split] = split_summary
        traces.to_csv(report_dir / f"step_13c_policy_traces_{split}.csv", index=False)
        terminal_rows = traces[traces["step"] == 0].copy()
        terminal_rows.to_csv(report_dir / f"step_13c_episode_summary_{split}.csv", index=False)
        diagnostics = build_policy_diagnostics(traces)
        save_json(report_dir / f"step_13c_diagnostics_{split}.json", diagnostics)
        logging.info("Split=%s summary: %s", split, split_summary)

    save_json(report_dir / "step_13c_summary.json", summary_bundle)
    return summary_bundle


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "smoke":
        args.smoke = True
    if args.smoke:
        if hasattr(args, "episodes_per_split"):
            args.episodes_per_split = min(args.episodes_per_split, 12)
        if hasattr(args, "train_steps"):
            args.train_steps = min(args.train_steps, 200)
        if hasattr(args, "train_episodes"):
            args.train_episodes = min(args.train_episodes, 64)

    args.data = resolve_repo_path(args.data)
    args.model_dir = resolve_repo_path(args.model_dir)
    args.control_model_dir = resolve_repo_path(args.control_model_dir)
    args.report_dir = resolve_repo_path(args.report_dir)
    args.severity_model_dir = resolve_repo_path(args.severity_model_dir)
    args.terminal_model_dir = resolve_repo_path(args.terminal_model_dir)
    args.log = resolve_repo_path(args.log)
    if hasattr(args, "ddqn_path"):
        args.ddqn_path = resolve_repo_path(args.ddqn_path)
    if hasattr(args, "bandit_path"):
        args.bandit_path = resolve_repo_path(args.bandit_path)

    setup_logging(args.log)
    t0 = time.time()
    device = resolve_device(args.device)
    logging.info("Step 13c started. mode=%s device=%s smoke=%s", args.mode, device, args.smoke)
    logging.info("Loading DAG-aware ensemble from %s", args.model_dir)
    ensemble = DAGAwareEnsemble.from_dir(args.model_dir, device=device)

    severity_model = None
    terminal_outcome_model = None
    if args.use_severity_reward:
        severity_model = load_severity_model(
            mode=args.severity_mode,
            model_dir=args.severity_model_dir,
            state_feature_names=["Hb", "BUN", "Creatinine", "Phosphate", "HR", "Chloride", "age", "charlson_score", "prior_ed_visits_6m"],
            device=device,
        )
        logging.info(
            "Using severity-derived reward. mode=%s model_dir=%s",
            args.severity_mode,
            args.severity_model_dir if args.severity_mode == "surrogate" else "n/a",
        )
    if args.use_terminal_readmit_reward:
        terminal_outcome_model = LightGBMReadmitModel.from_dir(
            args.terminal_model_dir,
            state_feature_names=["s_Hb", "s_BUN", "s_Creatinine", "s_Phosphate", "s_HR", "s_Chloride", "s_age", "s_charlson_score", "s_prior_ed_visits_6m"],
            device=device,
        )
        logging.info("Using terminal readmission reward from %s (scale=%.1f)", args.terminal_model_dir, args.terminal_reward_scale)

    if args.mode == "planner":
        summary = run_evaluation(
            args,
            ensemble,
            device,
            ddqn_model=None,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
        )
        logging.info("Planner evaluation complete. val=%s", summary.get("val"))
    elif args.mode == "train-ddqn":
        episodes = load_seed_episodes(
            data_path=args.data,
            split="train",
            history_len=args.history_len,
            max_episodes=args.train_episodes,
            seed=args.seed,
        )
        logging.info("Loaded %d training seed episodes", len(episodes))
        config = ddqn_config_from_args(args)
        model, metrics = train_ddqn(
            ensemble=ensemble,
            episodes=episodes,
            config=config,
            device=device,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=args.terminal_reward_scale,
        )
        model_path = save_ddqn_artifacts(model, metrics, config, Path(args.control_model_dir))
        logging.info("DDQN saved to %s", model_path)
    elif args.mode == "train-bandit":
        episodes = load_seed_episodes(
            data_path=args.data,
            split="train",
            history_len=args.history_len,
            max_episodes=args.train_episodes,
            seed=args.seed,
        )
        logging.info("Loaded %d training seed episodes", len(episodes))
        config = bandit_config_from_args(args)
        env = CareSimEnvironment(
            ensemble,
            max_steps=args.rollout_steps,
            uncertainty_threshold=args.uncertainty_threshold,
            device=device,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=args.terminal_reward_scale,
        )
        action_values, action_counts, metrics = train_bandit(
            env=env,
            episodes=episodes,
            config=config,
            device=device,
        )
        model_path = save_bandit_artifacts(action_values, action_counts, metrics, config, Path(args.control_model_dir))
        logging.info("Bandit saved to %s", model_path)
    elif args.mode == "eval":
        ddqn_model = load_ddqn_model(args.ddqn_path, device=device)
        bandit_values = load_bandit_model(args.bandit_path) if Path(args.bandit_path).exists() else None
        summary = run_evaluation(
            args,
            ensemble,
            device,
            ddqn_model=ddqn_model,
            bandit_values=bandit_values,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
        )
        logging.info("Evaluation complete. val=%s", summary.get("val"))
    elif args.mode == "smoke":
        train_episodes = load_seed_episodes(
            data_path=args.data,
            split="train",
            history_len=args.history_len,
            max_episodes=64,
            seed=args.seed,
        )
        config = ddqn_config_from_args(args)
        ddqn_model, metrics = train_ddqn(
            ensemble=ensemble,
            episodes=train_episodes,
            config=config,
            device=device,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=args.terminal_reward_scale,
        )
        model_path = save_ddqn_artifacts(ddqn_model, metrics, config, Path(args.control_model_dir))
        logging.info("Smoke DDQN saved to %s", model_path)
        bandit_cfg = bandit_config_from_args(args)
        env = CareSimEnvironment(
            ensemble,
            max_steps=args.rollout_steps,
            uncertainty_threshold=args.uncertainty_threshold,
            device=device,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=args.terminal_reward_scale,
        )
        bandit_values, action_counts, bandit_metrics = train_bandit(
            env=env,
            episodes=train_episodes,
            config=bandit_cfg,
            device=device,
        )
        bandit_path = save_bandit_artifacts(bandit_values, action_counts, bandit_metrics, bandit_cfg, Path(args.control_model_dir))
        logging.info("Smoke bandit saved to %s", bandit_path)
        summary = run_evaluation(
            args,
            ensemble,
            device,
            ddqn_model=ddqn_model,
            bandit_values=bandit_values,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
        )
        logging.info("Smoke evaluation complete. val=%s", summary.get("val"))

    logging.info("Step 13c complete in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
