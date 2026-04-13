"""Step 14 -- Selected offline RL comparison branch.

Train and evaluate an offline DDQN policy directly on the selected logged ICU
data, aligned as closely as possible with the current Step 13 CARE-Sim control
track.

Compared with the old Step 11/12 branch, this script:
  - uses the selected 9-state / 5-action setup
  - builds a Step-13-style fixed observation window of recent state/action pairs
  - recomputes reward to match Step 13:
      dense  = handcrafted severity improvement
      terminal = terminal readmission reward model
  - supports comparison against the Step 13 CARE-Sim DDQN and MarkovSim DDQN on held-out real data

Outputs:
  models/icu_readmit/offline_selected/
    ddqn/
    eval_support/
  reports/icu_readmit/offline_selected/
    step_14_eval_results.json
    step_14_action_stats.json
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
import time
from pathlib import Path

import torch
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.readmit import LightGBMReadmitModel  # noqa: E402
from careai.icu_readmit.caresim.severity import load_severity_model  # noqa: E402
from careai.icu_readmit.rl.continuous import compute_q_values, train_dqn  # noqa: E402
from careai.icu_readmit.rl.evaluation import (  # noqa: E402
    doubly_robust_evaluation,
    predict_next_states,
    predict_physician_probs,
    predict_rewards,
    train_env_model,
    train_physician_policy,
    train_reward_estimator,
)
from careai.icu_readmit.rl.networks import DuelingDQN  # noqa: E402


DEFAULT_DATA = "data/processed/icu_readmit/rl_dataset_selected.parquet"
DEFAULT_MODEL_DIR = "models/icu_readmit/offline_selected"
DEFAULT_REPORT_DIR = "reports/icu_readmit/offline_selected"
DEFAULT_TERMINAL_MODEL_DIR = "models/icu_readmit/terminal_readmit_selected"
DEFAULT_CARESIM_DDQN = "models/icu_readmit/caresim_control_selected_causal/ddqn_model.pt"
DEFAULT_MARKOVSIM_DDQN = "models/icu_readmit/markovsim_control_selected_causal/ddqn_model.pt"

WINDOW_LEN = 5
N_ACTIONS = 32
STATE_FEATURE_NAMES = [
    "Hb",
    "BUN",
    "Creatinine",
    "Phosphate",
    "HR",
    "Chloride",
    "age",
    "charlson_score",
    "prior_ed_visits_6m",
]
STATE_COLS = [f"s_{name}" for name in STATE_FEATURE_NAMES]
ACTION_COLS = ["vasopressor_b", "ivfluid_b", "antibiotic_b", "diuretic_b", "mechvent_b"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Step 14: selected offline RL")
    sub = parser.add_subparsers(dest="mode", required=True)

    def add_common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--data", default=DEFAULT_DATA)
        p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
        p.add_argument("--report-dir", default=DEFAULT_REPORT_DIR)
        p.add_argument("--terminal-model-dir", default=DEFAULT_TERMINAL_MODEL_DIR)
        p.add_argument("--caresim-ddqn-path", default=DEFAULT_CARESIM_DDQN)
        p.add_argument("--markovsim-ddqn-path", default=DEFAULT_MARKOVSIM_DDQN)
        p.add_argument("--severity-mode", choices=["handcrafted", "surrogate"], default="handcrafted")
        p.add_argument("--severity-model-dir", default="models/icu_readmit/severity_selected")
        p.add_argument("--terminal-reward-scale", type=float, default=15.0)
        p.add_argument("--window-len", type=int, default=WINDOW_LEN)
        p.add_argument("--device", default=None)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--smoke", action="store_true")
        p.add_argument("--max-stays-per-split", type=int, default=None)
        p.add_argument("--log", default="logs/step_14_offline_selected.log")

    train_p = sub.add_parser("train-ddqn", help="Train offline DDQN on selected logged data")
    add_common(train_p)
    train_p.add_argument("--dqn-steps", type=int, default=100000)

    eval_p = sub.add_parser("eval", help="Train OPE support models and compare offline vs world-model DDQN")
    add_common(eval_p)
    eval_p.add_argument("--physician-steps", type=int, default=35000)
    eval_p.add_argument("--reward-steps", type=int, default=30000)
    eval_p.add_argument("--env-steps", type=int, default=60000)

    smoke_p = sub.add_parser("smoke", help="Compact train + eval sanity run")
    add_common(smoke_p)
    smoke_p.add_argument("--dqn-steps", type=int, default=500)
    smoke_p.add_argument("--physician-steps", type=int, default=200)
    smoke_p.add_argument("--reward-steps", type=int, default=200)
    smoke_p.add_argument("--env-steps", type=int, default=200)
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


def _load_selected_df(data_path: str, max_stays_per_split: int | None, seed: int) -> pd.DataFrame:
    use_cols = ["icustayid", "bloc", "split", "done", "a", *STATE_COLS, *ACTION_COLS]
    df = pd.read_parquet(data_path, columns=use_cols)
    if max_stays_per_split is None:
        return df

    rng = np.random.default_rng(seed)
    keep_ids: list[int] = []
    for split_name, split_df in df.groupby("split", sort=False):
        stay_ids = split_df["icustayid"].drop_duplicates().to_numpy()
        if len(stay_ids) > max_stays_per_split:
            chosen = rng.choice(stay_ids, size=max_stays_per_split, replace=False)
            keep_ids.extend(chosen.tolist())
        else:
            keep_ids.extend(stay_ids.tolist())
        logging.info("Split=%s limited to %d stays", split_name, min(len(stay_ids), max_stays_per_split))
    return df[df["icustayid"].isin(keep_ids)].copy()


def _compute_rewards(
    df: pd.DataFrame,
    severity_mode: str,
    severity_model_dir: str,
    terminal_model_dir: str,
    terminal_reward_scale: float,
    device: torch.device,
) -> np.ndarray:
    severity_model = load_severity_model(
        mode=severity_mode,
        model_dir=severity_model_dir,
        state_feature_names=STATE_FEATURE_NAMES,
        device=device,
    )
    terminal_model = LightGBMReadmitModel.from_dir(
        terminal_model_dir,
        state_feature_names=STATE_COLS,
        device=device,
    )

    states_t = torch.tensor(df[STATE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    sev_now = severity_model.score(states_t).detach().cpu().numpy()

    next_cols = [c.replace("s_", "s_next_", 1) for c in STATE_COLS]
    next_df = pd.read_parquet(df.attrs["data_path"], columns=["icustayid", "bloc", *next_cols])
    next_df = next_df.set_index(["icustayid", "bloc"]).loc[list(zip(df["icustayid"], df["bloc"]))].reset_index()
    next_states_t = torch.tensor(next_df[next_cols].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    sev_next = severity_model.score(next_states_t).detach().cpu().numpy()

    terminal_reward = terminal_model.terminal_reward(states_t, reward_scale=terminal_reward_scale)[0].detach().cpu().numpy()
    rewards = np.where(df["done"].to_numpy(dtype=np.float32) > 0.5, terminal_reward, sev_now - sev_next).astype(np.float32)
    return rewards


def _build_windowed_dataset(
    df: pd.DataFrame,
    window_len: int,
    rewards: np.ndarray,
) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    step_dim = len(STATE_COLS) + len(ACTION_COLS)
    obs_dim = window_len * step_dim
    split_rows: dict[str, list] = {"train": [], "val": [], "test": []}
    split_stays: dict[str, set[int]] = {"train": set(), "val": set(), "test": set()}

    df = df.sort_values(["icustayid", "bloc"]).reset_index(drop=True).copy()
    df["reward_step17"] = rewards

    for _, stay_df in df.groupby("icustayid", sort=False):
        stay_df = stay_df.sort_values("bloc").reset_index(drop=True)
        split_name = str(stay_df.loc[0, "split"])
        split_stays[split_name].add(int(stay_df.loc[0, "icustayid"]))
        step_matrix = stay_df[STATE_COLS + ACTION_COLS].to_numpy(dtype=np.float32, copy=True)
        actions = stay_df["a"].to_numpy(dtype=np.int64, copy=True)
        rewards_arr = stay_df["reward_step17"].to_numpy(dtype=np.float32, copy=True)
        done_arr = stay_df["done"].to_numpy(dtype=np.float32, copy=True)

        obs_rows = np.zeros((len(stay_df), obs_dim), dtype=np.float32)
        for i in range(len(stay_df)):
            start = max(0, i - window_len + 1)
            hist = step_matrix[start:i + 1]
            padded = np.zeros((window_len, step_dim), dtype=np.float32)
            padded[-len(hist):] = hist
            obs_rows[i] = padded.reshape(-1)

        next_obs = np.zeros_like(obs_rows)
        if len(stay_df) > 1:
            next_obs[:-1] = obs_rows[1:]

        for i in range(len(stay_df)):
            split_rows[split_name].append((
                obs_rows[i],
                int(actions[i]),
                float(rewards_arr[i]),
                next_obs[i],
                float(done_arr[i]),
                int(stay_df.loc[i, "icustayid"]),
                int(stay_df.loc[i, "bloc"]),
            ))

    datasets: dict[str, dict[str, np.ndarray]] = {}
    summary = {"obs_dim": obs_dim, "step_dim": step_dim, "window_len": window_len, "splits": {}}
    for split_name, rows in split_rows.items():
        if not rows:
            raise ValueError(f"No rows available for split={split_name}")
        states = np.stack([row[0] for row in rows]).astype(np.float32)
        actions = np.array([row[1] for row in rows], dtype=np.int64)
        rewards_arr = np.array([row[2] for row in rows], dtype=np.float32)
        next_states = np.stack([row[3] for row in rows]).astype(np.float32)
        done = np.array([row[4] for row in rows], dtype=np.float32)
        stay_ids = np.array([row[5] for row in rows], dtype=np.int64)
        blocs = np.array([row[6] for row in rows], dtype=np.int64)
        datasets[split_name] = {
            "states": states,
            "actions": actions,
            "rewards": rewards_arr,
            "next_states": next_states,
            "done": done,
            "stay_ids": stay_ids,
            "blocs": blocs,
        }
        summary["splits"][split_name] = {
            "n_rows": int(len(rows)),
            "n_stays": int(len(split_stays[split_name])),
            "reward_mean": float(rewards_arr.mean()),
            "reward_std": float(rewards_arr.std()),
            "done_rate": float(done.mean()),
        }
    return datasets, summary


def _save_ddqn_checkpoint(model: DuelingDQN, out_dir: Path, obs_dim: int, n_actions: int) -> None:
    ckpt = {
        "state_dict": model.state_dict(),
        "obs_dim": int(obs_dim),
        "n_actions": int(n_actions),
    }
    torch.save(ckpt, out_dir / "dqn_model.pt")


def _load_any_ddqn(model_path: str, device: torch.device, fallback_obs_dim: int, fallback_n_actions: int) -> DuelingDQN:
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        obs_dim = int(ckpt.get("obs_dim", fallback_obs_dim))
        n_actions = int(ckpt.get("n_actions", fallback_n_actions))
        state_dict = ckpt["state_dict"]
    else:
        obs_dim = int(fallback_obs_dim)
        n_actions = int(fallback_n_actions)
        state_dict = ckpt
    model = DuelingDQN(n_input=obs_dim, n_actions=n_actions, hidden=128).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _save_split_outputs(model: DuelingDQN, split_data: dict[str, np.ndarray], out_dir: Path, prefix: str, split_name: str, device: torch.device) -> None:
    q_vals, actions = compute_q_values(model, split_data["states"], device=device)
    with open(out_dir / f"{prefix}_actions_{split_name}.pkl", "wb") as f:
        pickle.dump(actions, f)
    with open(out_dir / f"{prefix}_q_{split_name}.pkl", "wb") as f:
        pickle.dump(q_vals, f)


def _train_support_models(
    train_data: dict[str, np.ndarray],
    val_data: dict[str, np.ndarray],
    test_data: dict[str, np.ndarray],
    obs_dim: int,
    args,
    device: torch.device,
    support_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    support_dir.mkdir(parents=True, exist_ok=True)
    log_every = 100 if args.smoke else 5000

    phys_dir = support_dir / "physician_policy"
    rew_dir = support_dir / "reward_estimator"
    env_dir = support_dir / "env_model"

    phys_model = train_physician_policy(
        train_data, n_state=obs_dim, n_actions=N_ACTIONS,
        num_steps=args.physician_steps, save_dir=str(phys_dir), device=device, log_every=log_every,
    )
    train_probs = predict_physician_probs(phys_model, train_data, device=device, save_path=str(phys_dir / "train_policy.pkl"))
    val_probs = predict_physician_probs(phys_model, val_data, device=device, save_path=str(phys_dir / "val_policy.pkl"))
    test_probs = predict_physician_probs(phys_model, test_data, device=device, save_path=str(phys_dir / "test_policy.pkl"))

    rew_model = train_reward_estimator(
        train_data, n_state=obs_dim, num_steps=args.reward_steps,
        save_dir=str(rew_dir), device=device, log_every=log_every,
    )
    val_rewards_est = predict_rewards(rew_model, val_data, device=device, save_path=str(rew_dir / "val_rewards.pkl"))
    test_rewards_est = predict_rewards(rew_model, test_data, device=device, save_path=str(rew_dir / "test_rewards.pkl"))

    env_model = train_env_model(
        train_data, n_state=obs_dim, num_steps=args.env_steps,
        save_dir=str(env_dir), device=device, log_every=log_every,
    )
    predict_next_states(env_model, val_data, device=device, save_path=str(env_dir / "val_next_states.pkl"))
    predict_next_states(env_model, test_data, device=device, save_path=str(env_dir / "test_next_states.pkl"))
    return train_probs, val_probs, test_probs, val_rewards_est, test_rewards_est


def _empirical_trajectory_returns(data_split: dict[str, np.ndarray], gamma: float = 0.99) -> dict[str, float]:
    rewards = data_split["rewards"]
    done = data_split["done"]
    trajectory_vals = []
    start = 0
    for i in range(len(done)):
        if done[i] > 0.5:
            traj_rewards = rewards[start:i + 1]
            discounts = gamma ** np.arange(len(traj_rewards), dtype=np.float32)
            trajectory_vals.append(float(np.sum(discounts * traj_rewards)))
            start = i + 1
    arr = np.asarray(trajectory_vals, dtype=np.float32)
    return {
        "mean": float(arr.mean()) if len(arr) else 0.0,
        "std": float(arr.std()) if len(arr) else 0.0,
        "n_trajectories": int(len(arr)),
    }


def _action_stats(agent_actions: np.ndarray, data_split: dict[str, np.ndarray]) -> dict:
    phys_actions = data_split["actions"]
    agreement = (agent_actions == phys_actions)
    action_counts = np.bincount(agent_actions, minlength=N_ACTIONS)
    return {
        "exact_agreement_pct": float(100.0 * agreement.mean()),
        "n_unique_actions": int(np.count_nonzero(action_counts)),
        "top_actions": [
            {"action_id": int(idx), "count": int(action_counts[idx])}
            for idx in np.argsort(action_counts)[::-1][:5]
            if action_counts[idx] > 0
        ],
    }


def _prepare_selected_logged_data(args, device: torch.device) -> tuple[dict[str, dict[str, np.ndarray]], dict]:
    logging.info("Loading selected parquet: %s", args.data)
    df = _load_selected_df(args.data, max_stays_per_split=args.max_stays_per_split, seed=args.seed)
    df.attrs["data_path"] = args.data
    logging.info("  rows=%d stays=%d", len(df), df["icustayid"].nunique())
    rewards = _compute_rewards(
        df=df,
        severity_mode=args.severity_mode,
        severity_model_dir=args.severity_model_dir,
        terminal_model_dir=args.terminal_model_dir,
        terminal_reward_scale=args.terminal_reward_scale,
        device=device,
    )
    datasets, summary = _build_windowed_dataset(df, window_len=args.window_len, rewards=rewards)
    logging.info("Built selected offline dataset: obs_dim=%d", summary["obs_dim"])
    for split_name, split_meta in summary["splits"].items():
        logging.info("  %s: rows=%d stays=%d reward_mean=%.4f done=%.3f",
                     split_name, split_meta["n_rows"], split_meta["n_stays"],
                     split_meta["reward_mean"], split_meta["done_rate"])
    return datasets, summary


def run_train_ddqn(args, device: torch.device) -> None:
    datasets, summary = _prepare_selected_logged_data(args, device)
    train_data = {k: v for k, v in datasets["train"].items() if k in {"states", "actions", "rewards", "next_states", "done"}}

    model_root = Path(args.model_dir)
    ddqn_dir = model_root / "ddqn"
    ddqn_dir.mkdir(parents=True, exist_ok=True)
    log_every = 100 if args.smoke else 5000

    model, _, _ = train_dqn(
        train_data,
        n_state=summary["obs_dim"],
        n_actions=N_ACTIONS,
        hidden=128,
        leaky_slope=0.01,
        lr=1e-4,
        gamma=0.99,
        tau=0.001,
        batch_size=32,
        num_steps=args.dqn_steps,
        reward_threshold=20,
        reg_lambda=5.0,
        per_alpha=0.6,
        per_epsilon=0.01,
        beta_start=0.9,
        save_dir=str(ddqn_dir),
        checkpoint_every=0 if args.smoke else 20000,
        device=device,
        log_every=log_every,
    )
    _save_ddqn_checkpoint(model, ddqn_dir, obs_dim=summary["obs_dim"], n_actions=N_ACTIONS)
    with open(ddqn_dir / "step_14_train_config.json", "w", encoding="utf-8") as f:
        json.dump({
            "obs_dim": summary["obs_dim"],
            "window_len": args.window_len,
            "n_actions": N_ACTIONS,
            "severity_mode": args.severity_mode,
            "terminal_reward_scale": args.terminal_reward_scale,
        }, f, indent=2)
    for split_name in ["val", "test"]:
        split_data = {k: v for k, v in datasets[split_name].items() if k in {"states", "actions", "rewards", "next_states", "done"}}
        _save_split_outputs(model, split_data, ddqn_dir, "dqn", split_name, device)
    logging.info("Step 14 train-ddqn complete. Model saved to %s", ddqn_dir)


def run_eval(args, device: torch.device) -> None:
    ddqn_dir = Path(args.model_dir) / "ddqn"
    offline_ckpt = ddqn_dir / "dqn_model.pt"
    if not offline_ckpt.exists():
        raise FileNotFoundError(
            f"Offline DDQN checkpoint not found at {offline_ckpt}. "
            "Run 'train-ddqn' first."
        )

    datasets, summary = _prepare_selected_logged_data(args, device)
    train_data = {k: v for k, v in datasets["train"].items() if k in {"states", "actions", "rewards", "next_states", "done"}}
    val_data = {k: v for k, v in datasets["val"].items() if k in {"states", "actions", "rewards", "next_states", "done"}}
    test_data = {k: v for k, v in datasets["test"].items() if k in {"states", "actions", "rewards", "next_states", "done"}}

    support_dir = Path(args.model_dir) / "eval_support"
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    _, val_probs, test_probs, val_rewards_est, test_rewards_est = _train_support_models(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        obs_dim=summary["obs_dim"],
        args=args,
        device=device,
        support_dir=support_dir,
    )

    offline_model = _load_any_ddqn(str(offline_ckpt), device=device, fallback_obs_dim=summary["obs_dim"], fallback_n_actions=N_ACTIONS)

    policy_models: dict[str, DuelingDQN] = {
        "offline_ddqn": offline_model,
    }
    if os.path.exists(args.caresim_ddqn_path):
        policy_models["caresim_ddqn"] = _load_any_ddqn(
            args.caresim_ddqn_path,
            device=device,
            fallback_obs_dim=summary["obs_dim"],
            fallback_n_actions=N_ACTIONS,
        )
    else:
        logging.warning("CARE-Sim DDQN checkpoint not found: %s -- skipping caresim_ddqn comparison", args.caresim_ddqn_path)
    if os.path.exists(args.markovsim_ddqn_path):
        policy_models["markovsim_ddqn"] = _load_any_ddqn(
            args.markovsim_ddqn_path,
            device=device,
            fallback_obs_dim=summary["obs_dim"],
            fallback_n_actions=N_ACTIONS,
        )
    else:
        logging.warning("MarkovSim DDQN checkpoint not found: %s -- skipping markovsim_ddqn comparison", args.markovsim_ddqn_path)

    results: dict[str, dict] = {
        "meta": {
            "data": args.data,
            "obs_dim": summary["obs_dim"],
            "window_len": args.window_len,
            "comparison_policies": sorted(policy_models.keys()),
            "reward": {
                "severity_mode": args.severity_mode,
                "terminal_model_dir": args.terminal_model_dir,
                "terminal_reward_scale": args.terminal_reward_scale,
            },
        },
        "logged_policy": {},
        "ope": {},
        "action_stats": {},
    }

    for split_name, split_data, probs, reward_est in [
        ("val", val_data, val_probs, val_rewards_est),
        ("test", test_data, test_probs, test_rewards_est),
    ]:
        results["logged_policy"][split_name] = _empirical_trajectory_returns(split_data)
        for policy_name, model in policy_models.items():
            q_vals, policy_actions = compute_q_values(model, split_data["states"], device=device)
            results["ope"][f"{policy_name}_{split_name}"] = doubly_robust_evaluation(
                split_data, policy_actions, q_vals, probs, reward_est, gamma=0.99, value_clip=40.0,
            )
            results["action_stats"][f"{policy_name}_{split_name}"] = _action_stats(policy_actions, split_data)

    for key, payload in list(results["ope"].items()):
        payload.pop("values", None)

    with open(report_dir / "step_14_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(report_dir / "step_14_action_stats.json", "w", encoding="utf-8") as f:
        json.dump(results["action_stats"], f, indent=2)
    logging.info("Step 14 eval complete. Results saved to %s", report_dir)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "smoke":
        args.smoke = True
        args.max_stays_per_split = args.max_stays_per_split or 400
    elif args.smoke and args.max_stays_per_split is None:
        args.max_stays_per_split = 1000

    args.data = resolve_repo_path(args.data)
    args.model_dir = resolve_repo_path(args.model_dir)
    args.report_dir = resolve_repo_path(args.report_dir)
    args.terminal_model_dir = resolve_repo_path(args.terminal_model_dir)
    args.caresim_ddqn_path = resolve_repo_path(args.caresim_ddqn_path)
    args.markovsim_ddqn_path = resolve_repo_path(args.markovsim_ddqn_path)
    args.severity_model_dir = resolve_repo_path(args.severity_model_dir)
    args.log = resolve_repo_path(args.log)

    setup_logging(args.log)
    device = resolve_device(args.device)
    t0 = time.time()
    logging.info("Step 14 started. mode=%s device=%s smoke=%s", args.mode, device, args.smoke)

    if args.mode == "train-ddqn":
        run_train_ddqn(args, device)
    elif args.mode == "eval":
        run_eval(args, device)
    elif args.mode == "smoke":
        run_train_ddqn(args, device)
        run_eval(args, device)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")

    logging.info("Step 14 finished in %.1f sec (%.1f min)", time.time() - t0, (time.time() - t0) / 60.0)


if __name__ == "__main__":
    main()
