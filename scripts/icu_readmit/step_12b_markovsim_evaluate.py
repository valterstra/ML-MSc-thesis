"""Step 12b -- MarkovSim evaluation and sanity checks."""

from __future__ import annotations

import argparse
import json
import logging
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
from careai.icu_readmit.markovsim.model import (  # noqa: E402
    ACTION_COLS,
    ACTION_FEATURE_NAMES,
    DYNAMIC_STATE_IDX,
    MarkovSimEnsemble,
    NEXT_STATE_COLS,
    STATE_COLS,
    STATE_FEATURE_NAMES,
    STATIC_STATE_IDX,
    STATIC_STATE_NAMES,
)
from careai.icu_readmit.markovsim.simulator import MarkovSimEnvironment  # noqa: E402


BLOC_COL = "bloc"
DONE_COL = "done"
SPLIT_COL = "split"
STAY_COL = "icustayid"

ACTION_GRID = np.array(
    [[(a >> bit) & 1 for bit in range(len(ACTION_COLS))] for a in range(2 ** len(ACTION_COLS))],
    dtype=np.float32,
)


def parse_args():
    p = argparse.ArgumentParser(description="Step 12b: MarkovSim evaluation")
    p.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
    p.add_argument("--model-dir", default="models/icu_readmit/markovsim_selected_causal")
    p.add_argument("--report-dir", default="reports/icu_readmit/markovsim_selected_causal")
    p.add_argument("--history-len", type=int, default=5)
    p.add_argument("--rollout-steps", type=int, default=5)
    p.add_argument("--rollout-patients", type=int, default=200)
    p.add_argument("--counterfactual-patients", type=int, default=10)
    p.add_argument("--uncertainty-threshold", type=float, default=1.0)
    p.add_argument("--use-severity-reward", action="store_true")
    p.add_argument("--severity-mode", choices=["surrogate", "handcrafted"], default="handcrafted")
    p.add_argument("--severity-model-dir", default="models/icu_readmit/severity_selected")
    p.add_argument("--use-terminal-readmit-reward", action="store_true")
    p.add_argument("--terminal-model-dir", default="models/icu_readmit/terminal_readmit_selected")
    p.add_argument("--terminal-reward-scale", type=float, default=15.0)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log", default="logs/step_12b_markovsim_eval.log")
    return p.parse_args()


def resolve_repo_path(path_str: str) -> str:
    path = Path(path_str)
    if path.is_absolute():
        return str(path)
    return str((PROJECT_ROOT / path).resolve())


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def setup_logging(log_path: str) -> None:
    path = Path(log_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.FileHandler(path, mode="w", encoding="utf-8"), logging.StreamHandler()],
    )


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_eval_dataframe(data_path: str) -> pd.DataFrame:
    use_cols = [STAY_COL, BLOC_COL, SPLIT_COL, DONE_COL, *STATE_COLS, *ACTION_COLS, *NEXT_STATE_COLS]
    df = pd.read_parquet(data_path, columns=use_cols)
    return df.sort_values([STAY_COL, BLOC_COL]).reset_index(drop=True)


def compute_observed_rewards(
    df: pd.DataFrame,
    severity_model,
    terminal_outcome_model: LightGBMReadmitModel | None,
    terminal_reward_scale: float,
    device: torch.device,
) -> np.ndarray:
    states_t = torch.tensor(df[STATE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
    next_states_t = torch.tensor(df[NEXT_STATE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)

    dense_reward = torch.zeros(states_t.shape[0], dtype=torch.float32, device=device)
    if severity_model is not None:
        dense_reward = severity_model.score(states_t) - severity_model.score(next_states_t)

    rewards = dense_reward
    if terminal_outcome_model is not None:
        terminal_reward, _ = terminal_outcome_model.terminal_reward(states_t, reward_scale=terminal_reward_scale)
        done_mask = torch.tensor(df[DONE_COL].to_numpy(dtype=np.float32) > 0.5, dtype=torch.bool, device=device)
        rewards = torch.where(done_mask, terminal_reward, rewards)

    return rewards.detach().cpu().numpy().astype(np.float32)


def compute_predicted_rewards(
    current_states: np.ndarray,
    predicted_next_states: np.ndarray,
    done_true: np.ndarray,
    severity_model,
    terminal_outcome_model: LightGBMReadmitModel | None,
    terminal_reward_scale: float,
    device: torch.device,
) -> np.ndarray:
    states_t = torch.tensor(current_states, dtype=torch.float32, device=device)
    next_states_t = torch.tensor(predicted_next_states, dtype=torch.float32, device=device)

    dense_reward = torch.zeros(states_t.shape[0], dtype=torch.float32, device=device)
    if severity_model is not None:
        dense_reward = severity_model.score(states_t) - severity_model.score(next_states_t)

    rewards = dense_reward
    if terminal_outcome_model is not None:
        terminal_reward, _ = terminal_outcome_model.terminal_reward(next_states_t, reward_scale=terminal_reward_scale)
        done_mask = torch.tensor(done_true > 0.5, dtype=torch.bool, device=device)
        rewards = torch.where(done_mask, terminal_reward, rewards)

    return rewards.detach().cpu().numpy().astype(np.float32)


def one_step_metrics(
    ensemble: MarkovSimEnsemble,
    df: pd.DataFrame,
    split: str,
    uncertainty_threshold: float,
    device: torch.device,
    severity_model=None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> dict:
    split_df = df[df[SPLIT_COL] == split].copy()
    states = torch.tensor(split_df[STATE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(1)
    actions = torch.tensor(split_df[ACTION_COLS].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(1)
    time_steps = torch.tensor(split_df[BLOC_COL].to_numpy(dtype=np.float32), dtype=torch.float32, device=device).unsqueeze(1)
    out = ensemble.predict(states, actions, time_steps=time_steps)

    ns_true = split_df[NEXT_STATE_COLS].to_numpy(dtype=np.float32, copy=True)
    done_true = split_df[DONE_COL].to_numpy(dtype=np.float32, copy=True)

    ns_pred = out["next_state_mean"][:, 0, :].detach().cpu().numpy()
    term_prob = out["terminal_prob"][:, 0].detach().cpu().numpy()
    unc = out["next_state_std"][:, 0, :].mean(dim=-1).detach().cpu().numpy()

    rw_true = compute_observed_rewards(
        split_df,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=terminal_reward_scale,
        device=device,
    )
    rw_pred = compute_predicted_rewards(
        current_states=split_df[STATE_COLS].to_numpy(dtype=np.float32, copy=True),
        predicted_next_states=ns_pred,
        done_true=done_true,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=terminal_reward_scale,
        device=device,
    )

    per_feature_mse = {
        STATE_FEATURE_NAMES[i]: float(np.mean((ns_pred[:, i] - ns_true[:, i]) ** 2))
        for i in range(len(STATE_COLS))
    }
    static_drift = {
        STATIC_STATE_NAMES[i]: float(np.mean(np.abs(ns_pred[:, STATIC_STATE_IDX[i]] - ns_true[:, STATIC_STATE_IDX[i]])))
        for i in range(len(STATIC_STATE_IDX))
    }
    term_pred = (term_prob > 0.5).astype(np.float32)
    return {
        "split": split,
        "n_rows": int(len(split_df)),
        "next_state_mse": float(np.mean((ns_pred - ns_true) ** 2)),
        "next_state_per_feature_mse": per_feature_mse,
        "reward_mse": float(np.mean((rw_pred - rw_true) ** 2)),
        "reward_mae": float(np.mean(np.abs(rw_pred - rw_true))),
        "terminal_accuracy": float(np.mean(term_pred == done_true)),
        "terminal_brier": float(np.mean((term_prob - done_true) ** 2)),
        "mean_uncertainty": float(np.mean(unc)),
        "uncertainty_flag_rate": float(np.mean(unc > uncertainty_threshold)),
        "static_abs_drift": static_drift,
    }


def sample_rollout_episodes(
    df: pd.DataFrame,
    split: str,
    history_len: int,
    rollout_steps: int,
    n_patients: int,
    seed: int,
) -> list[pd.DataFrame]:
    split_df = df[df[SPLIT_COL] == split].copy()
    episodes = []
    for _, stay_df in split_df.groupby(STAY_COL, sort=False):
        stay_df = stay_df.sort_values(BLOC_COL).reset_index(drop=True)
        if len(stay_df) >= history_len + rollout_steps:
            episodes.append(stay_df)
    rng = np.random.default_rng(seed)
    if len(episodes) > n_patients:
        idx = rng.choice(len(episodes), size=n_patients, replace=False)
        episodes = [episodes[i] for i in idx]
    return episodes


def rollout_metrics(
    ensemble: MarkovSimEnsemble,
    df: pd.DataFrame,
    split: str,
    history_len: int,
    rollout_steps: int,
    n_patients: int,
    uncertainty_threshold: float,
    device: torch.device,
    seed: int,
    severity_model=None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> dict:
    episodes = sample_rollout_episodes(df, split, history_len, rollout_steps, n_patients, seed)
    if not episodes:
        return {"split": split, "n_patients": 0, "n_steps": rollout_steps}

    state_se = np.zeros((rollout_steps, len(STATE_COLS)), dtype=np.float64)
    reward_ae = np.zeros(rollout_steps, dtype=np.float64)
    done_correct = np.zeros(rollout_steps, dtype=np.float64)
    uncertainty_sum = np.zeros(rollout_steps, dtype=np.float64)
    uncertainty_flagged = np.zeros(rollout_steps, dtype=np.float64)
    static_drift = np.zeros((rollout_steps, len(STATIC_STATE_IDX)), dtype=np.float64)

    for stay_df in episodes:
        seed_rows = stay_df.iloc[:history_len]
        future_rows = stay_df.iloc[history_len:history_len + rollout_steps].reset_index(drop=True)
        future_reward = compute_observed_rewards(
            future_rows,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=terminal_reward_scale,
            device=device,
        )

        seed_states = torch.tensor(seed_rows[STATE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_actions = torch.tensor(seed_rows[ACTION_COLS].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_time_steps = torch.tensor(seed_rows[BLOC_COL].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)

        env = MarkovSimEnvironment(
            ensemble,
            max_steps=rollout_steps + 10,
            uncertainty_threshold=uncertainty_threshold,
            device=device,
            severity_model=severity_model,
            terminal_outcome_model=terminal_outcome_model,
            terminal_reward_scale=terminal_reward_scale,
        )
        env.reset(seed_states, seed_actions, seed_time_steps=seed_time_steps)

        for t, row in future_rows.iterrows():
            action = torch.tensor(row[ACTION_COLS].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
            next_state, reward, done, info = env.step(action)
            actual_next = row[NEXT_STATE_COLS].to_numpy(dtype=np.float32, copy=True)
            pred_state = next_state[0].detach().cpu().numpy()
            state_se[t] += (pred_state - actual_next) ** 2
            reward_ae[t] += abs(float(reward[0]) - float(future_reward[t]))
            done_correct[t] += float(bool(done[0]) == bool(row[DONE_COL]))
            uncertainty_sum[t] += float(info["uncertainty"][0])
            uncertainty_flagged[t] += float(info["uncertain_flag"][0])
            static_drift[t] += np.abs(pred_state[list(STATIC_STATE_IDX)] - actual_next[list(STATIC_STATE_IDX)])

    n = len(episodes)
    return {
        "split": split,
        "n_patients": n,
        "n_steps": rollout_steps,
        "per_step_state_mse": (state_se.sum(axis=1) / (n * len(STATE_COLS))).tolist(),
        "per_step_reward_mae": (reward_ae / n).tolist(),
        "per_step_done_accuracy": (done_correct / n).tolist(),
        "per_step_mean_uncertainty": (uncertainty_sum / n).tolist(),
        "per_step_uncertainty_flag_rate": (uncertainty_flagged / n).tolist(),
        "per_step_static_abs_drift": {
            STATIC_STATE_NAMES[i]: (static_drift[:, i] / n).tolist() for i in range(len(STATIC_STATE_IDX))
        },
    }


def counterfactual_sweep(
    ensemble: MarkovSimEnsemble,
    df: pd.DataFrame,
    split: str,
    history_len: int,
    n_patients: int,
    uncertainty_threshold: float,
    device: torch.device,
    seed: int,
    severity_model=None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> pd.DataFrame:
    episodes = sample_rollout_episodes(df, split, history_len, 1, n_patients, seed)
    rows = []
    for stay_df in episodes:
        seed_rows = stay_df.iloc[:history_len]
        stay_id = int(seed_rows[STAY_COL].iloc[0])
        seed_states = torch.tensor(seed_rows[STATE_COLS].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_actions = torch.tensor(seed_rows[ACTION_COLS].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_time_steps = torch.tensor(seed_rows[BLOC_COL].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        for action_id, action_vec in enumerate(ACTION_GRID):
            env = MarkovSimEnvironment(
                ensemble,
                max_steps=2,
                uncertainty_threshold=uncertainty_threshold,
                device=device,
                severity_model=severity_model,
                terminal_outcome_model=terminal_outcome_model,
                terminal_reward_scale=terminal_reward_scale,
            )
            env.reset(seed_states, seed_actions, seed_time_steps=seed_time_steps)
            action = torch.tensor(action_vec, dtype=torch.float32).unsqueeze(0)
            next_state, reward, done, info = env.step(action)
            row = {
                "split": split,
                "stay_id": stay_id,
                "action_id": int(action_id),
                "reward_pred": float(reward[0]),
                "terminal_prob": float(info["terminal_prob"][0]),
                "uncertainty": float(info["uncertainty"][0]),
                "uncertain_flag": bool(info["uncertain_flag"][0]),
                "done_pred": bool(done[0]),
            }
            for j, col in enumerate(ACTION_COLS):
                row[col] = float(action_vec[j])
            for j, name in enumerate(STATE_FEATURE_NAMES):
                row[f"next_{name}_pred"] = float(next_state[0, j])
            rows.append(row)
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["stay_id", "reward_pred", "uncertainty"], ascending=[True, False, True])
    return out


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.rollout_patients = min(args.rollout_patients, 20)
        args.counterfactual_patients = min(args.counterfactual_patients, 3)
        args.rollout_steps = min(args.rollout_steps, 3)

    args.data = resolve_repo_path(args.data)
    args.model_dir = resolve_repo_path(args.model_dir)
    args.report_dir = resolve_repo_path(args.report_dir)
    args.severity_model_dir = resolve_repo_path(args.severity_model_dir)
    args.terminal_model_dir = resolve_repo_path(args.terminal_model_dir)
    args.log = resolve_repo_path(args.log)

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    setup_logging(args.log)

    t0 = time.time()
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Loading MarkovSim model from %s", args.model_dir)
    ensemble = MarkovSimEnsemble.from_dir(args.model_dir, device=device)
    df = load_eval_dataframe(args.data)
    logging.info("Loaded rows=%d from %s", len(df), args.data)

    severity_model = None
    terminal_outcome_model = None
    if args.use_severity_reward:
        severity_model = load_severity_model(
            mode=args.severity_mode,
            model_dir=args.severity_model_dir,
            state_feature_names=STATE_FEATURE_NAMES,
            device=device,
        )
        logging.info("Using severity-derived reward. mode=%s", args.severity_mode)
    if args.use_terminal_readmit_reward:
        terminal_outcome_model = LightGBMReadmitModel.from_dir(
            args.terminal_model_dir,
            state_feature_names=STATE_COLS,
            device=device,
        )
        logging.info("Using terminal readmission reward from %s (scale=%.1f)", args.terminal_model_dir, args.terminal_reward_scale)

    one_step_val = one_step_metrics(
        ensemble, df, "val", args.uncertainty_threshold, device,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    one_step_test = one_step_metrics(
        ensemble, df, "test", args.uncertainty_threshold, device,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    rollout_val = rollout_metrics(
        ensemble, df, "val", args.history_len, args.rollout_steps, args.rollout_patients,
        args.uncertainty_threshold, device, args.seed,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    rollout_test = rollout_metrics(
        ensemble, df, "test", args.history_len, args.rollout_steps, args.rollout_patients,
        args.uncertainty_threshold, device, args.seed + 1,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    cf_val = counterfactual_sweep(
        ensemble, df, "val", args.history_len, args.counterfactual_patients,
        args.uncertainty_threshold, device, args.seed,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )

    write_json(report_dir / "markovsim_one_step_val.json", one_step_val)
    write_json(report_dir / "markovsim_one_step_test.json", one_step_test)
    write_json(report_dir / "markovsim_rollout_val.json", rollout_val)
    write_json(report_dir / "markovsim_rollout_test.json", rollout_test)
    cf_val.to_csv(report_dir / "markovsim_counterfactual_val.csv", index=False)

    summary = {
        "meta": {
            "data": args.data,
            "model_dir": args.model_dir,
            "history_len": args.history_len,
            "rollout_steps": args.rollout_steps,
            "rollout_patients": args.rollout_patients,
            "counterfactual_patients": args.counterfactual_patients,
            "uncertainty_threshold": args.uncertainty_threshold,
            "use_severity_reward": bool(args.use_severity_reward),
            "severity_mode": args.severity_mode if args.use_severity_reward else None,
            "use_terminal_readmit_reward": bool(args.use_terminal_readmit_reward),
            "terminal_reward_scale": args.terminal_reward_scale if args.use_terminal_readmit_reward else None,
            "device": str(device),
            "smoke": args.smoke,
            "total_seconds": round(time.time() - t0, 1),
        },
        "one_step_val": one_step_val,
        "one_step_test": one_step_test,
        "rollout_val": rollout_val,
        "rollout_test": rollout_test,
    }
    write_json(report_dir / "markovsim_summary.json", summary)
    logging.info("Step 12b MarkovSim complete in %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
