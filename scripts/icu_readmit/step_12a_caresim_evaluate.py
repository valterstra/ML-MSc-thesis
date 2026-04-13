"""
Step 12a -- CARE-Sim evaluation and simulator sanity checks.

Evaluates the trained CARE-Sim ensemble on held-out data and produces:
  1. One-step prediction metrics on val/test
  2. Closed-loop rollout diagnostics under clinician actions
  3. Counterfactual one-step action sweeps on real patient seeds

Outputs:
  reports/icu_readmit/caresim/
    caresim_summary.json
    caresim_one_step_val.json
    caresim_one_step_test.json
    caresim_rollout_val.json
    caresim_rollout_test.json
    caresim_counterfactual_val.csv

Usage:
  python scripts/icu_readmit/step_12a_caresim_evaluate.py --device cpu
  python scripts/icu_readmit/step_12a_caresim_evaluate.py --smoke
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.dataset import (  # noqa: E402
    BLOC_COL,
    DONE_COL,
    ICUSequenceDataset,
    REWARD_COL,
    SPLIT_COL,
    STAY_COL,
    collate_sequences,
    infer_schema_from_path,
)
from careai.icu_readmit.caresim.ensemble import CareSimEnsemble  # noqa: E402
from careai.icu_readmit.caresim.readmit import LightGBMReadmitModel  # noqa: E402
from careai.icu_readmit.caresim.severity import load_severity_model  # noqa: E402
from careai.icu_readmit.caresim.simulator import CareSimEnvironment  # noqa: E402

STATIC_STATE_NAMES = ["age", "charlson_score", "prior_ed_visits_6m"]


def parse_args():
    p = argparse.ArgumentParser(description="Step 12a: CARE-Sim evaluation")
    p.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_tier2.parquet")
    p.add_argument("--model-dir", default="models/icu_readmit/caresim")
    p.add_argument("--report-dir", default="reports/icu_readmit/caresim")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--history-len", type=int, default=5,
                   help="Number of observed rows used to seed rollouts/counterfactuals")
    p.add_argument("--rollout-steps", type=int, default=5)
    p.add_argument("--rollout-patients", type=int, default=200)
    p.add_argument("--counterfactual-patients", type=int, default=10)
    p.add_argument("--uncertainty-threshold", type=float, default=1.0)
    p.add_argument("--use-severity-reward", action="store_true")
    p.add_argument("--severity-mode", choices=["surrogate", "handcrafted"], default="surrogate")
    p.add_argument("--severity-model-dir", default="models/icu_readmit/severity_selected")
    p.add_argument("--use-terminal-readmit-reward", action="store_true")
    p.add_argument("--terminal-model-dir", default="models/icu_readmit/terminal_readmit_selected")
    p.add_argument("--terminal-reward-scale", type=float, default=15.0)
    p.add_argument("--device", default=None, help="cpu or cuda; auto-detect if omitted")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log", default="logs/step_12a_caresim_eval.log")
    return p.parse_args()


def one_step_metrics(
    ensemble: CareSimEnsemble,
    data_path: str,
    split: str,
    batch_size: int,
    device: torch.device,
    uncertainty_threshold: float,
    state_cols: list[str],
    action_cols: list[str],
    next_state_cols: list[str],
    severity_model = None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> dict:
    state_names = [c.removeprefix("s_") for c in state_cols]
    static_state_idx = [state_names.index(name) for name in STATIC_STATE_NAMES if name in state_names]
    ds = ICUSequenceDataset.from_parquet(
        data_path,
        split=split,
        max_seq_len=80,
        state_cols=state_cols,
        action_cols=action_cols,
        next_state_cols=next_state_cols,
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_sequences)

    n_rows = 0
    n_nonterminal = 0
    state_se = np.zeros(len(state_cols), dtype=np.float64)
    reward_se = 0.0
    reward_ae = 0.0
    term_correct = 0
    term_brier = 0.0
    uncertainty_sum = 0.0
    uncertainty_flagged = 0
    static_abs_drift = np.zeros(len(static_state_idx), dtype=np.float64)

    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["src_key_padding_mask"].to(device)
            time_steps = batch["time_steps"].to(device)
            real_mask = ~mask

            out = ensemble.predict(states, actions, src_key_padding_mask=mask, time_steps=time_steps)

            ns_true = batch["next_states"].to(device)
            rw_true = batch["rewards"].to(device)
            done_true = batch["terminals"].to(device)
            if severity_model is not None:
                rw_pred = severity_model.score(states) - severity_model.score(out["next_state_mean"])
            elif out["reward_mean"] is not None:
                rw_pred = out["reward_mean"]
            else:
                rw_pred = torch.zeros_like(rw_true)
            if terminal_outcome_model is not None:
                terminal_reward_pred, _ = terminal_outcome_model.terminal_reward(
                    out["next_state_mean"],
                    reward_scale=terminal_reward_scale,
                )
                rw_pred = rw_pred + terminal_reward_pred * done_true

            ns_err = (out["next_state_mean"] - ns_true) ** 2
            rw_err = (rw_pred - rw_true) ** 2
            rw_abs = (rw_pred - rw_true).abs()
            term_prob = out["terminal_prob"]
            term_pred = (term_prob > 0.5).float()
            unc = out["next_state_std"].mean(dim=-1)

            n_real = int(real_mask.sum().item())
            n_rows += n_real
            nonterminal_mask = real_mask & (done_true < 0.5)
            n_nonterminal += int(nonterminal_mask.sum().item())

            real_mask_f = real_mask.float()
            nonterminal_mask_f = nonterminal_mask.float()
            state_se += (ns_err * real_mask_f.unsqueeze(-1)).sum(dim=(0, 1)).cpu().numpy()
            reward_se += float((rw_err * nonterminal_mask_f).sum().item())
            reward_ae += float((rw_abs * nonterminal_mask_f).sum().item())
            term_correct += int(((term_pred == done_true).float() * real_mask_f).sum().item())
            term_brier += float((((term_prob - done_true) ** 2) * real_mask_f).sum().item())
            uncertainty_sum += float((unc * real_mask_f).sum().item())
            uncertainty_flagged += int(((unc > uncertainty_threshold).float() * real_mask_f).sum().item())

            if static_state_idx:
                static_pred = out["next_state_mean"][..., static_state_idx]
                static_true = ns_true[..., static_state_idx]
                static_abs_drift += (
                    (static_pred - static_true).abs() * real_mask_f.unsqueeze(-1)
                ).sum(dim=(0, 1)).cpu().numpy()

    per_feature_mse = {state_names[i]: float(state_se[i] / max(n_rows, 1)) for i in range(len(state_cols))}
    static_drift = {
        state_names[static_state_idx[i]]: float(static_abs_drift[i] / max(n_rows, 1))
        for i in range(len(static_state_idx))
    }
    return {
        "split": split,
        "n_stays": len(ds),
        "n_rows": n_rows,
        "n_nonterminal_rows": n_nonterminal,
        "next_state_mse": float(state_se.sum() / (max(n_rows, 1) * len(state_cols))),
        "next_state_per_feature_mse": per_feature_mse,
        "reward_mse": float(reward_se / max(n_nonterminal, 1)),
        "reward_mae": float(reward_ae / max(n_nonterminal, 1)),
        "terminal_accuracy": float(term_correct / max(n_rows, 1)),
        "terminal_brier": float(term_brier / max(n_rows, 1)),
        "mean_uncertainty": float(uncertainty_sum / max(n_rows, 1)),
        "uncertainty_flag_rate": float(uncertainty_flagged / max(n_rows, 1)),
        "static_abs_drift": static_drift,
        "reward_source": "severity+terminal" if (severity_model is not None and terminal_outcome_model is not None) else (
            "severity" if severity_model is not None else ("head" if ensemble.models[0].predict_reward else "zero")
        ),
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
    ensemble: CareSimEnsemble,
    df: pd.DataFrame,
    split: str,
    history_len: int,
    rollout_steps: int,
    n_patients: int,
    uncertainty_threshold: float,
    device: torch.device,
    seed: int,
    state_cols: list[str],
    action_cols: list[str],
    next_state_cols: list[str],
    severity_model = None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> dict:
    state_names = [c.removeprefix("s_") for c in state_cols]
    static_state_idx = [state_names.index(name) for name in STATIC_STATE_NAMES if name in state_names]
    episodes = sample_rollout_episodes(df, split, history_len, rollout_steps, n_patients, seed)
    if not episodes:
        return {
            "split": split,
            "n_patients": 0,
            "n_steps": rollout_steps,
            "per_step_state_mse": [],
            "per_step_reward_mae": [],
            "per_step_done_accuracy": [],
            "per_step_mean_uncertainty": [],
            "per_step_uncertainty_flag_rate": [],
            "per_step_static_abs_drift": {state_names[idx]: [] for idx in static_state_idx},
        }

    state_se = np.zeros((rollout_steps, len(state_cols)), dtype=np.float64)
    reward_ae = np.zeros(rollout_steps, dtype=np.float64)
    done_correct = np.zeros(rollout_steps, dtype=np.float64)
    uncertainty_sum = np.zeros(rollout_steps, dtype=np.float64)
    uncertainty_flagged = np.zeros(rollout_steps, dtype=np.float64)
    static_drift = np.zeros((rollout_steps, len(static_state_idx)), dtype=np.float64)

    for stay_df in episodes:
        seed_rows = stay_df.iloc[:history_len]
        future_rows = stay_df.iloc[history_len:history_len + rollout_steps].reset_index(drop=True)

        seed_states = torch.tensor(seed_rows[state_cols].values, dtype=torch.float32).unsqueeze(0)
        seed_actions = torch.tensor(seed_rows[action_cols].values, dtype=torch.float32).unsqueeze(0)
        seed_time_steps = torch.tensor(seed_rows[BLOC_COL].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32).unsqueeze(0)

        env = CareSimEnvironment(
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
            action_np = row[action_cols].to_numpy(dtype=np.float32, copy=True)
            action = torch.tensor(action_np, dtype=torch.float32).unsqueeze(0)
            next_state, reward, done, info = env.step(action)

            actual_next_np = row[next_state_cols].to_numpy(dtype=np.float32, copy=True)
            actual_next_state = torch.tensor(actual_next_np, dtype=torch.float32)
            actual_reward = float(row[REWARD_COL])
            actual_done = bool(row[DONE_COL])

            pred_state = next_state[0].cpu().numpy()
            state_se[t] += (pred_state - actual_next_state.numpy()) ** 2
            reward_ae[t] += abs(float(reward[0]) - actual_reward)
            done_correct[t] += float(bool(done[0]) == actual_done)
            uncertainty_sum[t] += float(info["uncertainty"][0])
            uncertainty_flagged[t] += float(info["uncertain_flag"][0])
            if static_state_idx:
                static_drift[t] += np.abs(pred_state[static_state_idx] - actual_next_state.numpy()[static_state_idx])

    n = len(episodes)
    per_step_state_mse = (state_se.sum(axis=1) / (n * len(state_cols))).tolist()
    per_step_reward_mae = (reward_ae / n).tolist()
    per_step_done_accuracy = (done_correct / n).tolist()
    per_step_mean_uncertainty = (uncertainty_sum / n).tolist()
    per_step_uncertainty_flag_rate = (uncertainty_flagged / n).tolist()
    per_step_static_abs_drift = {
        state_names[static_state_idx[i]]: (static_drift[:, i] / n).tolist()
        for i in range(len(static_state_idx))
    }

    return {
        "split": split,
        "n_patients": n,
        "n_steps": rollout_steps,
        "per_step_state_mse": per_step_state_mse,
        "per_step_reward_mae": per_step_reward_mae,
        "per_step_done_accuracy": per_step_done_accuracy,
        "per_step_mean_uncertainty": per_step_mean_uncertainty,
        "per_step_uncertainty_flag_rate": per_step_uncertainty_flag_rate,
        "per_step_static_abs_drift": per_step_static_abs_drift,
    }


def counterfactual_sweep(
    ensemble: CareSimEnsemble,
    df: pd.DataFrame,
    split: str,
    history_len: int,
    n_patients: int,
    uncertainty_threshold: float,
    device: torch.device,
    seed: int,
    state_cols: list[str],
    action_cols: list[str],
    severity_model = None,
    terminal_outcome_model: LightGBMReadmitModel | None = None,
    terminal_reward_scale: float = 15.0,
) -> pd.DataFrame:
    state_names = [c.removeprefix("s_") for c in state_cols]
    action_grid = np.array(
        [[(a >> bit) & 1 for bit in range(len(action_cols))] for a in range(2 ** len(action_cols))],
        dtype=np.float32,
    )
    episodes = sample_rollout_episodes(df, split, history_len, rollout_steps=1, n_patients=n_patients, seed=seed)
    rows = []
    for stay_df in episodes:
        seed_rows = stay_df.iloc[:history_len]
        stay_id = int(seed_rows[STAY_COL].iloc[0])
        seed_states = torch.tensor(seed_rows[state_cols].values, dtype=torch.float32).unsqueeze(0)
        seed_actions = torch.tensor(seed_rows[action_cols].values, dtype=torch.float32).unsqueeze(0)
        seed_time_steps = torch.tensor(seed_rows[BLOC_COL].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32).unsqueeze(0)

        for action_id, action_vec in enumerate(action_grid):
            env = CareSimEnvironment(
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
            for j, col in enumerate(action_cols):
                row[col] = float(action_vec[j])
            for j, name in enumerate(state_names):
                row[f"next_{name}_pred"] = float(next_state[0, j])
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["stay_id", "reward_pred", "uncertainty"], ascending=[True, False, True])
    return out


def write_json(path: Path, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main():
    args = parse_args()
    if args.smoke:
        args.rollout_patients = 20
        args.counterfactual_patients = 3
        args.rollout_steps = 3
        args.batch_size = 16

    device = torch.device(args.device) if args.device else (
        torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    )
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(log_path, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )

    t0 = time.time()
    logging.info("Step 12a started. device=%s smoke=%s", device, args.smoke)
    logging.info("Loading CARE-Sim ensemble: %s", args.model_dir)
    ensemble = CareSimEnsemble.from_dir(args.model_dir, device=device)
    state_cols, action_cols, next_state_cols = infer_schema_from_path(args.data)
    state_names = [c.removeprefix("s_") for c in state_cols]
    severity_model = None
    terminal_outcome_model = None
    if args.use_severity_reward:
        severity_model = load_severity_model(
            mode=args.severity_mode,
            model_dir=args.severity_model_dir,
            state_feature_names=state_names,
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
            state_feature_names=state_cols,
            device=device,
        )
        logging.info("Using terminal readmission reward from %s (scale=%.1f)", args.terminal_model_dir, args.terminal_reward_scale)

    logging.info("Loading data: %s", args.data)
    use_cols = [STAY_COL, BLOC_COL, SPLIT_COL, REWARD_COL, DONE_COL, *state_cols, *action_cols, *next_state_cols]
    df = pd.read_parquet(args.data, columns=use_cols)
    logging.info("Data rows=%d stays=%d", len(df), df[STAY_COL].nunique())

    one_step_val = one_step_metrics(
        ensemble,
        args.data,
        "val",
        args.batch_size,
        device,
        args.uncertainty_threshold,
        state_cols,
        action_cols,
        next_state_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    one_step_test = one_step_metrics(
        ensemble,
        args.data,
        "test",
        args.batch_size,
        device,
        args.uncertainty_threshold,
        state_cols,
        action_cols,
        next_state_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    write_json(report_dir / "caresim_one_step_val.json", one_step_val)
    write_json(report_dir / "caresim_one_step_test.json", one_step_test)
    logging.info("One-step val next_state_mse=%.5f reward_mae=%.5f term_acc=%.5f unc=%.5f",
                 one_step_val["next_state_mse"], one_step_val["reward_mae"],
                 one_step_val["terminal_accuracy"], one_step_val["mean_uncertainty"])
    logging.info("One-step test next_state_mse=%.5f reward_mae=%.5f term_acc=%.5f unc=%.5f",
                 one_step_test["next_state_mse"], one_step_test["reward_mae"],
                 one_step_test["terminal_accuracy"], one_step_test["mean_uncertainty"])

    rollout_val = rollout_metrics(
        ensemble, df, "val", args.history_len, args.rollout_steps,
        args.rollout_patients, args.uncertainty_threshold, device, args.seed,
        state_cols, action_cols, next_state_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    rollout_test = rollout_metrics(
        ensemble, df, "test", args.history_len, args.rollout_steps,
        args.rollout_patients, args.uncertainty_threshold, device, args.seed + 1,
        state_cols, action_cols, next_state_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    write_json(report_dir / "caresim_rollout_val.json", rollout_val)
    write_json(report_dir / "caresim_rollout_test.json", rollout_test)
    if rollout_val["n_patients"] > 0:
        logging.info("Rollout val step1_mse=%.5f stepN_mse=%.5f mean_unc_step1=%.5f",
                     rollout_val["per_step_state_mse"][0],
                     rollout_val["per_step_state_mse"][-1],
                     rollout_val["per_step_mean_uncertainty"][0])
    if rollout_test["n_patients"] > 0:
        logging.info("Rollout test step1_mse=%.5f stepN_mse=%.5f mean_unc_step1=%.5f",
                     rollout_test["per_step_state_mse"][0],
                     rollout_test["per_step_state_mse"][-1],
                     rollout_test["per_step_mean_uncertainty"][0])

    cf_val = counterfactual_sweep(
        ensemble, df, "val", args.history_len, args.counterfactual_patients,
        args.uncertainty_threshold, device, args.seed,
        state_cols, action_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    cf_path = report_dir / "caresim_counterfactual_val.csv"
    cf_val.to_csv(cf_path, index=False)
    logging.info("Counterfactual rows saved: %d", len(cf_val))

    summary = {
        "data": args.data,
        "model_dir": args.model_dir,
        "device": str(device),
        "state_cols": state_cols,
        "action_cols": action_cols,
        "history_len": args.history_len,
        "rollout_steps": args.rollout_steps,
        "rollout_patients": args.rollout_patients,
        "counterfactual_patients": args.counterfactual_patients,
        "reward_source": "severity+terminal" if (severity_model is not None and terminal_outcome_model is not None) else (
            "severity" if severity_model is not None else ("head" if ensemble.models[0].predict_reward else "zero")
        ),
        "severity_mode": args.severity_mode if severity_model is not None else None,
        "use_terminal_readmit_reward": bool(terminal_outcome_model is not None),
        "one_step_val": {
            "next_state_mse": one_step_val["next_state_mse"],
            "reward_mae": one_step_val["reward_mae"],
            "terminal_accuracy": one_step_val["terminal_accuracy"],
            "mean_uncertainty": one_step_val["mean_uncertainty"],
        },
        "one_step_test": {
            "next_state_mse": one_step_test["next_state_mse"],
            "reward_mae": one_step_test["reward_mae"],
            "terminal_accuracy": one_step_test["terminal_accuracy"],
            "mean_uncertainty": one_step_test["mean_uncertainty"],
        },
        "rollout_val": {
            "n_patients": rollout_val["n_patients"],
            "step1_state_mse": rollout_val["per_step_state_mse"][0] if rollout_val["n_patients"] else None,
            "last_state_mse": rollout_val["per_step_state_mse"][-1] if rollout_val["n_patients"] else None,
            "step1_uncertainty": rollout_val["per_step_mean_uncertainty"][0] if rollout_val["n_patients"] else None,
        },
        "rollout_test": {
            "n_patients": rollout_test["n_patients"],
            "step1_state_mse": rollout_test["per_step_state_mse"][0] if rollout_test["n_patients"] else None,
            "last_state_mse": rollout_test["per_step_state_mse"][-1] if rollout_test["n_patients"] else None,
            "step1_uncertainty": rollout_test["per_step_mean_uncertainty"][0] if rollout_test["n_patients"] else None,
        },
        "counterfactual_rows": int(len(cf_val)),
        "total_seconds": round(time.time() - t0, 1),
    }
    write_json(report_dir / "caresim_summary.json", summary)

    logging.info("Reports written to: %s", report_dir)
    logging.info("Step 12a complete in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
