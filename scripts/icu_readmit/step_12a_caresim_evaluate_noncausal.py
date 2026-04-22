"""Step 12a -- Non-causal CARE-Sim evaluation and simulator sanity checks.

Evaluates the trained non-causal CARE-Sim model on held-out data and produces:
  1. One-step prediction metrics on val/test
  2. Closed-loop rollout diagnostics under clinician actions
  3. Counterfactual one-step action sweeps on real patient seeds
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.noncausal_dataset import (  # noqa: E402
    BLOC_COL,
    DONE_COL,
    READMIT_COL,
    SPLIT_COL,
    STAY_COL,
    NonCausalICUSequenceDataset,
    collate_noncausal_sequences,
)
from careai.icu_readmit.caresim.noncausal_inference import NonCausalCareSimModel  # noqa: E402
from careai.icu_readmit.caresim.noncausal_simulator import NonCausalCareSimEnvironment  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 12a: non-causal CARE-Sim evaluation")
    p.add_argument("--data", default="data/processed/icu_readmit/step_10_noncausal/rl_dataset_noncausal.parquet")
    p.add_argument("--model-dir", default="models/icu_readmit/caresim_noncausal")
    p.add_argument("--report-dir", default="reports/icu_readmit/caresim_noncausal")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--history-len", type=int, default=5)
    p.add_argument("--rollout-steps", type=int, default=5)
    p.add_argument("--rollout-patients", type=int, default=200)
    p.add_argument("--counterfactual-patients", type=int, default=10)
    p.add_argument("--max-counterfactual-actions", type=int, default=128)
    p.add_argument("--device", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--log", default="logs/step_12a_caresim_noncausal_eval.log")
    return p.parse_args()


def resolve_device(device_arg: str | None) -> torch.device:
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


def _binary_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true, dtype=np.float32)
    y_prob = np.asarray(y_prob, dtype=np.float32)
    y_pred = (y_prob > 0.5).astype(np.float32)
    out: dict[str, float | None] = {
        "accuracy": float(np.mean(y_pred == y_true)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "logloss": float(log_loss(y_true, y_prob, labels=[0, 1])),
    }
    if len(np.unique(y_true)) >= 2:
        out["auc"] = float(roc_auc_score(y_true, y_prob))
        out["auprc"] = float(average_precision_score(y_true, y_prob))
    else:
        out["auc"] = None
        out["auprc"] = None
    return out


def one_step_metrics(
    model: NonCausalCareSimModel,
    data_path: str,
    split: str,
    batch_size: int,
    device: torch.device,
) -> dict:
    ds = NonCausalICUSequenceDataset.from_parquet(
        data_path,
        state_cols=model.state_cols,
        action_cols=model.action_cols,
        next_state_cols=model.next_state_cols,
        split=split,
        max_seq_len=model.max_seq_len,
        window_mode="last",
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, collate_fn=collate_noncausal_sequences)

    n_rows = 0
    state_se = np.zeros(len(model.next_state_cols), dtype=np.float64)
    term_correct = 0
    term_brier = 0.0
    readmit_probs: list[float] = []
    readmit_true: list[float] = []

    with torch.no_grad():
        for batch in loader:
            states = batch["states"].to(device)
            actions = batch["actions"].to(device)
            mask = batch["src_key_padding_mask"].to(device)
            time_steps = batch["time_steps"].to(device)
            real_mask = ~mask

            out = model.predict(states, actions, src_key_padding_mask=mask, time_steps=time_steps)
            next_true = batch["next_states"].to(device)
            next_pred = out["next_state_dynamic_mean"]
            done_true = batch["terminals"].to(device)
            term_prob = out["terminal_prob"]

            next_err = (next_pred - next_true) ** 2
            real_mask_f = real_mask.float()
            state_se += (next_err * real_mask_f.unsqueeze(-1)).sum(dim=(0, 1)).cpu().numpy()
            n_real = int(real_mask.sum().item())
            n_rows += n_real
            term_pred = (term_prob > 0.5).float()
            term_correct += int(((term_pred == done_true).float() * real_mask_f).sum().item())
            term_brier += float((((term_prob - done_true) ** 2) * real_mask_f).sum().item())

            readmit_probs.extend(out["readmit_prob"].detach().cpu().numpy().tolist())
            readmit_true.extend(batch["readmit"].numpy().tolist())

    state_names = [c.removeprefix("s_next_") for c in model.next_state_cols]
    per_feature_mse = {state_names[i]: float(state_se[i] / max(n_rows, 1)) for i in range(len(model.next_state_cols))}
    return {
        "split": split,
        "n_stays": len(ds),
        "n_rows": n_rows,
        "next_state_mse": float(state_se.sum() / (max(n_rows, 1) * len(model.next_state_cols))),
        "next_state_per_feature_mse": per_feature_mse,
        "terminal_accuracy": float(term_correct / max(n_rows, 1)),
        "terminal_brier": float(term_brier / max(n_rows, 1)),
        "readmit": _binary_metrics(np.array(readmit_true), np.array(readmit_probs)),
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
    model: NonCausalCareSimModel,
    df: pd.DataFrame,
    split: str,
    history_len: int,
    rollout_steps: int,
    n_patients: int,
    device: torch.device,
    seed: int,
) -> dict:
    episodes = sample_rollout_episodes(df, split, history_len, rollout_steps, n_patients, seed)
    if not episodes:
        return {"split": split, "n_patients": 0, "n_steps": rollout_steps}

    state_se = np.zeros((rollout_steps, len(model.next_state_cols)), dtype=np.float64)
    done_correct = np.zeros(rollout_steps, dtype=np.float64)
    final_readmit_prob = []
    final_readmit_true = []

    for stay_df in episodes:
        seed_rows = stay_df.iloc[:history_len]
        future_rows = stay_df.iloc[history_len:history_len + rollout_steps].reset_index(drop=True)
        seed_states = torch.tensor(seed_rows[model.state_cols].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_actions = torch.tensor(seed_rows[model.action_cols].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_time_steps = torch.tensor(seed_rows[BLOC_COL].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32).unsqueeze(0)

        env = NonCausalCareSimEnvironment(model, max_steps=rollout_steps + 10, device=device)
        env.reset(seed_states, seed_actions, seed_time_steps=seed_time_steps)

        last_readmit_prob = float("nan")
        for t, row in future_rows.iterrows():
            action = torch.tensor(row[model.action_cols].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
            next_state, _, done, info = env.step(action)
            actual_next = row[model.next_state_cols].to_numpy(dtype=np.float32, copy=True)
            pred_dynamic = next_state[0, model.dynamic_state_idx].detach().cpu().numpy()
            state_se[t] += (pred_dynamic - actual_next) ** 2
            done_correct[t] += float(bool(done[0]) == bool(row[DONE_COL]))
            last_readmit_prob = float(info["readmit_prob"][0])

        final_readmit_prob.append(last_readmit_prob)
        final_readmit_true.append(float(stay_df[READMIT_COL].iloc[0]))

    n = len(episodes)
    return {
        "split": split,
        "n_patients": n,
        "n_steps": rollout_steps,
        "per_step_state_mse": (state_se.sum(axis=1) / (n * len(model.next_state_cols))).tolist(),
        "per_step_done_accuracy": (done_correct / n).tolist(),
        "final_readmit": _binary_metrics(np.array(final_readmit_true), np.array(final_readmit_prob)),
    }


def observed_action_grid(df: pd.DataFrame, action_cols: list[str], split: str, max_actions: int) -> pd.DataFrame:
    split_df = df[df[SPLIT_COL] == split].copy()
    grid = split_df.groupby(action_cols, dropna=False).size().reset_index(name="count")
    grid = grid.sort_values("count", ascending=False).head(max_actions).reset_index(drop=True)
    grid["action_id"] = np.arange(len(grid))
    return grid


def counterfactual_sweep(
    model: NonCausalCareSimModel,
    df: pd.DataFrame,
    split: str,
    history_len: int,
    n_patients: int,
    max_actions: int,
    device: torch.device,
    seed: int,
) -> pd.DataFrame:
    grid = observed_action_grid(df, model.action_cols, split, max_actions=max_actions)
    episodes = sample_rollout_episodes(df, split, history_len, rollout_steps=1, n_patients=n_patients, seed=seed)
    rows: list[dict] = []
    next_names = [c.removeprefix("s_next_") for c in model.next_state_cols]

    for stay_df in episodes:
        seed_rows = stay_df.iloc[:history_len]
        stay_id = int(seed_rows[STAY_COL].iloc[0])
        seed_states = torch.tensor(seed_rows[model.state_cols].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_actions = torch.tensor(seed_rows[model.action_cols].to_numpy(dtype=np.float32), dtype=torch.float32).unsqueeze(0)
        seed_time_steps = torch.tensor(seed_rows[BLOC_COL].to_numpy(dtype=np.float32, copy=True), dtype=torch.float32).unsqueeze(0)

        for _, action_row in grid.iterrows():
            env = NonCausalCareSimEnvironment(model, max_steps=2, device=device)
            env.reset(seed_states, seed_actions, seed_time_steps=seed_time_steps)
            action_vec = action_row[model.action_cols].to_numpy(dtype=np.float32, copy=True)
            action = torch.tensor(action_vec, dtype=torch.float32).unsqueeze(0)
            next_state, _, done, info = env.step(action)

            row = {
                "split": split,
                "stay_id": stay_id,
                "action_id": int(action_row["action_id"]),
                "action_count": int(action_row["count"]),
                "terminal_prob": float(info["terminal_prob"][0]),
                "readmit_prob": float(info["readmit_prob"][0]),
                "done_pred": bool(done[0]),
            }
            for j, col in enumerate(model.action_cols):
                row[col] = float(action_vec[j])
            pred_dynamic = next_state[0, model.dynamic_state_idx].detach().cpu().numpy()
            for j, name in enumerate(next_names):
                row[f"next_{name}_pred"] = float(pred_dynamic[j])
            rows.append(row)

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(["stay_id", "readmit_prob", "terminal_prob"], ascending=[True, True, True])
    return out


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.rollout_patients = 20
        args.counterfactual_patients = 3
        args.rollout_steps = 3
        args.batch_size = 16
        args.max_counterfactual_actions = min(args.max_counterfactual_actions, 32)

    device = resolve_device(args.device)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(args.log)

    t0 = time.time()
    logging.info("Step 12a non-causal started. device=%s smoke=%s", device, args.smoke)
    logging.info("Loading non-causal CARE-Sim model: %s", args.model_dir)
    model = NonCausalCareSimModel.from_dir(args.model_dir, device=device, which="best")
    logging.info("Loading replay data: %s", args.data)
    use_cols = [STAY_COL, BLOC_COL, SPLIT_COL, DONE_COL, READMIT_COL, *model.state_cols, *model.action_cols, *model.next_state_cols]
    df = pd.read_parquet(args.data, columns=use_cols)
    logging.info("Data rows=%d stays=%d", len(df), df[STAY_COL].nunique())

    one_step_val = one_step_metrics(model, args.data, "val", args.batch_size, device)
    one_step_test = one_step_metrics(model, args.data, "test", args.batch_size, device)
    write_json(report_dir / "caresim_noncausal_one_step_val.json", one_step_val)
    write_json(report_dir / "caresim_noncausal_one_step_test.json", one_step_test)

    rollout_val = rollout_metrics(model, df, "val", args.history_len, args.rollout_steps, args.rollout_patients, device, args.seed)
    rollout_test = rollout_metrics(model, df, "test", args.history_len, args.rollout_steps, args.rollout_patients, device, args.seed + 1)
    write_json(report_dir / "caresim_noncausal_rollout_val.json", rollout_val)
    write_json(report_dir / "caresim_noncausal_rollout_test.json", rollout_test)

    cf_val = counterfactual_sweep(
        model,
        df,
        "val",
        args.history_len,
        args.counterfactual_patients,
        args.max_counterfactual_actions,
        device,
        args.seed,
    )
    cf_path = report_dir / "caresim_noncausal_counterfactual_val.csv"
    cf_val.to_csv(cf_path, index=False)

    summary = {
        "data": args.data,
        "model_dir": args.model_dir,
        "device": str(device),
        "history_len": args.history_len,
        "rollout_steps": args.rollout_steps,
        "rollout_patients": args.rollout_patients,
        "counterfactual_patients": args.counterfactual_patients,
        "max_counterfactual_actions": args.max_counterfactual_actions,
        "state_cols": model.state_cols,
        "dynamic_target_cols": model.next_state_cols,
        "action_cols": model.action_cols,
        "one_step_val": {
            "next_state_mse": one_step_val["next_state_mse"],
            "terminal_accuracy": one_step_val["terminal_accuracy"],
            "readmit_auc": one_step_val["readmit"]["auc"],
            "readmit_brier": one_step_val["readmit"]["brier"],
        },
        "one_step_test": {
            "next_state_mse": one_step_test["next_state_mse"],
            "terminal_accuracy": one_step_test["terminal_accuracy"],
            "readmit_auc": one_step_test["readmit"]["auc"],
            "readmit_brier": one_step_test["readmit"]["brier"],
        },
        "rollout_val": {
            "n_patients": rollout_val["n_patients"],
            "step1_state_mse": rollout_val["per_step_state_mse"][0] if rollout_val["n_patients"] else None,
            "last_state_mse": rollout_val["per_step_state_mse"][-1] if rollout_val["n_patients"] else None,
            "final_readmit_auc": rollout_val["final_readmit"]["auc"] if rollout_val["n_patients"] else None,
        },
        "rollout_test": {
            "n_patients": rollout_test["n_patients"],
            "step1_state_mse": rollout_test["per_step_state_mse"][0] if rollout_test["n_patients"] else None,
            "last_state_mse": rollout_test["per_step_state_mse"][-1] if rollout_test["n_patients"] else None,
            "final_readmit_auc": rollout_test["final_readmit"]["auc"] if rollout_test["n_patients"] else None,
        },
        "counterfactual_rows": int(len(cf_val)),
        "total_seconds": round(time.time() - t0, 1),
    }
    write_json(report_dir / "caresim_noncausal_summary.json", summary)

    logging.info("Reports written to: %s", report_dir)
    logging.info("Step 12a non-causal complete in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
