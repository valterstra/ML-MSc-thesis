"""
Step 12c -- DAG-aware temporal simulator evaluation and sanity checks.

This mirrors the CARE-Sim evaluation flow for the new DAG-aware simulator family:
  1. one-step prediction metrics on val/test
  2. closed-loop rollout diagnostics under clinician actions
  3. counterfactual one-step action sweeps on real patient seeds
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from careai.icu_readmit.caresim.dataset import BLOC_COL, DONE_COL, REWARD_COL, SPLIT_COL, STAY_COL, infer_schema_from_path
from careai.icu_readmit.caresim.readmit import LightGBMReadmitModel
from careai.icu_readmit.caresim.severity import load_severity_model
from careai.icu_readmit.dagaware.ensemble import DAGAwareEnsemble
from careai.icu_readmit.dagaware.simulator import DAGAwareEnvironment


STATIC_STATE_NAMES = ["age", "charlson_score", "prior_ed_visits_6m"]


def load_caresim_eval_module():
    path = PROJECT_ROOT / "scripts" / "icu_readmit" / "step_12a_caresim_evaluate.py"
    spec = importlib.util.spec_from_file_location("step_12a_caresim_evaluate", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load evaluation helpers from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.CareSimEnvironment = DAGAwareEnvironment
    module.CareSimEnsemble = DAGAwareEnsemble
    return module


def parse_args():
    parser = argparse.ArgumentParser(description="Step 12c: DAG-aware simulator evaluation")
    parser.add_argument("--data", default="data/processed/icu_readmit/rl_dataset_selected.parquet")
    parser.add_argument("--model-dir", default="models/icu_readmit/dagaware_selected_causal")
    parser.add_argument("--report-dir", default="reports/icu_readmit/dagaware_selected_causal")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--history-len", type=int, default=5)
    parser.add_argument("--rollout-steps", type=int, default=5)
    parser.add_argument("--rollout-patients", type=int, default=200)
    parser.add_argument("--counterfactual-patients", type=int, default=10)
    parser.add_argument("--uncertainty-threshold", type=float, default=1.0)
    parser.add_argument("--use-severity-reward", action="store_true")
    parser.add_argument("--severity-mode", choices=["surrogate", "handcrafted"], default="surrogate")
    parser.add_argument("--severity-model-dir", default="models/icu_readmit/severity_selected")
    parser.add_argument("--use-terminal-readmit-reward", action="store_true")
    parser.add_argument("--terminal-model-dir", default="models/icu_readmit/terminal_readmit_selected")
    parser.add_argument("--terminal-reward-scale", type=float, default=15.0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--log", default="logs/step_12c_dagaware_eval.log")
    return parser.parse_args()


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

    helper = load_caresim_eval_module()
    t0 = time.time()
    logging.info("Step 12c started. device=%s smoke=%s", device, args.smoke)
    logging.info("Loading DAG-aware ensemble: %s", args.model_dir)
    ensemble = DAGAwareEnsemble.from_dir(args.model_dir, device=device)
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
        logging.info(
            "Using terminal readmission reward from %s (scale=%.1f)",
            args.terminal_model_dir,
            args.terminal_reward_scale,
        )

    logging.info("Loading data: %s", args.data)
    use_cols = [STAY_COL, BLOC_COL, SPLIT_COL, REWARD_COL, DONE_COL, *state_cols, *action_cols, *next_state_cols]
    df = pd.read_parquet(args.data, columns=use_cols)
    logging.info("Data rows=%d stays=%d", len(df), df[STAY_COL].nunique())

    one_step_val = helper.one_step_metrics(
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
    one_step_test = helper.one_step_metrics(
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
    helper.write_json(report_dir / "dagaware_one_step_val.json", one_step_val)
    helper.write_json(report_dir / "dagaware_one_step_test.json", one_step_test)

    rollout_val = helper.rollout_metrics(
        ensemble,
        df,
        "val",
        args.history_len,
        args.rollout_steps,
        args.rollout_patients,
        args.uncertainty_threshold,
        device,
        args.seed,
        state_cols,
        action_cols,
        next_state_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    rollout_test = helper.rollout_metrics(
        ensemble,
        df,
        "test",
        args.history_len,
        args.rollout_steps,
        args.rollout_patients,
        args.uncertainty_threshold,
        device,
        args.seed + 1,
        state_cols,
        action_cols,
        next_state_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    helper.write_json(report_dir / "dagaware_rollout_val.json", rollout_val)
    helper.write_json(report_dir / "dagaware_rollout_test.json", rollout_test)

    cf_val = helper.counterfactual_sweep(
        ensemble,
        df,
        "val",
        args.history_len,
        args.counterfactual_patients,
        args.uncertainty_threshold,
        device,
        args.seed,
        state_cols,
        action_cols,
        severity_model=severity_model,
        terminal_outcome_model=terminal_outcome_model,
        terminal_reward_scale=args.terminal_reward_scale,
    )
    cf_path = report_dir / "dagaware_counterfactual_val.csv"
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
            "severity" if severity_model is not None else "zero"
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
    with open(report_dir / "dagaware_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logging.info("Reports written to: %s", report_dir)
    logging.info("Step 12c complete in %.1f sec", time.time() - t0)


if __name__ == "__main__":
    main()
