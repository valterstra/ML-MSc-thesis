"""
Step 12 -- Offline policy evaluation: physician policy + reward/env models + DR.

Chains from: step 11 outputs (models/icu_readmit/continuous/)
Produces:
  models/icu_readmit/eval/
    physician_policy/   physician_policy_model.pt, {train,val,test}_policy.pkl
    reward_estimator/   reward_estimator_model.pt, {val,test}_rewards.pkl
    env_model/          env_model.pt, {val,test}_next_states.pkl
  reports/icu_readmit/
    evaluation_results.json

Key differences from sepsis step_12:
  - Input: rl_dataset_broad.parquet (not CSV; rewards and actions precomputed)
  - n_actions=32 (2^5 binary drug combos)
  - Action represented as 5-bit binary vector in reward/env models
    (no orig_data argument; decoded directly from data["actions"])
  - value_clip=40 for DR (wider Q range with 32 actions vs 25)
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

import torch
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))
from src.careai.icu_readmit.rl.continuous import prepare_rl_data
from src.careai.icu_readmit.rl.evaluation import (
    doubly_robust_evaluation,
    predict_next_states,
    predict_physician_probs,
    predict_rewards,
    train_env_model,
    train_physician_policy,
    train_reward_estimator,
)

N_ACTIONS = 32


def main():
    parser = argparse.ArgumentParser(description="Step 12: Offline Policy Evaluation")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24"),
                        help="Directory containing rl_dataset_broad.parquet")
    parser.add_argument("--continuous-model-dir", default=str(PROJECT_ROOT / "models" / "icu_readmit" / "legacy" / "step_25_33" / "continuous"),
                        help="Directory with step 11 DDQN/SARSA models")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models" / "icu_readmit" / "legacy" / "step_25_33" / "eval"),
                        help="Output directory for evaluation models")
    parser.add_argument("--report-dir", default=str(PROJECT_ROOT / "reports" / "icu_readmit" / "legacy" / "step_30_33"),
                        help="Output directory for evaluation reports")
    parser.add_argument("--physician-steps", type=int, default=35000)
    parser.add_argument("--reward-steps",    type=int, default=30000)
    parser.add_argument("--env-steps",       type=int, default=60000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 200 steps each")
    parser.add_argument("--log", default=str(PROJECT_ROOT / "logs" / "legacy" / "icu_readmit" / "step_30_ope_broad_legacy.log"))
    args = parser.parse_args()

    if args.smoke:
        args.physician_steps = 200
        args.reward_steps    = 200
        args.env_steps       = 200

    os.makedirs(args.model_dir,  exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.log), exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    mode_str = "SMOKE TEST" if args.smoke else "FULL"
    logging.info("Step 12 started. mode=%s, device=%s", mode_str, args.device)
    t0 = time.time()

    # ── Load broad parquet ────────────────────────────────────────────
    suffix = "_broad_smoke" if args.smoke else "_broad"
    parquet_path = os.path.join(args.data_dir, f"rl_dataset{suffix}.parquet")
    logging.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logging.info("  %d rows, %d cols", len(df), len(df.columns))

    state_cols      = [c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")]
    next_state_cols = [c.replace("s_", "s_next_", 1) for c in state_cols]
    n_state         = len(state_cols)
    logging.info("  %d state features", n_state)

    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)
    logging.info("  Splits: train=%d, val=%d, test=%d rows",
                 len(df_train), len(df_val), len(df_test))
    del df

    train_data = prepare_rl_data(df_train, state_cols, next_state_cols)
    val_data   = prepare_rl_data(df_val,   state_cols, next_state_cols)
    test_data  = prepare_rl_data(df_test,  state_cols, next_state_cols)

    log_every = 100 if args.smoke else 5000

    # ── Step 1: Physician policy ───────────────────────────────────────
    logging.info("--- Training physician policy ---")
    phys_dir  = os.path.join(args.model_dir, "physician_policy")
    phys_model = train_physician_policy(
        train_data, n_state=n_state, n_actions=N_ACTIONS,
        num_steps=args.physician_steps,
        save_dir=phys_dir, device=args.device, log_every=log_every,
    )

    train_probs = predict_physician_probs(
        phys_model, train_data, device=args.device,
        save_path=os.path.join(phys_dir, "train_policy.pkl"),
    )
    val_probs = predict_physician_probs(
        phys_model, val_data, device=args.device,
        save_path=os.path.join(phys_dir, "val_policy.pkl"),
    )
    test_probs = predict_physician_probs(
        phys_model, test_data, device=args.device,
        save_path=os.path.join(phys_dir, "test_policy.pkl"),
    )
    logging.info("  Physician probs computed for all splits")

    # ── Step 2: Reward estimator ───────────────────────────────────────
    logging.info("--- Training reward estimator ---")
    rew_dir   = os.path.join(args.model_dir, "reward_estimator")
    rew_model = train_reward_estimator(
        train_data, n_state=n_state,
        num_steps=args.reward_steps,
        save_dir=rew_dir, device=args.device, log_every=log_every,
    )

    val_rewards_est = predict_rewards(
        rew_model, val_data, device=args.device,
        save_path=os.path.join(rew_dir, "val_rewards.pkl"),
    )
    test_rewards_est = predict_rewards(
        rew_model, test_data, device=args.device,
        save_path=os.path.join(rew_dir, "test_rewards.pkl"),
    )

    # ── Step 3: Environment model ──────────────────────────────────────
    logging.info("--- Training environment model ---")
    env_dir   = os.path.join(args.model_dir, "env_model")
    env_model = train_env_model(
        train_data, n_state=n_state,
        num_steps=args.env_steps,
        save_dir=env_dir, device=args.device, log_every=log_every,
    )

    predict_next_states(
        env_model, val_data, device=args.device,
        save_path=os.path.join(env_dir, "val_next_states.pkl"),
    )
    predict_next_states(
        env_model, test_data, device=args.device,
        save_path=os.path.join(env_dir, "test_next_states.pkl"),
    )

    # ── Step 4: Doubly Robust evaluation ──────────────────────────────
    logging.info("--- Doubly Robust off-policy evaluation ---")
    results = {}

    ddqn_dir  = os.path.join(args.continuous_model_dir, "ddqn")
    sarsa_dir = os.path.join(args.continuous_model_dir, "sarsa_phys")

    policies_to_eval = []

    # DDQN on train set
    if os.path.exists(os.path.join(ddqn_dir, "dqn_actions.pkl")):
        policies_to_eval.append((
            "ddqn_train",
            os.path.join(ddqn_dir, "dqn_actions.pkl"),
            os.path.join(ddqn_dir, "dqn_q_values.pkl"),
            train_data, train_probs, None,  # None = compute fresh
        ))

    # DDQN on test set
    if os.path.exists(os.path.join(ddqn_dir, "dqn_actions_test.pkl")):
        policies_to_eval.append((
            "ddqn_test",
            os.path.join(ddqn_dir, "dqn_actions_test.pkl"),
            os.path.join(ddqn_dir, "dqn_q_test.pkl"),
            test_data, test_probs, test_rewards_est,
        ))

    # SARSA physician on test set
    if os.path.exists(os.path.join(sarsa_dir, "phys_actions_test.pkl")):
        policies_to_eval.append((
            "sarsa_physician_test",
            os.path.join(sarsa_dir, "phys_actions_test.pkl"),
            os.path.join(sarsa_dir, "phys_q_test.pkl"),
            test_data, test_probs, test_rewards_est,
        ))

    for policy_name, actions_path, q_path, data_split, probs, rewards_est in policies_to_eval:
        logging.info("  Evaluating: %s", policy_name)
        with open(actions_path, "rb") as f:
            agent_actions = pickle.load(f)
        with open(q_path, "rb") as f:
            agent_q = pickle.load(f)

        # Compute reward estimates if not precomputed
        if rewards_est is None:
            rewards_est = predict_rewards(rew_model, data_split, device=args.device)

        dr_result = doubly_robust_evaluation(
            data_split,
            agent_actions=agent_actions,
            agent_q_values=agent_q,
            physician_probs=probs,
            reward_estimates=rewards_est,
            gamma=0.99,
            value_clip=40.0,
        )

        results[policy_name] = {
            "dr_mean":        dr_result["mean"],
            "dr_std":         dr_result["std"],
            "n_trajectories": dr_result["n_trajectories"],
            "n_valid":        dr_result["n_valid"],
        }

    # ── Save results ───────────────────────────────────────────────────
    results_path = os.path.join(args.report_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Results saved: %s", results_path)

    logging.info("=== EVALUATION SUMMARY ===")
    for name, r in results.items():
        logging.info("  %-30s  DR = %.4f +/- %.4f  (%d/%d trajectories)",
                     name, r["dr_mean"], r["dr_std"], r["n_valid"], r["n_trajectories"])

    dt = time.time() - t0
    logging.info("Step 12 complete. %.1f sec (%.1f min)", dt, dt / 60)


if __name__ == "__main__":
    main()
