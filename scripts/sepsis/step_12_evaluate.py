"""
Step 12 - Offline policy evaluation: physician policy + reward/env models + DR.

Chains from: step 11 outputs (continuous RL models + data)
Produces:
  models/sepsis_rl/eval/
    physician_policy/     (supervised physician policy model)
    reward_estimator/     (reward function approximator)
    env_model/            (transition dynamics model)
  reports/sepsis_rl/
    evaluation_results.json   (DR value estimates for all policies)

Ported from: sepsisrl/eval/physician_policy_tf.ipynb
             sepsisrl/eval/reward_estimator_new.ipynb
             sepsisrl/eval/env_model_regression_for_eval.ipynb
             sepsisrl/eval/doubly_robust.ipynb
"""
import argparse
import json
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.careai.sepsis.rl.preprocessing import STATE_FEATURES
from src.careai.sepsis.rl.continuous import prepare_rl_data
from src.careai.sepsis.rl.evaluation import (
    doubly_robust_evaluation,
    predict_next_states,
    predict_physician_probs,
    predict_rewards,
    train_env_model,
    train_physician_policy,
    train_reward_estimator,
)

RL_STATE_FEATURES = [f for f in STATE_FEATURES if f != "bloc"]


def main():
    parser = argparse.ArgumentParser(description="Step 12: Offline Policy Evaluation")
    parser.add_argument(
        "--data-dir", default="data/processed/sepsis",
        help="Directory with RL data",
    )
    parser.add_argument(
        "--continuous-model-dir", default="models/sepsis_rl/continuous",
        help="Directory with DQN/SARSA models from step 11",
    )
    parser.add_argument(
        "--model-dir", default="models/sepsis_rl/eval",
        help="Output directory for evaluation models",
    )
    parser.add_argument(
        "--report-dir", default="reports/sepsis_rl",
        help="Output directory for evaluation reports",
    )
    parser.add_argument(
        "--physician-steps", type=int, default=35000,
        help="Physician policy training steps",
    )
    parser.add_argument(
        "--reward-steps", type=int, default=30000,
        help="Reward estimator training steps",
    )
    parser.add_argument(
        "--env-steps", type=int, default=60000,
        help="Environment model training steps",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test mode: tiny step counts to verify pipeline works",
    )
    parser.add_argument(
        "--log", default="logs/step_12_evaluate.log",
        help="Log file path",
    )
    args = parser.parse_args()

    # Smoke test: override all step counts
    if args.smoke:
        args.physician_steps = 200
        args.reward_steps = 200
        args.env_steps = 200

    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.report_dir, exist_ok=True)

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
    logging.info("Step 12 started. mode=%s", mode_str)
    t0 = time.time()

    # ── Load data ─────────────────────────────────────────────────────
    logging.info("Loading continuous RL datasets")
    train_cont = pd.read_csv(f"{args.data_dir}/rl_train_data_final_cont.csv")
    val_cont = pd.read_csv(f"{args.data_dir}/rl_val_data_final_cont.csv")
    test_cont = pd.read_csv(f"{args.data_dir}/rl_test_data_final_cont.csv")

    state_cols = [c for c in RL_STATE_FEATURES if c in train_cont.columns]
    n_state = len(state_cols)
    logging.info("  %d state features, train=%d, val=%d, test=%d",
                 n_state, len(train_cont), len(val_cont), len(test_cont))

    train_data = prepare_rl_data(train_cont, state_cols, device=args.device)
    val_data = prepare_rl_data(val_cont, state_cols, device=args.device)
    test_data = prepare_rl_data(test_cont, state_cols, device=args.device)

    log_every = 100 if args.smoke else 5000

    # Original action columns for reward/env models
    train_orig = {
        "iv_input": train_cont["iv_input"].values if "iv_input" in train_cont.columns
                    else np.zeros(len(train_cont)),
        "vaso_input": train_cont["vaso_input"].values if "vaso_input" in train_cont.columns
                      else np.zeros(len(train_cont)),
    }
    val_orig = {
        "iv_input": val_cont["iv_input"].values if "iv_input" in val_cont.columns
                    else np.zeros(len(val_cont)),
        "vaso_input": val_cont["vaso_input"].values if "vaso_input" in val_cont.columns
                      else np.zeros(len(val_cont)),
    }
    test_orig = {
        "iv_input": test_cont["iv_input"].values if "iv_input" in test_cont.columns
                    else np.zeros(len(test_cont)),
        "vaso_input": test_cont["vaso_input"].values if "vaso_input" in test_cont.columns
                      else np.zeros(len(test_cont)),
    }

    # ── Step 1: Physician policy ──────────────────────────────────────
    logging.info("--- Training physician policy ---")
    phys_dir = f"{args.model_dir}/physician_policy"
    phys_model = train_physician_policy(
        train_data, n_state=n_state,
        num_steps=args.physician_steps,
        save_dir=phys_dir, device=args.device,
        log_every=log_every,
    )

    # Predict probabilities for all splits
    train_probs = predict_physician_probs(
        phys_model, train_data, device=args.device,
        save_path=f"{phys_dir}/train_policy.pkl",
    )
    val_probs = predict_physician_probs(
        phys_model, val_data, device=args.device,
        save_path=f"{phys_dir}/val_policy.pkl",
    )
    test_probs = predict_physician_probs(
        phys_model, test_data, device=args.device,
        save_path=f"{phys_dir}/test_policy.pkl",
    )
    logging.info("  Physician policy probs computed for all splits")

    # ── Step 2: Reward estimator ──────────────────────────────────────
    logging.info("--- Training reward estimator ---")
    rew_dir = f"{args.model_dir}/reward_estimator"
    rew_model = train_reward_estimator(
        train_data, train_orig, n_state=n_state,
        num_steps=args.reward_steps,
        save_dir=rew_dir, device=args.device,
        log_every=log_every,
    )

    val_rewards_est = predict_rewards(
        rew_model, val_data, val_orig, device=args.device,
        save_path=f"{rew_dir}/val_rewards.pkl",
    )
    test_rewards_est = predict_rewards(
        rew_model, test_data, test_orig, device=args.device,
        save_path=f"{rew_dir}/test_rewards.pkl",
    )

    # ── Step 3: Environment model ─────────────────────────────────────
    logging.info("--- Training environment model ---")
    env_dir = f"{args.model_dir}/env_model"
    env_model = train_env_model(
        train_data, train_orig, n_state=n_state,
        num_steps=args.env_steps,
        save_dir=env_dir, device=args.device,
        log_every=log_every,
    )

    val_next_est = predict_next_states(
        env_model, val_data, val_orig, device=args.device,
        save_path=f"{env_dir}/val_next_states.pkl",
    )
    test_next_est = predict_next_states(
        env_model, test_data, test_orig, device=args.device,
        save_path=f"{env_dir}/test_next_states.pkl",
    )

    # ── Step 4: Doubly Robust evaluation ──────────────────────────────
    logging.info("--- Doubly Robust off-policy evaluation ---")
    results = {}

    # Load DQN agent actions and Q-values
    dqn_dir = f"{args.continuous_model_dir}/dqn"
    policies_to_eval = []

    for policy_name, actions_path, q_path in [
        ("dqn", f"{dqn_dir}/dqn_actions.pkl", f"{dqn_dir}/dqn_q_values.pkl"),
        ("dqn_test", f"{dqn_dir}/dqn_actions_test.pkl", f"{dqn_dir}/dqn_q_test.pkl"),
    ]:
        if os.path.exists(actions_path) and os.path.exists(q_path):
            policies_to_eval.append((policy_name, actions_path, q_path))

    # Also check for SARSA physician
    sarsa_dir = f"{args.continuous_model_dir}/sarsa_phys"
    if os.path.exists(f"{sarsa_dir}/phys_actions.pkl"):
        policies_to_eval.append((
            "sarsa_physician",
            f"{sarsa_dir}/phys_actions.pkl",
            f"{sarsa_dir}/phys_q_values.pkl",
        ))

    for policy_name, actions_path, q_path in policies_to_eval:
        logging.info("  Evaluating policy: %s", policy_name)
        with open(actions_path, "rb") as f:
            agent_actions = pickle.load(f)
        with open(q_path, "rb") as f:
            agent_q = pickle.load(f)

        # Use train set for evaluation (matching physician probs)
        if len(agent_actions) == len(train_data["actions"]):
            probs = train_probs
            data_for_eval = train_data
            rewards_est = predict_rewards(
                rew_model, train_data, train_orig, device=args.device,
            )
        elif len(agent_actions) == len(test_data["actions"]):
            probs = test_probs
            data_for_eval = test_data
            rewards_est = test_rewards_est
        else:
            logging.warning("  Skipping %s: action length %d doesn't match any split",
                            policy_name, len(agent_actions))
            continue

        dr_result = doubly_robust_evaluation(
            data_for_eval,
            agent_actions=agent_actions,
            agent_q_values=agent_q,
            physician_probs=probs,
            reward_estimates=rewards_est,
            gamma=0.99,
        )

        results[policy_name] = {
            "dr_mean": dr_result["mean"],
            "dr_std": dr_result["std"],
            "n_trajectories": dr_result["n_trajectories"],
            "n_valid": dr_result["n_valid"],
        }

    # ── Save results ──────────────────────────────────────────────────
    results_path = f"{args.report_dir}/evaluation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logging.info("Evaluation results saved: %s", results_path)

    # Print summary
    logging.info("=== EVALUATION SUMMARY ===")
    for policy_name, r in results.items():
        logging.info("  %s: DR value = %.4f +/- %.4f (%d/%d trajectories)",
                     policy_name, r["dr_mean"], r["dr_std"],
                     r["n_valid"], r["n_trajectories"])

    dt = time.time() - t0
    logging.info("Step 12 complete. %.1f sec", dt)


if __name__ == "__main__":
    main()
