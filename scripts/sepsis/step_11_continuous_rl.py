"""
Step 11 - Continuous RL branch: reward shaping + DQN + autoencoder + SARSA.

Chains from: step 09 outputs (rl_*_set_scaled.csv, rl_*_set_original.csv)
Produces:
  models/sepsis_rl/continuous/
    dqn/              (DQN model, actions, Q-values)
    dqn_auto/         (DQN on autoencoded states)
    sarsa_phys/       (SARSA physician baseline)
    autoencoder/      (autoencoder model + encoded states)
  data/processed/sepsis/
    rl_train_data_final_cont.csv          (shaped reward, with terminal)
    rl_val_data_final_cont.csv
    rl_test_data_final_cont.csv
    rl_train_data_final_cont_noterm.csv   (shaped reward, no terminal)
    rl_val_data_final_cont_noterm.csv
    rl_test_data_final_cont_noterm.csv

Ported from: sepsisrl/continuous/q_network.ipynb
             sepsisrl/continuous/autoencoder.ipynb
             sepsisrl/continuous/q_network_autoencoder.ipynb
             sepsisrl/continuous/sarsa_physician.ipynb
             sepsisrl/preprocessing/new_rewards.ipynb
             sepsisrl/preprocessing/new_rewards_no_terminal.ipynb
"""
import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, ".")
from src.careai.sepsis.rl.preprocessing import (
    STATE_FEATURES,
    add_shaped_reward,
)
from src.careai.sepsis.rl.continuous import (
    prepare_rl_data,
    train_autoencoder,
    train_dqn,
    train_sarsa_physician,
)

# 47 state features used for RL (bloc excluded from state representation)
RL_STATE_FEATURES = [f for f in STATE_FEATURES if f != "bloc"]


def main():
    parser = argparse.ArgumentParser(description="Step 11: Continuous RL")
    parser.add_argument(
        "--data-dir", default="data/processed/sepsis",
        help="Directory with preprocessed data",
    )
    parser.add_argument(
        "--model-dir", default="models/sepsis_rl/continuous",
        help="Output directory for models",
    )
    parser.add_argument(
        "--dqn-steps", type=int, default=60000,
        help="DQN training steps",
    )
    parser.add_argument(
        "--sarsa-steps", type=int, default=70000,
        help="SARSA training steps",
    )
    parser.add_argument(
        "--ae-steps", type=int, default=100000,
        help="Autoencoder training steps",
    )
    parser.add_argument(
        "--dqn-auto-steps", type=int, default=200000,
        help="DQN on autoencoded state training steps",
    )
    parser.add_argument(
        "--skip-autoencoder", action="store_true",
        help="Skip autoencoder + DQN-auto training",
    )
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device (cpu or cuda)",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke test mode: tiny step counts to verify pipeline works",
    )
    parser.add_argument(
        "--log", default="logs/step_11_continuous_rl.log",
        help="Log file path",
    )
    args = parser.parse_args()

    # Smoke test: override all step counts to tiny values
    if args.smoke:
        args.dqn_steps = 200
        args.sarsa_steps = 200
        args.ae_steps = 200
        args.dqn_auto_steps = 200

    os.makedirs(args.model_dir, exist_ok=True)

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
    logging.info("Step 11 started. mode=%s, device=%s", mode_str, args.device)
    t0 = time.time()

    # ── Load scaled data ──────────────────────────────────────────────
    logging.info("Loading scaled datasets")
    train_scaled = pd.read_csv(f"{args.data_dir}/rl_train_set_scaled.csv")
    val_scaled = pd.read_csv(f"{args.data_dir}/rl_val_set_scaled.csv")
    test_scaled = pd.read_csv(f"{args.data_dir}/rl_test_set_scaled.csv")

    # Load original (unscaled) for reward shaping
    train_orig = pd.read_csv(f"{args.data_dir}/rl_train_set_original.csv")
    val_orig = pd.read_csv(f"{args.data_dir}/rl_val_set_original.csv")
    test_orig = pd.read_csv(f"{args.data_dir}/rl_test_set_original.csv")

    logging.info("  train=%d, val=%d, test=%d rows",
                 len(train_scaled), len(val_scaled), len(test_scaled))

    # ── Step 1: Reward shaping (with terminal) ────────────────────────
    logging.info("--- Shaping rewards (with terminal) ---")
    train_cont = add_shaped_reward(train_scaled, train_orig, include_terminal=True)
    val_cont = add_shaped_reward(val_scaled, val_orig, include_terminal=True)
    test_cont = add_shaped_reward(test_scaled, test_orig, include_terminal=True)

    train_cont.to_csv(f"{args.data_dir}/rl_train_data_final_cont.csv", index=False)
    val_cont.to_csv(f"{args.data_dir}/rl_val_data_final_cont.csv", index=False)
    test_cont.to_csv(f"{args.data_dir}/rl_test_data_final_cont.csv", index=False)
    logging.info("  Saved rl_*_data_final_cont.csv")

    # ── Step 2: Reward shaping (no terminal) ──────────────────────────
    logging.info("--- Shaping rewards (no terminal) ---")
    train_noterm = add_shaped_reward(train_scaled, train_orig, include_terminal=False)
    val_noterm = add_shaped_reward(val_scaled, val_orig, include_terminal=False)
    test_noterm = add_shaped_reward(test_scaled, test_orig, include_terminal=False)

    train_noterm.to_csv(f"{args.data_dir}/rl_train_data_final_cont_noterm.csv", index=False)
    val_noterm.to_csv(f"{args.data_dir}/rl_val_data_final_cont_noterm.csv", index=False)
    test_noterm.to_csv(f"{args.data_dir}/rl_test_data_final_cont_noterm.csv", index=False)
    logging.info("  Saved rl_*_data_final_cont_noterm.csv")

    # Determine state features available in the data
    state_cols = [c for c in RL_STATE_FEATURES if c in train_noterm.columns]
    n_state = len(state_cols)
    logging.info("Using %d state features for continuous RL", n_state)

    # Log frequency: every 100 steps in smoke, every 5000 in full
    log_every = 100 if args.smoke else 5000

    # ── Step 3: DQN (no terminal rewards for stability) ───────────────
    logging.info("--- DQN training (no terminal) ---")
    train_data = prepare_rl_data(train_noterm, state_cols, device=args.device)

    dqn_dir = f"{args.model_dir}/dqn"
    dqn_model, dqn_actions, dqn_q = train_dqn(
        train_data,
        n_state=n_state,
        num_steps=args.dqn_steps,
        save_dir=dqn_dir,
        device=args.device,
        log_every=log_every,
    )

    # Also compute actions/Q for val and test sets
    for split_name, split_df in [("val", val_noterm), ("test", test_noterm)]:
        split_data = prepare_rl_data(split_df, state_cols, device=args.device)
        dqn_model.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(split_data["states"]).to(args.device)
            q_vals = []
            bs = 4096
            for i in range(0, len(states_t), bs):
                q_vals.append(dqn_model(states_t[i:i + bs]).cpu().numpy())
            q_vals = np.concatenate(q_vals, axis=0)
            actions = q_vals.argmax(axis=1)

        with open(f"{dqn_dir}/dqn_actions_{split_name}.pkl", "wb") as f:
            pickle.dump(actions, f)
        with open(f"{dqn_dir}/dqn_q_{split_name}.pkl", "wb") as f:
            pickle.dump(q_vals, f)
        logging.info("  DQN %s: mean Q=%.4f", split_name, q_vals.mean())

    # ── Step 4: SARSA physician baseline ──────────────────────────────
    logging.info("--- SARSA physician training ---")
    sarsa_dir = f"{args.model_dir}/sarsa_phys"
    sarsa_model, sarsa_actions, sarsa_q = train_sarsa_physician(
        train_data,
        n_state=n_state,
        num_steps=args.sarsa_steps,
        save_dir=sarsa_dir,
        device=args.device,
        log_every=log_every,
    )

    # Val/test Q-values for physician
    for split_name, split_df in [("val", val_noterm), ("test", test_noterm)]:
        split_data = prepare_rl_data(split_df, state_cols, device=args.device)
        sarsa_model.eval()
        with torch.no_grad():
            states_t = torch.FloatTensor(split_data["states"]).to(args.device)
            q_vals = []
            for i in range(0, len(states_t), 4096):
                q_vals.append(sarsa_model(states_t[i:i + 4096]).cpu().numpy())
            q_vals = np.concatenate(q_vals, axis=0)
            actions = q_vals.argmax(axis=1)

        with open(f"{sarsa_dir}/phys_actions_{split_name}.pkl", "wb") as f:
            pickle.dump(actions, f)
        with open(f"{sarsa_dir}/phys_q_{split_name}.pkl", "wb") as f:
            pickle.dump(q_vals, f)

    # ── Step 5: Autoencoder + DQN on encoded state ────────────────────
    if not args.skip_autoencoder:
        logging.info("--- Autoencoder training ---")
        ae_dir = f"{args.model_dir}/autoencoder"
        ae_model, encoded_train = train_autoencoder(
            train_data,
            n_input=n_state,
            n_hidden=200,
            num_steps=args.ae_steps,
            save_dir=ae_dir,
            device=args.device,
            log_every=log_every,
        )

        # Encode val/test
        ae_model.eval()
        for split_name, split_df in [("val", val_noterm), ("test", test_noterm)]:
            split_data = prepare_rl_data(split_df, state_cols, device=args.device)
            with torch.no_grad():
                states_t = torch.FloatTensor(split_data["states"]).to(args.device)
                encoded = []
                for i in range(0, len(states_t), 4096):
                    _, z = ae_model(states_t[i:i + 4096])
                    encoded.append(z.cpu().numpy())
                encoded = np.concatenate(encoded, axis=0)
            with open(f"{ae_dir}/encoded_{split_name}.pkl", "wb") as f:
                pickle.dump(encoded, f)

        # DQN on autoencoded state
        logging.info("--- DQN on autoencoded state ---")
        ae_data = {
            "states": encoded_train,
            "actions": train_data["actions"],
            "rewards": train_data["rewards"],
            "next_states": np.zeros_like(encoded_train),  # will be filled
            "done": train_data["done"],
        }
        # Build next encoded states
        for i in range(len(encoded_train) - 1):
            if ae_data["done"][i] == 0:
                ae_data["next_states"][i] = encoded_train[i + 1]

        dqn_auto_dir = f"{args.model_dir}/dqn_auto"
        train_dqn(
            ae_data,
            n_state=200,  # autoencoded dimension
            hidden=400,  # larger network for larger input
            leaky_slope=0.5,
            batch_size=30,
            num_steps=args.dqn_auto_steps,
            save_dir=dqn_auto_dir,
            device=args.device,
            log_every=log_every,
        )

    # ── Summary ───────────────────────────────────────────────────────
    dt = time.time() - t0
    logging.info("Step 11 complete. %.1f sec", dt)
    logging.info("Models saved in %s/", args.model_dir)


if __name__ == "__main__":
    main()
