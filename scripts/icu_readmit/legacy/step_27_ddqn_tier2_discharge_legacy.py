"""
Step 11c -- Joint DDQN + SARSA: Tier 2 with discharge terminal action.

Chains from: step_10d output (rl_dataset_tier2_discharge.parquet)

Two-phase MDP:
  Phase 0: state=(Hb,BUN,Creatinine,HR,Shock_Index), action=drug combo 0-15
  Phase 1: state=same, action=discharge destination 0-2

Two separate DuelingDQN networks trained jointly with cross-phase Bellman:
  - Last in-stay bloc bootstraps from Q_discharge (not Q_drug)
  - Drug policy is shaped by downstream discharge value
  - SARSA physician baseline follows observed drug+discharge actions

Outputs:
  models/icu_readmit/tier2_discharge/
    ddqn/
      dqn_drug_model.pt, dqn_discharge_model.pt
      dqn_drug_actions.pkl, dqn_drug_q_values.pkl
      dqn_discharge_actions.pkl, dqn_discharge_q_values.pkl
      dqn_losses.pkl
      dqn_drug_actions_{val,test}.pkl, dqn_drug_q_{val,test}.pkl
      dqn_discharge_actions_{val,test}.pkl, dqn_discharge_q_{val,test}.pkl
    sarsa_phys/ (same pattern, phys_ prefix)

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_11c_ddqn_discharge.py --smoke
    python scripts/icu_readmit/step_11c_ddqn_discharge.py
    python scripts/icu_readmit/step_11c_ddqn_discharge.py --dqn-steps 100000 --sarsa-steps 80000
"""
import argparse
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
from src.careai.icu_readmit.legacy.rl.continuous_discharge import (
    compute_q_values,
    prepare_discharge_data,
    train_joint_dqn,
    train_joint_sarsa_physician,
)

N_DRUG_ACTIONS      = 16   # 2^4
N_DISCHARGE_ACTIONS = 3    # home / home+services / institutional


def _save_split_outputs(drug_model, discharge_model, data, split_name, out_dir,
                        prefix, device):
    """Save Q-values + greedy actions for drug and discharge networks on one split."""
    # All rows: drug Q-values
    dq, da = compute_q_values(drug_model, data["states"], device=device)
    with open(os.path.join(out_dir, f"{prefix}_drug_actions_{split_name}.pkl"), "wb") as f:
        pickle.dump(da, f)
    with open(os.path.join(out_dir, f"{prefix}_drug_q_{split_name}.pkl"), "wb") as f:
        pickle.dump(dq, f)
    logging.info("  %s drug %s: mean Q=%.4f, %d unique actions",
                 prefix, split_name, dq.mean(), len(np.unique(da)))

    # Phase-1 rows only: discharge Q-values
    phase1_mask = data["phase"] == 1
    if phase1_mask.any():
        dischq, discha = compute_q_values(
            discharge_model, data["states"][phase1_mask], device=device)
        with open(os.path.join(out_dir,
                               f"{prefix}_discharge_actions_{split_name}.pkl"), "wb") as f:
            pickle.dump(discha, f)
        with open(os.path.join(out_dir,
                               f"{prefix}_discharge_q_{split_name}.pkl"), "wb") as f:
            pickle.dump(dischq, f)
        logging.info("  %s discharge %s: mean Q=%.4f, action dist=%s",
                     prefix, split_name, dischq.mean(),
                     str(np.bincount(discha, minlength=N_DISCHARGE_ACTIONS).tolist()))


def main():
    parser = argparse.ArgumentParser(description="Step 11c: Joint DDQN + SARSA, Tier 2 + discharge")
    parser.add_argument("--data-dir",   default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24"))
    parser.add_argument("--model-dir",  default=str(PROJECT_ROOT / "models" / "icu_readmit" / "legacy" / "step_25_33" / "tier2_discharge"))
    parser.add_argument("--dqn-steps",  type=int, default=100000)
    parser.add_argument("--sarsa-steps",type=int, default=80000)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--smoke",      action="store_true")
    parser.add_argument("--log",        default=str(PROJECT_ROOT / "logs" / "legacy" / "icu_readmit" / "step_27_ddqn_tier2_discharge_legacy.log"))
    args = parser.parse_args()

    if args.smoke:
        args.dqn_steps   = 500
        args.sarsa_steps = 500

    os.makedirs(args.model_dir,              exist_ok=True)
    os.makedirs(os.path.dirname(args.log),   exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%H:%M:%S",
        handlers=[
            logging.FileHandler(args.log, mode="w", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    mode_str = "SMOKE" if args.smoke else "FULL"
    logging.info("Step 11c (Tier 2 + discharge) started. mode=%s device=%s",
                 mode_str, args.device)
    t0 = time.time()

    # -- Load parquet --------------------------------------------------------
    suffix       = "_tier2_discharge_smoke" if args.smoke else "_tier2_discharge"
    parquet_path = os.path.join(args.data_dir, f"rl_dataset{suffix}.parquet")
    logging.info("Loading %s ...", parquet_path)
    df = pd.read_parquet(parquet_path)
    logging.info("  %d rows, %d cols", len(df), len(df.columns))
    logging.info("  Phase 0: %d rows | Phase 1: %d rows",
                 (df["phase"] == 0).sum(), (df["phase"] == 1).sum())

    state_cols      = [c for c in df.columns
                       if c.startswith("s_") and not c.startswith("s_next_")]
    next_state_cols = [c.replace("s_", "s_next_", 1) for c in state_cols]
    n_state         = len(state_cols)
    logging.info("State features (%d): %s", n_state, state_cols)

    missing = [c for c in next_state_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing next-state columns: {missing}")

    # -- Split ---------------------------------------------------------------
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)
    logging.info("Splits -- train=%d  val=%d  test=%d rows",
                 len(df_train), len(df_val), len(df_test))
    del df

    train_data = prepare_discharge_data(df_train, state_cols, next_state_cols)
    val_data   = prepare_discharge_data(df_val,   state_cols, next_state_cols)
    test_data  = prepare_discharge_data(df_test,  state_cols, next_state_cols)

    log_every = 100 if args.smoke else 5000

    # -- Joint DDQN ----------------------------------------------------------
    logging.info("--- Joint DDQN training ---")
    ddqn_dir = os.path.join(args.model_dir, "ddqn")

    drug_model, discharge_model, _, _, _, _ = train_joint_dqn(
        train_data,
        n_state              = n_state,
        n_drug_actions       = N_DRUG_ACTIONS,
        n_discharge_actions  = N_DISCHARGE_ACTIONS,
        hidden               = 128,
        leaky_slope          = 0.01,
        lr                   = 1e-4,
        gamma                = 0.99,
        tau                  = 0.001,
        batch_size           = 32,
        num_steps            = args.dqn_steps,
        reward_threshold     = 20,
        reg_lambda           = 5.0,
        per_alpha            = 0.6,
        per_epsilon          = 0.01,
        beta_start           = 0.9,
        save_dir             = ddqn_dir,
        checkpoint_every     = 20000 if not args.smoke else 0,
        device               = args.device,
        log_every            = log_every,
    )

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        _save_split_outputs(drug_model, discharge_model, split_data,
                            split_name, ddqn_dir, "dqn", args.device)

    # -- Joint SARSA physician ------------------------------------------------
    logging.info("--- Joint SARSA physician training ---")
    sarsa_dir = os.path.join(args.model_dir, "sarsa_phys")

    sarsa_drug, sarsa_discharge, _, _, _, _ = train_joint_sarsa_physician(
        train_data,
        n_state              = n_state,
        n_drug_actions       = N_DRUG_ACTIONS,
        n_discharge_actions  = N_DISCHARGE_ACTIONS,
        hidden               = 128,
        leaky_slope          = 0.01,
        lr                   = 1e-4,
        gamma                = 0.99,
        tau                  = 0.001,
        batch_size           = 32,
        num_steps            = args.sarsa_steps,
        reward_threshold     = 20,
        reg_lambda           = 5.0,
        per_alpha            = 0.6,
        per_epsilon          = 0.01,
        beta_start           = 0.9,
        save_dir             = sarsa_dir,
        checkpoint_every     = 20000 if not args.smoke else 0,
        device               = args.device,
        log_every            = log_every,
    )

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        _save_split_outputs(sarsa_drug, sarsa_discharge, split_data,
                            split_name, sarsa_dir, "phys", args.device)

    # -- Summary -------------------------------------------------------------
    dt = time.time() - t0
    logging.info("Step 11c complete. Total time: %.1f sec (%.1f min)", dt, dt / 60)
    logging.info("Models saved in %s/", args.model_dir)


if __name__ == "__main__":
    main()
