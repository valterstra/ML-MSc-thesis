"""
Step 11b -- Dueling DDQN + SARSA physician baseline (Tier 2 FCI-guided state).

Chains from: step_10c output (rl_dataset_tier2.parquet)

State:   5 features (Hb, BUN, Creatinine, HR, Shock_Index) -- FCI-selected
Actions: 16 combinations (2^4: vasopressor, ivfluid, antibiotic, diuretic)
Reward:  SOFA delta (dense) + +-15 terminal (readmit_30d)

Key differences from step_11 (broad baseline):
  - Input: rl_dataset_tier2.parquet
  - n_actions=16 (2^4, vs 32 for broad)
  - n_state=5 (vs 51 for broad)
  - Output: models/icu_readmit/tier2/
  - state_cols still extracted dynamically from parquet -- no hardcoding

Everything else (network architecture, PER, SARSA, hyperparameters) is identical
to step_11 to keep the comparison clean.

Produces:
  models/icu_readmit/tier2/
    ddqn/       dqn_model.pt, dqn_actions.pkl, dqn_q_values.pkl
                dqn_actions_{val,test}.pkl, dqn_q_{val,test}.pkl
    sarsa_phys/ sarsa_phys_model.pt, phys_actions.pkl, phys_q_values.pkl
                phys_actions_{val,test}.pkl, phys_q_{val,test}.pkl

Usage:
    source ../.venv/Scripts/activate
    python scripts/icu_readmit/step_11b_ddqn_tier2.py --smoke
    python scripts/icu_readmit/step_11b_ddqn_tier2.py
    python scripts/icu_readmit/step_11b_ddqn_tier2.py --dqn-steps 100000 --sarsa-steps 80000
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
from src.careai.icu_readmit.rl.continuous import (
    compute_q_values,
    prepare_rl_data,
    train_dqn,
    train_sarsa_physician,
)

N_ACTIONS = 16   # 2^4 Tier 2 action space


def _save_split_outputs(model, data, split_name, out_dir, prefix, device):
    """Compute and save Q-values + greedy actions for a val/test split."""
    q_vals, actions = compute_q_values(model, data["states"], device=device)
    with open(os.path.join(out_dir, f"{prefix}_actions_{split_name}.pkl"), "wb") as f:
        pickle.dump(actions, f)
    with open(os.path.join(out_dir, f"{prefix}_q_{split_name}.pkl"), "wb") as f:
        pickle.dump(q_vals, f)
    logging.info("  %s %s: mean Q=%.4f, actions %d unique",
                 prefix, split_name, q_vals.mean(), len(np.unique(actions)))


def main():
    parser = argparse.ArgumentParser(description="Step 11b: DDQN + SARSA, Tier 2 FCI state")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "processed" / "icu_readmit" / "legacy" / "step_20_24"),
                        help="Directory containing rl_dataset_tier2.parquet")
    parser.add_argument("--model-dir", default=str(PROJECT_ROOT / "models" / "icu_readmit" / "legacy" / "step_25_33" / "tier2"),
                        help="Output directory for models")
    parser.add_argument("--dqn-steps", type=int, default=100000,
                        help="DDQN training steps (default 100000)")
    parser.add_argument("--sarsa-steps", type=int, default=80000,
                        help="SARSA physician training steps (default 80000)")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch device (cpu or cuda)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: 500 steps each")
    parser.add_argument("--log", default=str(PROJECT_ROOT / "logs" / "legacy" / "icu_readmit" / "step_26_ddqn_tier2_legacy.log"),
                        help="Log file path")
    args = parser.parse_args()

    if args.smoke:
        args.dqn_steps   = 500
        args.sarsa_steps = 500

    os.makedirs(args.model_dir, exist_ok=True)
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
    logging.info("Step 11b (Tier 2) started. mode=%s, device=%s", mode_str, args.device)
    logging.info("n_actions=%d, model_dir=%s", N_ACTIONS, args.model_dir)
    t0 = time.time()

    # ── Load Tier 2 parquet ───────────────────────────────────────────
    suffix = "_tier2_smoke" if args.smoke else "_tier2"
    parquet_path = os.path.join(args.data_dir, f"rl_dataset{suffix}.parquet")
    logging.info("Loading %s", parquet_path)
    df = pd.read_parquet(parquet_path)
    logging.info("  %d rows, %d cols", len(df), len(df.columns))

    # Dynamically extract state column names from parquet
    # s_* but NOT s_next_* (those are next-state columns)
    state_cols      = [c for c in df.columns if c.startswith("s_") and not c.startswith("s_next_")]
    next_state_cols = [c.replace("s_", "s_next_", 1) for c in state_cols]

    missing = [c for c in next_state_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing next-state columns: {missing}")

    n_state = len(state_cols)
    logging.info("State features (%d): %s", n_state, state_cols)
    logging.info("Actions: %d unique in dataset (expect %d)", df["a"].nunique(), N_ACTIONS)
    logging.info("Reward: mean=%.3f, std=%.3f, min=%.1f, max=%.1f",
                 df["r"].mean(), df["r"].std(), df["r"].min(), df["r"].max())
    logging.info("Done: %.1f%% terminal transitions", 100 * df["done"].mean())

    # ── Split ─────────────────────────────────────────────────────────
    df_train = df[df["split"] == "train"].reset_index(drop=True)
    df_val   = df[df["split"] == "val"].reset_index(drop=True)
    df_test  = df[df["split"] == "test"].reset_index(drop=True)
    logging.info("Splits -- train=%d, val=%d, test=%d rows",
                 len(df_train), len(df_val), len(df_test))
    del df

    # ── Prepare data arrays ───────────────────────────────────────────
    logging.info("Preparing RL data arrays...")
    train_data = prepare_rl_data(df_train, state_cols, next_state_cols)
    val_data   = prepare_rl_data(df_val,   state_cols, next_state_cols)
    test_data  = prepare_rl_data(df_test,  state_cols, next_state_cols)

    log_every = 100 if args.smoke else 5000

    # ── DDQN ──────────────────────────────────────────────────────────
    logging.info("--- DDQN training ---")
    ddqn_dir = os.path.join(args.model_dir, "ddqn")

    dqn_model, _, _ = train_dqn(
        train_data,
        n_state=n_state,
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
        save_dir=ddqn_dir,
        checkpoint_every=20000,
        device=args.device,
        log_every=log_every,
    )

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        _save_split_outputs(dqn_model, split_data, split_name, ddqn_dir, "dqn", args.device)

    # ── SARSA physician baseline ───────────────────────────────────────
    logging.info("--- SARSA physician training ---")
    sarsa_dir = os.path.join(args.model_dir, "sarsa_phys")

    sarsa_model, _, _ = train_sarsa_physician(
        train_data,
        n_state=n_state,
        n_actions=N_ACTIONS,
        hidden=128,
        lr=1e-4,
        gamma=0.99,
        tau=0.001,
        batch_size=32,
        num_steps=args.sarsa_steps,
        reward_threshold=20,
        reg_lambda=5.0,
        per_alpha=0.6,
        per_epsilon=0.01,
        beta_start=0.9,
        save_dir=sarsa_dir,
        checkpoint_every=20000,
        device=args.device,
        log_every=log_every,
    )

    for split_name, split_data in [("val", val_data), ("test", test_data)]:
        _save_split_outputs(sarsa_model, split_data, split_name, sarsa_dir, "phys", args.device)

    # ── Summary ───────────────────────────────────────────────────────
    dt = time.time() - t0
    logging.info("Step 11b complete. Total time: %.1f sec (%.1f min)", dt, dt / 60)
    logging.info("Models saved in %s/", args.model_dir)
    logging.info("  DDQN:  %s/", ddqn_dir)
    logging.info("  SARSA: %s/", sarsa_dir)


if __name__ == "__main__":
    main()
