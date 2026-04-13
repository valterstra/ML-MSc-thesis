"""
Step 10 - Discrete RL branch: K-means clustering + SARSA + Value Iteration.

Chains from: step 09 outputs (rl_*_set_unscaled.csv)
Produces:
  models/sepsis_rl/discrete/
    kmeans_model.pkl         (K-means cluster model)
    trans_prob.pkl           (transition probability dict)
    q_table_physician.pkl    (SARSA physician Q-function)
    policy_value_iter.pkl    (optimal policy from VI)
    V_value_iter.pkl         (state values from VI)
  data/processed/sepsis/
    rl_train_data_discrete.csv
    rl_val_data_discrete.csv
    rl_test_data_discrete.csv

Ported from: sepsisrl/discrete/find_transition_matrix.ipynb
             sepsisrl/discrete/sarsa_episodic.ipynb
             sepsisrl/discrete/value_iteration.ipynb
"""
import argparse
import logging
import os
import pickle
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, ".")
from src.careai.sepsis.rl.preprocessing import STATE_FEATURES
from src.careai.sepsis.rl.discrete import (
    N_ACTIONS,
    N_CLUSTERS,
    build_transition_matrix,
    cluster_states,
    sarsa_episodic,
    value_iteration,
)


def main():
    parser = argparse.ArgumentParser(description="Step 10: Discrete RL")
    parser.add_argument(
        "--data-dir", default="data/processed/sepsis",
        help="Directory with rl_*_set_unscaled.csv files",
    )
    parser.add_argument(
        "--model-dir", default="models/sepsis_rl/discrete",
        help="Output directory for models",
    )
    parser.add_argument(
        "--n-clusters", type=int, default=N_CLUSTERS,
        help="Number of K-means clusters",
    )
    parser.add_argument(
        "--sarsa-episodes", type=int, default=250000,
        help="Number of SARSA training episodes",
    )
    parser.add_argument(
        "--vi-gamma", type=float, default=0.9,
        help="Discount factor for Value Iteration",
    )
    parser.add_argument(
        "--log", default="logs/step_10_discrete_rl.log",
        help="Log file path",
    )
    args = parser.parse_args()

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
    logging.info("Step 10 started. data_dir=%s, model_dir=%s", args.data_dir, args.model_dir)
    t0 = time.time()

    # ── Load unscaled data ────────────────────────────────────────────
    logging.info("Loading unscaled datasets")
    train_df = pd.read_csv(f"{args.data_dir}/rl_train_set_unscaled.csv")
    val_df = pd.read_csv(f"{args.data_dir}/rl_val_set_unscaled.csv")
    test_df = pd.read_csv(f"{args.data_dir}/rl_test_set_unscaled.csv")
    logging.info("  train=%d, val=%d, test=%d rows", len(train_df), len(val_df), len(test_df))

    # ── Step 1: K-means clustering ────────────────────────────────────
    logging.info("--- K-means clustering ---")
    # Use state features that exist in our data
    feature_cols = [c for c in STATE_FEATURES if c in train_df.columns]
    logging.info("Using %d/%d state features for clustering", len(feature_cols), len(STATE_FEATURES))

    train_disc, val_disc, test_disc, km = cluster_states(
        train_df, val_df, test_df, feature_cols,
        n_clusters=args.n_clusters,
    )

    # Save discrete datasets
    train_disc.to_csv(f"{args.data_dir}/rl_train_data_discrete.csv", index=False)
    val_disc.to_csv(f"{args.data_dir}/rl_val_data_discrete.csv", index=False)
    test_disc.to_csv(f"{args.data_dir}/rl_test_data_discrete.csv", index=False)
    logging.info("Discrete datasets saved")

    # Save K-means model
    km_path = f"{args.model_dir}/kmeans_model.pkl"
    with open(km_path, "wb") as f:
        pickle.dump(km, f)
    logging.info("KMeans model saved: %s", km_path)

    # ── Step 2: Transition matrix ─────────────────────────────────────
    logging.info("--- Building transition matrix ---")
    trans_prob = build_transition_matrix(train_disc, n_states=args.n_clusters)

    tp_path = f"{args.model_dir}/trans_prob.pkl"
    with open(tp_path, "wb") as f:
        pickle.dump(trans_prob, f)
    logging.info("Transition matrix saved: %s", tp_path)

    # ── Step 3: SARSA (physician baseline) ────────────────────────────
    logging.info("--- SARSA physician policy ---")
    Q_phys = sarsa_episodic(
        train_disc,
        alpha=0.1,
        gamma=1.0,
        num_episodes=args.sarsa_episodes,
        n_states=args.n_clusters,
    )

    q_path = f"{args.model_dir}/q_table_physician.pkl"
    with open(q_path, "wb") as f:
        pickle.dump(Q_phys, f)
    logging.info("Physician Q-table saved: %s", q_path)

    # Physician policy summary
    phys_policy = Q_phys.argmax(axis=1)
    unique, counts = np.unique(phys_policy, return_counts=True)
    logging.info("Physician policy action distribution (top 5):")
    top5 = np.argsort(-counts)[:5]
    for idx in top5:
        a = unique[idx]
        iv_lvl, vaso_lvl = divmod(a, 5)
        logging.info("  action %d (IV=%d, vaso=%d): %d states (%.1f%%)",
                     a, iv_lvl, vaso_lvl, counts[idx], 100 * counts[idx] / args.n_clusters)

    # ── Step 4: Value Iteration (optimal policy) ──────────────────────
    logging.info("--- Value Iteration ---")
    V, policy = value_iteration(
        trans_prob,
        gamma=args.vi_gamma,
        n_states=args.n_clusters,
    )

    v_path = f"{args.model_dir}/V_value_iter.pkl"
    with open(v_path, "wb") as f:
        pickle.dump(V, f)

    p_path = f"{args.model_dir}/policy_value_iter.pkl"
    with open(p_path, "wb") as f:
        pickle.dump(policy, f)
    logging.info("Value function and policy saved")

    # ── Summary ───────────────────────────────────────────────────────
    dt = time.time() - t0
    logging.info("Step 10 complete. %.1f sec", dt)
    logging.info("Outputs:")
    logging.info("  Discrete data: %s/rl_{train,val,test}_data_discrete.csv", args.data_dir)
    logging.info("  Models: %s/", args.model_dir)


if __name__ == "__main__":
    main()
