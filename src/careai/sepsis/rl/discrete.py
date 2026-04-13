"""
Discrete MDP-based RL for sepsis treatment.
Ported from sepsisrl/discrete/ notebooks.

Handles: K-means state clustering, transition matrix construction,
SARSA (physician baseline), Value Iteration (optimal policy).
"""
import logging
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.cluster import MiniBatchKMeans

# Sentinel states for episode boundaries
END_REWARD = "END_REWARD"    # patient survived
END_PENALTY = "END_PENALTY"  # patient died

N_CLUSTERS = 1250
N_ACTIONS = 25  # 5 IV x 5 vasopressor


def cluster_states(train_df, val_df, test_df, feature_cols, n_clusters=N_CLUSTERS,
                   n_init=32, random_state=42):
    """K-means clustering on state features.

    Fits on train, predicts for all splits.
    Returns (train_disc, val_disc, test_disc, kmeans_model).
    """
    logging.info("Fitting KMeans with k=%d on %d features", n_clusters, len(feature_cols))

    # Extract features
    X_train = train_df[feature_cols].values.astype(np.float32)
    X_val = val_df[feature_cols].values.astype(np.float32)
    X_test = test_df[feature_cols].values.astype(np.float32)

    # Replace NaN with column mean (from train)
    col_means = np.nanmean(X_train, axis=0)
    for i in range(X_train.shape[1]):
        X_train[np.isnan(X_train[:, i]), i] = col_means[i]
        X_val[np.isnan(X_val[:, i]), i] = col_means[i]
        X_test[np.isnan(X_test[:, i]), i] = col_means[i]

    km = MiniBatchKMeans(
        n_clusters=n_clusters,
        n_init=n_init,
        random_state=random_state,
        batch_size=10000,
    )
    km.fit(X_train)
    logging.info("KMeans converged. Inertia=%.1f", km.inertia_)

    train_states = km.predict(X_train)
    val_states = km.predict(X_val)
    test_states = km.predict(X_test)

    def build_discrete_df(df, states):
        return pd.DataFrame({
            "bloc": df["bloc"].values,
            "icustayid": df["icustayid"].values,
            "state": states,
            "action_id": df["action_id"].values if "action_id" in df.columns
                         else 5 * df["iv_input"].values + df["vaso_input"].values,
            "vaso_input": df["vaso_input"].values,
            "iv_input": df["iv_input"].values,
            "reward": df["reward"].values,
            "died_in_hosp": df["died_in_hosp"].values,
        })

    train_disc = build_discrete_df(train_df, train_states)
    val_disc = build_discrete_df(val_df, val_states)
    test_disc = build_discrete_df(test_df, test_states)

    # Cluster distribution
    unique, counts = np.unique(train_states, return_counts=True)
    logging.info("Cluster sizes: min=%d, max=%d, median=%d, empty=%d",
                 counts.min(), counts.max(), int(np.median(counts)),
                 n_clusters - len(unique))

    return train_disc, val_disc, test_disc, km


def build_transition_matrix(disc_df, n_states=N_CLUSTERS, n_actions=N_ACTIONS):
    """Build empirical transition probabilities P(s'|s,a).

    Returns dict: {(state, action): {next_state: probability}}.
    Episode boundaries map to END_REWARD or END_PENALTY.
    """
    logging.info("Building transition matrix from %d rows", len(disc_df))
    trans_counts = defaultdict(lambda: defaultdict(int))

    states = disc_df["state"].values
    actions = disc_df["action_id"].values
    icuids = disc_df["icustayid"].values
    died = disc_df["died_in_hosp"].values

    for i in range(len(disc_df) - 1):
        s = int(states[i])
        a = int(actions[i])

        if icuids[i + 1] != icuids[i]:
            # Episode boundary - terminal transition
            terminal = END_PENALTY if died[i] == 1 else END_REWARD
            trans_counts[(s, a)][terminal] += 1
        else:
            s_next = int(states[i + 1])
            trans_counts[(s, a)][s_next] += 1

    # Last row terminal
    i = len(disc_df) - 1
    s = int(states[i])
    a = int(actions[i])
    terminal = END_PENALTY if died[i] == 1 else END_REWARD
    trans_counts[(s, a)][terminal] += 1

    # Normalize to probabilities
    trans_prob = {}
    for (s, a), next_dict in trans_counts.items():
        total = sum(next_dict.values())
        trans_prob[(s, a)] = {ns: c / total for ns, c in next_dict.items()}

    n_sa = len(trans_prob)
    logging.info("Transition matrix: %d (s,a) pairs observed out of %d possible",
                 n_sa, n_states * n_actions)

    return trans_prob


def sarsa_episodic(disc_df, alpha=0.1, gamma=1.0, num_episodes=250000,
                   n_states=N_CLUSTERS, n_actions=N_ACTIONS,
                   reward_threshold=15, seed=42):
    """Learn physician Q-function via episodic SARSA.

    On-policy learning from observed trajectories.
    Returns Q-table (n_states x n_actions).
    """
    logging.info("SARSA: alpha=%.2f, gamma=%.2f, %d episodes", alpha, gamma, num_episodes)
    rng = np.random.RandomState(seed)

    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    # Build episode index: list of (start_idx, end_idx) per ICU stay
    icuids = disc_df["icustayid"].values
    episodes = []
    start = 0
    for i in range(1, len(disc_df)):
        if icuids[i] != icuids[i - 1]:
            episodes.append((start, i - 1))
            start = i
    episodes.append((start, len(disc_df) - 1))
    logging.info("  %d episodes available", len(episodes))

    states = disc_df["state"].values.astype(int)
    actions = disc_df["action_id"].values.astype(int)
    rewards = disc_df["reward"].values.astype(float)

    for step in range(num_episodes):
        # Sample random episode
        ep_idx = rng.randint(len(episodes))
        start, end = episodes[ep_idx]

        # Process backward through trajectory
        for t in range(end, start - 1, -1):
            s = states[t]
            a = actions[t]
            r = np.clip(rewards[t], -reward_threshold, reward_threshold)

            if t == end:
                # Terminal step
                Q[s, a] += alpha * (r - Q[s, a])
            else:
                s_next = states[t + 1]
                a_next = actions[t + 1]
                Q[s, a] += alpha * (r + gamma * Q[s_next, a_next] - Q[s, a])

        if (step + 1) % 50000 == 0:
            mean_q = Q[Q != 0].mean() if (Q != 0).any() else 0
            logging.info("  SARSA step %d/%d, mean nonzero Q=%.4f",
                         step + 1, num_episodes, mean_q)

    logging.info("SARSA complete. Q range: [%.2f, %.2f]", Q.min(), Q.max())
    return Q


def value_iteration(trans_prob, gamma=0.9, n_states=N_CLUSTERS, n_actions=N_ACTIONS,
                     reward_end=100.0, penalty_end=-100.0, tol=1e-5, max_iter=1000):
    """Solve optimal policy via tabular Value Iteration.

    Uses empirical transition matrix.
    Returns (V, policy) where V[s] is the value and policy[s] is the best action.
    """
    logging.info("Value Iteration: gamma=%.2f, tol=%.6f", gamma, tol)

    V = np.zeros(n_states, dtype=np.float64)

    for iteration in range(max_iter):
        delta = 0.0
        V_new = np.copy(V)

        for s in range(n_states):
            best_val = -np.inf
            for a in range(n_actions):
                if (s, a) not in trans_prob:
                    continue
                val = 0.0
                for ns, prob in trans_prob[(s, a)].items():
                    if ns == END_REWARD:
                        val += prob * reward_end
                    elif ns == END_PENALTY:
                        val += prob * penalty_end
                    else:
                        val += prob * (0 + gamma * V[ns])  # R=0 for non-terminal
                if val > best_val:
                    best_val = val

            if best_val > -np.inf:
                V_new[s] = best_val
                delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new
        if (iteration + 1) % 10 == 0:
            logging.info("  VI iter %d, delta=%.6f", iteration + 1, delta)
        if delta < tol:
            logging.info("  VI converged at iter %d, delta=%.8f", iteration + 1, delta)
            break

    # Extract policy
    policy = np.zeros(n_states, dtype=int)
    for s in range(n_states):
        best_val = -np.inf
        best_a = 0
        for a in range(n_actions):
            if (s, a) not in trans_prob:
                continue
            val = 0.0
            for ns, prob in trans_prob[(s, a)].items():
                if ns == END_REWARD:
                    val += prob * reward_end
                elif ns == END_PENALTY:
                    val += prob * penalty_end
                else:
                    val += prob * gamma * V[ns]
            if val > best_val:
                best_val = val
                best_a = a
        policy[s] = best_a

    logging.info("Value Iteration complete. V range: [%.2f, %.2f]", V.min(), V.max())

    # Policy action distribution
    unique, counts = np.unique(policy, return_counts=True)
    logging.info("Policy action distribution:")
    for a, c in zip(unique, counts):
        iv_lvl, vaso_lvl = divmod(a, 5)
        logging.info("  action %d (IV=%d, vaso=%d): %d states", a, iv_lvl, vaso_lvl, c)

    return V, policy
