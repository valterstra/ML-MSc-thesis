from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

from .model import (
    ACTION_COLS,
    ACTION_FEATURE_OFFSET,
    DYNAMIC_STATE_IDX,
    MarkovSimConfig,
    MarkovSimEnsemble,
    NEXT_STATE_COLS,
    SELECTED_CAUSAL_ACTION_MASK,
    STATE_COLS,
)


def fit_markovsim_from_dataframe(
    df: pd.DataFrame,
    split: str | None = "train",
    config: MarkovSimConfig | None = None,
) -> tuple[MarkovSimEnsemble, dict]:
    config = config or MarkovSimConfig()

    if split is not None and "split" in df.columns:
        df = df[df["split"] == split].copy()
    if df.empty:
        raise ValueError("No rows available to fit the Markov simulator")

    X_raw = df[STATE_COLS + ACTION_COLS].to_numpy(dtype=np.float32, copy=True)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw)

    y_next = df[NEXT_STATE_COLS].to_numpy(dtype=np.float32, copy=True)[:, DYNAMIC_STATE_IDX]
    y_done = df["done"].to_numpy(dtype=np.int64, copy=True)

    transition_models: list[Ridge] = []
    next_state_std: list[float] = []
    action_mask_matrix = SELECTED_CAUSAL_ACTION_MASK.copy()

    for dyn_pos, state_idx in enumerate(DYNAMIC_STATE_IDX):
        X_target = X_scaled.copy()
        disallowed = np.where(action_mask_matrix[state_idx] < 0.5)[0]
        if len(disallowed):
            X_target[:, ACTION_FEATURE_OFFSET + disallowed] = 0.0

        model = Ridge(alpha=config.ridge_alpha)
        model.fit(X_target, y_next[:, dyn_pos])
        pred = model.predict(X_target)
        resid = y_next[:, dyn_pos] - pred
        transition_models.append(model)
        next_state_std.append(float(np.std(resid)))

    terminal_model = LogisticRegression(
        C=config.terminal_c,
        max_iter=config.max_iter,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
    )
    terminal_model.fit(X_scaled, y_done)

    ensemble = MarkovSimEnsemble(
        transition_models=transition_models,
        terminal_model=terminal_model,
        feature_scaler=scaler,
        next_state_std=np.asarray(next_state_std, dtype=np.float32),
        action_mask_matrix=action_mask_matrix,
        config=config,
    )

    dyn_pred = np.zeros_like(y_next, dtype=np.float32)
    for dyn_pos, state_idx in enumerate(DYNAMIC_STATE_IDX):
        X_target = X_scaled.copy()
        disallowed = np.where(action_mask_matrix[state_idx] < 0.5)[0]
        if len(disallowed):
            X_target[:, ACTION_FEATURE_OFFSET + disallowed] = 0.0
        dyn_pred[:, dyn_pos] = transition_models[dyn_pos].predict(X_target).astype(np.float32)

    metrics = {
        "n_rows": int(len(df)),
        "feature_dim": int(X_scaled.shape[1]),
        "transition_train_mse": float(np.mean((y_next - dyn_pred) ** 2)),
        "terminal_train_accuracy": float(terminal_model.score(X_scaled, y_done)),
        "config": asdict(config),
    }
    return ensemble, metrics
