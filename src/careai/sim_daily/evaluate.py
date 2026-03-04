"""Evaluation utilities: single-step metrics + multi-step rollout comparison."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, r2_score, roc_auc_score

from .env import DailySimEnv
from .features import (
    ACTION_COLS,
    INPUT_COLS,
    OUTPUT_BINARY,
    OUTPUT_CONTINUOUS,
    STATE_BINARY,
    STATE_CONTINUOUS,
)
from .transition import TransitionModel


# ---------------------------------------------------------------------------
# Single-step metrics
# ---------------------------------------------------------------------------

def single_step_metrics(model: TransitionModel, test_df: pd.DataFrame) -> dict[str, Any]:
    """Compute per-output R²/MAE (continuous) and AUC (binary) on test set."""
    X = test_df[model.input_cols]
    results: dict[str, Any] = {"continuous": {}, "binary": {}, "done": {}}

    # Continuous outputs
    for col in model.output_continuous:
        target = f"next_{col}"
        mask = test_df[target].notna()
        if mask.sum() < 10:
            results["continuous"][col] = {"r2": None, "mae": None, "n": int(mask.sum())}
            continue
        y_true = test_df.loc[mask, target].values
        y_pred = model.continuous_models[col].predict(X.loc[mask])
        results["continuous"][col] = {
            "r2": round(float(r2_score(y_true, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_true, y_pred)), 4),
            "n": int(mask.sum()),
        }

    # Binary outputs
    for col in model.output_binary:
        target = f"next_{col}"
        mask = test_df[target].notna()
        y_true = test_df.loc[mask, target].astype(int).values
        if mask.sum() < 10 or len(np.unique(y_true)) < 2:
            results["binary"][col] = {"auc": None, "n": int(mask.sum())}
            continue
        y_prob = model.binary_models[col].predict_proba(X.loc[mask])[:, 1]
        results["binary"][col] = {
            "auc": round(float(roc_auc_score(y_true, y_prob)), 4),
            "n": int(mask.sum()),
        }

    # Done model
    y_done = test_df["done_next"].astype(int).values
    if len(np.unique(y_done)) >= 2:
        y_done_prob = model.done_model.predict_proba(X)[:, 1]
        results["done"] = {
            "auc": round(float(roc_auc_score(y_done, y_done_prob)), 4),
            "n": len(y_done),
        }
    else:
        results["done"] = {"auc": None, "n": len(y_done)}

    return results


# ---------------------------------------------------------------------------
# Multi-step rollouts
# ---------------------------------------------------------------------------

def run_rollouts(
    env: DailySimEnv,
    n_rollouts: int = 500,
    max_days: int = 60,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate *n_rollouts* simulated trajectories with zero actions."""
    rng = np.random.default_rng(seed)
    zero_action = {c: 0.0 for c in ACTION_COLS}

    records: list[dict[str, Any]] = []
    for rid in range(n_rollouts):
        state = env.reset(rng)
        records.append({"rollout_id": rid, "day": 0, **state})
        for day in range(1, max_days + 1):
            state, _reward, done, info = env.step(zero_action)
            records.append({"rollout_id": rid, "day": day, "done_prob": info["done_prob"], **state})
            if done:
                break

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Rollout vs real distribution comparison
# ---------------------------------------------------------------------------

def rollout_comparison(
    sim_traj: pd.DataFrame,
    real_data: pd.DataFrame,
    cols: list[str] | None = None,
) -> dict[str, dict[str, float]]:
    """Per-column KS test + mean/std comparison (simulated vs real)."""
    if cols is None:
        cols = STATE_CONTINUOUS + STATE_BINARY
    results: dict[str, dict[str, float]] = {}

    for col in cols:
        sim_vals = sim_traj[col].dropna().values
        real_vals = real_data[col].dropna().values
        if len(sim_vals) < 5 or len(real_vals) < 5:
            results[col] = {"ks_stat": None, "ks_pval": None}
            continue
        ks_stat, ks_pval = stats.ks_2samp(sim_vals, real_vals)
        results[col] = {
            "ks_stat": round(float(ks_stat), 4),
            "ks_pval": round(float(ks_pval), 6),
            "sim_mean": round(float(np.nanmean(sim_vals)), 4),
            "real_mean": round(float(np.nanmean(real_vals)), 4),
            "sim_std": round(float(np.nanstd(sim_vals)), 4),
            "real_std": round(float(np.nanstd(real_vals)), 4),
        }

    return results
