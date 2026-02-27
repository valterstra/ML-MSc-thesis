"""Weight computation utilities for causal-weighted sim v1 training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class SimWeightResult:
    sample_weight: np.ndarray
    propensity: np.ndarray
    diagnostics: dict[str, Any]


def _encode_covariates(df: pd.DataFrame, num_cols: list[str], cat_cols: list[str]) -> pd.DataFrame:
    x_num = df[num_cols].copy()
    x_cat = pd.get_dummies(df[cat_cols], prefix=cat_cols, drop_first=False, dtype=float)
    return pd.concat([x_num, x_cat], axis=1)


def build_sim_training_weights(train_df: pd.DataFrame, cfg: dict[str, Any]) -> SimWeightResult:
    model_cfg = cfg.get("model", {})
    num_cols = list(model_cfg.get("features_numeric", []))
    cat_cols = list(model_cfg.get("features_categorical", []))
    include_actions = set(model_cfg.get("weight_include_actions", ["A_LOW_SUPPORT", "A_HIGH_SUPPORT"]))
    clip_percentiles = list(model_cfg.get("weight_clip_percentiles", [0.01, 0.99]))
    stabilized = bool(model_cfg.get("weight_stabilized", True))
    high_action_label = str(model_cfg.get("weight_treatment_high_label", "A_HIGH_SUPPORT"))

    required = ["a_t"] + num_cols + cat_cols
    missing = [c for c in required if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing required columns for sim weights: {missing}")

    d = train_df.reset_index(drop=True).copy()
    use_mask = d["a_t"].isin(include_actions)
    if int(use_mask.sum()) == 0:
        raise ValueError("No training rows available for weighting after action filtering.")

    use = d.loc[use_mask].copy()
    t = (use["a_t"] == high_action_label).astype(int)
    if int(t.nunique()) < 2:
        raise ValueError("Need both treatment arms in training data for weighting.")

    x = _encode_covariates(use, num_cols, cat_cols)
    model = LogisticRegression(
        max_iter=int(model_cfg.get("max_iter", 2000)),
        C=float(model_cfg.get("C", 1.0)),
        solver="lbfgs",
    )
    model.fit(x, t)
    ps = model.predict_proba(x)[:, 1].astype(float)
    eps = 1e-6
    ps = np.clip(ps, eps, 1.0 - eps)

    t_arr = t.to_numpy(dtype=float)
    p_treat = float(t_arr.mean())
    if stabilized:
        sw_raw = (t_arr * p_treat / ps) + ((1.0 - t_arr) * (1.0 - p_treat) / (1.0 - ps))
    else:
        sw_raw = (t_arr / ps) + ((1.0 - t_arr) / (1.0 - ps))

    lo_q, hi_q = float(clip_percentiles[0]), float(clip_percentiles[1])
    lo = float(np.quantile(sw_raw, lo_q))
    hi = float(np.quantile(sw_raw, hi_q))
    sw = np.clip(sw_raw, lo, hi)

    full_w = np.ones((len(d),), dtype=float)
    full_ps = np.full((len(d),), np.nan, dtype=float)
    use_idx = use.index.to_numpy(dtype=int)
    full_w[use_idx] = sw
    full_ps[use_idx] = ps

    ess = float((np.sum(sw) ** 2) / np.sum(sw**2))
    diagnostics = {
        "mode": "ipw_stabilized" if stabilized else "ipw_unstabilized",
        "n_rows_total": int(len(d)),
        "n_rows_weighted": int(len(use)),
        "include_actions": sorted(include_actions),
        "high_action_label": high_action_label,
        "p_treat": p_treat,
        "clip_percentiles": [lo_q, hi_q],
        "clip_bounds": [lo, hi],
        "weight_mean": float(np.mean(sw)),
        "weight_p95": float(np.quantile(sw, 0.95)),
        "weight_p99": float(np.quantile(sw, 0.99)),
        "weight_max": float(np.max(sw)),
        "effective_sample_size": ess,
    }
    return SimWeightResult(sample_weight=full_w, propensity=full_ps, diagnostics=diagnostics)

