"""Diagnostics for overlap, weights, and covariate balance."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def summarize_propensity(ps: np.ndarray, t: pd.Series) -> pd.DataFrame:
    t_arr = t.to_numpy(dtype=int)
    out_rows: list[dict[str, float | str]] = []
    for label, mask in [("treated", t_arr == 1), ("control", t_arr == 0)]:
        arr = ps[mask]
        out_rows.append(
            {
                "group": label,
                "n": int(arr.shape[0]),
                "min": float(np.min(arr)),
                "p01": float(np.quantile(arr, 0.01)),
                "p10": float(np.quantile(arr, 0.10)),
                "p50": float(np.quantile(arr, 0.50)),
                "p90": float(np.quantile(arr, 0.90)),
                "p99": float(np.quantile(arr, 0.99)),
                "max": float(np.max(arr)),
            }
        )
    return pd.DataFrame(out_rows)


def summarize_weights(sw_raw: np.ndarray, sw: np.ndarray, t: pd.Series) -> pd.DataFrame:
    t_arr = t.to_numpy(dtype=int)
    rows: list[dict[str, float | str]] = []
    for label, mask in [("all", np.ones_like(t_arr, dtype=bool)), ("treated", t_arr == 1), ("control", t_arr == 0)]:
        raw = sw_raw[mask]
        clip = sw[mask]
        ess = float((np.sum(clip) ** 2) / np.sum(clip**2))
        rows.append(
            {
                "group": label,
                "n": int(raw.shape[0]),
                "raw_mean": float(np.mean(raw)),
                "raw_p95": float(np.quantile(raw, 0.95)),
                "raw_p99": float(np.quantile(raw, 0.99)),
                "raw_max": float(np.max(raw)),
                "clip_mean": float(np.mean(clip)),
                "clip_p95": float(np.quantile(clip, 0.95)),
                "clip_p99": float(np.quantile(clip, 0.99)),
                "clip_max": float(np.max(clip)),
                "ess": ess,
            }
        )
    return pd.DataFrame(rows)


def _weighted_mean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _weighted_var(x: np.ndarray, w: np.ndarray) -> float:
    m = _weighted_mean(x, w)
    return float(np.sum(w * (x - m) ** 2) / np.sum(w))


def _smd_unweighted(x: np.ndarray, t: np.ndarray) -> float:
    x1 = x[t == 1]
    x0 = x[t == 0]
    m1, m0 = np.mean(x1), np.mean(x0)
    v1, v0 = np.var(x1), np.var(x0)
    denom = np.sqrt((v1 + v0) / 2.0)
    if denom == 0:
        return 0.0
    return float((m1 - m0) / denom)


def _smd_weighted(x: np.ndarray, t: np.ndarray, w: np.ndarray) -> float:
    w1 = w[t == 1]
    w0 = w[t == 0]
    x1 = x[t == 1]
    x0 = x[t == 0]
    m1, m0 = _weighted_mean(x1, w1), _weighted_mean(x0, w0)
    v1, v0 = _weighted_var(x1, w1), _weighted_var(x0, w0)
    denom = np.sqrt((v1 + v0) / 2.0)
    if denom == 0:
        return 0.0
    return float((m1 - m0) / denom)


def compute_balance_smd(X: pd.DataFrame, t: pd.Series, w: np.ndarray) -> pd.DataFrame:
    t_arr = t.to_numpy(dtype=int)
    rows: list[dict[str, float | str]] = []
    for col in X.columns:
        x = X[col].to_numpy(dtype=float)
        before = _smd_unweighted(x, t_arr)
        after = _smd_weighted(x, t_arr, w)
        rows.append({"feature": col, "smd_before": before, "smd_after": after, "abs_smd_before": abs(before), "abs_smd_after": abs(after)})
    out = pd.DataFrame(rows).sort_values("abs_smd_after", ascending=False).reset_index(drop=True)
    return out


def diagnostics_summary(balance_df: pd.DataFrame, ps: np.ndarray) -> dict[str, Any]:
    pct_extreme = float(((ps < 0.02) | (ps > 0.98)).mean())
    pass_rate = float((balance_df["abs_smd_after"] < 0.1).mean())
    return {
        "propensity_extreme_rate_ps_lt_002_or_gt_098": pct_extreme,
        "balance_pass_rate_abs_smd_lt_01": pass_rate,
        "balance_median_abs_smd_before": float(balance_df["abs_smd_before"].median()),
        "balance_median_abs_smd_after": float(balance_df["abs_smd_after"].median()),
    }

