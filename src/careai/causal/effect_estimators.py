"""Crude/IPW/AIPW effect estimators."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm

from .propensity import fit_propensity


@dataclass(frozen=True)
class EffectResult:
    crude_mu1: float
    crude_mu0: float
    crude_rd: float
    crude_rr: float
    ipw_mu1: float
    ipw_mu0: float
    ipw_rd: float
    ipw_rr: float
    aipw_mu1: float
    aipw_mu0: float
    aipw_rd: float
    aipw_rr: float


def _safe_rr(mu1: float, mu0: float) -> float:
    if mu0 <= 0:
        return float("inf")
    return float(mu1 / mu0)


def estimate_effects(X: pd.DataFrame, t: pd.Series, y: pd.Series, cfg: dict[str, Any]) -> tuple[EffectResult, dict[str, Any]]:
    t_arr = t.to_numpy(dtype=float)
    y_arr = y.to_numpy(dtype=float)

    # Crude
    mu1_crude = float(y_arr[t_arr == 1].mean())
    mu0_crude = float(y_arr[t_arr == 0].mean())

    # Propensity + weights
    ps_res = fit_propensity(X, t, cfg)
    ps = ps_res.ps
    sw = ps_res.sw

    # IPW
    w1 = sw * t_arr
    w0 = sw * (1.0 - t_arr)
    mu1_ipw = float(np.sum(w1 * y_arr) / np.sum(w1))
    mu0_ipw = float(np.sum(w0 * y_arr) / np.sum(w0))

    # Outcome model for AIPW
    Z = pd.concat([pd.Series(t_arr, name="treat"), X.reset_index(drop=True)], axis=1)
    Z = sm.add_constant(Z, has_constant="add")
    glm = sm.GLM(y_arr, Z, family=sm.families.Binomial())
    fit = glm.fit()

    Z1 = Z.copy()
    Z1["treat"] = 1.0
    Z0 = Z.copy()
    Z0["treat"] = 0.0
    m1 = np.asarray(fit.predict(Z1), dtype=float)
    m0 = np.asarray(fit.predict(Z0), dtype=float)

    aipw_psi1 = m1 + t_arr * (y_arr - m1) / ps
    aipw_psi0 = m0 + (1.0 - t_arr) * (y_arr - m0) / (1.0 - ps)
    mu1_aipw = float(np.mean(aipw_psi1))
    mu0_aipw = float(np.mean(aipw_psi0))

    out = EffectResult(
        crude_mu1=mu1_crude,
        crude_mu0=mu0_crude,
        crude_rd=float(mu1_crude - mu0_crude),
        crude_rr=_safe_rr(mu1_crude, mu0_crude),
        ipw_mu1=mu1_ipw,
        ipw_mu0=mu0_ipw,
        ipw_rd=float(mu1_ipw - mu0_ipw),
        ipw_rr=_safe_rr(mu1_ipw, mu0_ipw),
        aipw_mu1=mu1_aipw,
        aipw_mu0=mu0_aipw,
        aipw_rd=float(mu1_aipw - mu0_aipw),
        aipw_rr=_safe_rr(mu1_aipw, mu0_aipw),
    )
    diag = {
        "ps": ps_res.ps,
        "sw_raw": ps_res.sw_raw,
        "sw": ps_res.sw,
        "p_treat": ps_res.p_treat,
        "clip_bounds": ps_res.clip_bounds,
    }
    return out, diag


def bootstrap_effects(
    X: pd.DataFrame,
    t: pd.Series,
    y: pd.Series,
    cfg: dict[str, Any],
    n_resamples: int,
    seed: int,
    progress_every: int | None = None,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(X)
    rows: list[dict[str, float]] = []
    Xr = X.reset_index(drop=True)
    tr = t.reset_index(drop=True)
    yr = y.reset_index(drop=True)
    started = perf_counter()

    def _fmt_time(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        h = seconds // 3600
        m = (seconds % 3600) // 60
        s = seconds % 60
        if h > 0:
            return f"{h:02d}:{m:02d}:{s:02d}"
        return f"{m:02d}:{s:02d}"

    for i in range(1, n_resamples + 1):
        idx = rng.integers(0, n, size=n)
        Xb = Xr.iloc[idx].reset_index(drop=True)
        tb = tr.iloc[idx].reset_index(drop=True)
        yb = yr.iloc[idx].reset_index(drop=True)
        est, _ = estimate_effects(Xb, tb, yb, cfg)
        rows.append(
            {
                "ipw_rd": est.ipw_rd,
                "ipw_rr": est.ipw_rr,
                "aipw_rd": est.aipw_rd,
                "aipw_rr": est.aipw_rr,
            }
        )
        if progress_every and (i % progress_every == 0 or i == n_resamples):
            elapsed = perf_counter() - started
            per_iter = elapsed / float(i)
            remaining = per_iter * float(n_resamples - i)
            pct = 100.0 * float(i) / float(n_resamples)
            print(
                f"[bootstrap] {i}/{n_resamples} ({pct:.1f}%) "
                f"elapsed={_fmt_time(elapsed)} eta={_fmt_time(remaining)}",
                flush=True,
            )
    return pd.DataFrame(rows)


def percentile_ci(samples: pd.Series, alpha: float = 0.05) -> tuple[float, float]:
    lo = float(samples.quantile(alpha / 2.0))
    hi = float(samples.quantile(1.0 - alpha / 2.0))
    return lo, hi
