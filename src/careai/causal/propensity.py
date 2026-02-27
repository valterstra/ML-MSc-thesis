"""Propensity score modeling and weight computation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass(frozen=True)
class PropensityResult:
    ps: np.ndarray
    sw_raw: np.ndarray
    sw: np.ndarray
    p_treat: float
    clip_bounds: tuple[float, float]
    model: LogisticRegression


def fit_propensity(X: pd.DataFrame, t: pd.Series, cfg: dict[str, Any]) -> PropensityResult:
    prop_cfg = cfg["propensity"]
    w_cfg = cfg["weights"]
    model = LogisticRegression(
        max_iter=int(prop_cfg.get("max_iter", 2000)),
        C=float(prop_cfg.get("C", 1.0)),
        solver="lbfgs",
    )
    model.fit(X, t)
    ps = model.predict_proba(X)[:, 1].astype(float)
    eps = 1e-6
    ps = np.clip(ps, eps, 1.0 - eps)

    t_arr = t.to_numpy(dtype=float)
    p_treat = float(t_arr.mean())
    stabilized = bool(w_cfg.get("stabilized", True))

    if stabilized:
        sw_raw = (t_arr * p_treat / ps) + ((1.0 - t_arr) * (1.0 - p_treat) / (1.0 - ps))
    else:
        sw_raw = (t_arr / ps) + ((1.0 - t_arr) / (1.0 - ps))

    lo_q, hi_q = w_cfg.get("clip_percentiles", [0.01, 0.99])
    lo = float(np.quantile(sw_raw, float(lo_q)))
    hi = float(np.quantile(sw_raw, float(hi_q)))
    sw = np.clip(sw_raw, lo, hi)

    return PropensityResult(
        ps=ps,
        sw_raw=sw_raw,
        sw=sw,
        p_treat=p_treat,
        clip_bounds=(lo, hi),
        model=model,
    )

