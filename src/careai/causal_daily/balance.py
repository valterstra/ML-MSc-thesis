"""Covariate balance diagnostics: overlap checks and standardised mean differences."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from .features import ALL_CONFOUNDERS, ACTION_COLS
from .propensity import PropensityModel, predict_propensity

_OVERLAP_INNER = (0.1, 0.9)   # "good overlap" region
_POOR_OVERLAP_THRESHOLD = 0.20  # flag if >20% of rows outside inner region
_GOOD_SMD = 0.10                # SMD < 0.1 considered well-balanced


# ---------------------------------------------------------------------------
# Overlap
# ---------------------------------------------------------------------------

def check_overlap(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment: str,
) -> dict:
    """Summarise propensity score overlap for treated vs untreated.

    Returns a dict with keys:
        n_treated, n_control,
        ps_mean_treated, ps_mean_control,
        ps_min_treated, ps_max_treated,
        ps_min_control, ps_max_control,
        frac_outside_inner (fraction of ALL rows outside [0.1, 0.9]),
        poor_overlap (bool)
    """
    t = df[treatment].fillna(0).astype(int).values
    p = propensity_scores

    treated_ps = p[t == 1]
    control_ps = p[t == 0]

    lo, hi = _OVERLAP_INNER
    frac_outside = float(np.mean((p < lo) | (p > hi)))
    poor = frac_outside > _POOR_OVERLAP_THRESHOLD

    return {
        "treatment":         treatment,
        "n_treated":         int(np.sum(t == 1)),
        "n_control":         int(np.sum(t == 0)),
        "ps_mean_treated":   float(np.mean(treated_ps)) if len(treated_ps) else float("nan"),
        "ps_mean_control":   float(np.mean(control_ps)) if len(control_ps) else float("nan"),
        "ps_min_treated":    float(np.min(treated_ps)) if len(treated_ps) else float("nan"),
        "ps_max_treated":    float(np.max(treated_ps)) if len(treated_ps) else float("nan"),
        "ps_min_control":    float(np.min(control_ps)) if len(control_ps) else float("nan"),
        "ps_max_control":    float(np.max(control_ps)) if len(control_ps) else float("nan"),
        "frac_outside_0.1_0.9": frac_outside,
        "poor_overlap":      poor,
    }


# ---------------------------------------------------------------------------
# Standardised Mean Difference
# ---------------------------------------------------------------------------

def standardised_mean_difference(
    df: pd.DataFrame,
    treatment: str,
    confounder: str,
) -> float:
    """Raw (unadjusted) SMD for one confounder.

    SMD = (mean_treated - mean_control) / pooled_std.
    Returns NaN if the column is missing or has zero variance.
    """
    if confounder not in df.columns:
        return float("nan")

    t = df[treatment].fillna(0).astype(int)
    col = pd.to_numeric(df[confounder], errors="coerce")

    v1 = col[t == 1].dropna()
    v0 = col[t == 0].dropna()

    if len(v1) < 2 or len(v0) < 2:
        return float("nan")

    pooled_std = np.sqrt((v1.var(ddof=1) + v0.var(ddof=1)) / 2)
    if pooled_std == 0:
        return float("nan")
    return float((v1.mean() - v0.mean()) / pooled_std)


def weighted_smd(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment: str,
    confounder: str,
) -> float:
    """IPW-weighted SMD for one confounder (measures balance after adjustment).

    Uses stabilised weights: treated weight = 1/p, control weight = 1/(1-p).
    """
    if confounder not in df.columns:
        return float("nan")

    t = df[treatment].fillna(0).astype(int).values
    col = pd.to_numeric(df[confounder], errors="coerce").values
    p = np.clip(propensity_scores, 0.01, 0.99)

    w = np.where(t == 1, 1.0 / p, 1.0 / (1 - p))

    # Weighted means
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mask1 = (t == 1) & ~np.isnan(col)
        mask0 = (t == 0) & ~np.isnan(col)

        if mask1.sum() < 2 or mask0.sum() < 2:
            return float("nan")

        mu1 = np.average(col[mask1], weights=w[mask1])
        mu0 = np.average(col[mask0], weights=w[mask0])

        # Weighted variance
        def _wvar(vals: np.ndarray, wts: np.ndarray) -> float:
            avg = np.average(vals, weights=wts)
            return float(np.average((vals - avg) ** 2, weights=wts))

        v1 = _wvar(col[mask1], w[mask1])
        v0 = _wvar(col[mask0], w[mask0])

    pooled_std = np.sqrt((v1 + v0) / 2)
    if pooled_std == 0:
        return float("nan")
    return float((mu1 - mu0) / pooled_std)


# ---------------------------------------------------------------------------
# Balance table
# ---------------------------------------------------------------------------

def balance_table(
    df: pd.DataFrame,
    propensity_model: PropensityModel,
) -> pd.DataFrame:
    """Compute raw and weighted SMD for every confounder × drug combination.

    Returns a long-format DataFrame with columns:
        drug, confounder, raw_smd, weighted_smd, well_balanced (bool)
    """
    rows: list[dict] = []

    for drug in ACTION_COLS:
        ps = predict_propensity(propensity_model, df, drug)
        confounders = [c for c in ALL_CONFOUNDERS if c != drug and c in df.columns]

        for conf in confounders:
            raw = standardised_mean_difference(df, drug, conf)
            weighted = weighted_smd(df, ps, drug, conf)
            rows.append({
                "drug":         drug,
                "confounder":   conf,
                "raw_smd":      raw,
                "weighted_smd": weighted,
                "well_balanced": abs(weighted) < _GOOD_SMD if not np.isnan(weighted) else None,
            })

    return pd.DataFrame(rows)
