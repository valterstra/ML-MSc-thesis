"""ATE estimators: naive, IPW, and AIPW (doubly robust) with bootstrap CIs."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline

from .features import ALL_CONFOUNDERS
from .propensity import PropensityModel, predict_propensity

# Propensity clipping to avoid extreme weights
_CLIP_LOW = 0.01
_CLIP_HIGH = 0.99


def _outcome_col(outcome: str) -> str:
    """Resolve the next-state column name for a given outcome."""
    return f"next_{outcome}"


def _valid_mask(df: pd.DataFrame, treatment: str, outcome: str) -> pd.Series:
    """Boolean mask: rows where both treatment and next-outcome are non-null."""
    oc = _outcome_col(outcome)
    return df[treatment].notna() & df[oc].notna()


# ---------------------------------------------------------------------------
# Naive estimator
# ---------------------------------------------------------------------------

def naive_ate(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """Raw mean difference — no confounder adjustment (biased baseline).

    Returns mean(outcome | treatment=1) - mean(outcome | treatment=0).
    """
    mask = _valid_mask(df, treatment, outcome)
    sub = df[mask]
    oc = _outcome_col(outcome)
    t = sub[treatment].astype(int)
    y = sub[oc]
    mu1 = y[t == 1].mean()
    mu0 = y[t == 0].mean()
    return float(mu1 - mu0)


# ---------------------------------------------------------------------------
# IPW estimator
# ---------------------------------------------------------------------------

def ipw_ate(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment: str,
    outcome: str,
) -> float:
    """Inverse Probability Weighting ATE estimate.

    Parameters
    ----------
    df:
        Must contain treatment flag and next_{outcome}.
    propensity_scores:
        P(treatment=1 | confounders) aligned to df's index.
    treatment, outcome:
        Column names.
    """
    mask = _valid_mask(df, treatment, outcome)
    sub = df[mask].copy()
    p = np.clip(propensity_scores[mask], _CLIP_LOW, _CLIP_HIGH)
    oc = _outcome_col(outcome)
    t = sub[treatment].astype(int).values
    y = sub[oc].values

    # Horvitz-Thompson weighted means
    w1 = t / p
    w0 = (1 - t) / (1 - p)
    mu1 = np.sum(w1 * y) / np.sum(w1)
    mu0 = np.sum(w0 * y) / np.sum(w0)
    return float(mu1 - mu0)


# ---------------------------------------------------------------------------
# AIPW estimator (doubly robust)
# ---------------------------------------------------------------------------

def aipw_ate(
    df: pd.DataFrame,
    propensity_scores: np.ndarray,
    treatment: str,
    outcome: str,
) -> float:
    """Augmented IPW (doubly robust) ATE estimate.

    Combines an outcome model mu(x, t) with propensity scores.
    Correct if *either* model is correctly specified.

    ATE = mean_i [ mu1_i - mu0_i
                   + (t_i / p_i) * (y_i - mu1_i)
                   - ((1-t_i)/(1-p_i)) * (y_i - mu0_i) ]
    """
    mask = _valid_mask(df, treatment, outcome)
    sub = df[mask].copy()
    p = np.clip(propensity_scores[mask], _CLIP_LOW, _CLIP_HIGH)
    oc = _outcome_col(outcome)
    t = sub[treatment].fillna(0).astype(int).values
    y = sub[oc].values

    # --- Outcome model ---
    confounder_cols = [c for c in ALL_CONFOUNDERS if c != treatment and c in sub.columns]
    X_base = sub[confounder_cols].copy()

    # Append treatment as a feature
    X_t = X_base.copy()
    X_t[treatment] = t

    outcome_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("lr", LinearRegression()),
    ])
    outcome_pipe.fit(X_t, y)

    # Predict potential outcomes
    X_1 = X_base.copy()
    X_1[treatment] = 1
    X_0 = X_base.copy()
    X_0[treatment] = 0

    mu1 = outcome_pipe.predict(X_1)
    mu0 = outcome_pipe.predict(X_0)

    # AIPW pseudo-outcomes
    psi = (
        mu1 - mu0
        + (t / p) * (y - mu1)
        - ((1 - t) / (1 - p)) * (y - mu0)
    )
    return float(np.mean(psi))


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------

def bootstrap_ci(
    df: pd.DataFrame,
    propensity_model: PropensityModel,
    treatment: str,
    outcome: str,
    n_boot: int = 500,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for AIPW ATE.

    Returns
    -------
    (point_estimate, ci_lower, ci_upper)
        Point estimate from the original sample; 95% CI from bootstrap.
    """
    rng = np.random.default_rng(seed)
    mask = _valid_mask(df, treatment, outcome)
    sub = df[mask].reset_index(drop=True)

    # Point estimate on the full (filtered) sample
    ps_full = predict_propensity(propensity_model, sub, treatment)
    point = aipw_ate(sub, ps_full, treatment, outcome)

    boot_estimates: list[float] = []
    n = len(sub)

    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boot_df = sub.iloc[idx].reset_index(drop=True)
        # Recompute propensity on bootstrap sample using the *fixed* fitted model
        ps_boot = predict_propensity(propensity_model, boot_df, treatment)
        try:
            est = aipw_ate(boot_df, ps_boot, treatment, outcome)
            boot_estimates.append(est)
        except Exception:
            # Skip degenerate resamples (e.g. single treatment group)
            continue
        if (i + 1) % 10 == 0:
            print(f"    bootstrap {i+1}/{n_boot} ...", flush=True)

    arr = np.array(boot_estimates)
    ci_lower = float(np.percentile(arr, 2.5))
    ci_upper = float(np.percentile(arr, 97.5))
    return point, ci_lower, ci_upper
