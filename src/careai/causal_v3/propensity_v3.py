"""V3 propensity models for IPW-weighted structural equation training.

Fits P(drug=1 | confounders) for each of the 5 V3 drugs using logistic
regression, then computes stabilized inverse probability weights for each
training row.

The confounder set uses the 14 V3 state variables + static features +
infection context + co-prescriptions (the other 4 drug flags), matching
the variable space that the causal discovery operates on.

Stabilized IPW (vs. raw IPW):
  Raw:        w = 1/e  (for treated)  or  1/(1-e)  (for untreated)
  Stabilized: w = P(d) / e  (treated)  or  P(1-d) / (1-e)  (untreated)

Stabilized weights have lower variance because the numerator (marginal
treatment probability) is typically well away from 0 and 1, preventing
extreme weight values that destabilize regression. Weights are additionally
clipped to [0.05, 20] to handle residual extremes.

For models with multiple drug parents (e.g. next_glucose has steroid_active
as a parent), per-drug weights are multiplied together. This is the product-
of-marginals estimator, which is consistent under the assumption that drug
assignments are conditionally independent given confounders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ── Confounder set for V3 propensity models ───────────────────────────────
# 14 state vars that the structural equations operate on, plus infection
# context and static features available in the V3 triplet dataset.
# The focal drug is excluded per-model inside fit_propensity_v3().

V3_STATE_CONFOUNDERS: list[str] = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
    "magnesium", "is_icu",
]

V3_STATIC_CONFOUNDERS: list[str] = [
    "age_at_admit", "charlson_score", "day_of_stay",
]

V3_INFECTION_CONFOUNDERS: list[str] = [
    "positive_culture_cumulative",
    "blood_culture_positive_cumulative",
]

# All 5 optimised drugs — focal drug excluded per-model
V3_DRUGS: list[str] = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active",
]

ALL_V3_CONFOUNDERS: list[str] = (
    V3_STATE_CONFOUNDERS
    + V3_STATIC_CONFOUNDERS
    + V3_INFECTION_CONFOUNDERS
    + V3_DRUGS   # co-prescriptions; focal drug dropped inside fit_propensity_v3()
)

# Clipping bounds for stabilized weights
WEIGHT_MIN: float = 0.05
WEIGHT_MAX: float = 20.0


@dataclass
class PropensityV3:
    """Fitted propensity pipelines and marginal probabilities for V3 drugs."""
    pipelines: dict[str, Pipeline]    # drug -> sklearn Pipeline
    marginals: dict[str, float]       # drug -> P(drug=1) in training data
    confounder_cols: list[str]        # base confounder list (focal drug excluded inside)


def fit_propensity_v3(train_df: pd.DataFrame) -> PropensityV3:
    """Fit one logistic propensity model per drug on the training split.

    Args:
        train_df: Training rows from hosp_daily_v3_triplets.csv.

    Returns:
        PropensityV3 with fitted pipelines and marginal probabilities.
    """
    pipelines: dict[str, Pipeline] = {}
    marginals: dict[str, float] = {}

    for drug in V3_DRUGS:
        # Confounders: all except the focal drug
        confounders = [c for c in ALL_V3_CONFOUNDERS
                       if c != drug and c in train_df.columns]
        available = [c for c in confounders if c in train_df.columns]

        X = train_df[available].copy()
        y = train_df[drug].fillna(0).astype(int)

        marginals[drug] = float(y.mean())

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
            ("lr",      LogisticRegression(max_iter=1000, C=1.0,
                                           solver="lbfgs", n_jobs=-1)),
        ])
        pipe.fit(X, y)
        pipelines[drug] = pipe

        # Quick diagnostics
        e = pipe.predict_proba(X)[:, 1]
        log.info(
            "  Propensity [%s]: P(d=1)=%.3f  e: mean=%.3f  min=%.3f  max=%.3f",
            drug, marginals[drug], e.mean(), e.min(), e.max(),
        )

    return PropensityV3(
        pipelines=pipelines,
        marginals=marginals,
        confounder_cols=ALL_V3_CONFOUNDERS,
    )


def compute_stabilized_weights(
    prop: PropensityV3,
    df: pd.DataFrame,
    drug: str,
) -> np.ndarray:
    """Compute stabilized IPW weights for one drug on df.

    Stabilized weight for row i:
      treated   (d_i=1): w_i = P(d=1) / e_i
      untreated (d_i=0): w_i = P(d=0) / (1 - e_i)

    Clipped to [WEIGHT_MIN, WEIGHT_MAX].

    Args:
        prop:  Fitted PropensityV3.
        df:    DataFrame with drug column and confounders.
        drug:  Which drug to compute weights for.

    Returns:
        np.ndarray of shape (len(df),) with stabilized weights.
    """
    pipe = prop.pipelines[drug]
    confounders = [c for c in ALL_V3_CONFOUNDERS
                   if c != drug and c in df.columns]
    X = df[confounders].copy()
    e = pipe.predict_proba(X)[:, 1]  # P(d=1 | x)

    d = df[drug].fillna(0).astype(float).values
    p1 = prop.marginals[drug]   # P(d=1)
    p0 = 1.0 - p1               # P(d=0)

    w = np.where(d == 1, p1 / e, p0 / (1.0 - e))
    w = np.clip(w, WEIGHT_MIN, WEIGHT_MAX)
    return w


def compute_model_weights(
    prop: PropensityV3,
    train_df: pd.DataFrame,
    drug_parents: list[str],
) -> np.ndarray | None:
    """Compute combined IPW weights for a structural equation model.

    If the model has no drug parents, returns None (unweighted training).
    If the model has one drug parent, returns that drug's stabilized weights.
    If the model has multiple drug parents, returns the product of per-drug
    stabilized weights (product-of-marginals estimator).

    Args:
        prop:          Fitted PropensityV3.
        train_df:      Training DataFrame (full, index preserved).
        drug_parents:  Drug columns in this model's parent set.

    Returns:
        np.ndarray of shape (len(train_df),) or None.
    """
    # Filter to drugs that have fitted propensity models
    actionable = [d for d in drug_parents if d in prop.pipelines]

    if not actionable:
        return None

    weights = np.ones(len(train_df), dtype=float)
    for drug in actionable:
        w = compute_stabilized_weights(prop, train_df, drug)
        weights *= w
        log.info(
            "    [drug=%s] w: mean=%.3f  min=%.3f  max=%.3f",
            drug, w.mean(), w.min(), w.max(),
        )

    if len(actionable) > 1:
        # Re-clip product weights
        weights = np.clip(weights, WEIGHT_MIN, WEIGHT_MAX)
        log.info(
            "    [combined %d drugs] w: mean=%.3f  min=%.3f  max=%.3f",
            len(actionable), weights.mean(), weights.min(), weights.max(),
        )

    return weights
