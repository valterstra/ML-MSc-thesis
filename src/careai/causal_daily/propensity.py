"""Propensity score models — P(drug=1 | confounders) for each treatment."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import ACTION_COLS, ALL_CONFOUNDERS


@dataclass(frozen=True)
class PropensityModel:
    """Fitted propensity pipelines, one per drug."""
    pipelines: dict[str, Pipeline]   # drug → sklearn Pipeline(imputer + LR)
    confounder_cols: list[str]       # column names used as X


def _get_confounders_for(treatment: str) -> list[str]:
    """Return confounder columns excluding the focal drug (it's the outcome)."""
    return [c for c in ALL_CONFOUNDERS if c != treatment]


def fit_propensity_models(df: pd.DataFrame) -> PropensityModel:
    """Fit one logistic propensity model per drug on *df*.

    Parameters
    ----------
    df:
        One-step transition frame (train split recommended).

    Returns
    -------
    PropensityModel
        Frozen dataclass with fitted pipelines for all 6 ACTION_COLS.
    """
    pipelines: dict[str, Pipeline] = {}

    for drug in ACTION_COLS:
        confounder_cols = _get_confounders_for(drug)
        available = [c for c in confounder_cols if c in df.columns]

        X = df[available].copy()
        y = df[drug].fillna(0).astype(int)

        pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs")),
        ])
        pipe.fit(X, y)
        pipelines[drug] = pipe

    return PropensityModel(
        pipelines=pipelines,
        confounder_cols=ALL_CONFOUNDERS,
    )


def predict_propensity(
    model: PropensityModel,
    df: pd.DataFrame,
    treatment: str,
) -> np.ndarray:
    """Return P(treatment=1 | confounders) for every row in *df*.

    Parameters
    ----------
    model:
        Fitted PropensityModel.
    df:
        DataFrame whose rows we want propensity scores for.
    treatment:
        Which drug flag to score.

    Returns
    -------
    np.ndarray of shape (n_rows,)
    """
    pipe = model.pipelines[treatment]
    confounder_cols = _get_confounders_for(treatment)
    available = [c for c in confounder_cols if c in df.columns]
    X = df[available].copy()
    return pipe.predict_proba(X)[:, 1]
