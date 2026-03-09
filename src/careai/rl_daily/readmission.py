"""Readmission risk model — fit, predict, save, load.

Features: 43 state features available in the simulated next_state.
ACTION_COLS are deliberately excluded — they are inputs to the simulator,
not outputs, so they cannot be present at prediction time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
import lightgbm as lgb

from careai.sim_daily.features import (
    INFECTION_CONTEXT,
    MEASURED_FLAGS,
    STATIC_FEATURES,
    STATE_BINARY,
    STATE_CONTINUOUS,
)

# Features that are output by predict_next() (no ACTION_COLS)
READMISSION_FEATURES: list[str] = (
    STATE_CONTINUOUS       # 15 continuous labs
    + STATE_BINARY         # 4 binary state flags
    + STATIC_FEATURES      # 7 static / context features
    + MEASURED_FLAGS       # 15 missingness flags
    + INFECTION_CONTEXT    # 2 infection context features
)  # Total: 43 features


@dataclass(frozen=True)
class ReadmissionModel:
    pipeline: Pipeline          # SimpleImputer(median) + LGBMClassifier
    feature_cols: list[str]     # columns actually present at fit time
    test_auc: float
    test_brier: float
    prevalence: float           # fraction readmitted in train set


def fit_readmission_model(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> ReadmissionModel:
    """Train a LightGBM readmission risk classifier on all days of train_df."""
    # Use only features that are present in the data
    feature_cols = [c for c in READMISSION_FEATURES if c in train_df.columns]

    X_train = train_df[feature_cols]
    y_train = train_df["readmit_30d"].astype(int)

    X_test = test_df[feature_cols]
    y_test = test_df["readmit_30d"].astype(int)

    prevalence = float(y_train.mean())

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight="balanced",
            random_state=42,
            verbosity=-1,
        )),
    ])
    # Keep feature names through the pipeline so LightGBM doesn't warn
    pipe.set_output(transform="pandas")

    pipe.fit(X_train, y_train)

    y_prob = pipe.predict_proba(X_test)[:, 1]
    test_auc = float(roc_auc_score(y_test, y_prob))
    test_brier = float(brier_score_loss(y_test, y_prob))

    return ReadmissionModel(
        pipeline=pipe,
        feature_cols=feature_cols,
        test_auc=test_auc,
        test_brier=test_brier,
        prevalence=prevalence,
    )


def predict_readmission_risk(
    model: ReadmissionModel,
    state: dict[str, float] | pd.DataFrame,
) -> float | np.ndarray:
    """Return P(readmit_30d=1) for one state dict or a DataFrame of states."""
    if isinstance(state, dict):
        df = pd.DataFrame([state])
        # Fill missing feature columns with NaN so the imputer handles them
        for col in model.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[model.feature_cols]
        return float(model.pipeline.predict_proba(df)[0, 1])
    else:
        df = state.copy()
        for col in model.feature_cols:
            if col not in df.columns:
                df[col] = np.nan
        df = df[model.feature_cols]
        return model.pipeline.predict_proba(df)[:, 1]


def save_readmission_model(model: ReadmissionModel, dir_path: Path | str) -> None:
    """Save pipeline as .joblib and metadata as .json."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    joblib.dump(model.pipeline, d / "readmission_model.joblib")

    meta = {
        "feature_cols": model.feature_cols,
        "test_auc": model.test_auc,
        "test_brier": model.test_brier,
        "prevalence": model.prevalence,
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_readmission_model(dir_path: Path | str) -> ReadmissionModel:
    """Load a previously saved ReadmissionModel."""
    d = Path(dir_path)
    pipeline = joblib.load(d / "readmission_model.joblib")
    meta = json.loads((d / "metadata.json").read_text())

    return ReadmissionModel(
        pipeline=pipeline,
        feature_cols=meta["feature_cols"],
        test_auc=meta["test_auc"],
        test_brier=meta["test_brier"],
        prevalence=meta["prevalence"],
    )
