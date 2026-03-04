"""Train, predict, save and load LightGBM transition models for sim_daily."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from .features import (
    ACTION_COLS,
    INFECTION_CONTEXT,
    INPUT_COLS,
    MEASURED_FLAGS,
    OUTPUT_BINARY,
    OUTPUT_CONTINUOUS,
    STATIC_FEATURES,
    STATE_BINARY,
    STATE_CONTINUOUS,
)

# Default LightGBM hyperparameters
_LGB_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


@dataclass(frozen=True)
class TransitionModel:
    continuous_models: dict[str, lgb.LGBMRegressor]
    binary_models: dict[str, lgb.LGBMClassifier]
    done_model: lgb.LGBMClassifier
    input_cols: list[str]
    output_continuous: list[str]
    output_binary: list[str]
    clip_bounds: dict[str, tuple[float, float]] = field(default_factory=dict)


def fit_transition_model(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None = None,
    lgb_params: dict | None = None,
) -> TransitionModel:
    """Train per-output LightGBM models on one-step transition data."""
    params = {**_LGB_PARAMS, **(lgb_params or {})}
    input_cols = list(INPUT_COLS)

    X_train = train_df[input_cols]
    X_valid = valid_df[input_cols] if valid_df is not None else None

    # --- Continuous regressors ---
    continuous_models: dict[str, lgb.LGBMRegressor] = {}
    for col in OUTPUT_CONTINUOUS:
        target = f"next_{col}"
        mask_tr = train_df[target].notna()
        y_tr = train_df.loc[mask_tr, target]
        X_tr = X_train.loc[mask_tr]

        reg = lgb.LGBMRegressor(**params)
        fit_kw: dict = {}
        if X_valid is not None:
            mask_va = valid_df[target].notna()
            if mask_va.sum() > 0:
                fit_kw["eval_set"] = [(X_valid.loc[mask_va], valid_df.loc[mask_va, target])]
                fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]

        reg.fit(X_tr, y_tr, **fit_kw)
        continuous_models[col] = reg

    # --- Binary classifiers ---
    binary_models: dict[str, lgb.LGBMClassifier] = {}
    for col in OUTPUT_BINARY:
        target = f"next_{col}"
        mask_tr = train_df[target].notna()
        y_tr = train_df.loc[mask_tr, target].astype(int)
        X_tr = X_train.loc[mask_tr]

        clf = lgb.LGBMClassifier(**params)
        fit_kw = {}
        if X_valid is not None:
            mask_va = valid_df[target].notna()
            if mask_va.sum() > 0:
                fit_kw["eval_set"] = [(X_valid.loc[mask_va], valid_df.loc[mask_va, target].astype(int))]
                fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]

        clf.fit(X_tr, y_tr, **fit_kw)
        binary_models[col] = clf

    # --- Done model (discharge/death) ---
    y_done_tr = train_df["done_next"].astype(int)
    done_clf = lgb.LGBMClassifier(**params)
    fit_kw = {}
    if X_valid is not None:
        fit_kw["eval_set"] = [(X_valid, valid_df["done_next"].astype(int))]
        fit_kw["callbacks"] = [lgb.early_stopping(50, verbose=False)]

    done_clf.fit(X_train, y_done_tr, **fit_kw)

    # --- Clip bounds (1st / 99th percentile from training data) ---
    clip_bounds: dict[str, tuple[float, float]] = {}
    for col in OUTPUT_CONTINUOUS:
        target = f"next_{col}"
        vals = train_df[target].dropna()
        if len(vals) > 0:
            clip_bounds[col] = (float(vals.quantile(0.01)), float(vals.quantile(0.99)))

    return TransitionModel(
        continuous_models=continuous_models,
        binary_models=binary_models,
        done_model=done_clf,
        input_cols=input_cols,
        output_continuous=list(OUTPUT_CONTINUOUS),
        output_binary=list(OUTPUT_BINARY),
        clip_bounds=clip_bounds,
    )


def predict_next(
    model: TransitionModel,
    state_dict: dict[str, float],
    action_dict: dict[str, float],
) -> tuple[dict[str, float], float]:
    """Predict next state and done probability from current state + actions."""
    row = {**state_dict, **action_dict}
    X = pd.DataFrame([row], columns=model.input_cols)

    next_state: dict[str, float] = {}

    # Continuous predictions
    for col in model.output_continuous:
        pred = float(model.continuous_models[col].predict(X)[0])
        if col in model.clip_bounds:
            lo, hi = model.clip_bounds[col]
            pred = float(np.clip(pred, lo, hi))
        next_state[col] = pred

    # Binary predictions
    for col in model.output_binary:
        prob = float(model.binary_models[col].predict_proba(X)[0, 1])
        next_state[col] = 1.0 if prob >= 0.5 else 0.0

    # Done probability
    done_prob = float(model.done_model.predict_proba(X)[0, 1])

    return next_state, done_prob


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(model: TransitionModel, dir_path: str | Path) -> None:
    """Save all sub-models and metadata to *dir_path*."""
    d = Path(dir_path)
    d.mkdir(parents=True, exist_ok=True)

    for col, m in model.continuous_models.items():
        joblib.dump(m, d / f"cont_{col}.joblib")
    for col, m in model.binary_models.items():
        joblib.dump(m, d / f"bin_{col}.joblib")
    joblib.dump(model.done_model, d / "done_model.joblib")

    meta = {
        "input_cols": model.input_cols,
        "output_continuous": model.output_continuous,
        "output_binary": model.output_binary,
        "clip_bounds": {k: list(v) for k, v in model.clip_bounds.items()},
    }
    (d / "metadata.json").write_text(json.dumps(meta, indent=2))


def load_model(dir_path: str | Path) -> TransitionModel:
    """Load a previously saved TransitionModel."""
    d = Path(dir_path)
    meta = json.loads((d / "metadata.json").read_text())

    continuous_models = {
        col: joblib.load(d / f"cont_{col}.joblib")
        for col in meta["output_continuous"]
    }
    binary_models = {
        col: joblib.load(d / f"bin_{col}.joblib")
        for col in meta["output_binary"]
    }
    done_model = joblib.load(d / "done_model.joblib")
    clip_bounds = {k: tuple(v) for k, v in meta["clip_bounds"].items()}

    return TransitionModel(
        continuous_models=continuous_models,
        binary_models=binary_models,
        done_model=done_model,
        input_cols=meta["input_cols"],
        output_continuous=meta["output_continuous"],
        output_binary=meta["output_binary"],
        clip_bounds=clip_bounds,
    )
