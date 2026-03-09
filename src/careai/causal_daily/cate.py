"""CATE models — Conditional Average Treatment Effects via CausalForestDML.

Uses EconML's CausalForestDML to estimate patient-specific treatment effects
P(delta_outcome | state, treatment). Parallel to the AIPW ATE approach but
provides heterogeneous (per-patient) estimates rather than a scalar.

Design:
  - One CATEModel per (treatment, outcome) pair
  - CATERegistry holds all fitted models, keyed by (treatment, outcome)
  - predict_cate() returns a single float for a given patient state dict
  - Population mean of CATEs should approximate the AIPW ATE for comparison
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import dump, load

from .features import ALL_CONFOUNDERS, TREATMENT_OUTCOME_PAIRS


@dataclass(frozen=True)
class CATEModel:
    """Fitted CausalForestDML for one (treatment, outcome) pair."""

    treatment: str
    outcome: str
    estimator: object           # fitted CausalForestDML instance
    confounder_cols: list[str]  # X columns used during fit
    n_train: int
    population_ate: float       # mean(est.effect(X_train))
    ate_std: float              # std of per-patient CATEs on training set


@dataclass(frozen=True)
class CATERegistry:
    """Collection of fitted CATEModels, one per (treatment, outcome) pair."""

    models: dict[tuple[str, str], CATEModel]


def _get_confounders_for(treatment: str) -> list[str]:
    """Confounders for a given treatment: ALL_CONFOUNDERS minus the focal drug."""
    return [c for c in ALL_CONFOUNDERS if c != treatment]


def fit_cate_model(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    n_estimators: int = 200,
    max_depth: int = 5,
    seed: int = 42,
) -> CATEModel:
    """Fit a CausalForestDML for (treatment → next_{outcome}).

    Parameters
    ----------
    df:
        One-step transition frame (train split). Must contain columns for
        treatment, ``next_{outcome}``, and the confounder set.
    treatment:
        Drug flag column name (e.g. ``"insulin_active"``).
    outcome:
        Lab or binary outcome column name (e.g. ``"glucose"``).
        The target will be ``next_{outcome}``.
    n_estimators, max_depth, seed:
        CausalForestDML hyper-parameters.

    Returns
    -------
    CATEModel with population_ate and ate_std set.
    """
    from econml.dml import CausalForestDML
    from lightgbm import LGBMClassifier, LGBMRegressor

    next_col = f"next_{outcome}"
    confounder_cols = _get_confounders_for(treatment)

    # Keep only rows where T and Y are available
    available_X = [c for c in confounder_cols if c in df.columns]
    needed = [treatment, next_col] + available_X
    sub = df.dropna(subset=[treatment, next_col]).copy()

    if len(sub) < 50:
        raise ValueError(
            f"Insufficient rows ({len(sub)}) for ({treatment}, {outcome})"
        )

    T = sub[treatment].astype(float).values
    Y = sub[next_col].astype(float).values
    X = sub[available_X].copy()

    # Impute missings in X with column medians
    X = X.fillna(X.median())

    est = CausalForestDML(
        model_y=LGBMRegressor(n_estimators=100, random_state=seed, verbose=-1),
        model_t=LGBMClassifier(n_estimators=100, random_state=seed, verbose=-1),
        discrete_treatment=True,
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=seed,
        inference=True,
    )
    est.fit(Y, T, X=X)

    # Population ATE = mean of individual CATEs over training set
    cates = est.effect(X)
    population_ate = float(np.mean(cates))
    ate_std = float(np.std(cates))

    return CATEModel(
        treatment=treatment,
        outcome=outcome,
        estimator=est,
        confounder_cols=available_X,
        n_train=len(sub),
        population_ate=population_ate,
        ate_std=ate_std,
    )


def fit_cate_registry(
    train_df: pd.DataFrame,
    pairs: list[tuple[str, str]] = TREATMENT_OUTCOME_PAIRS,
    n_estimators: int = 200,
    max_depth: int = 5,
    seed: int = 42,
) -> CATERegistry:
    """Fit CATEModels for all (treatment, outcome) pairs.

    Skips pairs where required columns are missing (warns instead of raises).

    Parameters
    ----------
    train_df:
        Training split of the one-step transition frame.
    pairs:
        List of (treatment, outcome) tuples. Defaults to TREATMENT_OUTCOME_PAIRS.
    n_estimators, max_depth, seed:
        Passed to each ``fit_cate_model`` call.

    Returns
    -------
    CATERegistry
    """
    models: dict[tuple[str, str], CATEModel] = {}

    for treatment, outcome in pairs:
        next_col = f"next_{outcome}"
        if treatment not in train_df.columns:
            warnings.warn(
                f"Skipping ({treatment}, {outcome}): treatment column missing.",
                stacklevel=2,
            )
            continue
        if next_col not in train_df.columns:
            warnings.warn(
                f"Skipping ({treatment}, {outcome}): '{next_col}' column missing.",
                stacklevel=2,
            )
            continue

        print(f"  Fitting CATE: {treatment} -> {outcome} ...", flush=True)
        try:
            model = fit_cate_model(
                train_df,
                treatment=treatment,
                outcome=outcome,
                n_estimators=n_estimators,
                max_depth=max_depth,
                seed=seed,
            )
            models[(treatment, outcome)] = model
            print(
                f"    n_train={model.n_train:,}  "
                f"population_ATE={model.population_ate:+.4f}  "
                f"std={model.ate_std:.4f}"
            )
        except Exception as exc:
            warnings.warn(
                f"Failed to fit ({treatment}, {outcome}): {exc}",
                stacklevel=2,
            )

    return CATERegistry(models=models)


def predict_cate(
    registry: CATERegistry,
    treatment: str,
    outcome: str,
    state_dict: dict[str, float],
) -> float:
    """Predict per-patient CATE for (treatment → outcome) given current state.

    Parameters
    ----------
    registry:
        Fitted CATERegistry.
    treatment:
        Drug flag (e.g. ``"insulin_active"``).
    outcome:
        Lab/binary outcome (e.g. ``"glucose"``).
    state_dict:
        Current patient state as a flat dict {col: value}.

    Returns
    -------
    float — estimated individual treatment effect (delta in next_{outcome}).
    Returns 0.0 if the pair has no fitted model.
    """
    model = registry.models.get((treatment, outcome))
    if model is None:
        return 0.0

    # Build 1-row DataFrame from state_dict using stored confounder_cols.
    # fillna(X.median()) on a 1-row DataFrame returns NaN for missing cols
    # (median of a single NaN is NaN), so we chain fillna(0) as a fallback.
    row = {col: state_dict.get(col, np.nan) for col in model.confounder_cols}
    X = pd.DataFrame([row])
    X = X.fillna(X.median()).fillna(0)

    cate = float(model.estimator.effect(X)[0])
    return cate


def save_cate_registry(registry: CATERegistry, dir_path: Path | str) -> None:
    """Persist each CATEModel estimator to joblib and metadata to JSON.

    Filename format: ``cate_{treatment}__{outcome}.joblib`` (double underscore).

    Parameters
    ----------
    registry:
        Fitted CATERegistry.
    dir_path:
        Target directory (created if absent).
    """
    dir_path = Path(dir_path)
    dir_path.mkdir(parents=True, exist_ok=True)

    metadata: list[dict] = []
    for (treatment, outcome), model in registry.models.items():
        fname = f"cate_{treatment}__{outcome}.joblib"
        fpath = dir_path / fname
        dump(model.estimator, fpath)

        metadata.append(
            {
                "treatment": treatment,
                "outcome": outcome,
                "file": fname,
                "confounder_cols": model.confounder_cols,
                "n_train": model.n_train,
                "population_ate": model.population_ate,
                "ate_std": model.ate_std,
            }
        )

    meta_path = dir_path / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))
    print(f"Saved {len(metadata)} CATE models to {dir_path}")


def load_cate_registry(dir_path: Path | str) -> CATERegistry:
    """Load a CATERegistry previously saved by ``save_cate_registry``.

    Parameters
    ----------
    dir_path:
        Directory containing ``metadata.json`` and ``.joblib`` files.

    Returns
    -------
    CATERegistry
    """
    dir_path = Path(dir_path)
    meta_path = dir_path / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No metadata.json found in {dir_path}")

    metadata = json.loads(meta_path.read_text())
    models: dict[tuple[str, str], CATEModel] = {}

    for entry in metadata:
        fpath = dir_path / entry["file"]
        estimator = load(fpath)
        model = CATEModel(
            treatment=entry["treatment"],
            outcome=entry["outcome"],
            estimator=estimator,
            confounder_cols=entry["confounder_cols"],
            n_train=entry["n_train"],
            population_ate=entry["population_ate"],
            ate_std=entry["ate_std"],
        )
        models[(entry["treatment"], entry["outcome"])] = model

    print(f"Loaded {len(models)} CATE models from {dir_path}")
    return CATERegistry(models=models)
