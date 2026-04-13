"""Step C: Fit structural equations on the discovered causal DAG.

For each next-state variable, trains a LightGBM model using only the
causal parents discovered by PC (from parent_sets.csv), with next_*
parents stripped (Option A — tier-0/1/2 parents only).

This produces 15 LightGBM models that together constitute the Structural
Causal Model (SCM) simulator. At inference time, given (state_t, action_t),
run all 15 models to produce state_{t+1}.

Design notes:
  - next_* parents are stripped before training (Option A). This ensures
    each structural equation uses only causally prior information and the
    simulator has no cycles at inference time.
  - Binary target (is_icu) trained as regression (predicts P(is_icu=1|parents))
    so simulation outputs a continuous probability. Threshold at 0.5 if needed.
  - Median imputation applied per column before training.
  - Models saved individually as .pkl files plus a manifest JSON.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

BINARY_TARGETS = {"is_icu"}

# LightGBM hyperparameters — same as v1 transition models
LGBM_PARAMS = {
    "n_estimators":    300,
    "learning_rate":   0.05,
    "max_depth":       6,
    "num_leaves":      63,
    "min_child_samples": 20,
    "subsample":       0.8,
    "colsample_bytree": 0.8,
    "random_state":    42,
    "n_jobs":          -1,
    "verbose":         -1,
}


# ---------------------------------------------------------------------------
# Load and strip parent sets (Option A)
# ---------------------------------------------------------------------------

def load_parent_sets(parent_sets_csv: str | Path) -> dict[str, list[str]]:
    """Load parent_sets.csv and strip any next_* parents (Option A).

    Returns:
        dict mapping target -> list of tier-0/1/2 parent features
    """
    df = pd.read_csv(parent_sets_csv)
    parent_sets = {}
    for _, row in df.iterrows():
        target = row["target"]
        raw_parents = [p.strip() for p in str(row["parents"]).split(",") if p.strip()]
        # Option A: remove any parent that starts with "next_"
        filtered = [p for p in raw_parents if not p.startswith("next_")]
        parent_sets[target] = filtered
        if len(filtered) < len(raw_parents):
            stripped = [p for p in raw_parents if p.startswith("next_")]
            log.info("  %s: stripped %d next_* parents: %s", target, len(stripped), stripped)
    return parent_sets


# ---------------------------------------------------------------------------
# Prepare training data
# ---------------------------------------------------------------------------

def prepare_training_data(
    df: pd.DataFrame,
    split: str = "train",
) -> pd.DataFrame:
    """Filter to training split, build next_* columns, drop last-day rows."""
    from careai.causal_v2.causal_discovery import CAUSAL_STATE_VARS

    train_df = df[df["split"] == split].copy()
    log.info("  Training rows before transition build: %d", len(train_df))

    train_df = train_df.sort_values(["hadm_id", "day_of_stay"])
    for var in CAUSAL_STATE_VARS:
        if var in train_df.columns:
            train_df[f"next_{var}"] = train_df.groupby("hadm_id")[var].shift(-1)

    train_df = train_df[train_df["is_last_day"] == 0].copy()
    log.info("  Transition rows (is_last_day==0): %d", len(train_df))
    return train_df


# ---------------------------------------------------------------------------
# Train one structural equation
# ---------------------------------------------------------------------------

def train_equation(
    train_df: pd.DataFrame,
    target: str,
    parents: list[str],
) -> tuple:
    """Train one LightGBM structural equation.

    Returns:
        (model, metrics_dict) where metrics_dict has r2 or auc depending on target type.
    """
    from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.metrics import r2_score, roc_auc_score

    # Drop rows where target is missing
    avail = train_df[[target] + parents].dropna()
    n = len(avail)

    if n < 100:
        log.warning("  [%s] Only %d rows available — skipping", target, n)
        return None, {"n": n, "skipped": True}

    X = avail[parents].values
    y = avail[target].values

    var_name = target.replace("next_", "")
    is_binary = var_name in BINARY_TARGETS

    if is_binary:
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X, y.astype(int))
        y_pred = model.predict_proba(X)[:, 1]
        try:
            metric_val = roc_auc_score(y.astype(int), y_pred)
            metric_key = "train_auc"
        except Exception:
            metric_val = float("nan")
            metric_key = "train_auc"
    else:
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X, y)
        y_pred = model.predict(X)
        metric_val = r2_score(y, y_pred)
        metric_key = "train_r2"

    if parents:
        importances = dict(zip(parents, model.feature_importances_))
        top = sorted(importances.items(), key=lambda x: -x[1])[:3]
        top_str = ", ".join(f"{k}({v:.0f})" for k, v in top)
    else:
        top_str = "(no parents)"

    log.info(
        "  [%s] n=%d parents=%d | %s=%.3f | top: %s",
        target, n, len(parents), metric_key, metric_val, top_str,
    )

    metrics = {
        "target": target,
        "n": n,
        "n_parents": len(parents),
        "parents": parents,
        metric_key: round(metric_val, 4),
        "top_features": top_str,
    }
    return model, metrics


# ---------------------------------------------------------------------------
# Train all structural equations
# ---------------------------------------------------------------------------

def train_all_equations(
    train_df: pd.DataFrame,
    parent_sets: dict[str, list[str]],
) -> tuple[dict, list[dict]]:
    """Train LightGBM for each target in parent_sets.

    Returns:
        models  : dict target -> fitted model (None if skipped)
        metrics : list of metric dicts (one per target)
    """
    models = {}
    all_metrics = []

    for target, parents in parent_sets.items():
        if target not in train_df.columns:
            log.warning("  [%s] target column not found — skipping", target)
            continue

        if not parents:
            log.warning(
                "  [%s] no parents after stripping next_* — will use intercept-only prediction",
                target,
            )
            # Fit a trivial model: predict the column mean
            avail = train_df[target].dropna()
            mean_val = float(avail.mean()) if len(avail) > 0 else 0.0
            models[target] = {"type": "mean", "value": mean_val}
            all_metrics.append({
                "target": target, "n": len(avail), "n_parents": 0,
                "parents": [], "note": "intercept-only (no non-next parents)",
            })
            log.info("  [%s] intercept-only: mean=%.4f", target, mean_val)
            continue

        model, metrics = train_equation(train_df, target, parents)
        models[target] = model
        all_metrics.append(metrics)

    return models, all_metrics


# ---------------------------------------------------------------------------
# Save and load
# ---------------------------------------------------------------------------

def save_models(
    models: dict,
    all_metrics: list[dict],
    parent_sets: dict[str, list[str]],
    model_dir: Path,
) -> None:
    """Save all models, manifest, and metrics to model_dir."""
    model_dir.mkdir(parents=True, exist_ok=True)

    for target, model in models.items():
        if model is None:
            continue
        model_path = model_dir / f"{target}.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

    # Manifest: records parent sets and model paths
    manifest = {
        "parent_sets": {t: p for t, p in parent_sets.items()},
        "models": [t for t, m in models.items() if m is not None],
    }
    with open(model_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    metrics_df = pd.DataFrame(all_metrics)
    metrics_df.to_csv(model_dir / "training_metrics.csv", index=False)

    log.info("Saved %d models to %s", len([m for m in models.values() if m is not None]), model_dir)
    log.info("Saved manifest.json and training_metrics.csv")


def load_models(model_dir: Path) -> tuple[dict, dict]:
    """Load all structural equation models and manifest.

    Returns:
        models      : dict target -> fitted model
        parent_sets : dict target -> list of parent feature names
    """
    with open(model_dir / "manifest.json") as f:
        manifest = json.load(f)

    parent_sets = manifest["parent_sets"]
    models = {}
    for target in manifest["models"]:
        model_path = model_dir / f"{target}.pkl"
        with open(model_path, "rb") as f:
            models[target] = pickle.load(f)

    log.info("Loaded %d structural equation models from %s", len(models), model_dir)
    return models, parent_sets
