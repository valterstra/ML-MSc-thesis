"""Step C: Structural equations for V3 causal simulator.

For each next-state variable, trains a LightGBM model using causally-informed
parent sets. The feature set per model is:

  1. Own lag (always) -- e.g. creatinine for next_creatinine
  2. All 5 drug flags (always) -- let LightGBM learn zero importance if no effect
  3. Static features: age_at_admit, charlson_score (always)
  4. PC-discovered state parents (variable per outcome, from parent_sets.csv)
  5. Deduplicated

This "middle ground" approach ensures:
  - Drug effects can only be learned through observed features (no spurious paths)
  - The causal discovery constrains which STATE features each model sees
  - But every model CAN learn drug effects -- the discovery tells us which to TRUST
  - No next_* parents (can't use tomorrow's values to predict tomorrow)

The 14 models together form the Structural Causal Model (SCM) simulator.
At inference: given (state_t, action_t), run all 14 models -> state_{t+1}.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────

STATIC_FEATURES = ["age_at_admit", "charlson_score"]

ACTION_FEATURES = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active",
]

STATE_VARS = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
    "magnesium", "is_icu",
]

NEXT_TARGETS = [f"next_{v}" for v in STATE_VARS]

BINARY_TARGETS = {"is_icu"}

# LightGBM hyperparameters -- same as V2
LGBM_PARAMS = {
    "n_estimators":      300,
    "learning_rate":     0.05,
    "max_depth":         6,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "random_state":      42,
    "n_jobs":            -1,
    "verbose":           -1,
}


# ── Build parent sets ─────────────────────────────────────────────────────

def build_parent_sets(
    pc_parent_sets_csv: str | Path,
    strict: bool = False,
) -> dict[str, list[str]]:
    """Build parent sets from PC discovery results.

    Two modes:
      strict=False (default, "augmented"):
        Own lag (always) + PC state parents + static features (always)
        + all 5 drugs (always). Ensures every model has >= 8 features.

      strict=True ("strict PC"):
        ONLY features that PC discovered as directed parents, after stripping
        next_* columns. No padding with extra drugs or static features.
        A model may have as few as 1 feature if PC found only the own lag.

    Args:
        pc_parent_sets_csv: Path to PC parent_sets.csv with columns:
            target, n_parents, drug_parents, state_parents, undirected
        strict: If True, use only PC-discovered parents (no padding).

    Returns:
        dict mapping next_* target -> list of feature column names
    """
    pc_df = pd.read_csv(pc_parent_sets_csv)
    parent_sets = {}

    for _, row in pc_df.iterrows():
        target = row["target"]  # e.g. "next_creatinine"
        var_name = target.replace("next_", "")  # e.g. "creatinine"

        # Parse PC parents
        drug_parents = [
            p.strip() for p in str(row["drug_parents"]).split(",")
            if p.strip() and p.strip() != "nan"
        ]
        state_parents = [
            p.strip() for p in str(row["state_parents"]).split(",")
            if p.strip() and p.strip() != "nan"
        ]

        # Strip next_* parents (can't use at simulation time)
        state_clean = [p for p in state_parents if not p.startswith("next_")]
        state_dropped = [p for p in state_parents if p.startswith("next_")]

        # Build feature set
        features = []
        seen = set()

        def _add(f):
            if f not in seen:
                features.append(f)
                seen.add(f)

        if strict:
            # Strict PC: only what PC discovered (state + drugs), nothing else
            for p in state_clean:
                _add(p)
            for p in drug_parents:
                _add(p)

            log.info(
                "  %s: %d features (%d state + %d drugs, strict PC)%s",
                target, len(features), len(state_clean), len(drug_parents),
                "  [dropped: %s]" % ", ".join(state_dropped) if state_dropped else "",
            )
        else:
            # Augmented: own lag + PC state + static + all drugs
            # 1. Own lag (always)
            _add(var_name)

            # 2. PC-discovered state parents
            for p in state_clean:
                _add(p)

            # 3. Static features (always)
            for p in STATIC_FEATURES:
                _add(p)

            # 4. All 5 drug features (always)
            for p in ACTION_FEATURES:
                _add(p)

            pc_state_unique = [p for p in state_clean if p != var_name and p not in STATIC_FEATURES]
            log.info(
                "  %s: %d features (own_lag + %d PC_state + 2 static + 5 drugs)%s",
                target, len(features), len(pc_state_unique),
                "  [dropped: %s]" % ", ".join(state_dropped) if state_dropped else "",
            )

        parent_sets[target] = features

    return parent_sets


def build_parent_sets_notears_augmented(
    pc_parent_sets_csv: str | Path,
    notears_edges_csv: str | Path,
) -> dict[str, list[str]]:
    """Build parent sets: strict PC parents + NOTEARS state parents (all weights).

    Starts from strict PC-only parents (no padding), then adds every
    state/static variable that NOTEARS discovered as a parent of each
    next_* target, regardless of weight.

    This fills the gap left by PC's stripping of next_* parents: NOTEARS
    directly models lag-1 state→next_state relationships, recovering the
    usable form of the contemporaneous next→next links that PC found but
    we had to discard.

    Drug parents come only from PC discovery (not NOTEARS), keeping the
    causal drug-effect attribution clean.

    Args:
        pc_parent_sets_csv:  Path to PC parent_sets.csv
        notears_edges_csv:   Path to NOTEARS edges CSV (e.g. edges_lambda_0p0100.csv)
                             Columns: source, target, weight, abs_weight

    Returns:
        dict mapping next_* target -> list of feature column names
    """
    # Start from strict PC parent sets
    parent_sets = build_parent_sets(pc_parent_sets_csv, strict=True)

    # Load NOTEARS edges
    notears_df = pd.read_csv(notears_edges_csv)

    # Valid state/static sources (not drugs, not next_* variables)
    valid_sources = set(STATE_VARS) | set(STATIC_FEATURES)

    # For each next_* target, collect NOTEARS state parents
    notears_parents: dict[str, list[str]] = {t: [] for t in parent_sets}
    for _, row in notears_df.iterrows():
        source = str(row["source"])
        target = str(row["target"])
        if target not in parent_sets:
            continue
        if source.startswith("next_"):
            continue
        if source not in valid_sources:
            continue
        notears_parents[target].append(source)

    # Merge: strict PC first, then NOTEARS additions
    for target in parent_sets:
        existing = set(parent_sets[target])
        added = []
        for p in notears_parents[target]:
            if p not in existing:
                parent_sets[target].append(p)
                existing.add(p)
                added.append(p)

        n_pc = len(parent_sets[target]) - len(added)
        log.info(
            "  %s: %d features (%d PC_strict + %d NOTEARS_added: %s)",
            target,
            len(parent_sets[target]),
            n_pc,
            len(added),
            ", ".join(added) if added else "none",
        )

    return parent_sets


# ── Training ──────────────────────────────────────────────────────────────

def train_equation(
    train_df: pd.DataFrame,
    target: str,
    parents: list[str],
    sample_weight: "np.ndarray | None" = None,
) -> tuple:
    """Train one LightGBM structural equation.

    Args:
        train_df:      Training DataFrame.
        target:        Column name of the outcome to predict.
        parents:       Feature column names.
        sample_weight: Optional per-row weights (e.g. IPW). If None, all rows
                       are weighted equally (standard unweighted training).

    Returns:
        (model, metrics_dict)
    """
    from lightgbm import LGBMRegressor, LGBMClassifier
    from sklearn.metrics import r2_score, roc_auc_score

    avail = train_df[[target] + parents].dropna()
    n = len(avail)

    if n < 100:
        log.warning("  [%s] Only %d rows -- skipping", target, n)
        return None, {"target": target, "n": n, "skipped": True}

    X = avail[parents].values
    y = avail[target].values

    # Align sample weights to the rows that survived dropna
    w = None
    if sample_weight is not None:
        w = sample_weight[avail.index]
        log.info(
            "  [%s] IPW weights: mean=%.3f  min=%.3f  max=%.3f",
            target, w.mean(), w.min(), w.max(),
        )

    var_name = target.replace("next_", "")
    is_binary = var_name in BINARY_TARGETS

    if is_binary:
        model = LGBMClassifier(**LGBM_PARAMS)
        model.fit(X, y.astype(int), sample_weight=w)
        y_pred = model.predict_proba(X)[:, 1]
        try:
            metric_val = roc_auc_score(y.astype(int), y_pred)
        except Exception:
            metric_val = float("nan")
        metric_key = "train_auc"
    else:
        model = LGBMRegressor(**LGBM_PARAMS)
        model.fit(X, y, sample_weight=w)
        y_pred = model.predict(X)
        metric_val = r2_score(y, y_pred)
        metric_key = "train_r2"

    # Feature importances
    importances = dict(zip(parents, model.feature_importances_))
    top = sorted(importances.items(), key=lambda x: -x[1])[:5]
    top_str = ", ".join("%s(%d)" % (k, v) for k, v in top)

    # Drug importances specifically
    drug_imps = {d: importances.get(d, 0) for d in ACTION_FEATURES}
    total_imp = sum(importances.values())
    drug_pct = sum(drug_imps.values()) / total_imp * 100 if total_imp > 0 else 0

    log.info(
        "  [%s] n=%d features=%d | %s=%.4f | drug_imp=%.1f%% | top: %s",
        target, n, len(parents), metric_key, metric_val, drug_pct, top_str,
    )

    metrics = {
        "target": target,
        "n": n,
        "n_features": len(parents),
        "features": parents,
        metric_key: round(metric_val, 4),
        "drug_importance_pct": round(drug_pct, 2),
        "top_features": top_str,
        "feature_importances": importances,
    }
    return model, metrics


def evaluate_equation(
    model,
    test_df: pd.DataFrame,
    target: str,
    parents: list[str],
) -> dict:
    """Evaluate one structural equation on test data."""
    from sklearn.metrics import r2_score, roc_auc_score, mean_absolute_error

    avail = test_df[[target] + parents].dropna()
    n = len(avail)
    if n < 10:
        return {"target": target, "test_n": n, "note": "insufficient test data"}

    X = avail[parents].values
    y = avail[target].values

    var_name = target.replace("next_", "")
    is_binary = var_name in BINARY_TARGETS

    if is_binary:
        y_pred = model.predict_proba(X)[:, 1]
        try:
            auc = roc_auc_score(y.astype(int), y_pred)
        except Exception:
            auc = float("nan")
        return {"target": target, "test_n": n, "test_auc": round(auc, 4)}
    else:
        y_pred = model.predict(X)
        r2 = r2_score(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        return {
            "target": target,
            "test_n": n,
            "test_r2": round(r2, 4),
            "test_mae": round(mae, 4),
        }


def train_all_equations(
    train_df: pd.DataFrame,
    parent_sets: dict[str, list[str]],
) -> tuple[dict, list[dict]]:
    """Train LightGBM for each target.

    Returns:
        models  : dict target -> fitted model
        metrics : list of metric dicts
    """
    models = {}
    all_metrics = []

    for target, parents in parent_sets.items():
        if target not in train_df.columns:
            log.warning("  [%s] column not found -- skipping", target)
            continue

        model, metrics = train_equation(train_df, target, parents)
        models[target] = model
        all_metrics.append(metrics)

    return models, all_metrics


def evaluate_all_equations(
    models: dict,
    test_df: pd.DataFrame,
    parent_sets: dict[str, list[str]],
) -> list[dict]:
    """Evaluate all models on test split."""
    test_metrics = []
    for target, model in models.items():
        if model is None:
            continue
        parents = parent_sets[target]
        metrics = evaluate_equation(model, test_df, target, parents)
        test_metrics.append(metrics)

        metric_key = "test_auc" if "test_auc" in metrics else "test_r2"
        metric_val = metrics.get(metric_key, "N/A")
        log.info("  [%s] %s=%.4f (n=%d)",
                 target, metric_key, float(metric_val), metrics["test_n"])

    return test_metrics


# ── Save / Load ───────────────────────────────────────────────────────────

def save_models(
    models: dict,
    train_metrics: list[dict],
    test_metrics: list[dict],
    parent_sets: dict[str, list[str]],
    model_dir: Path,
) -> None:
    """Save all models, manifest, and metrics."""
    model_dir.mkdir(parents=True, exist_ok=True)

    for target, model in models.items():
        if model is None:
            continue
        with open(model_dir / ("%s.pkl" % target), "wb") as f:
            pickle.dump(model, f)

    # Manifest
    manifest = {
        "parent_sets": parent_sets,
        "models": [t for t, m in models.items() if m is not None],
        "static_features": STATIC_FEATURES,
        "action_features": ACTION_FEATURES,
        "state_vars": STATE_VARS,
    }
    with open(model_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # Metrics
    # Strip non-serializable fields for CSV
    train_csv = []
    for m in train_metrics:
        row = {k: v for k, v in m.items() if k != "feature_importances"}
        if "features" in row:
            row["features"] = ", ".join(row["features"])
        train_csv.append(row)
    pd.DataFrame(train_csv).to_csv(model_dir / "train_metrics.csv", index=False)

    if test_metrics:
        pd.DataFrame(test_metrics).to_csv(model_dir / "test_metrics.csv", index=False)

    # Feature importances (full detail)
    imp_rows = []
    for m in train_metrics:
        if "feature_importances" in m:
            for feat, imp in m["feature_importances"].items():
                imp_rows.append({
                    "target": m["target"],
                    "feature": feat,
                    "importance": imp,
                })
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(model_dir / "feature_importances.csv", index=False)

    n_saved = len([m for m in models.values() if m is not None])
    log.info("Saved %d models to %s", n_saved, model_dir)


def load_models(model_dir: Path) -> tuple[dict, dict]:
    """Load all structural equation models and parent sets.

    Returns:
        models      : dict target -> fitted model
        parent_sets : dict target -> list of feature names
    """
    with open(model_dir / "manifest.json") as f:
        manifest = json.load(f)

    parent_sets = manifest["parent_sets"]
    models = {}
    for target in manifest["models"]:
        with open(model_dir / ("%s.pkl" % target), "rb") as f:
            models[target] = pickle.load(f)

    log.info("Loaded %d structural equation models from %s", len(models), model_dir)
    return models, parent_sets
