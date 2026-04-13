"""Step A: ML-based variable selection for causal discovery.

Two analyses:
  Analysis 1 (transition): For each dynamic state variable, train LightGBM to
    predict next-day value from current state + actions. Extract feature importances.
    Tells us which variables are in the causal pathway between drugs and outcomes.

  Analysis 2 (readmission): Train LightGBM on discharge-day rows to predict
    readmit_30d. Extract feature importances.
    Tells us which discharge-state features drive Phase 2 (post-discharge) outcomes.

Design decisions (agreed 2026-03-15):
  - Only rows with day_of_stay >= 1 (valid previous-day state)
  - Next-day targets: Option A — train only on rows where tomorrow was actually measured
  - Actions: binary active flags + route_iv flags (Option B)
  - Static confounders included as features: age, charlson, drg_severity, gender
  - 30 transition targets + 1 readmission model
  - Raw importances saved for user review — no automatic variable selection
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

try:
    import shap as _shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature groups
# ---------------------------------------------------------------------------

STATIC_CONFOUNDERS = [
    "age_at_admit", "charlson_score", "drg_severity", "gender",
]

TIME_CONTEXT = [
    "day_of_stay", "is_icu", "days_in_current_unit",
]

ACTION_FEATURES = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active",
    "antibiotic_route_iv", "anticoagulant_route_iv", "diuretic_route_iv",
    "steroid_route_iv", "insulin_route_iv",
]

# Continuous lab targets
LAB_TARGETS_CONTINUOUS = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "glucose", "hemoglobin", "wbc", "platelets", "magnesium", "calcium",
    "phosphate", "inr", "bilirubin", "nlr", "albumin",
    "crp", "ptt", "troponin_t", "bnp", "alt", "ast",
    "partial_sofa",
]

# Binary state targets
BINARY_TARGETS = [
    "is_icu", "lactate_elevated", "positive_culture_cumulative",
    "blood_culture_positive_cumulative", "icu_escalation",
]

ALL_TRANSITION_TARGETS = LAB_TARGETS_CONTINUOUS + BINARY_TARGETS

# curr_service_group gets one-hot encoded — excluded from target list
SERVICE_COL = "curr_service_group"


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_transitions(df: pd.DataFrame) -> pd.DataFrame:
    """Build transition dataset: features at t, targets at t+1.

    Filters to day_of_stay >= 1 so every row has a valid predecessor.
    Shifts targets within each admission to get next-day values.
    Does NOT impute missing next-day targets — each model trains on its own
    non-null subset (Option A).
    """
    log.info("Preparing transition dataset")
    df = df[df["day_of_stay"] >= 1].copy()
    df = df.sort_values(["hadm_id", "day_of_stay"])
    log.info("  Rows after day_of_stay >= 1 filter: %d", len(df))

    # Build next-day target columns by shifting within each admission
    for col in ALL_TRANSITION_TARGETS:
        if col in df.columns:
            df[f"next_{col}"] = df.groupby("hadm_id")[col].shift(-1)

    # Drop last day of each admission (no next-day state)
    df = df[df["is_last_day"] == 0].copy()
    log.info("  Rows after dropping last-day rows: %d", len(df))

    return df


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Assemble feature matrix from current-day state + actions + context."""
    feature_cols = []

    # Static confounders
    for c in STATIC_CONFOUNDERS:
        if c in df.columns:
            feature_cols.append(c)

    # Gender encoding
    if "gender" in df.columns:
        df["gender_M"] = (df["gender"] == "M").astype(int)
        feature_cols = [c for c in feature_cols if c != "gender"]
        feature_cols.append("gender_M")

    # Time context
    for c in TIME_CONTEXT:
        if c in df.columns:
            feature_cols.append(c)

    # Service group one-hot
    if SERVICE_COL in df.columns:
        dummies = pd.get_dummies(df[SERVICE_COL], prefix="svc", drop_first=False)
        df = pd.concat([df, dummies], axis=1)
        feature_cols += list(dummies.columns)

    # Current-day dynamic state (all transition targets as current state)
    for c in ALL_TRANSITION_TARGETS:
        if c in df.columns:
            feature_cols.append(c)

    # Actions
    for c in ACTION_FEATURES:
        if c in df.columns:
            feature_cols.append(c)

    # Remove duplicates preserving order
    seen = set()
    feature_cols = [c for c in feature_cols if not (c in seen or seen.add(c))]

    return df, feature_cols


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------

LGBM_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    random_state=42,
    verbose=-1,
)

RF_PARAMS = dict(
    n_estimators=100,
    max_features="sqrt",
    min_samples_leaf=50,
    n_jobs=-1,
    random_state=42,
)

SHAP_SAMPLE_N = 5_000  # rows sampled for SHAP computation per model


def train_transition_models(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
    return_models: bool = False,
) -> "dict[str, pd.Series] | tuple[dict, dict]":
    """Train one LightGBM per transition target. Returns dict of importance Series.

    If return_models=True, returns (importances, models) where models is a dict
    mapping target -> {"model": fitted model, "X_sample": DataFrame for SHAP}.
    """
    train_df = df[df["split"] == split].copy()
    log.info("  [LGBM] Training on %d rows (split=%s)", len(train_df), split)

    importances: dict[str, pd.Series] = {}
    models: dict[str, dict] = {}
    X_full = train_df[feature_cols].copy()

    for target in ALL_TRANSITION_TARGETS:
        next_col = f"next_{target}"
        if next_col not in train_df.columns:
            log.warning("  Skipping %s — next-day column not found", target)
            continue

        # Option A: only rows where next-day value was actually measured
        mask = train_df[next_col].notna()
        n_rows = mask.sum()
        if n_rows < 1000:
            log.warning("  Skipping %s — only %d non-null next-day rows", target, n_rows)
            continue

        X = X_full[mask].copy()
        y = train_df.loc[mask, next_col]
        X = X.fillna(X.median(numeric_only=True))

        is_binary = target in BINARY_TARGETS
        if is_binary:
            model = LGBMClassifier(**LGBM_PARAMS)
        else:
            model = LGBMRegressor(**LGBM_PARAMS)

        model.fit(X, y)

        imp = pd.Series(
            model.feature_importances_,
            index=feature_cols,
            name=target,
        ).sort_values(ascending=False)
        importances[target] = imp

        if return_models:
            models[target] = {
                "model": model,
                "X_sample": X.sample(min(SHAP_SAMPLE_N, len(X)), random_state=42),
                "is_binary": is_binary,
            }

        log.info(
            "  [LGBM] [%s] n=%d | top: %s",
            target, n_rows,
            ", ".join(imp.head(3).index.tolist()),
        )

    if return_models:
        return importances, models
    return importances


def train_readmission_model(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
    return_model: bool = False,
) -> "pd.Series | tuple[pd.Series, object]":
    """Train readmission model on discharge-day rows. Returns feature importances.

    If return_model=True, returns (importances, fitted_model).
    """
    discharge_df = df[(df["is_last_day"] == 1) & (df["split"] == split)].copy()
    log.info(
        "  [LGBM] Discharge-day rows for readmission: %d (prevalence=%.1f%%)",
        len(discharge_df),
        discharge_df["readmit_30d"].mean() * 100,
    )

    X = discharge_df[feature_cols].fillna(discharge_df[feature_cols].median(numeric_only=True))
    y = discharge_df["readmit_30d"]

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X, y)

    imp = pd.Series(
        model.feature_importances_,
        index=feature_cols,
        name="readmit_30d",
    ).sort_values(ascending=False)

    log.info(
        "  [LGBM] Readmission top: %s",
        ", ".join(imp.head(5).index.tolist()),
    )
    if return_model:
        return imp, model
    return imp


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------

def aggregate_importances(
    importances: dict[str, pd.Series],
) -> pd.DataFrame:
    """Stack all per-target importances into a wide DataFrame.

    Rows = features, columns = targets.
    Also adds mean_importance and n_top10 (how many targets rank this feature top-10).
    """
    wide = pd.DataFrame(importances)
    wide = wide.div(wide.sum(axis=0), axis=1)  # normalise per target

    wide["mean_importance"] = wide.mean(axis=1)
    wide["n_top10"] = (wide.rank(ascending=False) <= 10).sum(axis=1)
    wide = wide.sort_values("mean_importance", ascending=False)

    return wide


def save_results(
    transition_importances: dict[str, pd.Series],
    readmission_importance: pd.Series,
    report_dir: Path,
) -> None:
    """Save all importance tables as CSV files."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # Per-target importances (wide format)
    agg = aggregate_importances(transition_importances)
    agg.to_csv(report_dir / "transition_importances_wide.csv")
    log.info("  Saved transition_importances_wide.csv (%d features x %d targets)",
             len(agg), len(transition_importances))

    # Long format — easier for filtering
    long_rows = []
    for target, imp in transition_importances.items():
        imp_norm = imp / imp.sum() if imp.sum() > 0 else imp
        for feat, val in imp_norm.items():
            long_rows.append({"target": target, "feature": feat, "importance": val})
    long_df = pd.DataFrame(long_rows)
    long_df.to_csv(report_dir / "transition_importances_long.csv", index=False)
    log.info("  Saved transition_importances_long.csv")

    # Top-10 per target — quick human-readable summary
    summary_rows = []
    for target, imp in transition_importances.items():
        for rank, (feat, val) in enumerate(imp.head(10).items(), 1):
            summary_rows.append({
                "target": target,
                "rank": rank,
                "feature": feat,
                "importance": val / imp.sum() if imp.sum() > 0 else val,
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(report_dir / "transition_top10_per_target.csv", index=False)
    log.info("  Saved transition_top10_per_target.csv")

    # Readmission importances
    readmission_importance.to_csv(report_dir / "readmission_importances.csv", header=True)
    log.info("  Saved readmission_importances.csv")

    # Summary: feature ranking by mean transition importance
    summary = agg[["mean_importance", "n_top10"]].copy()
    summary.index.name = "feature"
    summary.to_csv(report_dir / "feature_summary.csv")
    log.info("  Saved feature_summary.csv")

    log.info("All results saved to %s", report_dir)


# ---------------------------------------------------------------------------
# Multi-model robustness: SHAP, Random Forest, Logistic/Ridge
# ---------------------------------------------------------------------------

def compute_shap_importances(
    models: dict,
    feature_cols: list[str],
) -> dict[str, pd.Series]:
    """Compute mean |SHAP| importances from trained LightGBM model objects.

    models: dict mapping target -> {"model": fitted LightGBM, "X_sample": DataFrame}
    Returns dict of same shape as train_transition_models output.
    """
    if not HAS_SHAP:
        log.warning("shap not installed — skipping SHAP analysis")
        return {}

    importances: dict[str, pd.Series] = {}
    for target, model_data in models.items():
        model = model_data["model"]
        X_sample = model_data["X_sample"]
        is_binary = model_data["is_binary"]
        try:
            explainer = _shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_sample)
            # For LGBMClassifier shap_values may be shape (n, p) or list of two arrays
            if isinstance(shap_values, list):
                shap_values = shap_values[1]  # positive class
            mean_abs = np.abs(shap_values).mean(axis=0)
            imp = pd.Series(mean_abs, index=feature_cols, name=target).sort_values(ascending=False)
            importances[target] = imp
            log.info("  [SHAP] [%s] top: %s", target, ", ".join(imp.head(3).index.tolist()))
        except Exception as exc:
            log.warning("  [SHAP] Failed for %s: %s", target, exc)

    return importances


def compute_readmission_shap(
    lgbm_model,
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
) -> "pd.Series | None":
    """Compute SHAP importances for the readmission LightGBM model."""
    if not HAS_SHAP:
        return None
    discharge_df = df[(df["is_last_day"] == 1) & (df["split"] == split)].copy()
    X = discharge_df[feature_cols].fillna(discharge_df[feature_cols].median(numeric_only=True))
    X_sample = X.sample(min(SHAP_SAMPLE_N, len(X)), random_state=42)
    try:
        explainer = _shap.TreeExplainer(lgbm_model)
        shap_values = explainer.shap_values(X_sample)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]
        mean_abs = np.abs(shap_values).mean(axis=0)
        imp = pd.Series(mean_abs, index=feature_cols, name="readmit_30d_shap").sort_values(ascending=False)
        log.info("  [SHAP] Readmission top: %s", ", ".join(imp.head(5).index.tolist()))
        return imp
    except Exception as exc:
        log.warning("  [SHAP] Readmission failed: %s", exc)
        return None


def train_transition_models_rf(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
) -> dict[str, pd.Series]:
    """Train one Random Forest per transition target. Returns MDI importance Series."""
    train_df = df[df["split"] == split].copy()
    log.info("  [RF] Training on %d rows (split=%s)", len(train_df), split)

    importances: dict[str, pd.Series] = {}
    X_full = train_df[feature_cols].copy()

    for target in ALL_TRANSITION_TARGETS:
        next_col = f"next_{target}"
        if next_col not in train_df.columns:
            continue
        mask = train_df[next_col].notna()
        n_rows = mask.sum()
        if n_rows < 1000:
            continue

        X = X_full[mask].copy().fillna(X_full[mask].median(numeric_only=True))
        y = train_df.loc[mask, next_col]

        is_binary = target in BINARY_TARGETS
        model = RandomForestClassifier(**RF_PARAMS) if is_binary else RandomForestRegressor(**RF_PARAMS)
        model.fit(X, y)

        imp = pd.Series(
            model.feature_importances_, index=feature_cols, name=target
        ).sort_values(ascending=False)
        importances[target] = imp
        log.info("  [RF] [%s] n=%d | top: %s", target, n_rows, ", ".join(imp.head(3).index.tolist()))

    return importances


def train_readmission_model_rf(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
) -> pd.Series:
    """Random Forest readmission model. Returns MDI importances."""
    discharge_df = df[(df["is_last_day"] == 1) & (df["split"] == split)].copy()
    X = discharge_df[feature_cols].fillna(discharge_df[feature_cols].median(numeric_only=True))
    y = discharge_df["readmit_30d"]

    model = RandomForestClassifier(**RF_PARAMS)
    model.fit(X, y)

    imp = pd.Series(
        model.feature_importances_, index=feature_cols, name="readmit_30d_rf"
    ).sort_values(ascending=False)
    log.info("  [RF] Readmission top: %s", ", ".join(imp.head(5).index.tolist()))
    return imp


LR_MAX_ROWS = 200_000  # cap for LR training — 200k rows is more than enough for coefficients


def train_transition_models_lr(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
    random_state: int = 42,
) -> dict[str, pd.Series]:
    """Train Ridge (continuous) / LogisticRegression (binary) per transition target.

    Features are standardized; importances = absolute coefficients.
    Capped at LR_MAX_ROWS rows per model to prevent saga solver stalling on large datasets.
    """
    train_df = df[df["split"] == split].copy()
    log.info("  [LR] Training on %d rows (split=%s, cap=%d)", len(train_df), split, LR_MAX_ROWS)

    importances: dict[str, pd.Series] = {}
    X_full = train_df[feature_cols].copy()

    for target in ALL_TRANSITION_TARGETS:
        next_col = f"next_{target}"
        if next_col not in train_df.columns:
            continue
        mask = train_df[next_col].notna()
        n_rows = mask.sum()
        if n_rows < 1000:
            continue

        X = X_full[mask].copy().fillna(X_full[mask].median(numeric_only=True))
        y = train_df.loc[mask, next_col]

        # Cap rows
        if len(X) > LR_MAX_ROWS:
            idx = X.sample(LR_MAX_ROWS, random_state=random_state).index
            X = X.loc[idx]
            y = y.loc[idx]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        is_binary = target in BINARY_TARGETS
        if is_binary:
            model = LogisticRegression(solver="saga", max_iter=100, random_state=random_state, n_jobs=-1)
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_[0])
        else:
            model = Ridge()
            model.fit(X_scaled, y)
            coefs = np.abs(model.coef_)

        imp = pd.Series(coefs, index=feature_cols, name=target).sort_values(ascending=False)
        importances[target] = imp
        log.info("  [LR] [%s] n=%d | top: %s", target, len(X), ", ".join(imp.head(3).index.tolist()))

    return importances


def train_readmission_model_lr(
    df: pd.DataFrame,
    feature_cols: list[str],
    split: str = "train",
    random_state: int = 42,
) -> pd.Series:
    """Logistic Regression readmission model. Returns absolute standardized coefficients."""
    discharge_df = df[(df["is_last_day"] == 1) & (df["split"] == split)].copy()
    X = discharge_df[feature_cols].fillna(discharge_df[feature_cols].median(numeric_only=True))
    y = discharge_df["readmit_30d"]

    if len(X) > LR_MAX_ROWS:
        idx = X.sample(LR_MAX_ROWS, random_state=random_state).index
        X = X.loc[idx]
        y = y.loc[idx]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = LogisticRegression(solver="saga", max_iter=100, random_state=random_state, n_jobs=-1)
    model.fit(X_scaled, y)

    imp = pd.Series(
        np.abs(model.coef_[0]), index=feature_cols, name="readmit_30d_lr"
    ).sort_values(ascending=False)
    log.info("  [LR] Readmission top: %s", ", ".join(imp.head(5).index.tolist()))
    return imp


# ---------------------------------------------------------------------------
# Consensus ranking across methods
# ---------------------------------------------------------------------------

def build_consensus(
    method_importances: dict[str, dict[str, pd.Series]],
    feature_cols: list[str],
) -> pd.DataFrame:
    """Build a consensus importance table across multiple methods.

    method_importances: {"lgbm": {target: Series}, "shap": {...}, "rf": {...}, "lr": {...}}

    For each method:
      - Normalize each target's importances to sum to 1
      - Average across all targets -> per-method mean importance vector

    Output DataFrame (rows=features):
      - one column per method (mean normalized importance)
      - mean_rank: average rank across methods (lower = more important)
      - mean_norm_importance: mean across method columns
      - n_methods_top10: number of methods where feature ranks top-10
    """
    method_means: dict[str, pd.Series] = {}

    for method, imp_dict in method_importances.items():
        if not imp_dict:
            continue
        normed = {}
        for target, imp in imp_dict.items():
            s = imp.sum()
            normed[target] = imp / s if s > 0 else imp
        wide = pd.DataFrame(normed).reindex(feature_cols).fillna(0)
        method_means[method] = wide.mean(axis=1)

    if not method_means:
        return pd.DataFrame()

    consensus = pd.DataFrame(method_means)
    ranks = consensus.rank(ascending=False)
    consensus["mean_rank"] = ranks.mean(axis=1)
    consensus["mean_norm_importance"] = consensus[list(method_means.keys())].mean(axis=1)
    consensus["n_methods_top10"] = (ranks <= 10).sum(axis=1)
    consensus = consensus.sort_values("mean_rank")
    return consensus


def save_results_multi(
    transition_by_method: dict[str, dict[str, pd.Series]],
    readmission_by_method: dict[str, "pd.Series | None"],
    feature_cols: list[str],
    report_dir: Path,
) -> None:
    """Save multi-model importance results to report_dir."""
    report_dir.mkdir(parents=True, exist_ok=True)

    # Per-method transition summaries
    for method, imp_dict in transition_by_method.items():
        if not imp_dict:
            log.info("  Skipping %s — no results", method)
            continue
        agg = aggregate_importances(imp_dict)
        agg.to_csv(report_dir / f"transition_{method}.csv")
        log.info("  Saved transition_%s.csv", method)

    # Consensus ranking
    consensus = build_consensus(transition_by_method, feature_cols)
    if not consensus.empty:
        consensus.to_csv(report_dir / "transition_consensus.csv")
        log.info(
            "  Saved transition_consensus.csv (%d features, %d methods)",
            len(consensus), len([m for m in transition_by_method if transition_by_method[m]]),
        )

    # Readmission multi-model comparison
    readmit_series = {k: v for k, v in readmission_by_method.items() if v is not None}
    if readmit_series:
        readmit_df = pd.DataFrame(readmit_series).reindex(feature_cols).fillna(0)
        readmit_norm = readmit_df.div(readmit_df.sum(axis=0), axis=1)
        readmit_norm["mean_importance"] = readmit_norm.mean(axis=1)
        readmit_norm = readmit_norm.sort_values("mean_importance", ascending=False)
        readmit_norm.to_csv(report_dir / "readmission_multi.csv")
        log.info("  Saved readmission_multi.csv")

    log.info("Multi-model results saved to %s", report_dir)
