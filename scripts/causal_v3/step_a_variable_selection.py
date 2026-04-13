"""Step A: LightGBM double selection for V3 causal discovery.

For each of 15 next-state targets (next_creatinine, ..., next_is_icu) and
readmit_30d:
    Fit LightGBM on [current state vars + 8 drug actions + static features].
    Record gain importance for each drug action.

Take the union: any drug with above-average importance in ANY model is kept.

Also runs the reverse direction (each drug as target, state+static as features)
to identify which state variables are the strongest confounders per drug.

Answers three questions before running PC causal discovery:
  1. Which drug actions have predictive signal on lab transitions?
  2. Which drug actions predict readmission?
  3. Which current-state variables predict each drug action (confounders)?

Output:
    reports/causal_v3/variable_selection.json
    reports/causal_v3/variable_selection.txt
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("careai.v3.variable_selection")

# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

STATE_VARS = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "calcium", "glucose", "hemoglobin", "wbc", "platelets", "phosphate",
    "magnesium", "is_icu",
]

STATIC_FEATURES = [
    "age_at_admit", "charlson_score", "drg_severity", "drg_mortality",
    "gender_M", "day_of_stay",
]

ACTION_COLS = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active", "opioid_active",
    "electrolyte_active", "cardiovascular_active",
]

NEXT_STATE_TARGETS = [f"next_{v}" for v in STATE_VARS]
OUTCOME_COL = "readmit_30d"

# LightGBM: shallow trees, fast — we want variable selection not best AUC
LGB_PARAMS = {
    "n_estimators":      300,
    "learning_rate":     0.05,
    "num_leaves":        31,
    "max_depth":         5,
    "min_child_samples": 50,
    "colsample_bytree":  0.8,
    "subsample":         0.8,
    "subsample_freq":    1,
    "n_jobs":            -1,
    "random_state":      42,
    "verbosity":         -1,
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train(csv_path: Path) -> pd.DataFrame:
    log.info("Loading dataset: %s", csv_path)
    needed = STATE_VARS + STATIC_FEATURES + ACTION_COLS + NEXT_STATE_TARGETS + [OUTCOME_COL, "split", "gender_M"]
    # gender may be stored as gender_M already or as gender — handle both
    available = pd.read_csv(csv_path, nrows=0).columns.tolist()
    if "gender_M" not in available and "gender" in available:
        needed = [c for c in needed if c != "gender_M"] + ["gender"]
    cols = [c for c in needed if c in available]
    df = pd.read_csv(csv_path, usecols=cols, low_memory=False)
    if "gender" in df.columns and "gender_M" not in df.columns:
        df["gender_M"] = (df["gender"] == "M").astype(int)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows: %d", len(train))
    return train


def run_lgb_regression(X: pd.DataFrame, y: np.ndarray, target_name: str) -> dict[str, float]:
    """Fit LightGBM regressor, return gain importance {feature: gain}."""
    if y.std() < 1e-6:
        log.info("  [%s] skipped (zero variance)", target_name)
        return {}
    log.info("  [%s] fitting LightGBM regressor ...", target_name)
    params = {**LGB_PARAMS, "objective": "regression", "metric": "rmse"}
    model = lgb.LGBMRegressor(**{k: v for k, v in params.items() if k != "metric"})
    model.set_params(objective="regression")
    model.fit(X, y)
    importance = dict(zip(X.columns, model.booster_.feature_importance(importance_type="gain")))
    n_used = sum(1 for v in importance.values() if v > 0)
    log.info("  [%s] features used: %d / %d", target_name, n_used, len(X.columns))
    return importance


def run_lgb_classifier(X: pd.DataFrame, y: np.ndarray, target_name: str) -> dict[str, float]:
    """Fit LightGBM classifier, return gain importance {feature: gain}."""
    rate = float(y.mean())
    if rate < 0.005 or rate > 0.995:
        log.info("  [%s] skipped (prevalence=%.2f%%)", target_name, rate * 100)
        return {}
    log.info("  [%s] prevalence=%.1f%% | fitting LightGBM classifier ...", target_name, rate * 100)
    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)
    importance = dict(zip(X.columns, model.booster_.feature_importance(importance_type="gain")))
    n_used = sum(1 for v in importance.values() if v > 0)
    log.info("  [%s] features used: %d / %d", target_name, n_used, len(X.columns))
    return importance


def above_mean_features(importance: dict[str, float]) -> list[str]:
    """Return features with gain above mean."""
    if not importance:
        return []
    vals = np.array(list(importance.values()), dtype=float)
    threshold = vals.mean()
    return [f for f, g in importance.items() if g > threshold]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"
    out_dir  = PROJECT_ROOT / "reports" / "causal_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_train(csv_path)

    # Feature sets
    state_feats   = [c for c in STATE_VARS    if c in train.columns]
    static_feats  = [c for c in STATIC_FEATURES if c in train.columns]
    action_feats  = [c for c in ACTION_COLS   if c in train.columns]
    all_feats     = state_feats + static_feats + action_feats

    log.info("State features:  %d", len(state_feats))
    log.info("Static features: %d", len(static_feats))
    log.info("Action features: %d", len(action_feats))

    log.info("\nAction prevalence in train split:")
    for col in action_feats:
        log.info("  %-30s  %.1f%%", col, train[col].mean() * 100)

    X_full = train[all_feats].copy()

    # -----------------------------------------------------------------------
    # Part 1: Predict each next-state variable and readmit_30d
    #         Features = state + static + drug actions
    #         -> tells us which DRUGS matter for each outcome
    # -----------------------------------------------------------------------
    log.info("\n--- Part 1: Drug importance for state transitions + readmission ---")

    drug_importance_map: dict[str, dict[str, float]] = {}   # target -> {drug: gain}
    drug_selected_map:   dict[str, list[str]]         = {}  # target -> [drugs above mean]

    for target in NEXT_STATE_TARGETS + [OUTCOME_COL]:
        if target not in train.columns:
            log.warning("  %s not in dataset, skipping", target)
            continue
        y = train[target].fillna(train[target].median()).values.astype(float)

        if target == OUTCOME_COL:
            imp = run_lgb_classifier(X_full, y, target)
        else:
            imp = run_lgb_regression(X_full, y, target)

        # Extract only drug action importances
        drug_imp = {k: v for k, v in imp.items() if k in action_feats}
        drug_importance_map[target] = drug_imp
        drug_selected_map[target]   = above_mean_features(drug_imp)

    # Which drugs are selected in at least one model
    drug_union: set[str] = set()
    for sel in drug_selected_map.values():
        drug_union.update(sel)

    drug_freq = {
        drug: sum(1 for sel in drug_selected_map.values() if drug in sel)
        for drug in action_feats
    }

    # -----------------------------------------------------------------------
    # Part 2: Predict each drug action from state + static
    #         -> tells us which STATE VARIABLES confound each drug
    # -----------------------------------------------------------------------
    log.info("\n--- Part 2: State confounder importance per drug action ---")

    X_state = train[state_feats + static_feats].copy()
    confounder_importance_map: dict[str, dict[str, float]] = {}
    confounder_selected_map:   dict[str, list[str]]         = {}

    for drug in action_feats:
        y = train[drug].fillna(0).values.astype(float)
        imp = run_lgb_classifier(X_state, y, drug)
        confounder_importance_map[drug] = imp
        confounder_selected_map[drug]   = above_mean_features(imp)

    # -----------------------------------------------------------------------
    # Report
    # -----------------------------------------------------------------------
    log.info("")
    log.info("=" * 65)
    log.info("V3 LIGHTGBM DOUBLE SELECTION RESULTS")
    log.info("=" * 65)

    log.info("\nDRUG ACTION IMPORTANCE (models where drug is above-mean predictor):")
    log.info("  %-30s  %s", "Drug", "# models (out of %d)" % len(drug_selected_map))
    for drug in action_feats:
        marker = "  <-- SELECTED" if drug in drug_union else ""
        log.info("  %-30s  %2d / %d%s",
                 drug, drug_freq[drug], len(drug_selected_map), marker)

    log.info("\nPER-TARGET drug selection:")
    for target, sel in drug_selected_map.items():
        short = target.replace("next_", "")
        log.info("  %-20s  %s", short, sel)

    log.info("\nTOP STATE CONFOUNDERS PER DRUG:")
    for drug, sel in confounder_selected_map.items():
        log.info("  %-30s  %s", drug, sel)

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    result = {
        "action_cols":            action_feats,
        "state_vars":             state_feats,
        "static_features":        static_feats,
        "n_train_rows":           len(train),
        "action_prevalence_train": {
            col: float(train[col].mean()) for col in action_feats
        },
        "drug_selection_frequency": drug_freq,
        "drugs_selected_union":    sorted(drug_union),
        "drugs_not_selected":      sorted(set(action_feats) - drug_union),
        "per_target_drug_selection": drug_selected_map,
        "per_drug_top_confounders":  confounder_selected_map,
        "per_drug_confounder_importance": confounder_importance_map,
        "per_target_drug_importance":     drug_importance_map,
    }

    json_path = out_dir / "variable_selection.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s", json_path)

    txt_path = out_dir / "variable_selection.txt"
    with open(txt_path, "w") as f:
        f.write("V3 LIGHTGBM DOUBLE SELECTION RESULTS\n")
        f.write("=" * 65 + "\n\n")
        f.write(f"Train rows:          {len(train):,}\n")
        f.write(f"State features:      {len(state_feats)}\n")
        f.write(f"Static features:     {len(static_feats)}\n")
        f.write(f"Drug action flags:   {len(action_feats)}\n\n")

        f.write("DRUG ACTION PREVALENCE\n")
        f.write("-" * 65 + "\n")
        for col in action_feats:
            f.write(f"  {col:<32}  {train[col].mean()*100:>5.1f}%\n")

        f.write("\nDRUG SELECTION SUMMARY (# models where drug > mean importance)\n")
        f.write("-" * 65 + "\n")
        n_models = len(drug_selected_map)
        for drug in action_feats:
            flag = " <-- SELECTED" if drug in drug_union else ""
            f.write(f"  {drug:<32}  {drug_freq[drug]:>2} / {n_models}{flag}\n")

        f.write("\nPER-TARGET DRUG SELECTION\n")
        f.write("-" * 65 + "\n")
        for target, sel in drug_selected_map.items():
            short = target.replace("next_", "")
            f.write(f"  {short:<20}  {sel}\n")

        f.write("\nTOP STATE CONFOUNDERS PER DRUG\n")
        f.write("-" * 65 + "\n")
        for drug, sel in confounder_selected_map.items():
            f.write(f"  {drug:<32}  {sel}\n")

    log.info("Saved: %s", txt_path)
    log.info("Done.")


if __name__ == "__main__":
    main()
