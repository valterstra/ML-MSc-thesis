"""Variable selection for V4 PC causal discovery — LightGBM double selection.

For each of 16-18 binary targets (readmit_30d + Tier 3 actions with prevalence >=1%):
    Fit a LightGBM classifier on all Tier 0-2 confounders (train split only).
    Record which confounders have gain importance above the per-model mean.

Take the UNION across all models.
Any confounder that is predictive of the outcome OR any action is kept.

Advantages over LASSO:
- Handles nonlinear confounding
- Native missing value support (no imputation needed)
- Handles categoricals with label encoding (no one-hot expansion)
- Runs in ~3 minutes vs ~60 minutes for LASSO on 342k rows

Output:
    reports/readmission_v4/variable_selection.json
    reports/readmission_v4/variable_selection.txt
"""

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
log = logging.getLogger("careai.v4.variable_selection")

# ---------------------------------------------------------------------------
# Variable definitions
# ---------------------------------------------------------------------------

CATEGORICAL_COLS = [
    "gender", "insurance_cat", "race_cat", "dc_location",
    "admission_type", "first_service", "discharge_service",
]

CONFOUNDER_COLS = [
    # Admission context
    "age_at_admit", "los_days", "admit_weekday", "is_weekend_admit",
    "ed_dwell_hours", "via_ed", "english_only", "is_partnered",
    "is_observation",
    # Severity
    "drg_severity", "drg_mortality", "drg_available", "charlson_score",
    # Chronic flags (18)
    "flag_depression", "flag_drug_use", "flag_alcohol", "flag_obesity",
    "flag_malnutrition", "flag_liver_disease", "flag_chf", "flag_copd",
    "flag_diabetes", "flag_ckd", "flag_cancer", "flag_dementia",
    "flag_pvd", "flag_atrial_fibrillation", "flag_hypertension",
    "flag_schizophrenia", "flag_sleep_apnea", "flag_hypothyroidism",
    # ICU / transfers
    "had_icu", "icu_days", "n_icu_stays", "n_transfers",
    "n_distinct_careunits", "n_distinct_services",
    # Discharge labs
    "dc_lab_creatinine", "dc_lab_bun", "dc_lab_sodium", "dc_lab_potassium",
    "dc_lab_bicarbonate", "dc_lab_hemoglobin", "dc_lab_wbc",
    "dc_lab_platelets", "dc_lab_anion_gap", "dc_lab_calcium",
    "dc_lab_albumin", "n_abnormal_dc_labs",
    # Microbiology
    "positive_culture", "blood_culture_positive", "resistant_organism",
    "n_cultures",
    # In-hospital drug exposure (state, not action)
    "rx_antibiotic", "rx_anticoagulant", "rx_diuretic", "rx_steroid",
    "rx_insulin",
    # Categoricals
    *CATEGORICAL_COLS,
]

ACTION_COLS = [
    "consult_pt", "consult_ot", "consult_social_work", "consult_followup",
    "consult_palliative", "consult_diabetes_education", "consult_speech",
    "consult_addiction",
    "dc_med_antibiotic", "dc_med_anticoagulant", "dc_med_diuretic",
    "dc_med_steroid", "dc_med_insulin", "dc_med_antihypertensive",
    "dc_med_statin", "dc_med_antiplatelet", "dc_med_opiate",
]

OUTCOME_COL = "readmit_30d"

# LightGBM params: fast, shallow trees — we want variable selection not best AUC
LGB_PARAMS = {
    "objective":        "binary",
    "metric":           "auc",
    "n_estimators":     300,
    "learning_rate":    0.05,
    "num_leaves":       31,
    "max_depth":        5,
    "min_child_samples": 50,
    "colsample_bytree": 0.8,
    "subsample":        0.8,
    "subsample_freq":   1,
    "n_jobs":           -1,
    "random_state":     42,
    "verbosity":        -1,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_train(csv_path: Path) -> pd.DataFrame:
    log.info("Loading dataset: %s", csv_path)
    df = pd.read_csv(csv_path, low_memory=False)
    train = df[df["split"] == "train"].copy()
    log.info("Train rows: %d", len(train))
    return train


def prepare_X(train: pd.DataFrame, feature_names: list[str]) -> pd.DataFrame:
    """Label-encode categoricals. LightGBM handles NaN natively."""
    X = train[feature_names].copy()
    for col in CATEGORICAL_COLS:
        if col in X.columns:
            X[col] = X[col].astype("category")
    return X


def run_lgb(X: pd.DataFrame, y: np.ndarray, target_name: str) -> dict[str, float]:
    """Fit LightGBM, return gain importance dict {feature: gain}."""
    rate = float(y.mean())
    if rate < 0.005 or rate > 0.995:
        log.info("  [%s] skipped (prevalence=%.2f%%) — keep in PC anyway",
                 target_name, rate * 100)
        return {}

    log.info("  [%s] prevalence=%.1f%% | fitting LightGBM ...", target_name, rate * 100)

    model = lgb.LGBMClassifier(**LGB_PARAMS)
    model.fit(X, y)

    importance = dict(zip(X.columns, model.booster_.feature_importance(importance_type="gain")))
    auc_train = model.score(X, y)  # quick train AUC proxy
    n_selected = sum(1 for v in importance.values() if v > 0)
    log.info("  [%s] features used: %d / %d", target_name, n_selected, len(X.columns))
    return importance


def select_features(importance: dict[str, float]) -> list[str]:
    """Keep features with gain > mean gain (above-average importance)."""
    if not importance:
        return []
    vals = np.array(list(importance.values()), dtype=float)
    threshold = vals.mean()
    return [feat for feat, gain in importance.items() if gain > threshold]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    csv_path = PROJECT_ROOT / "data" / "processed" / "readmission_v4_admissions.csv"
    out_dir  = PROJECT_ROOT / "reports" / "readmission_v4"
    out_dir.mkdir(parents=True, exist_ok=True)

    train = load_train(csv_path)

    # Action prevalence — skip extremely rare as LGB targets (still enter PC)
    log.info("Action prevalence in train split:")
    action_targets = []
    for col in ACTION_COLS:
        if col not in train.columns:
            log.warning("  %s: NOT IN DATASET", col)
            continue
        rate = float(train[col].mean())
        log.info("  %s: %.1f%%", col, rate * 100)
        if rate >= 0.01:
            action_targets.append(col)
        else:
            log.info("    -> skipping as LGB target (too rare), keep in PC")

    # Confounder columns present in dataset
    feature_names = [c for c in CONFOUNDER_COLS if c in train.columns]
    missing_cols  = [c for c in CONFOUNDER_COLS if c not in train.columns]
    if missing_cols:
        log.warning("Confounder columns not in dataset: %s", missing_cols)
    log.info("Confounder features: %d", len(feature_names))

    X = prepare_X(train, feature_names)

    # Run LightGBM for each target
    all_targets   = [OUTCOME_COL] + action_targets
    selection_map : dict[str, list[str]] = {}
    importance_map: dict[str, dict[str, float]] = {}

    for target in all_targets:
        if target not in train.columns:
            log.warning("Target %s not in dataset, skipping", target)
            continue
        y = train[target].fillna(0).values.astype(float)
        imp = run_lgb(X, y, target)
        importance_map[target] = imp
        selection_map[target]  = select_features(imp)

    # Union of selected confounders
    union_set: set[str] = set()
    for sel in selection_map.values():
        union_set.update(sel)
    union_confounders = sorted(union_set)

    # How many models selected each confounder
    freq: dict[str, int] = {
        conf: sum(1 for sel in selection_map.values() if conf in sel)
        for conf in union_confounders
    }
    union_sorted = sorted(union_confounders, key=lambda c: -freq[c])

    # Dropped confounders
    dropped = sorted(set(feature_names) - union_set)

    # Final PC variable list
    pc_variables = union_sorted + ACTION_COLS + [OUTCOME_COL]

    # --- Report ---
    log.info("")
    log.info("=" * 60)
    log.info("LIGHTGBM DOUBLE SELECTION RESULTS")
    log.info("=" * 60)
    log.info("Confounders in dataset:   %d", len(feature_names))
    log.info("Confounders selected:     %d", len(union_confounders))
    log.info("Actions (all kept):       %d", len(ACTION_COLS))
    log.info("Total variables for PC:   %d", len(pc_variables))
    log.info("")
    log.info("Selected confounders (sorted by selection frequency across %d models):", len(all_targets))
    for conf in union_sorted:
        log.info("  %-40s  %2d / %d models", conf, freq[conf], len(all_targets))
    log.info("")
    log.info("Dropped confounders (below-average importance in ALL models):")
    for d in dropped:
        log.info("  %s", d)

    # --- Save JSON ---
    result = {
        "n_confounders_total":   len(feature_names),
        "n_confounders_selected": len(union_confounders),
        "n_actions":             len(ACTION_COLS),
        "n_pc_variables_total":  len(pc_variables),
        "selected_confounders":  union_sorted,
        "dropped_confounders":   dropped,
        "action_cols":           ACTION_COLS,
        "outcome_col":           OUTCOME_COL,
        "pc_variables":          pc_variables,
        "selection_frequency":   freq,
        "per_target_selection":  {k: v for k, v in selection_map.items()},
        "action_prevalence_train": {
            col: float(train[col].mean())
            for col in ACTION_COLS if col in train.columns
        },
    }
    json_path = out_dir / "variable_selection.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    log.info("Saved: %s", json_path)

    # --- Save human-readable summary ---
    txt_path = out_dir / "variable_selection.txt"
    with open(txt_path, "w") as f:
        f.write("V4 LIGHTGBM DOUBLE SELECTION RESULTS\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Confounders in dataset:  {len(feature_names)}\n")
        f.write(f"Confounders selected:    {len(union_confounders)}\n")
        f.write(f"Actions (all kept):      {len(ACTION_COLS)}\n")
        f.write(f"Total variables for PC:  {len(pc_variables)}\n\n")

        f.write("SELECTED CONFOUNDERS (sorted by frequency)\n")
        f.write("-" * 60 + "\n")
        for conf in union_sorted:
            f.write(f"  {conf:<42}  {freq[conf]:>2} / {len(all_targets)}\n")

        f.write("\nDROPPED CONFOUNDERS\n")
        f.write("-" * 60 + "\n")
        for d in dropped:
            f.write(f"  {d}\n")

        f.write("\nACTION PREVALENCE IN TRAIN\n")
        f.write("-" * 60 + "\n")
        for col in ACTION_COLS:
            if col in train.columns:
                f.write(f"  {col:<42}  {train[col].mean()*100:>5.1f}%\n")

        f.write("\nPER-TARGET SELECTION COUNTS\n")
        f.write("-" * 60 + "\n")
        for tgt, sel in selection_map.items():
            f.write(f"  {tgt:<42}  {len(sel)} confounders selected\n")

    log.info("Saved: %s", txt_path)
    log.info("Done. Review reports/readmission_v4/variable_selection.txt before running PC.")


if __name__ == "__main__":
    main()
