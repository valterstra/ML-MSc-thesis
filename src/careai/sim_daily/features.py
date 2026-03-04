"""Column name constants — single source of truth for sim_daily."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Continuous state features (15 labs to predict)
# ---------------------------------------------------------------------------
STATE_CONTINUOUS: list[str] = [
    "creatinine", "bun", "sodium", "potassium", "bicarbonate", "anion_gap",
    "glucose", "hemoglobin", "wbc", "platelets", "magnesium", "calcium",
    "phosphate", "inr", "bilirubin",
]

# ---------------------------------------------------------------------------
# Binary state features (4 binary states to predict)
# ---------------------------------------------------------------------------
STATE_BINARY: list[str] = [
    "is_icu",
    "lactate_elevated",
    "positive_culture_cumulative",
    "blood_culture_positive_cumulative",
]

# ---------------------------------------------------------------------------
# Static / context features (not predicted, used as inputs only)
# ---------------------------------------------------------------------------
STATIC_FEATURES: list[str] = [
    "age_at_admit", "charlson_score", "drg_severity", "drg_mortality",
    "gender_M",          # encoded from raw `gender` column
    "day_of_stay",
    "days_in_current_unit",
]

# ---------------------------------------------------------------------------
# Missingness flags (15 — one per continuous lab)
# ---------------------------------------------------------------------------
MEASURED_FLAGS: list[str] = [f"{c}_measured" for c in STATE_CONTINUOUS]

# ---------------------------------------------------------------------------
# Action columns (6 drug-class binary flags)
# ---------------------------------------------------------------------------
ACTION_COLS: list[str] = [
    "antibiotic_active", "anticoagulant_active", "diuretic_active",
    "steroid_active", "insulin_active", "opioid_active",
]

# ---------------------------------------------------------------------------
# Extra infection-context inputs
# ---------------------------------------------------------------------------
INFECTION_CONTEXT: list[str] = [
    "culture_ordered_today",
    "n_active_drug_classes",
]

# ---------------------------------------------------------------------------
# Derived aggregates
# ---------------------------------------------------------------------------
INPUT_COLS: list[str] = (
    STATE_CONTINUOUS + STATE_BINARY + STATIC_FEATURES
    + MEASURED_FLAGS + ACTION_COLS + INFECTION_CONTEXT
)

OUTPUT_CONTINUOUS: list[str] = list(STATE_CONTINUOUS)
OUTPUT_BINARY: list[str] = list(STATE_BINARY)
