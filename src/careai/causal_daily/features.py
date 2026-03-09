"""Column name constants for causal_daily — pure constants, no logic."""

from __future__ import annotations

from careai.sim_daily.features import (
    ACTION_COLS,
    MEASURED_FLAGS,
    STATIC_FEATURES,
    STATE_BINARY,
    STATE_CONTINUOUS,
)

# ---------------------------------------------------------------------------
# Confounder groups
# ---------------------------------------------------------------------------
CONFOUNDER_CONTINUOUS: list[str] = list(STATE_CONTINUOUS)
CONFOUNDER_BINARY: list[str] = list(STATE_BINARY)
CONFOUNDER_STATIC: list[str] = list(STATIC_FEATURES)
CONFOUNDER_FLAGS: list[str] = list(MEASURED_FLAGS)

# All confounders: state + static + missingness flags + the other 5 drug flags.
# The drug being treated is excluded per-drug inside the estimators; we include
# all ACTION_COLS here and the caller drops the focal drug before fitting.
ALL_CONFOUNDERS: list[str] = (
    CONFOUNDER_CONTINUOUS
    + CONFOUNDER_BINARY
    + CONFOUNDER_STATIC
    + CONFOUNDER_FLAGS
    + ACTION_COLS          # co-prescriptions are confounders for each drug
)

# ---------------------------------------------------------------------------
# Clinically motivated treatment-outcome pairs
# ---------------------------------------------------------------------------
TREATMENT_OUTCOME_PAIRS: list[tuple[str, str]] = [
    ("insulin_active",       "glucose"),
    ("antibiotic_active",    "wbc"),
    ("antibiotic_active",    "positive_culture_cumulative"),
    ("diuretic_active",      "bun"),
    ("diuretic_active",      "potassium"),
    ("diuretic_active",      "sodium"),
    ("steroid_active",       "glucose"),
    ("steroid_active",       "wbc"),
    ("anticoagulant_active", "inr"),
]

# Expected direction of ATE ("up" = drug increases outcome, "down" = decreases)
EXPECTED_DIRECTION: dict[tuple[str, str], str] = {
    ("insulin_active",       "glucose"):                    "down",
    ("antibiotic_active",    "wbc"):                        "down",
    ("antibiotic_active",    "positive_culture_cumulative"): "down",
    ("diuretic_active",      "bun"):                        "up",
    ("diuretic_active",      "potassium"):                  "down",
    ("diuretic_active",      "sodium"):                     "up",
    ("steroid_active",       "glucose"):                    "up",
    ("steroid_active",       "wbc"):                        "up",
    ("anticoagulant_active", "inr"):                        "up",
}
