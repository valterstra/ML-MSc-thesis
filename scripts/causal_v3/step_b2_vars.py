"""Focused variable set for V3 causal discovery — 11 nodes.

Tier 0 (confounders): charlson_score, day_of_stay
Tier 1 (state T):     creatinine, potassium, wbc
Tier 2 (action):      diuretic_active, electrolyte_active, antibiotic_active
Tier 3 (next state):  next_creatinine, next_potassium, next_wbc

Selected by LightGBM double selection (step_a_variable_selection.py):
  - diuretic_active:     7/15 models above-mean importance
  - electrolyte_active:  6/15 models
  - antibiotic_active:   2/15 models
  - insulin_active + glucose dropped (confounding by indication)
  - steroid_active dropped (severity confounding on readmission)
  - charlson_score + day_of_stay added as explicit confounders
"""

from __future__ import annotations

TIER0_STATIC = ["charlson_score", "day_of_stay"]
TIER1_STATE  = ["creatinine", "potassium", "wbc"]
TIER2_ACTION = ["diuretic_active", "electrolyte_active", "antibiotic_active"]
TIER3_NEXT   = ["next_creatinine", "next_potassium", "next_wbc"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_NEXT
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_NEXT]

VAR_TIER: dict[str, int] = {}
for _tidx, _tvars in enumerate(TIERS):
    for _v in _tvars:
        VAR_TIER[_v] = _tidx
