"""Ultra-focused variable set for V3 causal discovery — 8 nodes.

Tier 0 (confounders): charlson_score, day_of_stay
Tier 1 (state T):     creatinine, potassium
Tier 2 (action):      diuretic_active, antibiotic_active
Tier 3 (next state):  next_creatinine, next_potassium

Selected from step_b2 cross-algorithm results:
  - diuretic_active:   confirmed by PC, FCI, NOTEARS (3/5 algos)
  - antibiotic_active: confirmed by PC, FCI, NOTEARS (3/5 algos)
  - potassium:  target of both drugs (correct sign, both confirmed)
  - creatinine: target of diuretic (correct sign, NOTEARS confirmed)
  - electrolyte_active dropped: all edges were confounding-by-indication
  - wbc dropped: antibiotic->wbc not found in focused discovery
"""

from __future__ import annotations

TIER0_STATIC = ["charlson_score", "day_of_stay"]
TIER1_STATE  = ["creatinine", "potassium"]
TIER2_ACTION = ["diuretic_active", "antibiotic_active"]
TIER3_NEXT   = ["next_creatinine", "next_potassium"]

ALL_VARS = TIER0_STATIC + TIER1_STATE + TIER2_ACTION + TIER3_NEXT
TIERS    = [TIER0_STATIC, TIER1_STATE, TIER2_ACTION, TIER3_NEXT]

VAR_TIER: dict[str, int] = {}
for _tidx, _tvars in enumerate(TIERS):
    for _v in _tvars:
        VAR_TIER[_v] = _tidx
