"""
rl_daily — causally-grounded 1-step policy for the daily hospital simulator.

Approach (Option C: ATE correction):
  1. Compute baseline next-state with no drugs via the transition model.
  2. For each drug combination, adjust the baseline using the causal ATEs
     estimated in causal_daily (AIPW / doubly-robust). This replaces the
     transition model's confounded observational drug effects with the
     causal estimates.
  3. Score the causally-corrected next-state with a readmission risk model.
  4. Pick the action that minimises predicted readmission risk.

Modules:
  readmission  — LightGBM classifier predicting P(readmit_30d=1 | state)
  policy       — causal ATE correction + exhaustive 2^5 action search
  evaluate     — compare policy vs do-nothing vs real clinical actions
"""
