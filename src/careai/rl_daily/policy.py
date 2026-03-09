"""Causal exhaustive policy using ATE corrections to the transition model.

Design (Option C):
  Rather than using predict_next(state, action) directly — which embeds
  observational confounding between drug prescriptions and outcomes — we:

  1. Compute a single baseline next-state with no drugs:
       base_next = predict_next(model, state, {all drugs: 0})
  2. For each candidate action combo, shift the relevant lab/binary outcomes
     by the causal ATEs estimated in the causal_daily step (AIPW):
       causal_next[outcome] = base_next[outcome] + ATE(drug → outcome)
  3. Score the causally-corrected next-state with the readmission model.
  4. Return the action with the lowest predicted readmission risk.

Coverage:
  9 (treatment, outcome) pairs across 5 drugs are covered by the ATE table.
  opioid_active is excluded from optimisation: no causal ATE was estimated
  for it, so including it would reintroduce the observational confounding we
  are specifically trying to avoid. It is fixed to 0 in all policy calls.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

from careai.sim_daily.features import ACTION_COLS
from careai.sim_daily.transition import TransitionModel, predict_next

from .readmission import ReadmissionModel, predict_readmission_risk

# Drugs for which we have causal ATEs — these are the ones we optimise over.
# opioid_active is deliberately absent.
ATE_DRUGS: list[str] = [
    "antibiotic_active",
    "anticoagulant_active",
    "diuretic_active",
    "insulin_active",
    "steroid_active",
]


def load_ate_table(ate_json_path: Path | str) -> dict[tuple[str, str], float]:
    """Load treatment_effects.json → {(treatment, outcome): causal_ate}."""
    entries = json.loads(Path(ate_json_path).read_text())
    return {(e["treatment"], e["outcome"]): e["causal_ate"] for e in entries}


def apply_ate_corrections(
    base_next_state: dict[str, float],
    action_dict: dict[str, int],
    ate_table: dict[tuple[str, str], float],
) -> dict[str, float]:
    """
    Shift base_next_state by causal ATEs for each active drug.

    For every drug with value=1 in action_dict, look up all (drug, outcome)
    pairs in ate_table and add the causal_ate to the corresponding outcome.
    Outcomes not covered by the ATE table are left unchanged.
    Drug interactions are assumed additive (independent ATEs summed).
    """
    corrected = dict(base_next_state)
    for drug, val in action_dict.items():
        if val == 1:
            for (treatment, outcome), ate in ate_table.items():
                if treatment == drug and outcome in corrected:
                    corrected[outcome] = corrected[outcome] + ate
    return corrected


def causal_exhaustive_policy(
    state_dict: dict[str, float],
    transition_model: TransitionModel,
    readmission_model: ReadmissionModel,
    ate_table: dict[tuple[str, str], float],
    optimise_drugs: list[str] = ATE_DRUGS,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """
    Evaluate all 2^5 = 32 drug combinations for drugs with causal ATEs.

    Returns:
      - best_action: dict of drug flags (lowest predicted readmission risk)
      - best_risk: float, predicted P(readmit) under best action
      - all_risks: dict mapping action_key → risk (32 entries)
    """
    # Single baseline: transition model with no drugs prescribed
    no_drug_action = {col: 0 for col in ACTION_COLS}
    base_next, _ = predict_next(transition_model, state_dict, no_drug_action)

    all_risks: dict[str, float] = {}
    best_action: dict[str, float] = {}
    best_risk = float("inf")

    for combo in itertools.product([0, 1], repeat=len(optimise_drugs)):
        action_dict = dict(zip(optimise_drugs, combo))
        action_key = "_".join(str(v) for v in combo)

        causal_next = apply_ate_corrections(base_next, action_dict, ate_table)
        risk = float(predict_readmission_risk(readmission_model, causal_next))

        all_risks[action_key] = risk
        if risk < best_risk:
            best_risk = risk
            best_action = dict(action_dict)

    return best_action, best_risk, all_risks
