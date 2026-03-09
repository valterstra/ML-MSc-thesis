"""CATE-based exhaustive policy — patient-specific treatment effect corrections.

Mirrors policy.py (Option C / ATE), but replaces the scalar ATE lookup with a
per-patient CATE prediction from CausalForestDML.  Both policies share the
same additive independence assumption and the same 2^5 action search space.

The two policies are methodologically comparable:
  ATE policy:  causal_next[outcome] = base_next[outcome] + ATE(drug, outcome)
  CATE policy: causal_next[outcome] = base_next[outcome] + CATE(drug, outcome, state)

Performance note:
  CATEs are precomputed once per patient (9 effect() calls) before the 32-combo
  loop, then reused as a plain dict lookup.  This reduces forest inference calls
  from 32 x 9 = 288 to 9 per patient — a 32x speedup.

Output format is identical (same tuple signature), enabling direct comparison.
"""

from __future__ import annotations

import itertools

from careai.sim_daily.features import ACTION_COLS
from careai.sim_daily.transition import TransitionModel, predict_next

from .policy import ATE_DRUGS  # reuse — no opioid, same 5 drugs
from .readmission import ReadmissionModel, predict_readmission_risk
from careai.causal_daily.cate import CATERegistry, predict_cate


def _precompute_patient_cates(
    state_dict: dict[str, float],
    cate_registry: CATERegistry,
) -> dict[tuple[str, str], float]:
    """Call effect() once per fitted (treatment, outcome) pair for this patient.

    Returns a plain dict that the combo loop can use with zero-cost lookups,
    avoiding redundant forest inference across the 32 action combinations.
    """
    return {
        (treatment, outcome): predict_cate(cate_registry, treatment, outcome, state_dict)
        for (treatment, outcome) in cate_registry.models
    }


def _apply_precomputed_cates(
    base_next_state: dict[str, float],
    action_dict: dict[str, int],
    patient_cates: dict[tuple[str, str], float],
) -> dict[str, float]:
    """Shift base_next_state using precomputed per-patient CATEs (dict lookup only)."""
    corrected = dict(base_next_state)
    for drug, val in action_dict.items():
        if val == 1:
            for (treatment, outcome), delta in patient_cates.items():
                if treatment == drug and outcome in corrected:
                    corrected[outcome] = corrected[outcome] + delta
    return corrected


def apply_cate_corrections(
    base_next_state: dict[str, float],
    action_dict: dict[str, int],
    cate_registry: CATERegistry,
    state_dict: dict[str, float],
) -> dict[str, float]:
    """Shift base_next_state by per-patient CATEs for each active drug.

    Used for the "real actions" baseline in evaluate.py (called once per patient,
    so no batching needed here). For the exhaustive 32-combo search, use
    cate_exhaustive_policy() which precomputes CATEs before the loop.

    Parameters
    ----------
    base_next_state:
        Baseline next-state from the transition model with no drugs.
    action_dict:
        {drug: 0 or 1} for the candidate action combination.
    cate_registry:
        Fitted CATERegistry.
    state_dict:
        Current patient state (used by predict_cate for personalisation).

    Returns
    -------
    dict[str, float] — causally-corrected next-state.
    """
    patient_cates = _precompute_patient_cates(state_dict, cate_registry)
    return _apply_precomputed_cates(base_next_state, action_dict, patient_cates)


def cate_exhaustive_policy(
    state_dict: dict[str, float],
    transition_model: TransitionModel,
    readmission_model: ReadmissionModel,
    cate_registry: CATERegistry,
    optimise_drugs: list[str] = ATE_DRUGS,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Evaluate all 2^5 = 32 drug combinations using patient-specific CATEs.

    Algorithm:
      1. Compute a single baseline next-state with no drugs.
      2. Precompute all per-patient CATEs once (9 effect() calls total).
      3. For each of the 32 action combinations, shift the baseline using the
         precomputed CATE dict (plain dict lookup — no forest calls in the loop).
      4. Score the corrected next-state with the readmission model.
      5. Return the action with the lowest predicted readmission risk.

    Parameters
    ----------
    state_dict:
        Current patient state as a flat dict {col: value}.
    transition_model:
        Fitted TransitionModel (LightGBM per-output).
    readmission_model:
        Fitted readmission LightGBM classifier.
    cate_registry:
        Fitted CATERegistry with per-patient CATE estimators.
    optimise_drugs:
        Drugs to optimise over (default: ATE_DRUGS — no opioid).

    Returns
    -------
    best_action : dict[str, int]
        Drug flag assignment with lowest predicted readmission risk.
    best_risk : float
        Predicted P(readmit_30d=1) under best_action.
    all_risks : dict[str, float]
        Mapping from action_key (e.g. ``"10100"``) to risk — 32 entries.
    """
    # Baseline: transition model with no drugs
    no_drug_action = {col: 0 for col in ACTION_COLS}
    base_next, _ = predict_next(transition_model, state_dict, no_drug_action)

    # Precompute all CATEs for this patient ONCE (9 calls, not 32x9=288)
    patient_cates = _precompute_patient_cates(state_dict, cate_registry)

    all_risks: dict[str, float] = {}
    best_action: dict[str, int] = {}
    best_risk = float("inf")

    for combo in itertools.product([0, 1], repeat=len(optimise_drugs)):
        action_dict = dict(zip(optimise_drugs, combo))
        action_key = "_".join(str(v) for v in combo)

        causal_next = _apply_precomputed_cates(base_next, action_dict, patient_cates)
        risk = float(predict_readmission_risk(readmission_model, causal_next))

        all_risks[action_key] = risk
        if risk < best_risk:
            best_risk = risk
            best_action = dict(action_dict)

    return best_action, best_risk, all_risks
