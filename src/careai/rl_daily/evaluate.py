"""Compare causal policy vs do-nothing vs real clinical actions.

All three strategies use the same causal framework for consistency:
  - base_next = predict_next(transition_model, state, no_drugs)  [computed once]
  - Policy:     apply_*_corrections(base_next, best_action)
  - Do-nothing: base_next directly (no corrections — no drugs prescribed)
  - Real:       apply_*_corrections(base_next, actual_drugs_from_data)

This ensures that differences in predicted risk are driven purely by the
causal drug effects, not by the transition model's observational confounding.

Supports two policy types via keyword arguments:
  - ate_table     (dict) → uses ATE corrections (population-level, Option C)
  - cate_registry (CATERegistry) → uses CATE corrections (patient-level, Option D)
Exactly one of the two must be provided (raises ValueError otherwise).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from careai.sim_daily.features import ACTION_COLS
from careai.sim_daily.transition import TransitionModel, predict_next

from .policy import ATE_DRUGS, apply_ate_corrections, causal_exhaustive_policy
from .readmission import ReadmissionModel, predict_readmission_risk


def evaluate_policy(
    initial_states: pd.DataFrame,
    transition_model: TransitionModel,
    readmission_model: ReadmissionModel,
    ate_table: dict[tuple[str, str], float] | None = None,
    n_patients: int = 500,
    seed: int = 42,
    cate_registry=None,
) -> pd.DataFrame:
    """
    For n_patients sampled from initial_states, compare 3 strategies:

    1. Causal policy: exhaustive action search → best action over ATE_DRUGS
    2. Do-nothing: base_next with no corrections (no drugs)
    3. Real actions: corrections using the patient's actual drug flags

    Exactly one of ``ate_table`` or ``cate_registry`` must be provided.

    Returns DataFrame with columns:
      hadm_id,
      policy_risk, donothing_risk, real_risk,
      policy_<drug>  (5 binary cols — recommended flags for ATE_DRUGS)
    """
    if ate_table is None and cate_registry is None:
        raise ValueError("Provide exactly one of ate_table or cate_registry.")
    if ate_table is not None and cate_registry is not None:
        raise ValueError("Provide exactly one of ate_table or cate_registry, not both.")

    use_cate = cate_registry is not None

    if use_cate:
        from .policy_cate import apply_cate_corrections, cate_exhaustive_policy

    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    no_drug_action = {col: 0 for col in ACTION_COLS}

    records = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 50 == 0:
            print(f"  [{i}/{n}] evaluating patients ...", flush=True)
        state_dict = row.to_dict()

        # Baseline: no drugs (computed once, shared by all three strategies)
        base_next, _ = predict_next(transition_model, state_dict, no_drug_action)

        # 1. Causal policy
        if use_cate:
            best_action, policy_risk, _ = cate_exhaustive_policy(
                state_dict, transition_model, readmission_model, cate_registry
            )
        else:
            best_action, policy_risk, _ = causal_exhaustive_policy(
                state_dict, transition_model, readmission_model, ate_table
            )

        # 2. Do-nothing: base_next with no corrections
        donothing_risk = float(predict_readmission_risk(readmission_model, base_next))

        # 3. Real actions: corrections using actual drug flags
        real_action = {col: float(row.get(col, 0)) for col in ATE_DRUGS}
        if use_cate:
            real_next = apply_cate_corrections(base_next, real_action, cate_registry, state_dict)
        else:
            real_next = apply_ate_corrections(base_next, real_action, ate_table)
        real_risk = float(predict_readmission_risk(readmission_model, real_next))

        rec: dict = {
            "hadm_id": row.get("hadm_id"),
            "policy_risk": policy_risk,
            "donothing_risk": donothing_risk,
            "real_risk": real_risk,
        }
        for drug in ATE_DRUGS:
            rec[f"policy_{drug}"] = int(best_action.get(drug, 0))

        records.append(rec)

    return pd.DataFrame(records)


def print_policy_summary(results: pd.DataFrame) -> None:
    """Print summary statistics comparing the three strategies."""
    print("\n=== Causal Policy Evaluation Summary ===\n")

    print("Mean predicted readmission risk (causally corrected next-state):")
    print(f"  Causal policy:  {results['policy_risk'].mean():.4f}")
    print(f"  Do-nothing:     {results['donothing_risk'].mean():.4f}")
    print(f"  Real clinical:  {results['real_risk'].mean():.4f}")

    print("\nDrug recommendation frequency (policy, over ATE_DRUGS only):")
    drug_cols = [c for c in results.columns if c.startswith("policy_") and c != "policy_risk"]
    for col in drug_cols:
        drug_name = col[len("policy_"):]
        freq = results[col].mean()
        print(f"  {drug_name:30s}: {freq:.1%}")

    n = len(results)
    beats_dn = (results["policy_risk"] < results["donothing_risk"]).sum()
    beats_real = (results["policy_risk"] < results["real_risk"]).sum()
    print(f"\nCausal policy beats do-nothing:    {beats_dn}/{n} ({beats_dn/n:.1%}) patients")
    print(f"Causal policy beats real actions:  {beats_real}/{n} ({beats_real/n:.1%}) patients")
