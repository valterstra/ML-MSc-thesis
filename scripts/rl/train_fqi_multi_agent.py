#!/usr/bin/env python
"""Train an extended Fitted Q-Iteration (FQI) agent: 5 drugs, 26-feature state.

This extends train_fqi_agent.py by:
  - Optimising all 5 causal drugs jointly (antibiotic, anticoagulant, diuretic,
    insulin, steroid) instead of antibiotic alone.
  - Using the full 26-feature patient state (15 labs + 4 binary + 7 static)
    instead of 7 infection-focused features.
  - Sampling random action sequences instead of enumerating all paths
    (32^3 = 32,768 paths per patient is infeasible; 64 samples give ~2x
    coverage of each of the 32 combos per step in expectation).

Pipeline:
  1. Load hosp_daily CSV, transition model, readmission model, ATE table.
  2. Collect trajectories: sample 64 random 5-drug sequences per patient.
  3. Run FQI for N iterations (LightGBM Q-model per RL step).
  4. Evaluate FQI-multi policy vs do-nothing vs real clinical on test set.
  5. Save Q-models to models/fqi_multi/ and report to reports/rl_daily_full/fqi_multi/.

Usage:
  python scripts/rl/train_fqi_multi_agent.py
  python scripts/rl/train_fqi_multi_agent.py --n-patients 3000 --n-seqs 64 --n-iter 10
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

from careai.sim_daily.data import prepare_daily_data
from careai.sim_daily.transition import load_model
from careai.rl_daily.readmission import load_readmission_model, predict_readmission_risk
from careai.rl_daily.policy import load_ate_table
from careai.rl_daily.fqi_multi import (
    FittedQIterationMulti,
    FQI_DRUGS_MULTI,
    RL_STATE_COLS_MULTI,
    N_RL_STEPS,
    SIM_STEPS_PER_RL,
    N_SEQS,
    collect_trajectories_multi,
    _rl_step_multi,
)


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_fqi_multi_policy(
    fqi: FittedQIterationMulti,
    initial_states: pd.DataFrame,
    transition_model,
    readmission_model,
    ate_table,
    n_patients: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare FQI-multi policy vs do-nothing vs real clinical on terminal risk.

    For each patient, rolls out N_RL_STEPS RL steps under three strategies:
      - FQI-multi:  argmax_combo Q(s_t, combo) at each step (5-drug joint policy)
      - Do-nothing: all drugs = 0 at every step
      - Real:       actual 5-drug flags from day-0 data, held constant

    Returns DataFrame with columns:
      hadm_id, fqi_risk, donothing_risk, real_risk,
      {drug}_fqi_step{t} for each drug and step.
    """
    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    no_drug_action = {d: 0 for d in FQI_DRUGS_MULTI}

    records = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 25 == 0:
            print(f"  Evaluating FQI-multi policy: [{i}/{n}] ...", flush=True)

        s0 = row.to_dict()
        hadm_id = row.get("hadm_id")
        real_action = {d: int(row.get(d, 0)) for d in FQI_DRUGS_MULTI}

        # 1. FQI-multi policy rollout
        s = s0
        fqi_actions_per_step = []
        for step in range(N_RL_STEPS):
            combo_dict = fqi.best_combo(s, step=step)
            fqi_actions_per_step.append(combo_dict)
            s = _rl_step_multi(s, combo_dict, transition_model, ate_table)
        fqi_risk = float(predict_readmission_risk(readmission_model, s))

        # 2. Do-nothing rollout (all drugs = 0 at every step)
        s = s0
        for _ in range(N_RL_STEPS):
            s = _rl_step_multi(s, no_drug_action, transition_model, ate_table)
        donothing_risk = float(predict_readmission_risk(readmission_model, s))

        # 3. Real clinical rollout (actual day-0 drug flags, held constant)
        s = s0
        for _ in range(N_RL_STEPS):
            s = _rl_step_multi(s, real_action, transition_model, ate_table)
        real_risk = float(predict_readmission_risk(readmission_model, s))

        rec: dict = {
            "hadm_id": hadm_id,
            "fqi_risk": fqi_risk,
            "donothing_risk": donothing_risk,
            "real_risk": real_risk,
        }
        for t, combo_dict in enumerate(fqi_actions_per_step):
            for drug, val in combo_dict.items():
                rec[f"{drug}_fqi_step{t}"] = val

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_fqi_multi_summary(results: pd.DataFrame, fqi: FittedQIterationMulti) -> None:
    """Print summary statistics and Q-function feature importances."""
    print("\n=== FQI-Multi Policy Evaluation Summary ===\n")
    print(f"Episode structure: {N_RL_STEPS} RL steps x {SIM_STEPS_PER_RL} sim days each")
    print(f"Drugs optimised:   {', '.join(FQI_DRUGS_MULTI)}\n")

    print(f"Mean terminal readmission risk (after {N_RL_STEPS}-step rollout):")
    print(f"  FQI-multi policy: {results['fqi_risk'].mean():.4f}")
    print(f"  Do-nothing:       {results['donothing_risk'].mean():.4f}")
    print(f"  Real clinical:    {results['real_risk'].mean():.4f}")

    n = len(results)
    beats_dn = (results["fqi_risk"] < results["donothing_risk"]).sum()
    beats_real = (results["fqi_risk"] < results["real_risk"]).sum()
    print(f"\nFQI-multi beats do-nothing:   {beats_dn}/{n} ({beats_dn/n:.1%}) patients")
    print(f"FQI-multi beats real actions: {beats_real}/{n} ({beats_real/n:.1%}) patients")

    print("\nDrug recommendation frequency per step:")
    for drug in FQI_DRUGS_MULTI:
        freqs = []
        for t in range(N_RL_STEPS):
            col = f"{drug}_fqi_step{t}"
            if col in results.columns:
                freqs.append(f"step{t}={results[col].mean():.1%}")
        print(f"  {drug:<28}: {', '.join(freqs)}")

    print("\nQ-function feature importances (top 10 per step):")
    imps = fqi.feature_importances()
    for step in sorted(imps):
        imp = imps[step]
        sorted_feats = sorted(imp.items(), key=lambda x: -x[1])[:10]
        max_score = sorted_feats[0][1] if sorted_feats else 1.0
        print(f"\n  Step {step} (decision at day {step * SIM_STEPS_PER_RL}):")
        for feat, score in sorted_feats:
            bar = "#" * int(score / max_score * 20) if max_score > 0 else ""
            print(f"    {feat:40s}: {score:6.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train FQI-multi RL agent: 5 drugs, 26-feature state"
    )
    p.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions.csv"),
        help="Path to hosp_daily CSV",
    )
    p.add_argument(
        "--transition-model-dir",
        default=str(_PROJECT_ROOT / "models" / "sim_daily_full"),
        help="Transition model directory",
    )
    p.add_argument(
        "--readmission-model-dir",
        default=str(_PROJECT_ROOT / "models" / "rl_daily_full"),
        help="Readmission model directory",
    )
    p.add_argument(
        "--ate-json",
        default=str(
            _PROJECT_ROOT / "reports" / "causal_daily_full" / "treatment_effects.json"
        ),
        help="Path to treatment_effects.json from causal step",
    )
    p.add_argument(
        "--model-dir",
        default=str(_PROJECT_ROOT / "models" / "fqi_multi"),
        help="Directory to save FQI-multi Q-models",
    )
    p.add_argument(
        "--report-dir",
        default=str(_PROJECT_ROOT / "reports" / "rl_daily_full" / "fqi_multi"),
        help="Directory to save evaluation outputs",
    )
    p.add_argument(
        "--n-patients",
        type=int,
        default=3000,
        help="Number of train patients for trajectory collection (default 3000)",
    )
    p.add_argument(
        "--n-seqs",
        type=int,
        default=N_SEQS,
        help=f"Random action sequences per patient (default {N_SEQS})",
    )
    p.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of FQI iterations (default 10)",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default 0.99)",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    print(f"Loading data from {args.csv} ...")
    data = prepare_daily_data(args.csv)

    train_initial = data.initial_states[data.initial_states["split"] == "train"].copy()
    test_initial = data.initial_states[data.initial_states["split"] == "test"].copy()
    print(f"  Train initial states: {len(train_initial):,}")
    print(f"  Test  initial states: {len(test_initial):,}")

    # ------------------------------------------------------------------
    # 2. Load models
    # ------------------------------------------------------------------
    print(f"\nLoading transition model from {args.transition_model_dir} ...")
    transition_model = load_model(args.transition_model_dir)

    print(f"Loading readmission model from {args.readmission_model_dir} ...")
    readmission_model = load_readmission_model(args.readmission_model_dir)
    print(f"  AUC: {readmission_model.test_auc:.4f}  Brier: {readmission_model.test_brier:.4f}")

    print(f"Loading ATE table from {args.ate_json} ...")
    ate_table = load_ate_table(args.ate_json)
    print(f"  (treatment, outcome) pairs: {len(ate_table)}")

    # ------------------------------------------------------------------
    # 3. Collect trajectories
    # ------------------------------------------------------------------
    n_train = min(args.n_patients, len(train_initial))
    n_episodes = n_train * args.n_seqs
    print(
        f"\nCollecting trajectories: {n_train:,} patients x {args.n_seqs} sequences "
        f"= {n_episodes:,} episodes ..."
    )
    print(f"  (5 drugs, {N_RL_STEPS} RL steps, {SIM_STEPS_PER_RL} sim days/step)\n")

    trajectories = collect_trajectories_multi(
        transition_model=transition_model,
        ate_table=ate_table,
        readmission_model=readmission_model,
        initial_states=train_initial,
        n_patients=n_train,
        n_seqs=args.n_seqs,
        seed=args.seed,
    )
    n_transitions = len(trajectories)
    terminal_rows = trajectories[trajectories["done"] == 1]
    print(f"  Collected {n_episodes:,} episodes, {n_transitions:,} transitions")
    print(f"  Mean terminal reward: {terminal_rows['reward'].mean():.4f}")
    print(f"  Std terminal reward:  {terminal_rows['reward'].std():.4f}")

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    traj_path = report_dir / "trajectories.csv"
    trajectories.to_csv(traj_path, index=False)
    print(f"  Trajectories saved to {traj_path}")

    # ------------------------------------------------------------------
    # 4. FQI training
    # ------------------------------------------------------------------
    print(
        f"\nRunning FQI-multi for {args.n_iter} iterations "
        f"(gamma={args.gamma}, {N_RL_STEPS} steps, 1 Q-model per step) ..."
    )
    fqi = FittedQIterationMulti()
    fqi.fit(trajectories, n_iter=args.n_iter, gamma=args.gamma)

    model_dir = Path(args.model_dir)
    fqi.save(model_dir)
    print(f"FQI-multi Q-models saved to {model_dir}")

    # ------------------------------------------------------------------
    # 5. Policy evaluation
    # ------------------------------------------------------------------
    n_eval = min(500, len(test_initial))
    print(f"\nEvaluating FQI-multi policy on {n_eval} test patients ...")

    eval_results = evaluate_fqi_multi_policy(
        fqi=fqi,
        initial_states=test_initial,
        transition_model=transition_model,
        readmission_model=readmission_model,
        ate_table=ate_table,
        n_patients=n_eval,
        seed=args.seed,
    )

    print_fqi_multi_summary(eval_results, fqi)

    # Save evaluation outputs
    eval_csv = report_dir / "policy_evaluation.csv"
    eval_results.to_csv(eval_csv, index=False)
    print(f"\nEvaluation saved to {eval_csv}  ({len(eval_results)} rows)")

    imps = fqi.feature_importances()
    drug_freq = {}
    for drug in FQI_DRUGS_MULTI:
        drug_freq[drug] = {
            f"step{t}": float(eval_results[f"{drug}_fqi_step{t}"].mean())
            for t in range(N_RL_STEPS)
            if f"{drug}_fqi_step{t}" in eval_results.columns
        }

    summary = {
        "algorithm": "FittedQIterationMulti",
        "n_iter": args.n_iter,
        "gamma": args.gamma,
        "n_patients_train": n_train,
        "n_seqs_per_patient": args.n_seqs,
        "n_episodes": n_episodes,
        "n_transitions": n_transitions,
        "rl_state_cols": RL_STATE_COLS_MULTI,
        "fqi_drugs": FQI_DRUGS_MULTI,
        "n_rl_steps": N_RL_STEPS,
        "sim_steps_per_rl": SIM_STEPS_PER_RL,
        "n_eval_patients": len(eval_results),
        "mean_fqi_risk": float(eval_results["fqi_risk"].mean()),
        "mean_donothing_risk": float(eval_results["donothing_risk"].mean()),
        "mean_real_risk": float(eval_results["real_risk"].mean()),
        "fqi_beats_donothing_frac": float(
            (eval_results["fqi_risk"] < eval_results["donothing_risk"]).mean()
        ),
        "fqi_beats_real_frac": float(
            (eval_results["fqi_risk"] < eval_results["real_risk"]).mean()
        ),
        "drug_recommendation_freq": drug_freq,
        "feature_importances": {
            f"step{step}": {k: int(v) for k, v in imp.items()}
            for step, imp in imps.items()
        },
    }
    json_path = report_dir / "policy_evaluation.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON saved to {json_path}")


if __name__ == "__main__":
    main()
