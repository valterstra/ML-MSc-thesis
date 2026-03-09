#!/usr/bin/env python
"""Train a Fitted Q-Iteration (FQI) RL agent on the daily hospital simulator.

Pipeline:
  1. Load hosp_daily CSV, transition model, readmission model, ATE table.
  2. Collect trajectories: enumerate all 2^3 action sequences per patient.
  3. Run FQI for N iterations (LightGBM Q-model per RL step).
  4. Evaluate FQI policy vs do-nothing vs real clinical actions on test set.
  5. Save Q-models to models/fqi_daily/ and report to reports/rl_daily_full/fqi/.

Usage:
  python scripts/rl/train_fqi_agent.py \\
    --csv data/processed/hosp_daily_transitions.csv \\
    --transition-model-dir models/sim_daily_full \\
    --readmission-model-dir models/rl_daily_full \\
    --model-dir models/fqi_daily \\
    --n-patients 5000 \\
    --n-iter 10
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
from careai.rl_daily.fqi import (
    FittedQIteration,
    RL_STATE_COLS,
    FQI_DRUG,
    N_RL_STEPS,
    SIM_STEPS_PER_RL,
    collect_trajectories,
    _rl_step,
    _extract_rl_state,
)


# ---------------------------------------------------------------------------
# Policy evaluation
# ---------------------------------------------------------------------------

def evaluate_fqi_policy(
    fqi: FittedQIteration,
    initial_states: pd.DataFrame,
    transition_model,
    readmission_model,
    ate_table,
    n_patients: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare FQI policy vs do-nothing vs real clinical on terminal risk.

    For each patient, rolls out N_RL_STEPS RL steps under three strategies:
      - FQI:        argmax_a Q(s_t, a) at each step
      - Do-nothing: always a=0 (no antibiotic)
      - Real:       actual antibiotic flag from day-0 data, held constant

    Returns DataFrame with columns:
      hadm_id, fqi_risk, donothing_risk, real_risk, fqi_action_step{t}.
    """
    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    records = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 100 == 0:
            print(f"  Evaluating FQI policy: [{i}/{n}] ...", flush=True)

        s0 = row.to_dict()
        hadm_id = row.get("hadm_id")
        real_ab = int(row.get(FQI_DRUG, 0))

        # 1. FQI policy rollout
        s = s0
        fqi_actions = []
        for step in range(N_RL_STEPS):
            a = fqi.best_action(s, step=step)
            fqi_actions.append(a)
            s = _rl_step(s, a, transition_model, ate_table)
        fqi_risk = float(predict_readmission_risk(readmission_model, s))

        # 2. Do-nothing rollout (always a=0)
        s = s0
        for step in range(N_RL_STEPS):
            s = _rl_step(s, 0, transition_model, ate_table)
        donothing_risk = float(predict_readmission_risk(readmission_model, s))

        # 3. Real clinical rollout (actual day-0 antibiotic flag, held constant)
        s = s0
        for step in range(N_RL_STEPS):
            s = _rl_step(s, real_ab, transition_model, ate_table)
        real_risk = float(predict_readmission_risk(readmission_model, s))

        rec: dict = {
            "hadm_id": hadm_id,
            "fqi_risk": fqi_risk,
            "donothing_risk": donothing_risk,
            "real_risk": real_risk,
        }
        for t, a in enumerate(fqi_actions):
            rec[f"fqi_action_step{t}"] = a

        records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_fqi_summary(results: pd.DataFrame, fqi: FittedQIteration) -> None:
    """Print summary statistics and Q-function feature importances."""
    print("\n=== FQI Policy Evaluation Summary ===\n")
    print(f"Episode structure: {N_RL_STEPS} RL steps x {SIM_STEPS_PER_RL} sim days each")
    print(f"Drug optimised:    {FQI_DRUG}\n")

    print("Mean terminal readmission risk (after {}-step rollout):".format(N_RL_STEPS))
    print(f"  FQI policy:    {results['fqi_risk'].mean():.4f}")
    print(f"  Do-nothing:    {results['donothing_risk'].mean():.4f}")
    print(f"  Real clinical: {results['real_risk'].mean():.4f}")

    n = len(results)
    beats_dn = (results["fqi_risk"] < results["donothing_risk"]).sum()
    beats_real = (results["fqi_risk"] < results["real_risk"]).sum()
    print(f"\nFQI beats do-nothing:   {beats_dn}/{n} ({beats_dn/n:.1%}) patients")
    print(f"FQI beats real actions: {beats_real}/{n} ({beats_real/n:.1%}) patients")

    print("\nAntibiotic recommendation frequency per step:")
    for t in range(N_RL_STEPS):
        col = f"fqi_action_step{t}"
        if col in results.columns:
            freq = results[col].mean()
            print(f"  Step {t} (day {t * SIM_STEPS_PER_RL}): {freq:.1%} recommend antibiotic")

    print("\nQ-function feature importances:")
    imps = fqi.feature_importances()
    for step in sorted(imps):
        imp = imps[step]
        max_score = max(imp.values()) if imp else 1.0
        print(f"\n  Step {step} (decision at day {step * SIM_STEPS_PER_RL}):")
        for feat, score in sorted(imp.items(), key=lambda x: -x[1]):
            bar = "#" * int(score / max_score * 20) if max_score > 0 else ""
            print(f"    {feat:35s}: {score:6.0f}  {bar}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train FQI RL agent on daily hospital simulator"
    )
    parser.add_argument(
        "--csv",
        default=str(_PROJECT_ROOT / "data" / "processed" / "hosp_daily_transitions.csv"),
        help="Path to hosp_daily CSV",
    )
    parser.add_argument(
        "--transition-model-dir",
        default=str(_PROJECT_ROOT / "models" / "sim_daily_full"),
        help="Transition model directory",
    )
    parser.add_argument(
        "--readmission-model-dir",
        default=str(_PROJECT_ROOT / "models" / "rl_daily_full"),
        help="Readmission model directory",
    )
    parser.add_argument(
        "--ate-json",
        default=str(
            _PROJECT_ROOT / "reports" / "causal_daily_full" / "treatment_effects.json"
        ),
        help="Path to treatment_effects.json from causal step",
    )
    parser.add_argument(
        "--model-dir",
        default=str(_PROJECT_ROOT / "models" / "fqi_daily"),
        help="Directory to save FQI Q-models",
    )
    parser.add_argument(
        "--report-dir",
        default=str(_PROJECT_ROOT / "reports" / "rl_daily_full" / "fqi"),
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--n-patients",
        type=int,
        default=5000,
        help="Number of train patients for trajectory collection",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=10,
        help="Number of FQI iterations",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor (default 0.99 = near-undiscounted)",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

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
    n_seqs = 2 ** N_RL_STEPS
    n_episodes = n_train * n_seqs
    print(
        f"\nCollecting trajectories: {n_train:,} patients x {n_seqs} action sequences "
        f"= {n_episodes:,} episodes ..."
    )

    trajectories = collect_trajectories(
        transition_model=transition_model,
        ate_table=ate_table,
        readmission_model=readmission_model,
        initial_states=train_initial,
        n_patients=n_train,
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
        f"\nRunning FQI for {args.n_iter} iterations "
        f"(gamma={args.gamma}, {N_RL_STEPS} steps, 1 Q-model per step) ..."
    )
    fqi = FittedQIteration()
    fqi.fit(trajectories, n_iter=args.n_iter, gamma=args.gamma)

    model_dir = Path(args.model_dir)
    fqi.save(model_dir)
    print(f"FQI Q-models saved to {model_dir}")

    # ------------------------------------------------------------------
    # 5. Policy evaluation
    # ------------------------------------------------------------------
    n_eval = min(500, len(test_initial))
    print(f"\nEvaluating FQI policy on {n_eval} test patients ...")

    eval_results = evaluate_fqi_policy(
        fqi=fqi,
        initial_states=test_initial,
        transition_model=transition_model,
        readmission_model=readmission_model,
        ate_table=ate_table,
        n_patients=n_eval,
        seed=args.seed,
    )

    print_fqi_summary(eval_results, fqi)

    # Save evaluation outputs
    eval_csv = report_dir / "policy_evaluation.csv"
    eval_results.to_csv(eval_csv, index=False)
    print(f"\nEvaluation saved to {eval_csv}  ({len(eval_results)} rows)")

    imps = fqi.feature_importances()
    summary = {
        "algorithm": "FittedQIteration",
        "n_iter": args.n_iter,
        "gamma": args.gamma,
        "n_patients_train": n_train,
        "n_episodes": n_episodes,
        "n_transitions": n_transitions,
        "rl_state_cols": RL_STATE_COLS,
        "fqi_drug": FQI_DRUG,
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
        "antibiotic_recommendation_freq": {
            f"step{t}": float(eval_results[f"fqi_action_step{t}"].mean())
            for t in range(N_RL_STEPS)
        },
        "feature_importances": {
            f"step{step}": imp for step, imp in imps.items()
        },
    }
    json_path = report_dir / "policy_evaluation.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary JSON saved to {json_path}")


if __name__ == "__main__":
    main()
