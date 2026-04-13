"""Step D: Train V3 causal FQI reinforcement learning agent.

Trains a 3-step Fitted Q-Iteration (FQI) agent that recommends 5-drug
combinations using the V3 causally-constrained structural equation simulator
and dense intermediary rewards.

Key differences from V1 FQI-multi (scripts/rl/train_fqi_multi_agent.py):
  - Simulator: V3 structural equations (14 LightGBM with causal parent sets)
    instead of V1 transition model + post-hoc ATE corrections.
  - Rewards: Dense at every step (readmission risk + lab stability + ICU penalty)
    instead of sparse terminal-only reward.
  - State: 16 features (14 state vars + 2 static) instead of 26.

Pipeline:
  1. Load V3 triplet data and extract initial states (first triplet per admission).
  2. Load V3 structural equation models (from Step C).
  3. Load readmission model (reused from V1 pipeline).
  4. Collect trajectories: sample random action sequences, roll out with dense rewards.
  5. Train FQI via backward induction (LightGBM Q-model per step, 10 iterations).
  6. Evaluate: FQI-V3 vs do-nothing vs real clinical actions.
  7. Save models and reports.

Usage:
    # Smoke test (500 patients, 16 sequences)
    python scripts/causal_v3/step_d_fqi_agent.py --sample-n 500 --n-seqs 16

    # Full run
    python scripts/causal_v3/step_d_fqi_agent.py --n-patients 5000 --n-seqs 64
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

log = logging.getLogger(__name__)


# ── Policy Evaluation ─────────────────────────────────────────────────────

def evaluate_fqi_v3_policy(
    fqi,
    initial_states: pd.DataFrame,
    models: dict,
    parent_sets: dict[str, list[str]],
    readmission_model,
    n_patients: int = 500,
    seed: int = 42,
) -> pd.DataFrame:
    """Compare FQI-V3 vs do-nothing vs real clinical on terminal state.

    For each patient, rolls out 3 RL steps (6 sim days) under three strategies:
      - FQI-V3:     argmax_combo Q(s, combo) at each step
      - Do-nothing:  all drugs = 0 at every step
      - Real:        actual drug flags from day-0, held constant

    Returns DataFrame with per-patient risks and FQI drug recommendations.
    """
    from careai.causal_v3.fqi_v3 import (
        FQI_DRUGS_V3, N_RL_STEPS, SIM_STEPS_PER_RL,
        _single_rl_step_v3, compute_lab_delta_reward,
    )
    from careai.rl_daily.readmission import predict_readmission_risk

    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    no_drug_action = {d: 0 for d in FQI_DRUGS_V3}

    records = []
    for i, (_, row) in enumerate(sample.iterrows()):
        if i % 50 == 0:
            print(
                "  Evaluating: [%d/%d] ..." % (i, n),
                flush=True,
            )

        s0 = row.to_dict()
        hadm_id = row.get("hadm_id")
        real_action = {d: int(row.get(d, 0)) for d in FQI_DRUGS_V3}

        # 1. FQI-V3 policy rollout
        s = dict(s0)
        fqi_actions_per_step = []
        fqi_step_rewards = []
        for step in range(N_RL_STEPS):
            combo_dict = fqi.best_combo(s, step=step)
            fqi_actions_per_step.append(combo_dict)
            prev_s = dict(s)
            s = _single_rl_step_v3(s, combo_dict, models, parent_sets)
            fqi_step_rewards.append(compute_lab_delta_reward(prev_s, s))
        # Readmission risk at terminal state (evaluation metric only)
        fqi_risk = float(predict_readmission_risk(readmission_model, s))

        # 2. Do-nothing rollout
        s = dict(s0)
        dn_step_rewards = []
        for step in range(N_RL_STEPS):
            prev_s = dict(s)
            s = _single_rl_step_v3(s, no_drug_action, models, parent_sets)
            dn_step_rewards.append(compute_lab_delta_reward(prev_s, s))
        donothing_risk = float(predict_readmission_risk(readmission_model, s))

        # 3. Real clinical rollout
        s = dict(s0)
        real_step_rewards = []
        for step in range(N_RL_STEPS):
            prev_s = dict(s)
            s = _single_rl_step_v3(s, real_action, models, parent_sets)
            real_step_rewards.append(compute_lab_delta_reward(prev_s, s))
        real_risk = float(predict_readmission_risk(readmission_model, s))

        rec: dict = {
            "hadm_id": hadm_id,
            "fqi_risk": fqi_risk,
            "donothing_risk": donothing_risk,
            "real_risk": real_risk,
            "fqi_cumulative_reward": sum(fqi_step_rewards),
            "donothing_cumulative_reward": sum(dn_step_rewards),
            "real_cumulative_reward": sum(real_step_rewards),
        }
        # Per-step FQI actions
        for t, combo_dict in enumerate(fqi_actions_per_step):
            for drug, val in combo_dict.items():
                rec["%s_fqi_step%d" % (drug, t)] = val
        # Per-step rewards for analysis
        for t in range(N_RL_STEPS):
            rec["fqi_reward_step%d" % t] = fqi_step_rewards[t]
            rec["dn_reward_step%d" % t] = dn_step_rewards[t]

        records.append(rec)

    return pd.DataFrame(records)


# ── Summary Printing ──────────────────────────────────────────────────────

def print_summary(results: pd.DataFrame, fqi) -> None:
    """Print evaluation summary and feature importances."""
    from careai.causal_v3.fqi_v3 import (
        FQI_DRUGS_V3, N_RL_STEPS, SIM_STEPS_PER_RL, LAB_WEIGHTS,
    )

    n = len(results)

    print("")
    print("=" * 70)
    print("FQI-V3 POLICY EVALUATION SUMMARY")
    print("=" * 70)
    print("")
    print("Episode: %d RL steps x %d sim days = %d total days"
          % (N_RL_STEPS, SIM_STEPS_PER_RL, N_RL_STEPS * SIM_STEPS_PER_RL))
    print("Drugs:   %s" % ", ".join(FQI_DRUGS_V3))
    print("Reward:  weighted lab delta (13 labs, weights %s)"
          % {k: v for k, v in sorted(LAB_WEIGHTS.items(), key=lambda x: -x[1])})
    print("")

    print("TERMINAL READMISSION RISK (after %d-step rollout):" % N_RL_STEPS)
    print("  FQI-V3 policy: %.4f" % results["fqi_risk"].mean())
    print("  Do-nothing:    %.4f" % results["donothing_risk"].mean())
    print("  Real clinical: %.4f" % results["real_risk"].mean())
    print("")

    print("CUMULATIVE DENSE REWARD (sum over %d steps):" % N_RL_STEPS)
    print("  FQI-V3 policy: %.4f" % results["fqi_cumulative_reward"].mean())
    print("  Do-nothing:    %.4f" % results["donothing_cumulative_reward"].mean())
    print("  Real clinical: %.4f" % results["real_cumulative_reward"].mean())
    print("")

    beats_dn = (results["fqi_risk"] < results["donothing_risk"]).sum()
    beats_real = (results["fqi_risk"] < results["real_risk"]).sum()
    print("FQI-V3 beats do-nothing:    %d/%d (%.1f%%)"
          % (beats_dn, n, beats_dn / n * 100))
    print("FQI-V3 beats real clinical: %d/%d (%.1f%%)"
          % (beats_real, n, beats_real / n * 100))
    print("")

    # Per-step reward analysis
    print("PER-STEP MEAN REWARD:")
    print("  %-8s %12s %12s %12s" % ("Step", "FQI-V3", "Do-nothing", "Delta"))
    for t in range(N_RL_STEPS):
        fqi_r = results["fqi_reward_step%d" % t].mean()
        dn_r = results["dn_reward_step%d" % t].mean()
        print("  Step %-3d %12.4f %12.4f %12.4f" % (t, fqi_r, dn_r, fqi_r - dn_r))
    print("")

    # Drug recommendation frequency
    print("DRUG RECOMMENDATION FREQUENCY:")
    for drug in FQI_DRUGS_V3:
        freqs = []
        for t in range(N_RL_STEPS):
            col = "%s_fqi_step%d" % (drug, t)
            if col in results.columns:
                freqs.append("step%d=%.1f%%" % (t, results[col].mean() * 100))
        print("  %-28s: %s" % (drug, ", ".join(freqs)))
    print("")

    # Feature importances
    imps = fqi.feature_importances()
    print("Q-FUNCTION FEATURE IMPORTANCES (top 10 per step):")
    for step in sorted(imps):
        imp = imps[step]
        sorted_feats = sorted(imp.items(), key=lambda x: -x[1])[:10]
        max_score = sorted_feats[0][1] if sorted_feats else 1.0
        print("")
        print("  Step %d (decision at day %d):" % (step, step * SIM_STEPS_PER_RL))
        for feat, score in sorted_feats:
            bar = "#" * int(score / max_score * 20) if max_score > 0 else ""
            print("    %-35s: %6.0f  %s" % (feat, score, bar))


# ── CLI ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Step D: Train V3 causal FQI RL agent.",
    )
    p.add_argument(
        "--csv",
        default=str(PROJECT_ROOT / "data" / "processed" / "hosp_daily_v3_triplets.csv"),
        help="Path to V3 triplet CSV",
    )
    p.add_argument(
        "--structural-eq-dir",
        default=str(PROJECT_ROOT / "models" / "causal_v3" / "structural_equations_notears"),
        help="V3 structural equation models directory (from Step C)",
    )
    p.add_argument(
        "--readmission-model-dir",
        default=str(PROJECT_ROOT / "models" / "rl_daily_full"),
        help="Readmission model directory (reused from V1 pipeline)",
    )
    p.add_argument(
        "--model-dir",
        default=str(PROJECT_ROOT / "models" / "fqi_v3"),
        help="Directory to save FQI-V3 Q-models",
    )
    p.add_argument(
        "--report-dir",
        default=str(PROJECT_ROOT / "reports" / "causal_v3" / "step_d"),
        help="Directory to save evaluation outputs",
    )
    p.add_argument(
        "--n-patients", type=int, default=5000,
        help="Number of train patients for trajectory collection (default 5000)",
    )
    p.add_argument(
        "--n-seqs", type=int, default=64,
        help="Random action sequences per patient (default 64)",
    )
    p.add_argument(
        "--n-iter", type=int, default=10,
        help="Number of FQI iterations (default 10)",
    )
    p.add_argument(
        "--gamma", type=float, default=0.99,
        help="Discount factor (default 0.99)",
    )
    p.add_argument(
        "--n-eval", type=int, default=500,
        help="Number of test patients for evaluation (default 500)",
    )
    p.add_argument(
        "--sample-n", type=int, default=0,
        help="If > 0, override n-patients and n-eval for smoke test",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()

    # Smoke test override
    if args.sample_n > 0:
        args.n_patients = min(args.sample_n, args.n_patients)
        args.n_eval = min(args.sample_n // 5 or 50, args.n_eval)
        args.n_seqs = min(16, args.n_seqs)
        args.n_iter = min(3, args.n_iter)

    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir)

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                str(report_dir / "run_log.txt"), mode="w", encoding="utf-8",
            ),
        ],
    )

    print("=" * 70)
    print("Step D: V3 Causal FQI Agent")
    print("=" * 70)
    print("  csv:                %s" % args.csv)
    print("  structural-eq-dir:  %s" % args.structural_eq_dir)
    print("  readmission-model:  %s" % args.readmission_model_dir)
    print("  model-dir:          %s" % model_dir)
    print("  n-patients:         %d" % args.n_patients)
    print("  n-seqs:             %d" % args.n_seqs)
    print("  n-iter:             %d" % args.n_iter)
    print("  gamma:              %.2f" % args.gamma)
    print("  n-eval:             %d" % args.n_eval)
    print("  seed:               %d" % args.seed)
    print("=" * 70)

    from careai.causal_v3.structural_equations import load_models, STATE_VARS, STATIC_FEATURES
    from careai.causal_v3.fqi_v3 import (
        FittedQIterationV3,
        FQI_DRUGS_V3,
        RL_STATE_COLS_V3,
        N_RL_STEPS,
        SIM_STEPS_PER_RL,
        collect_trajectories_v3,
    )
    from careai.rl_daily.readmission import load_readmission_model

    # ── [1/6] Load V3 triplet data ───────────────────────────────────
    print("")
    print("[1/6] Loading V3 triplet data ...")
    t0 = time.time()
    df = pd.read_csv(args.csv, low_memory=False)
    print("  Loaded %d rows in %.1f sec" % (len(df), time.time() - t0))

    # Extract initial states: first triplet per admission
    df_sorted = df.sort_values(["hadm_id", "day_of_stay"])
    initial_states = df_sorted.groupby("hadm_id").first().reset_index()
    print("  Unique admissions: %d" % len(initial_states))

    train_initial = initial_states[initial_states["split"] == "train"].copy()
    test_initial = initial_states[initial_states["split"] == "test"].copy()
    print("  Train initial states: %d" % len(train_initial))
    print("  Test  initial states: %d" % len(test_initial))

    # ── [2/6] Load V3 structural equation models ─────────────────────
    print("")
    print("[2/6] Loading V3 structural equation models ...")
    se_models, parent_sets = load_models(Path(args.structural_eq_dir))
    n_models = len([m for m in se_models.values() if m is not None])
    print("  Loaded %d structural equation models" % n_models)
    print("  Targets: %s" % ", ".join(se_models.keys()))

    # ── [3/6] Load readmission model (evaluation only — not used in training) ──
    print("")
    print("[3/6] Loading readmission model (for evaluation only) ...")
    readmission_model = load_readmission_model(args.readmission_model_dir)
    print("  AUC: %.4f  Brier: %.4f" % (
        readmission_model.test_auc, readmission_model.test_brier,
    ))

    # ── [4/6] Collect trajectories ───────────────────────────────────
    n_train = min(args.n_patients, len(train_initial))
    n_episodes = n_train * args.n_seqs
    print("")
    print("[4/6] Collecting trajectories ...")
    print("  %d patients x %d sequences = %d episodes"
          % (n_train, args.n_seqs, n_episodes))
    print("  %d RL steps x %d sim days/step = %d total sim days"
          % (N_RL_STEPS, SIM_STEPS_PER_RL, N_RL_STEPS * SIM_STEPS_PER_RL))
    print("  Reward: weighted lab delta toward clinical normal ranges")
    print("")

    t0 = time.time()
    trajectories = collect_trajectories_v3(
        models=se_models,
        parent_sets=parent_sets,
        initial_states=train_initial,
        n_patients=n_train,
        n_seqs=args.n_seqs,
        seed=args.seed,
    )
    traj_time = time.time() - t0
    print("  Trajectory collection: %.1f sec" % traj_time)
    print("  Total transitions: %d" % len(trajectories))

    # Reward statistics per step
    print("")
    print("  REWARD STATISTICS:")
    for step in range(N_RL_STEPS):
        step_df = trajectories[trajectories["step"] == step]
        r = step_df["reward"]
        print("    Step %d: mean=%.4f  std=%.4f  min=%.4f  max=%.4f"
              % (step, r.mean(), r.std(), r.min(), r.max()))

    # Save trajectories
    traj_path = report_dir / "trajectories.csv"
    trajectories.to_csv(traj_path, index=False)
    print("  Trajectories saved to %s" % traj_path)

    # ── [5/6] Train FQI ──────────────────────────────────────────────
    print("")
    print("[5/6] Training FQI-V3 (%d iterations, gamma=%.2f) ..."
          % (args.n_iter, args.gamma))
    print("  Q-function features: %d state + %d action = %d"
          % (len(RL_STATE_COLS_V3), len(FQI_DRUGS_V3),
             len(RL_STATE_COLS_V3) + len(FQI_DRUGS_V3)))
    print("")

    t0 = time.time()
    fqi = FittedQIterationV3()
    fqi.fit(trajectories, n_iter=args.n_iter, gamma=args.gamma)
    train_time = time.time() - t0
    print("")
    print("  FQI training: %.1f sec" % train_time)

    fqi.save(model_dir)
    print("  Q-models saved to %s" % model_dir)

    # ── [6/6] Evaluate ───────────────────────────────────────────────
    n_eval = min(args.n_eval, len(test_initial))
    print("")
    print("[6/6] Evaluating on %d test patients ..." % n_eval)

    t0 = time.time()
    eval_results = evaluate_fqi_v3_policy(
        fqi=fqi,
        initial_states=test_initial,
        models=se_models,
        parent_sets=parent_sets,
        readmission_model=readmission_model,
        n_patients=n_eval,
        seed=args.seed,
    )
    eval_time = time.time() - t0
    print("  Evaluation: %.1f sec" % eval_time)

    # Print summary
    print_summary(eval_results, fqi)

    # Save evaluation
    eval_csv = report_dir / "policy_evaluation.csv"
    eval_results.to_csv(eval_csv, index=False)
    print("")
    print("Evaluation saved to %s (%d rows)" % (eval_csv, len(eval_results)))

    # Save summary JSON
    imps = fqi.feature_importances()
    drug_freq = {}
    for drug in FQI_DRUGS_V3:
        drug_freq[drug] = {
            "step%d" % t: float(eval_results["%s_fqi_step%d" % (drug, t)].mean())
            for t in range(N_RL_STEPS)
            if "%s_fqi_step%d" % (drug, t) in eval_results.columns
        }

    summary = {
        "algorithm": "FittedQIterationV3",
        "simulator": "V3 structural equations (causally-constrained)",
        "reward_type": "weighted_lab_delta (13 labs, no readmission in training)",
        "n_iter": args.n_iter,
        "gamma": args.gamma,
        "n_patients_train": n_train,
        "n_seqs_per_patient": args.n_seqs,
        "n_episodes": n_episodes,
        "n_transitions": len(trajectories),
        "rl_state_cols": RL_STATE_COLS_V3,
        "fqi_drugs": FQI_DRUGS_V3,
        "n_rl_steps": N_RL_STEPS,
        "sim_steps_per_rl": SIM_STEPS_PER_RL,
        "n_eval_patients": len(eval_results),
        "mean_fqi_risk": float(eval_results["fqi_risk"].mean()),
        "mean_donothing_risk": float(eval_results["donothing_risk"].mean()),
        "mean_real_risk": float(eval_results["real_risk"].mean()),
        "mean_fqi_cumulative_reward": float(
            eval_results["fqi_cumulative_reward"].mean()
        ),
        "mean_donothing_cumulative_reward": float(
            eval_results["donothing_cumulative_reward"].mean()
        ),
        "fqi_beats_donothing_frac": float(
            (eval_results["fqi_risk"] < eval_results["donothing_risk"]).mean()
        ),
        "fqi_beats_real_frac": float(
            (eval_results["fqi_risk"] < eval_results["real_risk"]).mean()
        ),
        "drug_recommendation_freq": drug_freq,
        "feature_importances": {
            "step%d" % step: {k: int(v) for k, v in imp.items()}
            for step, imp in imps.items()
        },
        "timings": {
            "trajectory_collection_sec": round(traj_time, 1),
            "fqi_training_sec": round(train_time, 1),
            "evaluation_sec": round(eval_time, 1),
        },
    }
    json_path = report_dir / "policy_evaluation.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print("Summary JSON saved to %s" % json_path)

    print("")
    print("=" * 70)
    print("Step D complete.")
    print("  Models: %s" % model_dir)
    print("  Report: %s" % report_dir)
    print("  Total time: %.1f sec" % (traj_time + train_time + eval_time))
    print("=" * 70)


if __name__ == "__main__":
    main()
