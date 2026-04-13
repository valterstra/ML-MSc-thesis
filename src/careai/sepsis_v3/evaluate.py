"""Off-policy evaluation for V3 DDQN policy.

Metrics:
  1. Policy agreement rate: % of test rows where DDQN matches clinician
  2. Per-drug agreement: agreement rate for each of the 8 drugs individually
  3. Mean max-Q under DDQN vs clinician action
  4. DR estimate: doubly-robust value estimate using factorized behavior policy
  5. Readmission analysis: readmission rate split by whether DDQN agreed with clinician

Behavior policy: 8 independent LightGBM binary classifiers (one per drug).
  P(a | s) = product over k of P(drug_k = a_k | s)
This factorized form is tractable with 256 actions and avoids sparse 256-class
multinomial estimation.

DR estimator:
  V_DR = (1/N) * sum_t [ rho_t * (r_t - Q(s_t, a_t)) + V(s_t) ]
  where rho_t = pi_target(a_t | s_t) / pi_behavior(a_t | s_t), clipped at 10.
  pi_target is deterministic: 1 if a_t == argmax Q(s_t, .), else 0.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from careai.sepsis_v3.preprocessing import DRUG_COLS, N_ACTIONS, STATE_FEATURES

log = logging.getLogger(__name__)

S_COLS  = ["s_"  + f for f in STATE_FEATURES]
SN_COLS = ["sn_" + f for f in STATE_FEATURES]

# IS clip to prevent variance explosion (same issue seen in sepsis pipeline)
RHO_CLIP = 10.0


# ---------------------------------------------------------------------------
# Behavior policy estimation
# ---------------------------------------------------------------------------

class FactorizedBehaviorPolicy:
    """8 independent LightGBM classifiers estimating P(drug_k=1 | s).

    P(a | s) = product_k P(drug_k = a_k | s).
    """

    _LGB_PARAMS = dict(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        num_leaves=31,
        min_child_samples=20,
        verbosity=-1,
        random_state=42,
    )

    def __init__(self) -> None:
        self.models: dict[str, lgb.LGBMClassifier] = {}

    def fit(self, df: pd.DataFrame) -> "FactorizedBehaviorPolicy":
        """Fit one binary classifier per drug on state features."""
        X = df[S_COLS].values
        for i, drug in enumerate(DRUG_COLS):
            # Recover original drug flag from action_id bitmask
            y = ((df["action_id"].values >> i) & 1).astype(int)
            clf = lgb.LGBMClassifier(**self._LGB_PARAMS)
            clf.fit(X, y)
            self.models[drug] = clf
            pos_rate = y.mean()
            log.info("  Behavior policy [%s]: positive rate=%.3f", drug, pos_rate)
        return self

    def log_prob(self, df: pd.DataFrame, action_ids: np.ndarray) -> np.ndarray:
        """log P(a | s) for each row using the given action_ids."""
        X = df[S_COLS]   # keep as DataFrame so LightGBM has feature names
        log_p = np.zeros(len(df))
        for i, drug in enumerate(DRUG_COLS):
            model = self.models[drug]
            # Probability of drug_k = 1 given state
            p1 = model.predict_proba(X)[:, 1]
            p1 = np.clip(p1, 1e-6, 1 - 1e-6)
            drug_val = (action_ids >> i) & 1
            # log P(drug_k = drug_val | s)
            log_p += np.where(drug_val == 1, np.log(p1), np.log(1 - p1))
        return log_p

    def prob(self, df: pd.DataFrame, action_ids: np.ndarray) -> np.ndarray:
        """P(a | s) for each row."""
        return np.exp(self.log_prob(df, action_ids))

    def save(self, model_dir: str | Path) -> None:
        model_dir = Path(model_dir)
        model_dir.mkdir(parents=True, exist_ok=True)
        for drug, model in self.models.items():
            import joblib
            joblib.dump(model, model_dir / ("behavior_%s.joblib" % drug))
        log.info("Saved behavior policy models to %s", model_dir)

    @classmethod
    def load(cls, model_dir: str | Path) -> "FactorizedBehaviorPolicy":
        import joblib
        model_dir = Path(model_dir)
        obj = cls()
        for drug in DRUG_COLS:
            p = model_dir / ("behavior_%s.joblib" % drug)
            if p.exists():
                obj.models[drug] = joblib.load(p)
        log.info("Loaded behavior policy from %s", model_dir)
        return obj


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def decode_action(action_id: int) -> dict[str, int]:
    """Decode integer action_id to {drug: 0/1} dict."""
    return {drug: int((action_id >> i) & 1) for i, drug in enumerate(DRUG_COLS)}


def policy_agreement(
    clinician_actions: np.ndarray,
    ddqn_actions: np.ndarray,
) -> dict[str, float]:
    """Compute full-combination and per-drug agreement rates."""
    full_agree = float((clinician_actions == ddqn_actions).mean())
    per_drug: dict[str, float] = {}
    for i, drug in enumerate(DRUG_COLS):
        clin_drug = (clinician_actions >> i) & 1
        ddqn_drug = (ddqn_actions    >> i) & 1
        per_drug[drug] = float((clin_drug == ddqn_drug).mean())
    return {"full_agreement": full_agree, "per_drug": per_drug}


def dr_estimate(
    df: pd.DataFrame,
    q_values: np.ndarray,       # (N, 256)
    ddqn_actions: np.ndarray,   # (N,) recommended action per row
    behavior_policy: FactorizedBehaviorPolicy,
    gamma: float = 0.99,
) -> dict[str, float]:
    """Doubly-robust off-policy value estimate.

    V(s) = max_a Q(s, a)  (value under target policy)
    Q(s, a_clinician) from Q-network

    DR = mean[ rho * (r - Q(s, a_clin)) + V(s) ]
    where rho = pi_target(a_clin|s) / pi_behavior(a_clin|s), clipped at RHO_CLIP.
    """
    clinician_actions = df["action_id"].values.astype(int)
    rewards           = df["reward"].values.astype(float)

    # V(s) = max Q(s, a)
    v_s = q_values.max(axis=1)

    # Q(s, a_clinician)
    q_clin = q_values[np.arange(len(df)), clinician_actions]

    # IS weight: pi_target(a_clin|s) / pi_behavior(a_clin|s)
    # pi_target is deterministic: 1 if a_clin == ddqn_action, else 0
    pi_target = (clinician_actions == ddqn_actions).astype(float)
    pi_behav  = behavior_policy.prob(df, clinician_actions)
    pi_behav  = np.clip(pi_behav, 1e-8, None)
    rho       = np.clip(pi_target / pi_behav, 0.0, RHO_CLIP)

    dr_vals = rho * (rewards - q_clin) + v_s

    n_valid = (pi_target > 0).sum()
    log.info(
        "DR: rho>0 for %d / %d rows (%.1f%% where DDQN agrees with clinician)",
        n_valid, len(df), 100.0 * n_valid / len(df),
    )

    return {
        "dr_mean":      float(dr_vals.mean()),
        "dr_std":       float(dr_vals.std()),
        "v_s_mean":     float(v_s.mean()),
        "q_clin_mean":  float(q_clin.mean()),
        "rho_mean":     float(rho.mean()),
        "rho_clip_pct": float((rho >= RHO_CLIP).mean()),
        "n_rows":       int(len(df)),
    }


def readmission_analysis(
    df: pd.DataFrame,
    ddqn_actions: np.ndarray,
) -> pd.DataFrame:
    """Readmission rate by whether DDQN agreed with the clinician.

    Splits test admissions into:
      agree    : DDQN recommended same action as clinician on LAST day
      disagree : DDQN recommended a different action on last day

    Returns a summary dataframe with readmission rates per group.
    """
    # Take last row per admission (terminal rows)
    term = df[df["done"] == 1].copy()
    term = term.copy()
    term["ddqn_action"] = ddqn_actions[term.index] if hasattr(ddqn_actions, "__len__") else ddqn_actions
    term["agreed"] = (term["action_id"] == term["ddqn_action"]).astype(int)

    result = (
        term.groupby("agreed")["readmit_30d"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "readmit_rate", "count": "n_admissions"})
        .reset_index()
    )
    result["agreed"] = result["agreed"].map({0: "DDQN disagreed", 1: "DDQN agreed"})
    return result


# ---------------------------------------------------------------------------
# Main evaluation function
# ---------------------------------------------------------------------------

def evaluate_policy(
    test_csv: str | Path,
    train_csv: str | Path,
    model,                          # OfflineDDQN instance
    report_dir: str | Path,
    behavior_policy: FactorizedBehaviorPolicy | None = None,
) -> dict:
    """Full evaluation pipeline.

    If behavior_policy is None, fits it on train_csv first.
    Returns dict of all metrics and saves JSON report.
    """
    report_dir = Path(report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading test data from %s", test_csv)
    test_df = pd.read_csv(test_csv)
    log.info("Test rows: %d", len(test_df))

    # Fit behavior policy if not provided
    if behavior_policy is None:
        log.info("Fitting behavior policy on train data...")
        train_df = pd.read_csv(train_csv)
        behavior_policy = FactorizedBehaviorPolicy()
        behavior_policy.fit(train_df)
        behavior_policy.save(report_dir / "behavior_policy")

    # DDQN recommendations
    log.info("Computing DDQN actions and Q-values...")
    ddqn_actions = model.predict_actions(test_df)
    q_vals       = model.q_values(test_df)

    # 1. Agreement
    log.info("Computing policy agreement...")
    agree = policy_agreement(
        test_df["action_id"].values.astype(int), ddqn_actions
    )
    log.info(
        "Full agreement: %.2f%% | Per-drug: %s",
        100 * agree["full_agreement"],
        {k: "%.2f%%" % (100 * v) for k, v in agree["per_drug"].items()},
    )

    # 2. Mean Q-values
    q_ddqn_action = q_vals[np.arange(len(test_df)), ddqn_actions]
    q_clin_action = q_vals[
        np.arange(len(test_df)),
        test_df["action_id"].values.astype(int)
    ]
    q_stats = {
        "mean_q_ddqn":      float(q_ddqn_action.mean()),
        "mean_q_clinician": float(q_clin_action.mean()),
        "mean_q_max":       float(q_vals.max(axis=1).mean()),
    }
    log.info(
        "Q-values: DDQN=%.4f, Clinician=%.4f, Max=%.4f",
        q_stats["mean_q_ddqn"], q_stats["mean_q_clinician"], q_stats["mean_q_max"],
    )

    # 3. DR estimate
    log.info("Computing DR estimate...")
    dr = dr_estimate(test_df, q_vals, ddqn_actions, behavior_policy)
    log.info("DR estimate: mean=%.4f std=%.4f", dr["dr_mean"], dr["dr_std"])

    # 4. Readmission analysis (on terminal rows)
    test_df_with_idx = test_df.reset_index(drop=True)
    ddqn_series = pd.Series(ddqn_actions, index=test_df_with_idx.index)
    term_mask = test_df_with_idx["done"] == 1
    term_df = test_df_with_idx[term_mask].copy()
    term_df["ddqn_action"] = ddqn_series[term_mask].values
    term_df["agreed"] = (
        term_df["action_id"].values.astype(int) == term_df["ddqn_action"].values
    ).astype(int)
    readmit_by_agree = (
        term_df.groupby("agreed")["readmit_30d"]
        .agg(["mean", "count"])
        .rename(columns={"mean": "readmit_rate", "count": "n_admissions"})
        .reset_index()
    )
    readmit_by_agree["group"] = readmit_by_agree["agreed"].map(
        {0: "DDQN_disagreed", 1: "DDQN_agreed"}
    )
    log.info("Readmission by agreement:\n%s", readmit_by_agree.to_string(index=False))

    # Save outputs
    readmit_by_agree.to_csv(report_dir / "readmit_by_agreement.csv", index=False)

    results = {
        "agreement":        agree,
        "q_stats":          q_stats,
        "dr_estimate":      dr,
        "readmit_analysis": readmit_by_agree.to_dict(orient="records"),
        "n_test_rows":      int(len(test_df)),
        "n_test_admissions": int(term_mask.sum()),
    }
    report_path = report_dir / "evaluation_results.json"
    report_path.write_text(json.dumps(results, indent=2))
    log.info("Saved evaluation report to %s", report_path)

    return results
