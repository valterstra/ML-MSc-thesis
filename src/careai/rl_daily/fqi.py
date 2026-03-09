"""Fitted Q-Iteration (FQI) RL agent for antibiotic recommendation.

Algorithm:
  1. Collect trajectories by enumerating all 2^3 action sequences per patient.
     Each RL step = 2 simulator days. 3 decision points: day 0, 2, 4.
  2. Run FQI (backward induction) for N iterations using LightGBM Q-models.
     One Q-model per RL step (step 0, 1, 2).
  3. Policy: pi(s) = argmax_a Q(s, a) at each step.

Drug: antibiotic_active (binary, 0/1).
Reward: -predict_readmission_risk(terminal_state) at step 3, 0 elsewhere (sparse).
Causal correction: ATE for antibiotic applied at each simulator step.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from careai.sim_daily.features import (
    ACTION_COLS,
    INFECTION_CONTEXT,
    MEASURED_FLAGS,
    STATIC_FEATURES,
)
from careai.sim_daily.transition import TransitionModel, predict_next
from careai.rl_daily.policy import apply_ate_corrections
from careai.rl_daily.readmission import ReadmissionModel, predict_readmission_risk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RL_STATE_COLS: list[str] = [
    "wbc",
    "positive_culture_cumulative",
    "lactate_elevated",
    "is_icu",
    "age_at_admit",
    "charlson_score",
    "day_of_stay",
]

FQI_DRUG: str = "antibiotic_active"
N_RL_STEPS: int = 3        # decision points at day 0, 2, 4
SIM_STEPS_PER_RL: int = 2  # simulator days per RL decision

_NO_DRUG_ACTION: dict[str, float] = {col: 0.0 for col in ACTION_COLS}

_LGB_PARAMS: dict = dict(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


# ---------------------------------------------------------------------------
# Causal rollout helpers
# ---------------------------------------------------------------------------

def _causal_step(
    state_dict: dict[str, float],
    action_ab: int,
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
) -> dict[str, float]:
    """One causally-corrected simulator step (1 day forward).

    1. Predict next-state with no drugs (removes observational confounding).
    2. Apply ATE correction for the chosen antibiotic action.
    3. Carry forward static features; set measured flags to 1.
    """
    base_next, _ = predict_next(transition_model, state_dict, _NO_DRUG_ACTION)
    action_dict: dict[str, int] = {FQI_DRUG: action_ab}
    causal_next = apply_ate_corrections(base_next, action_dict, ate_table)

    # Build complete next state (mirrors DailySimEnv.step logic)
    full_next: dict[str, float] = dict(causal_next)

    for c in STATIC_FEATURES:
        if c == "day_of_stay":
            full_next[c] = state_dict.get(c, 0.0) + 1.0
        elif c == "days_in_current_unit":
            prev_icu = state_dict.get("is_icu", 0.0)
            curr_icu = causal_next.get("is_icu", 0.0)
            if curr_icu == prev_icu:
                full_next[c] = state_dict.get(c, 0.0) + 1.0
            else:
                full_next[c] = 0.0
        else:
            full_next[c] = state_dict.get(c, 0.0)

    for c in MEASURED_FLAGS:
        full_next[c] = 1.0
    for c in INFECTION_CONTEXT:
        full_next[c] = 0.0

    return full_next


def _rl_step(
    state_dict: dict[str, float],
    action_ab: int,
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
    n_sim_steps: int = SIM_STEPS_PER_RL,
) -> dict[str, float]:
    """Advance state by n_sim_steps days with the same antibiotic action."""
    s = state_dict
    for _ in range(n_sim_steps):
        s = _causal_step(s, action_ab, transition_model, ate_table)
    return s


def _extract_rl_state(state_dict: dict[str, float]) -> dict[str, float]:
    """Extract the 7 RL state features from a full state dict."""
    return {c: state_dict.get(c, np.nan) for c in RL_STATE_COLS}


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories(
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
    readmission_model: ReadmissionModel,
    initial_states: pd.DataFrame,
    n_patients: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Enumerate all 2^N_RL_STEPS action sequences per patient.

    For each patient x action sequence, rolls out N_RL_STEPS RL steps
    (each = SIM_STEPS_PER_RL simulator days) and records transitions.

    Returns DataFrame with columns:
      patient_idx, episode_idx, step, action, reward, done,
      s_{col} for col in RL_STATE_COLS  (current state),
      sn_{col} for col in RL_STATE_COLS (next state).
    """
    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    action_sequences = list(itertools.product([0, 1], repeat=N_RL_STEPS))
    records: list[dict[str, Any]] = []

    for patient_idx, (_, row) in enumerate(sample.iterrows()):
        if patient_idx % 500 == 0:
            print(f"  Collecting trajectories: [{patient_idx}/{n}] ...", flush=True)

        s0 = row.to_dict()

        for ep_idx, seq in enumerate(action_sequences):
            # Roll out full episode under this action sequence
            states = [s0]
            s = s0
            for a in seq:
                s_next = _rl_step(s, a, transition_model, ate_table)
                states.append(s_next)
                s = s_next

            # Terminal reward: negative readmission risk at final state
            terminal_reward = -float(predict_readmission_risk(readmission_model, states[-1]))

            for t, (a, s_curr, s_next) in enumerate(zip(seq, states[:-1], states[1:])):
                is_terminal = (t == N_RL_STEPS - 1)
                reward = terminal_reward if is_terminal else 0.0

                rec: dict[str, Any] = {
                    "patient_idx": patient_idx,
                    "episode_idx": ep_idx,
                    "step": t,
                    "action": a,
                    "reward": reward,
                    "done": int(is_terminal),
                }
                for c in RL_STATE_COLS:
                    rec[f"s_{c}"] = s_curr.get(c, np.nan)
                    rec[f"sn_{c}"] = s_next.get(c, np.nan)

                records.append(rec)

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# FQI class
# ---------------------------------------------------------------------------

class FittedQIteration:
    """Fitted Q-Iteration using LightGBM, one Q-model per RL step.

    Q(s, a) = expected cumulative reward (negative readmission risk) from
    state s taking action a, then acting optimally thereafter.

    Policy: pi(s, step) = argmax_a Q(s, a) using the Q-model for that step.
    """

    def __init__(self) -> None:
        self.q_models: dict[int, lgb.LGBMRegressor | None] = {
            t: None for t in range(N_RL_STEPS)
        }
        self.state_cols: list[str] = list(RL_STATE_COLS)
        self.feature_cols: list[str] = list(RL_STATE_COLS) + ["action"]
        self.n_iter: int = 0
        self.gamma: float = 0.99
        self.n_patients: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        transitions_df: pd.DataFrame,
        n_iter: int = 10,
        gamma: float = 0.99,
    ) -> "FittedQIteration":
        """Run FQI backward induction for n_iter iterations."""
        self.gamma = gamma
        self.n_patients = int(transitions_df["patient_idx"].nunique())

        step_dfs = {
            t: transitions_df[transitions_df["step"] == t].copy()
            for t in range(N_RL_STEPS)
        }

        for it in range(n_iter):
            print(f"  FQI iteration {it + 1}/{n_iter} ...", flush=True)

            # Backward induction: step 2 -> 1 -> 0
            for step in range(N_RL_STEPS - 1, -1, -1):
                df = step_dfs[step]

                sn_rename = {f"sn_{c}": c for c in self.state_cols}
                next_states = df[[f"sn_{c}" for c in self.state_cols]].rename(
                    columns=sn_rename
                )

                if step == N_RL_STEPS - 1:
                    # Terminal step: target = reward (known from data)
                    targets = df["reward"].values.copy()
                else:
                    # Non-terminal: Bellman backup from next step
                    q_next_0 = self._q_predict_batch(step + 1, next_states, action=0)
                    q_next_1 = self._q_predict_batch(step + 1, next_states, action=1)
                    max_q_next = np.maximum(q_next_0, q_next_1)
                    targets = df["reward"].values + gamma * max_q_next

                # Feature matrix: RL state + action
                X = df[[f"s_{c}" for c in self.state_cols]].rename(
                    columns={f"s_{c}": c for c in self.state_cols}
                ).copy()
                X["action"] = df["action"].values
                X = X[self.feature_cols]

                reg = lgb.LGBMRegressor(**_LGB_PARAMS)
                reg.fit(X, targets)
                self.q_models[step] = reg

        self.n_iter = n_iter
        return self

    def _q_predict_batch(
        self,
        step: int,
        states: pd.DataFrame,
        action: int,
    ) -> np.ndarray:
        """Predict Q(state, action) for a batch of states at a given step."""
        model = self.q_models[step]
        if model is None:
            return np.zeros(len(states))
        X = states[self.state_cols].copy()
        X["action"] = action
        X = X[self.feature_cols]
        return model.predict(X)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        state_dict: dict[str, float],
        action: int,
        step: int = 0,
    ) -> float:
        """Return Q(state_dict, action) for the given RL step."""
        model = self.q_models[step]
        if model is None:
            return 0.0
        row = {c: state_dict.get(c, np.nan) for c in self.state_cols}
        row["action"] = action
        X = pd.DataFrame([row])[self.feature_cols]
        return float(model.predict(X)[0])

    def best_action(self, state_dict: dict[str, float], step: int = 0) -> int:
        """Return argmax_a Q(state_dict, a) for the given RL step.

        state_dict may be a full simulator state dict or a 7-feature RL state dict.
        """
        q0 = self.predict(state_dict, 0, step)
        q1 = self.predict(state_dict, 1, step)
        return 1 if q1 > q0 else 0

    def feature_importances(self) -> dict[int, dict[str, float]]:
        """Return LightGBM feature importances per RL step."""
        result: dict[int, dict[str, float]] = {}
        for step, model in self.q_models.items():
            if model is not None:
                result[step] = dict(zip(self.feature_cols, model.feature_importances_))
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: Path | str) -> None:
        """Save Q-models and metadata to dir_path."""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        for step, model in self.q_models.items():
            if model is not None:
                joblib.dump(model, d / f"q_model_step{step}.joblib")
        meta = {
            "n_iter": self.n_iter,
            "gamma": self.gamma,
            "n_patients": self.n_patients,
            "state_cols": self.state_cols,
            "feature_cols": self.feature_cols,
            "n_rl_steps": N_RL_STEPS,
            "sim_steps_per_rl": SIM_STEPS_PER_RL,
            "fqi_drug": FQI_DRUG,
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, dir_path: Path | str) -> "FittedQIteration":
        """Load a previously saved FittedQIteration."""
        d = Path(dir_path)
        meta = json.loads((d / "metadata.json").read_text())
        obj = cls()
        obj.n_iter = meta["n_iter"]
        obj.gamma = meta["gamma"]
        obj.n_patients = meta["n_patients"]
        obj.state_cols = meta["state_cols"]
        obj.feature_cols = meta["feature_cols"]
        for step in range(meta["n_rl_steps"]):
            p = d / f"q_model_step{step}.joblib"
            if p.exists():
                obj.q_models[step] = joblib.load(p)
        return obj
