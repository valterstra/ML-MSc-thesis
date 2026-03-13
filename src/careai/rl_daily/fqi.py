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
# Batched simulation helpers (used by collect_trajectories)
# ---------------------------------------------------------------------------

def _batch_causal_step(
    states_df: pd.DataFrame,
    actions_array: np.ndarray,
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Causally-corrected simulator step for a full batch of rows.

    Equivalent to calling _causal_step once per row, but passes the entire
    DataFrame to each LightGBM model in one call — much faster.

    Parameters
    ----------
    states_df:
        DataFrame with one row per (patient, action_sequence). Must contain
        all columns expected by the transition model.
    actions_array:
        Integer array of shape (len(states_df),) — antibiotic action (0 or 1)
        for each row.
    transition_model, ate_table:
        Same objects used in the sequential version.

    Returns
    -------
    DataFrame with the same index as states_df, containing the causally
    corrected next-state for every row.
    """
    # Build input matrix: all drug columns set to 0 (base / no-drug prediction)
    X = pd.DataFrame(index=states_df.index)
    for col in transition_model.input_cols:
        if col in states_df.columns:
            X[col] = states_df[col].values
        else:
            X[col] = 0.0
    for col in ACTION_COLS:
        if col in X.columns:
            X[col] = 0.0

    # Batch predict all continuous and binary outputs
    out: dict[str, np.ndarray] = {}

    for col in transition_model.output_continuous:
        preds = transition_model.continuous_models[col].predict(X).astype(float)
        if col in transition_model.clip_bounds:
            lo, hi = transition_model.clip_bounds[col]
            preds = np.clip(preds, lo, hi)
        out[col] = preds

    for col in transition_model.output_binary:
        probs = transition_model.binary_models[col].predict_proba(X)[:, 1]
        out[col] = (probs >= 0.5).astype(float)

    # Apply ATE corrections only to rows where antibiotic action = 1
    active = actions_array == 1
    for (treatment, outcome), ate in ate_table.items():
        if treatment == FQI_DRUG and outcome in out:
            corrected = out[outcome].copy()
            corrected[active] += ate
            out[outcome] = corrected

    next_df = pd.DataFrame(out, index=states_df.index)

    # Carry forward static features (mirrors _causal_step logic)
    for c in STATIC_FEATURES:
        if c not in states_df.columns:
            next_df[c] = 0.0
        elif c == "day_of_stay":
            next_df[c] = states_df[c].values + 1.0
        elif c == "days_in_current_unit":
            prev_icu = states_df["is_icu"].values if "is_icu" in states_df.columns else np.zeros(len(states_df))
            curr_icu = next_df["is_icu"].values
            next_df[c] = np.where(curr_icu == prev_icu, states_df[c].values + 1.0, 0.0)
        else:
            next_df[c] = states_df[c].values

    for c in MEASURED_FLAGS:
        next_df[c] = 1.0
    for c in INFECTION_CONTEXT:
        next_df[c] = 0.0

    # Carry forward any remaining columns (e.g. hadm_id, split)
    for c in states_df.columns:
        if c not in next_df.columns:
            next_df[c] = states_df[c].values

    return next_df


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
    """Enumerate all 2^N_RL_STEPS action sequences per patient (batched).

    All patients x action sequences are simulated simultaneously at each
    time step using _batch_causal_step, replacing the original patient-by-
    patient loop. Results are mathematically identical to the sequential
    version — only the order of LightGBM calls changes.

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
    n_seqs = len(action_sequences)  # 2^N_RL_STEPS = 8
    total_rows = n * n_seqs

    print(f"  Batch size: {n} patients x {n_seqs} sequences = {total_rows} rows", flush=True)

    # Expand initial states: each patient repeated n_seqs times
    # Row i -> patient i // n_seqs, sequence i % n_seqs
    current_states = sample.loc[sample.index.repeat(n_seqs)].reset_index(drop=True)
    patient_idx_arr = np.repeat(np.arange(n), n_seqs)
    seq_idx_arr = np.tile(np.arange(n_seqs), n)

    # Action for each row at each RL step: shape (total_rows, N_RL_STEPS)
    action_seqs_array = np.array([action_sequences[s] for s in seq_idx_arr])

    step_frames: list[pd.DataFrame] = []

    for rl_step in range(N_RL_STEPS):
        print(f"  RL step {rl_step + 1}/{N_RL_STEPS} ({n} patients) ...", flush=True)

        actions_this_step = action_seqs_array[:, rl_step]

        # Snapshot current RL state features
        s_data = {
            f"s_{c}": current_states[c].values if c in current_states.columns else np.full(total_rows, np.nan)
            for c in RL_STATE_COLS
        }

        # Advance SIM_STEPS_PER_RL days in batch
        next_states = current_states
        for sim_day in range(SIM_STEPS_PER_RL):
            next_states = _batch_causal_step(next_states, actions_this_step, transition_model, ate_table)
            print(f"    sim day {sim_day + 1}/{SIM_STEPS_PER_RL} done", flush=True)

        # Snapshot next RL state features
        sn_data = {
            f"sn_{c}": next_states[c].values if c in next_states.columns else np.full(total_rows, np.nan)
            for c in RL_STATE_COLS
        }

        # Terminal reward: score all rows with readmission model in one call
        is_terminal = rl_step == N_RL_STEPS - 1
        if is_terminal:
            print(f"  Computing terminal rewards ...", flush=True)
            rewards = -predict_readmission_risk(readmission_model, next_states)
            print(f"  Mean reward: {rewards.mean():.4f}", flush=True)
        else:
            rewards = np.zeros(total_rows)

        step_df = pd.DataFrame({
            "patient_idx": patient_idx_arr,
            "episode_idx": seq_idx_arr,
            "step": rl_step,
            "action": actions_this_step,
            "reward": rewards,
            "done": int(is_terminal),
            **s_data,
            **sn_data,
        })
        step_frames.append(step_df)
        print(f"  RL step {rl_step + 1}/{N_RL_STEPS} complete.", flush=True)

        current_states = next_states

    trajectories = pd.concat(step_frames, ignore_index=True)
    print(f"  Trajectory collection done. {len(trajectories)} transitions total.", flush=True)
    return trajectories


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
