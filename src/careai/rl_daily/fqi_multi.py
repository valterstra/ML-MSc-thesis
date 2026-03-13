"""Fitted Q-Iteration (FQI) for joint 5-drug recommendation.

Extension of fqi.py that:
  - Controls all 5 causal drugs: antibiotic, anticoagulant, diuretic, insulin, steroid
  - Uses the full 26-feature patient state (15 labs + 4 binary + 7 static)
  - Applies ATE corrections for all 5 drugs at each simulator step
  - Collects trajectories via random sampling rather than exhaustive enumeration
    (32^3 = 32,768 paths per patient is computationally infeasible at scale)

Algorithm:
  1. For each patient, sample N_SEQS random 5-drug action sequences over N_RL_STEPS steps.
  2. Roll out each sequence through the causally-corrected simulator (batched).
  3. Run FQI backward induction (N_ITER iterations, LightGBM, one Q-model per step).
  4. Policy: pi(s, step) = argmax over all 32 drug combos of Q(s, a).

Design rationale:
  Different drugs affect different physiological systems:
    antibiotic  -> WBC, infection clearance
    diuretic    -> BUN, potassium, sodium
    steroid     -> glucose, WBC (anti-inflammatory)
    insulin     -> glucose
    anticoagulant -> INR
  The optimal joint drug combination depends on the full patient state — creatinine
  matters for diuretic safety, existing INR for anticoagulant risk, etc.
  Using 26 state features (vs 7 in the antibiotic-only FQI) allows the Q-function
  to learn these cross-drug, cross-system interactions.

  Random trajectory sampling: with n_seqs=64, each of the 32 drug combos appears
  roughly twice per step in expectation across sampled sequences, providing adequate
  action-space coverage without combinatorial blowup.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from careai.sim_daily.features import (
    ACTION_COLS,
    INFECTION_CONTEXT,
    MEASURED_FLAGS,
    STATIC_FEATURES,
    STATE_BINARY,
    STATE_CONTINUOUS,
)
from careai.sim_daily.transition import TransitionModel, predict_next
from careai.rl_daily.policy import ATE_DRUGS, apply_ate_corrections
from careai.rl_daily.readmission import ReadmissionModel, predict_readmission_risk


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Full clinical state: 15 labs + 4 binary + 7 static = 26 features.
# This superset ensures the Q-function sees the complete patient picture,
# not just infection markers relevant to one drug.
RL_STATE_COLS_MULTI: list[str] = STATE_CONTINUOUS + STATE_BINARY + STATIC_FEATURES

# 5 drugs for which causal ATEs were estimated. opioid_active is deliberately
# excluded — no ATE was estimated for it, so including it would reintroduce
# observational confounding (same reasoning as in policy.py).
FQI_DRUGS_MULTI: list[str] = list(ATE_DRUGS)

N_RL_STEPS: int = 3        # decision points at day 0, 2, 4 (same as fqi.py for comparability)
SIM_STEPS_PER_RL: int = 2  # simulator days per RL decision
N_SEQS: int = 64           # random sequences per patient (covers all 32 combos ~2x per step)

# All 32 possible 5-drug combos — enumerated once, reused for policy extraction.
ALL_COMBOS: list[tuple[int, ...]] = list(
    itertools.product([0, 1], repeat=len(FQI_DRUGS_MULTI))
)

_NO_DRUG_ACTION: dict[str, float] = {col: 0.0 for col in ACTION_COLS}

# Slightly richer model than fqi.py (more leaves, more trees) because the
# Q-function now has 31 input features instead of 8.
_LGB_PARAMS: dict = dict(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    verbosity=-1,
)


# ---------------------------------------------------------------------------
# Single-patient rollout helper (used during evaluation only)
# ---------------------------------------------------------------------------

def _causal_step_multi(
    state_dict: dict[str, float],
    action_dict: dict[str, int],
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
) -> dict[str, float]:
    """One causally-corrected simulator step with all 5 drugs.

    1. Predict next-state with no drugs (removes observational confounding).
    2. Apply ATE corrections for each active drug.
    3. Carry forward static features.
    """
    base_next, _ = predict_next(transition_model, state_dict, _NO_DRUG_ACTION)
    causal_next = apply_ate_corrections(base_next, action_dict, ate_table)

    full_next: dict[str, float] = dict(causal_next)
    for c in STATIC_FEATURES:
        if c == "day_of_stay":
            full_next[c] = state_dict.get(c, 0.0) + 1.0
        elif c == "days_in_current_unit":
            prev_icu = state_dict.get("is_icu", 0.0)
            curr_icu = causal_next.get("is_icu", 0.0)
            full_next[c] = (state_dict.get(c, 0.0) + 1.0) if curr_icu == prev_icu else 0.0
        else:
            full_next[c] = state_dict.get(c, 0.0)

    for c in MEASURED_FLAGS:
        full_next[c] = 1.0
    for c in INFECTION_CONTEXT:
        full_next[c] = 0.0

    return full_next


def _rl_step_multi(
    state_dict: dict[str, float],
    action_dict: dict[str, int],
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
    n_sim_steps: int = SIM_STEPS_PER_RL,
) -> dict[str, float]:
    """Advance state by n_sim_steps days with the given 5-drug action."""
    s = state_dict
    for _ in range(n_sim_steps):
        s = _causal_step_multi(s, action_dict, transition_model, ate_table)
    return s


# ---------------------------------------------------------------------------
# Batched simulation (used during trajectory collection)
# ---------------------------------------------------------------------------

def _batch_causal_step_multi(
    states_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
) -> pd.DataFrame:
    """Causally-corrected simulator step for all 5 drugs, over a full batch.

    Equivalent to calling _causal_step_multi once per row, but passes the
    entire DataFrame to each LightGBM model in a single call.

    Parameters
    ----------
    states_df:
        Full patient state, one row per (patient, sequence).
    actions_df:
        Drug flags (0/1), columns = FQI_DRUGS_MULTI, same index as states_df.
    """
    # Build simulator input matrix: all drugs set to 0 (base / no-drug)
    X = pd.DataFrame(index=states_df.index)
    for col in transition_model.input_cols:
        X[col] = states_df[col].values if col in states_df.columns else 0.0
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
        out[col] = preds.copy()

    for col in transition_model.output_binary:
        probs = transition_model.binary_models[col].predict_proba(X)[:, 1]
        out[col] = (probs >= 0.5).astype(float)

    # Apply ATE corrections: for each drug, shift relevant outcomes only in rows
    # where that drug is active (= 1). This correctly handles each drug independently.
    for drug in FQI_DRUGS_MULTI:
        if drug not in actions_df.columns:
            continue
        active = actions_df[drug].values == 1
        if not active.any():
            continue
        for (treatment, outcome), ate in ate_table.items():
            if treatment == drug and outcome in out:
                corrected = out[outcome].copy()
                corrected[active] += ate
                out[outcome] = corrected

    next_df = pd.DataFrame(out, index=states_df.index)

    # Carry forward static features (mirrors _causal_step_multi logic)
    for c in STATIC_FEATURES:
        if c not in states_df.columns:
            next_df[c] = 0.0
        elif c == "day_of_stay":
            next_df[c] = states_df[c].values + 1.0
        elif c == "days_in_current_unit":
            prev_icu = (
                states_df["is_icu"].values
                if "is_icu" in states_df.columns
                else np.zeros(len(states_df))
            )
            curr_icu = next_df["is_icu"].values
            next_df[c] = np.where(
                curr_icu == prev_icu, states_df[c].values + 1.0, 0.0
            )
        else:
            next_df[c] = states_df[c].values

    for c in MEASURED_FLAGS:
        next_df[c] = 1.0
    for c in INFECTION_CONTEXT:
        next_df[c] = 0.0

    # Carry forward any remaining columns (hadm_id, split, etc.)
    for c in states_df.columns:
        if c not in next_df.columns:
            next_df[c] = states_df[c].values

    return next_df


# ---------------------------------------------------------------------------
# Trajectory collection
# ---------------------------------------------------------------------------

def collect_trajectories_multi(
    transition_model: TransitionModel,
    ate_table: dict[tuple[str, str], float],
    readmission_model: ReadmissionModel,
    initial_states: pd.DataFrame,
    n_patients: int = 3000,
    n_seqs: int = N_SEQS,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample random 5-drug action sequences per patient and roll out in batch.

    With 5 drugs and 3 decision steps, exhaustive enumeration yields
    32^3 = 32,768 paths per patient — infeasible at training scale.
    Instead we independently sample one of the 32 drug combos at each step
    for each sequence. With n_seqs=64, each combo appears roughly twice per
    step in expectation, providing adequate coverage of the action space.

    Returns DataFrame with columns:
      patient_idx, episode_idx, step,
      {drug}_action for each drug in FQI_DRUGS_MULTI,
      reward, done,
      s_{col} and sn_{col} for col in RL_STATE_COLS_MULTI.
    """
    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    total_rows = n * n_seqs
    print(
        f"  Batch size: {n} patients x {n_seqs} sequences = {total_rows} rows",
        flush=True,
    )

    # Sample random action sequences: shape (total_rows, N_RL_STEPS)
    # Each entry is an index into ALL_COMBOS (0..31), sampled independently per step.
    combo_idx = rng.integers(0, len(ALL_COMBOS), size=(total_rows, N_RL_STEPS))
    # action_seqs: (total_rows, N_RL_STEPS, n_drugs)
    action_seqs = np.array(ALL_COMBOS)[combo_idx]

    patient_idx_arr = np.repeat(np.arange(n), n_seqs)
    episode_idx_arr = np.tile(np.arange(n_seqs), n)

    # Expand initial states: each patient repeated n_seqs times
    current_states = sample.loc[sample.index.repeat(n_seqs)].reset_index(drop=True)

    step_frames: list[pd.DataFrame] = []

    for rl_step in range(N_RL_STEPS):
        print(
            f"  RL step {rl_step + 1}/{N_RL_STEPS} "
            f"({n} patients x {n_seqs} sequences) ...",
            flush=True,
        )

        # Actions for all rows at this RL step: shape (total_rows, n_drugs)
        actions_this_step = action_seqs[:, rl_step, :]
        actions_df = pd.DataFrame(
            actions_this_step,
            columns=FQI_DRUGS_MULTI,
            index=current_states.index,
        )

        # Snapshot current RL state (26 features)
        s_data = {
            f"s_{c}": (
                current_states[c].values
                if c in current_states.columns
                else np.full(total_rows, np.nan)
            )
            for c in RL_STATE_COLS_MULTI
        }

        # Advance SIM_STEPS_PER_RL days (batch)
        next_states = current_states
        for sim_day in range(SIM_STEPS_PER_RL):
            next_states = _batch_causal_step_multi(
                next_states, actions_df, transition_model, ate_table
            )
            print(f"    sim day {sim_day + 1}/{SIM_STEPS_PER_RL} done", flush=True)

        # Snapshot next RL state
        sn_data = {
            f"sn_{c}": (
                next_states[c].values
                if c in next_states.columns
                else np.full(total_rows, np.nan)
            )
            for c in RL_STATE_COLS_MULTI
        }

        # Terminal reward: -P(readmit_30d) only at final step
        is_terminal = rl_step == N_RL_STEPS - 1
        if is_terminal:
            print("  Computing terminal rewards ...", flush=True)
            rewards = -predict_readmission_risk(readmission_model, next_states)
            print(f"  Mean reward: {rewards.mean():.4f}", flush=True)
        else:
            rewards = np.zeros(total_rows)

        step_df = pd.DataFrame(
            {
                "patient_idx": patient_idx_arr,
                "episode_idx": episode_idx_arr,
                "step": rl_step,
                **{f"{d}_action": actions_df[d].values for d in FQI_DRUGS_MULTI},
                "reward": rewards,
                "done": int(is_terminal),
                **s_data,
                **sn_data,
            }
        )
        step_frames.append(step_df)
        print(f"  RL step {rl_step + 1}/{N_RL_STEPS} complete.", flush=True)

        current_states = next_states

    trajectories = pd.concat(step_frames, ignore_index=True)
    print(
        f"  Trajectory collection done. {len(trajectories)} transitions total.",
        flush=True,
    )
    return trajectories


# ---------------------------------------------------------------------------
# FQI class
# ---------------------------------------------------------------------------

class FittedQIterationMulti:
    """Fitted Q-Iteration for joint 5-drug recommendation.

    Q(s, a) approximates the expected cumulative reward (negative readmission
    risk) from state s when taking 5-drug action a and acting optimally thereafter.

    The Q-function input is 26 state features + 5 binary drug flags = 31 features.
    One LightGBM regressor is fitted per RL decision step via backward induction.

    Policy: pi(s, step) = argmax over all 32 drug combos of Q_step(s, combo).
    """

    def __init__(self) -> None:
        self.q_models: dict[int, lgb.LGBMRegressor | None] = {
            t: None for t in range(N_RL_STEPS)
        }
        self.state_cols: list[str] = list(RL_STATE_COLS_MULTI)
        self.drug_cols: list[str] = list(FQI_DRUGS_MULTI)
        # Q-function features: state features + one binary flag per drug
        self.feature_cols: list[str] = list(RL_STATE_COLS_MULTI) + [
            f"{d}_action" for d in FQI_DRUGS_MULTI
        ]
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
    ) -> "FittedQIterationMulti":
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

                next_states = df[[f"sn_{c}" for c in self.state_cols]].rename(
                    columns={f"sn_{c}": c for c in self.state_cols}
                )

                if step == N_RL_STEPS - 1:
                    # Terminal: target = sparse reward (known from data)
                    targets = df["reward"].values.copy()
                else:
                    # Non-terminal: Bellman backup — max Q over all 32 combos
                    max_q_next = self._max_q_batch(step + 1, next_states)
                    targets = df["reward"].values + gamma * max_q_next

                # Build feature matrix: state + 5 drug action flags
                X = df[[f"s_{c}" for c in self.state_cols]].rename(
                    columns={f"s_{c}": c for c in self.state_cols}
                ).copy()
                for d in self.drug_cols:
                    X[f"{d}_action"] = df[f"{d}_action"].values
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
        combo: tuple[int, ...],
    ) -> np.ndarray:
        """Q(states, combo) for all rows in states."""
        model = self.q_models[step]
        if model is None:
            return np.zeros(len(states))
        X = states[self.state_cols].copy()
        for d, v in zip(self.drug_cols, combo):
            X[f"{d}_action"] = v
        X = X[self.feature_cols]
        return model.predict(X)

    def _max_q_batch(self, step: int, states: pd.DataFrame) -> np.ndarray:
        """max_a Q(states, a) over all 32 combos, for every row."""
        # Stack Q-values for all 32 combos: shape (n_rows, 32)
        q_values = np.stack(
            [self._q_predict_batch(step, states, combo) for combo in ALL_COMBOS],
            axis=1,
        )
        return q_values.max(axis=1)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(
        self,
        state_dict: dict[str, float],
        combo: tuple[int, ...],
        step: int = 0,
    ) -> float:
        """Return Q(state_dict, combo) for the given RL step."""
        model = self.q_models[step]
        if model is None:
            return 0.0
        row = {c: state_dict.get(c, np.nan) for c in self.state_cols}
        for d, v in zip(self.drug_cols, combo):
            row[f"{d}_action"] = v
        X = pd.DataFrame([row])[self.feature_cols]
        return float(model.predict(X)[0])

    def best_combo(
        self, state_dict: dict[str, float], step: int = 0
    ) -> dict[str, int]:
        """Return argmax_a Q(state_dict, a) as a drug-name -> 0/1 dict."""
        model = self.q_models[step]
        if model is None:
            return {d: 0 for d in self.drug_cols}

        # Score all 32 combos in one batch call
        row = {c: state_dict.get(c, np.nan) for c in self.state_cols}
        rows = []
        for combo in ALL_COMBOS:
            r = dict(row)
            for d, v in zip(self.drug_cols, combo):
                r[f"{d}_action"] = v
            rows.append(r)
        X = pd.DataFrame(rows)[self.feature_cols]
        q_vals = model.predict(X)
        best_idx = int(np.argmax(q_vals))
        return dict(zip(self.drug_cols, ALL_COMBOS[best_idx]))

    def feature_importances(self) -> dict[int, dict[str, float]]:
        """LightGBM feature importances per RL step."""
        result: dict[int, dict[str, float]] = {}
        for step, model in self.q_models.items():
            if model is not None:
                result[step] = dict(
                    zip(self.feature_cols, model.feature_importances_)
                )
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, dir_path: Path | str) -> None:
        """Save Q-models and metadata."""
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
            "drug_cols": self.drug_cols,
            "feature_cols": self.feature_cols,
            "n_rl_steps": N_RL_STEPS,
            "sim_steps_per_rl": SIM_STEPS_PER_RL,
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, dir_path: Path | str) -> "FittedQIterationMulti":
        """Load a previously saved FittedQIterationMulti."""
        d = Path(dir_path)
        meta = json.loads((d / "metadata.json").read_text())
        obj = cls()
        obj.n_iter = meta["n_iter"]
        obj.gamma = meta["gamma"]
        obj.n_patients = meta["n_patients"]
        obj.state_cols = meta["state_cols"]
        obj.drug_cols = meta["drug_cols"]
        obj.feature_cols = meta["feature_cols"]
        for step in range(meta["n_rl_steps"]):
            p = d / f"q_model_step{step}.joblib"
            if p.exists():
                obj.q_models[step] = joblib.load(p)
        return obj
