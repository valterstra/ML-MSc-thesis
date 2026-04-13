"""Fitted Q-Iteration for V3 causal RL agent.

Key improvements over V1 FQI-multi (src/careai/rl_daily/fqi_multi.py):

  1. Causally-constrained simulator: V3 structural equations (14 LightGBM
     models) replace V1's transition model + post-hoc ATE patching. Each model
     uses ONLY PC-discovered parent features, so drug effects flow through
     causally-valid pathways rather than being bolted on after the fact.

  2. Dense intermediary rewards: Every RL step computes reward based on three
     clinically-grounded signals (not just sparse terminal risk):
       - Readmission risk:   P(readmit_30d | s') from the V1 readmission model
       - Lab instability:    Normalized distance of labs from clinical normal ranges
       - ICU penalty:        Small penalty for ICU status (higher acuity = worse)

  3. Cleaner state space: 16 features (14 state vars + 2 static) vs V1's 26,
     aligned exactly with the variables the causal discovery operates on.

Algorithm (same backward-induction FQI as V1, with dense rewards):
  1. Sample N_SEQS random 5-drug action sequences per patient over 3 RL steps.
  2. Roll out using V3 structural equation simulator (2 sim days per RL step).
  3. Compute dense reward at every step.
  4. FQI backward induction: 1 LightGBM Q-model per step, N_ITER iterations.
  5. Policy: pi(s, step) = argmax over 32 drug combos of Q(s, a).

The dense reward is the main algorithmic improvement. V1's sparse terminal-only
reward made credit assignment across 3 decisions extremely difficult. Dense
rewards give the Q-function direct feedback at every decision point, enabling
the agent to attribute which drug changes actually improve the patient.
"""

from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from careai.causal_v3.structural_equations import (
    ACTION_FEATURES,
    BINARY_TARGETS,
    STATE_VARS,
    STATIC_FEATURES,
    load_models,
)

log = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────

# V3 RL state: 14 state variables + 2 static = 16 features.
# Exactly the variables that the causal discovery operates on.
RL_STATE_COLS_V3: list[str] = list(STATE_VARS) + list(STATIC_FEATURES)

# 5 drugs for which causal effects were discovered.
FQI_DRUGS_V3: list[str] = list(ACTION_FEATURES)

N_RL_STEPS: int = 3        # decision points at day 0, 2, 4
SIM_STEPS_PER_RL: int = 2  # simulator days per RL decision
N_SEQS: int = 64           # random sequences per patient

# All 32 possible 5-drug combos.
ALL_COMBOS: list[tuple[int, ...]] = list(
    itertools.product([0, 1], repeat=len(FQI_DRUGS_V3))
)

# Clinical normal ranges (adult population, hospital context).
# Sources: NBME, Harrison's Principles, ACCP references.
# Glucose uses hospital target band (70-180), not fasting normal (70-100),
# per ADA/AACE inpatient guidelines (NICE-SUGAR trial).
NORMAL_RANGES: dict[str, tuple[float, float]] = {
    "creatinine":   (0.6,  1.2),
    "bun":          (7.0,  20.0),
    "sodium":       (136.0, 145.0),
    "potassium":    (3.5,  5.0),
    "bicarbonate":  (22.0, 28.0),
    "anion_gap":    (8.0,  16.0),
    "calcium":      (8.4,  10.2),
    "glucose":      (70.0, 180.0),
    "hemoglobin":   (12.0, 17.5),
    "wbc":          (4.5,  11.0),
    "platelets":    (150.0, 400.0),
    "phosphate":    (2.5,  4.5),
    "magnesium":    (1.7,  2.2),
}

# Per-lab reward weights based on mortality evidence.
# Tier 1 (2.0-2.5): most acutely lethal when abnormal (cardiac arrhythmia,
#   cerebral edema). Tier 2 (1.5): direct drug targets + strong mortality
#   signal. Tier 3 (1.0): meaningful but less directly actionable.
#   Tier 4 (0.5): indirect effects, weaker mortality signal.
LAB_WEIGHTS: dict[str, float] = {
    "potassium":    2.0,   # Tier 1: K >6.5 -> cardiac arrest
    "sodium":       2.0,   # Tier 1: hypernatremia OR 14 for mortality
    "anion_gap":    1.5,   # Tier 2: AG >20 = metabolic crisis
    "bicarbonate":  1.5,   # Tier 2: acidosis, strong mortality signal
    "creatinine":   1.5,   # Tier 2: AKI -> 50% mortality at stage 3
    "glucose":      1.5,   # Tier 2: hypoglycemia RR 2.09; directly drug-controlled
    "platelets":    1.5,   # Tier 2: trajectory matters; anticoag/steroid target
    "wbc":          1.5,   # Tier 2: primary antibiotic endpoint
    "bun":          1.0,   # Tier 3: correlated with creatinine; AKI marker
    "calcium":      1.0,   # Tier 3: ICU prevalence high; antibiotic target
    "hemoglobin":   1.0,   # Tier 3: OR 2.22 for mortality in anemia
    "phosphate":    1.0,   # Tier 3: hyperphosphatemia OR 2.85
    "magnesium":    0.5,   # Tier 4: indirect via hypoK/hypoCa
}

# LightGBM parameters for Q-function (same as V1 FQI-multi).
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


# ── V3 Structural Equation Simulator ─────────────────────────────────────

def _batch_v3_step(
    states_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    models: dict,
    parent_sets: dict[str, list[str]],
) -> pd.DataFrame:
    """One simulator day using V3 structural equations (batch).

    For each of the 14 next_* targets, builds the feature matrix from the
    current state + action and runs the model's .predict() in batch.

    All 14 models see the SAME current state (simultaneous SCM evaluation),
    not a partially-updated state.  This is the correct semantics for
    computing all endogenous variables in one time step.
    """
    next_cols: dict[str, np.ndarray] = {}

    for target, model in models.items():
        if model is None:
            continue
        parents = parent_sets[target]
        var_name = target.replace("next_", "")

        # Build feature matrix: columns in parent order
        X = pd.DataFrame(index=states_df.index)
        for p in parents:
            if p in actions_df.columns:
                X[p] = actions_df[p].values
            elif p in states_df.columns:
                X[p] = states_df[p].values
            else:
                X[p] = np.nan
        X = X[parents]

        if var_name in BINARY_TARGETS:
            # Keep probability (smoother for RL than hard threshold)
            probs = model.predict_proba(X)[:, 1]
            next_cols[var_name] = probs.astype(float)
        else:
            preds = model.predict(X)
            next_cols[var_name] = preds.astype(float)

    next_df = pd.DataFrame(next_cols, index=states_df.index)

    # Carry forward static features (they don't change)
    for c in STATIC_FEATURES:
        if c in states_df.columns:
            next_df[c] = states_df[c].values
        else:
            next_df[c] = 0.0

    # Carry forward metadata columns (hadm_id, split, etc.)
    for c in states_df.columns:
        if c not in next_df.columns and c not in ACTION_FEATURES:
            next_df[c] = states_df[c].values

    return next_df


def _batch_rl_step_v3(
    states_df: pd.DataFrame,
    actions_df: pd.DataFrame,
    models: dict,
    parent_sets: dict[str, list[str]],
    n_sim_steps: int = SIM_STEPS_PER_RL,
) -> pd.DataFrame:
    """Advance state by n_sim_steps days (batch).

    Each sim step uses the updated state from the previous step, so a 2-day
    RL step means: state_t -> state_{t+1} -> state_{t+2}.
    """
    s = states_df
    for _ in range(n_sim_steps):
        s = _batch_v3_step(s, actions_df, models, parent_sets)
    return s


def _single_v3_step(
    state_dict: dict[str, float],
    action_dict: dict[str, int],
    models: dict,
    parent_sets: dict[str, list[str]],
) -> dict[str, float]:
    """One simulator day for a single patient (dict in, dict out)."""
    next_state: dict[str, float] = {}

    for target, model in models.items():
        if model is None:
            continue
        parents = parent_sets[target]
        var_name = target.replace("next_", "")

        features = {}
        for p in parents:
            if p in action_dict:
                features[p] = float(action_dict[p])
            elif p in state_dict:
                features[p] = float(state_dict[p])
            else:
                features[p] = np.nan
        X = pd.DataFrame([features])[parents]

        if var_name in BINARY_TARGETS:
            pred = float(model.predict_proba(X)[0, 1])
        else:
            pred = float(model.predict(X)[0])
        next_state[var_name] = pred

    # Carry forward static features
    for c in STATIC_FEATURES:
        next_state[c] = state_dict.get(c, 0.0)

    return next_state


def _single_rl_step_v3(
    state_dict: dict[str, float],
    action_dict: dict[str, int],
    models: dict,
    parent_sets: dict[str, list[str]],
    n_sim_steps: int = SIM_STEPS_PER_RL,
) -> dict[str, float]:
    """Advance state by n_sim_steps days (single patient)."""
    s = state_dict
    for _ in range(n_sim_steps):
        s = _single_v3_step(s, action_dict, models, parent_sets)
    return s


# ── Lab Delta Reward ──────────────────────────────────────────────────────

def _batch_lab_distances(states_df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Per-lab normalized distance from clinical normal range (batch).

    distance = max(0, lo - val) / range_width   if below normal
             = max(0, val - hi) / range_width   if above normal
             = 0                                 if within normal

    Returns dict: lab_name -> array of shape (n_rows,), values >= 0.
    NaN values are treated as distance = 0 (no signal).
    """
    distances: dict[str, np.ndarray] = {}
    for lab, (lo, hi) in NORMAL_RANGES.items():
        if lab not in states_df.columns:
            continue
        vals = states_df[lab].values.astype(float)
        range_width = hi - lo
        below = np.maximum(0.0, lo - vals) / range_width
        above = np.maximum(0.0, vals - hi) / range_width
        dist = below + above
        dist = np.where(np.isnan(vals), 0.0, dist)
        distances[lab] = dist
    return distances


def _batch_lab_delta_reward(
    prev_states_df: pd.DataFrame,
    next_states_df: pd.DataFrame,
) -> np.ndarray:
    """Weighted lab delta reward for a batch of transitions.

    For each lab:
        delta       = d_prev - d_next       (positive = moved toward normal)
        contribution = w * clip(delta, -1, 1)

    Clipping to [-1, 1] prevents any single noisy lab from dominating.
    Weights reflect mortality evidence (see LAB_WEIGHTS).

    Returns array of shape (n_rows,).
    """
    prev_dist = _batch_lab_distances(prev_states_df)
    next_dist = _batch_lab_distances(next_states_df)

    reward = np.zeros(len(prev_states_df))
    for lab, w in LAB_WEIGHTS.items():
        if lab not in prev_dist or lab not in next_dist:
            continue
        delta = prev_dist[lab] - next_dist[lab]
        reward += w * np.clip(delta, -1.0, 1.0)
    return reward


def compute_lab_delta_reward(
    prev_state: dict[str, float],
    next_state: dict[str, float],
) -> float:
    """Weighted lab delta reward for a single patient transition (dict in).

    Same formula as _batch_lab_delta_reward but for scalar dicts.
    Used during policy evaluation rollouts.
    """
    reward = 0.0
    for lab, (lo, hi) in NORMAL_RANGES.items():
        w = LAB_WEIGHTS.get(lab, 1.0)
        prev_val = prev_state.get(lab, float("nan"))
        next_val = next_state.get(lab, float("nan"))
        if np.isnan(prev_val) or np.isnan(next_val):
            continue
        range_width = hi - lo

        def _dist(v: float) -> float:
            if v < lo:
                return (lo - v) / range_width
            if v > hi:
                return (v - hi) / range_width
            return 0.0

        delta = _dist(prev_val) - _dist(next_val)
        reward += w * float(np.clip(delta, -1.0, 1.0))
    return reward


# ── Trajectory Collection ─────────────────────────────────────────────────

def collect_trajectories_v3(
    models: dict,
    parent_sets: dict[str, list[str]],
    initial_states: pd.DataFrame,
    n_patients: int = 3000,
    n_seqs: int = N_SEQS,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample random 5-drug action sequences and roll out through V3 simulator.

    Reward is the weighted lab delta at every step: positive when labs move
    toward clinical normal ranges, negative when they move away.  No
    readmission model is used during training — the agent is trained purely
    on in-hospital physiological improvement.

    Returns DataFrame with columns:
      patient_idx, episode_idx, step,
      {drug}_action for each drug,
      reward, done,
      s_{col} and sn_{col} for each col in RL_STATE_COLS_V3.
    """
    rng = np.random.default_rng(seed)
    n = min(n_patients, len(initial_states))
    idx = rng.choice(len(initial_states), size=n, replace=False)
    sample = initial_states.iloc[idx].reset_index(drop=True)

    total_rows = n * n_seqs
    print(
        "  Batch size: %d patients x %d sequences = %d rows"
        % (n, n_seqs, total_rows),
        flush=True,
    )

    # Sample random action sequences: shape (total_rows, N_RL_STEPS)
    # Each entry indexes into ALL_COMBOS (0..31)
    combo_idx = rng.integers(0, len(ALL_COMBOS), size=(total_rows, N_RL_STEPS))
    action_seqs = np.array(ALL_COMBOS)[combo_idx]

    patient_idx_arr = np.repeat(np.arange(n), n_seqs)
    episode_idx_arr = np.tile(np.arange(n_seqs), n)

    # Expand initial states: each patient repeated n_seqs times
    current_states = sample.loc[sample.index.repeat(n_seqs)].reset_index(drop=True)

    step_frames: list[pd.DataFrame] = []

    for rl_step in range(N_RL_STEPS):
        print(
            "  RL step %d/%d (%d patients x %d sequences) ..."
            % (rl_step + 1, N_RL_STEPS, n, n_seqs),
            flush=True,
        )

        # Actions for this RL step
        actions_this_step = action_seqs[:, rl_step, :]
        actions_df = pd.DataFrame(
            actions_this_step,
            columns=FQI_DRUGS_V3,
            index=current_states.index,
        )

        # Snapshot current RL state
        s_data = {
            "s_%s" % c: (
                current_states[c].values
                if c in current_states.columns
                else np.full(total_rows, np.nan)
            )
            for c in RL_STATE_COLS_V3
        }

        # Advance SIM_STEPS_PER_RL days through V3 simulator
        next_states = current_states
        for sim_day in range(SIM_STEPS_PER_RL):
            next_states = _batch_v3_step(
                next_states, actions_df, models, parent_sets,
            )
            print(
                "    sim day %d/%d done" % (sim_day + 1, SIM_STEPS_PER_RL),
                flush=True,
            )

        # Snapshot next RL state
        sn_data = {
            "sn_%s" % c: (
                next_states[c].values
                if c in next_states.columns
                else np.full(total_rows, np.nan)
            )
            for c in RL_STATE_COLS_V3
        }

        # Lab delta reward at EVERY step
        rewards = _batch_lab_delta_reward(current_states, next_states)
        is_terminal = rl_step == N_RL_STEPS - 1
        print(
            "  Step %d mean reward: %.4f (terminal=%s)"
            % (rl_step, rewards.mean(), is_terminal),
            flush=True,
        )

        step_df = pd.DataFrame(
            {
                "patient_idx": patient_idx_arr,
                "episode_idx": episode_idx_arr,
                "step": rl_step,
                **{"%s_action" % d: actions_df[d].values for d in FQI_DRUGS_V3},
                "reward": rewards,
                "done": int(is_terminal),
                **s_data,
                **sn_data,
            }
        )
        step_frames.append(step_df)
        print("  RL step %d/%d complete." % (rl_step + 1, N_RL_STEPS), flush=True)

        current_states = next_states

    trajectories = pd.concat(step_frames, ignore_index=True)
    print(
        "  Trajectory collection done. %d transitions total." % len(trajectories),
        flush=True,
    )
    return trajectories


# ── FQI Class ─────────────────────────────────────────────────────────────

class FittedQIterationV3:
    """Fitted Q-Iteration for V3 causal RL.

    Q(s, a) approximates the expected cumulative dense reward from state s
    when taking action a and acting optimally thereafter.

    Input features: 16 state features + 5 drug flags = 21 features.
    One LightGBM regressor per RL step, trained via backward induction.
    """

    def __init__(self) -> None:
        self.q_models: dict[int, lgb.LGBMRegressor | None] = {
            t: None for t in range(N_RL_STEPS)
        }
        self.state_cols: list[str] = list(RL_STATE_COLS_V3)
        self.drug_cols: list[str] = list(FQI_DRUGS_V3)
        self.feature_cols: list[str] = list(RL_STATE_COLS_V3) + [
            "%s_action" % d for d in FQI_DRUGS_V3
        ]
        self.n_iter: int = 0
        self.gamma: float = 0.99
        self.n_patients: int = 0

    # ── Training ──────────────────────────────────────────────────────

    def fit(
        self,
        transitions_df: pd.DataFrame,
        n_iter: int = 10,
        gamma: float = 0.99,
    ) -> "FittedQIterationV3":
        """Run FQI backward induction.

        With dense rewards, the Bellman target at every step is:
          Q_target(s, a) = r(s, a) + gamma * max_a' Q(s', a')

        This is identical to standard FQI except r is non-zero at every step
        (not just terminal), giving the Q-function direct learning signal
        from intermediate decisions.
        """
        self.gamma = gamma
        self.n_patients = int(transitions_df["patient_idx"].nunique())

        step_dfs = {
            t: transitions_df[transitions_df["step"] == t].copy()
            for t in range(N_RL_STEPS)
        }

        for it in range(n_iter):
            print("  FQI iteration %d/%d ..." % (it + 1, n_iter), flush=True)

            # Backward induction: step 2 -> 1 -> 0
            for step in range(N_RL_STEPS - 1, -1, -1):
                df = step_dfs[step]

                next_states = df[["sn_%s" % c for c in self.state_cols]].rename(
                    columns={"sn_%s" % c: c for c in self.state_cols}
                )

                if step == N_RL_STEPS - 1:
                    # Terminal step: target = reward (no future)
                    targets = df["reward"].values.copy()
                else:
                    # Non-terminal: Bellman backup with dense reward
                    max_q_next = self._max_q_batch(step + 1, next_states)
                    targets = df["reward"].values + gamma * max_q_next

                # Build feature matrix: state + action flags
                X = df[["s_%s" % c for c in self.state_cols]].rename(
                    columns={"s_%s" % c: c for c in self.state_cols}
                ).copy()
                for d in self.drug_cols:
                    X["%s_action" % d] = df["%s_action" % d].values
                X = X[self.feature_cols]

                reg = lgb.LGBMRegressor(**_LGB_PARAMS)
                reg.fit(X, targets)
                self.q_models[step] = reg

            # Print Q-value stats for monitoring
            terminal_df = step_dfs[N_RL_STEPS - 1]
            q_vals = self.q_models[N_RL_STEPS - 1].predict(
                terminal_df[["s_%s" % c for c in self.state_cols]].rename(
                    columns={"s_%s" % c: c for c in self.state_cols}
                ).assign(**{
                    "%s_action" % d: terminal_df["%s_action" % d].values
                    for d in self.drug_cols
                })[self.feature_cols]
            )
            print(
                "    Q-value stats (terminal): mean=%.4f std=%.4f"
                % (q_vals.mean(), q_vals.std()),
                flush=True,
            )

        self.n_iter = n_iter
        return self

    def _q_predict_batch(
        self,
        step: int,
        states: pd.DataFrame,
        combo: tuple[int, ...],
    ) -> np.ndarray:
        """Q(states, combo) for all rows."""
        model = self.q_models[step]
        if model is None:
            return np.zeros(len(states))
        X = states[self.state_cols].copy()
        for d, v in zip(self.drug_cols, combo):
            X["%s_action" % d] = v
        X = X[self.feature_cols]
        return model.predict(X)

    def _max_q_batch(self, step: int, states: pd.DataFrame) -> np.ndarray:
        """max_a Q(states, a) over all 32 combos."""
        q_values = np.stack(
            [self._q_predict_batch(step, states, combo) for combo in ALL_COMBOS],
            axis=1,
        )
        return q_values.max(axis=1)

    # ── Inference ─────────────────────────────────────────────────────

    def predict(
        self,
        state_dict: dict[str, float],
        combo: tuple[int, ...],
        step: int = 0,
    ) -> float:
        """Return Q(state, combo) for a single state."""
        model = self.q_models[step]
        if model is None:
            return 0.0
        row = {c: state_dict.get(c, np.nan) for c in self.state_cols}
        for d, v in zip(self.drug_cols, combo):
            row["%s_action" % d] = v
        X = pd.DataFrame([row])[self.feature_cols]
        return float(model.predict(X)[0])

    def best_combo(
        self, state_dict: dict[str, float], step: int = 0
    ) -> dict[str, int]:
        """Return argmax_a Q(state, a) as a drug -> 0/1 dict."""
        model = self.q_models[step]
        if model is None:
            return {d: 0 for d in self.drug_cols}

        row = {c: state_dict.get(c, np.nan) for c in self.state_cols}
        rows = []
        for combo in ALL_COMBOS:
            r = dict(row)
            for d, v in zip(self.drug_cols, combo):
                r["%s_action" % d] = v
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

    # ── Persistence ───────────────────────────────────────────────────

    def save(self, dir_path: Path | str) -> None:
        """Save Q-models and metadata."""
        d = Path(dir_path)
        d.mkdir(parents=True, exist_ok=True)
        for step, model in self.q_models.items():
            if model is not None:
                joblib.dump(model, d / ("q_model_step%d.joblib" % step))
        meta = {
            "algorithm": "FittedQIterationV3",
            "n_iter": self.n_iter,
            "gamma": self.gamma,
            "n_patients": self.n_patients,
            "state_cols": self.state_cols,
            "drug_cols": self.drug_cols,
            "feature_cols": self.feature_cols,
            "n_rl_steps": N_RL_STEPS,
            "sim_steps_per_rl": SIM_STEPS_PER_RL,
            "reward_type": "weighted_lab_delta",
            "lab_weights": LAB_WEIGHTS,
            "normal_ranges": NORMAL_RANGES,
        }
        (d / "metadata.json").write_text(json.dumps(meta, indent=2))
        log.info("Saved FQI-V3 Q-models to %s", d)

    @classmethod
    def load(cls, dir_path: Path | str) -> "FittedQIterationV3":
        """Load a previously saved FittedQIterationV3."""
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
            p = d / ("q_model_step%d.joblib" % step)
            if p.exists():
                obj.q_models[step] = joblib.load(p)
        log.info("Loaded FQI-V3 from %s (%d iterations)", d, obj.n_iter)
        return obj
