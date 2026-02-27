from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class DynamicsModel:
    state_models: dict[str, Pipeline]
    done_model: Pipeline
    state_cols: list[str]
    action_cols: list[str]
    x_cols: list[str]
    train_one_step_df: pd.DataFrame
    state_bounds: dict[str, tuple[float, float]]
    train_mean: np.ndarray
    train_cov: np.ndarray
    clip_to_bounds: bool = False

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.concatenate([state, action], axis=0).reshape(1, -1)
        x = pd.DataFrame(values, columns=self.x_cols)
        next_state = np.array([float(self.state_models[c].predict(x)[0]) for c in self.state_cols], dtype=float)
        if self.clip_to_bounds:
            for i, c in enumerate(self.state_cols):
                lo, hi = self.state_bounds[c]
                next_state[i] = float(np.clip(next_state[i], lo, hi))
        done_prob = float(self.done_model.predict_proba(x)[0, 1])
        return next_state, done_prob


def _build_one_step_frame(df: pd.DataFrame, state_cols: list[str], action_cols: list[str]) -> pd.DataFrame:
    grouped = df.sort_values(["episode_id", "t"]).groupby("episode_id", sort=False)
    rows = []
    for _, g in grouped:
        g = g.reset_index(drop=True)
        if len(g) < 2:
            continue
        for i in range(len(g) - 1):
            cur = g.iloc[i]
            nxt = g.iloc[i + 1]
            rec: dict[str, float] = {}
            for c in state_cols:
                rec[f"s_{c}"] = cur[c]
                rec[f"n_{c}"] = nxt[c]
            for c in action_cols:
                rec[f"a_{c}"] = cur[c]
            rec["done_next"] = int(nxt["done"] == 1)
            rows.append(rec)
    return pd.DataFrame(rows)


def fit_dynamics_model(df: pd.DataFrame, cfg: dict[str, Any], state_cols: list[str], action_cols: list[str]) -> DynamicsModel:
    train = _build_one_step_frame(df, state_cols, action_cols)
    if train.empty:
        raise ValueError("No valid state transitions for dynamics training.")

    x_cols = [f"s_{c}" for c in state_cols] + [f"a_{c}" for c in action_cols]
    X = train[x_cols]

    dcfg = cfg.get("dynamics_model", {})
    alpha = float(dcfg.get("alpha", 1.0))
    use_scaling = bool(dcfg.get("use_scaling", False))
    done_class_weight = dcfg.get("done_class_weight", None)
    done_c = float(dcfg.get("done_c", 1.0))
    state_models: dict[str, Pipeline] = {}
    for c in state_cols:
        y = pd.to_numeric(train[f"n_{c}"], errors="coerce")
        mask = y.notna()
        steps: list[tuple[str, Any]] = [("imp", SimpleImputer(strategy="median"))]
        if use_scaling:
            steps.append(("scaler", StandardScaler()))
        steps.append(("reg", Ridge(alpha=alpha)))
        pipe = Pipeline(steps)
        pipe.fit(X[mask], y[mask])
        state_models[c] = pipe

    y_done = pd.to_numeric(train["done_next"], errors="coerce").fillna(0).astype(int)
    done_steps: list[tuple[str, Any]] = [("imp", SimpleImputer(strategy="median"))]
    if use_scaling:
        done_steps.append(("scaler", StandardScaler()))
    done_steps.append(
        (
            "clf",
            LogisticRegression(max_iter=1000, random_state=42, class_weight=done_class_weight, C=done_c),
        )
    )
    done_model = Pipeline(done_steps)
    done_model.fit(X, y_done)
    ql = float(cfg.get("qa", {}).get("reference_quantiles", [0.01, 0.99])[0])
    qh = float(cfg.get("qa", {}).get("reference_quantiles", [0.01, 0.99])[1])
    state_bounds = {c: (float(pd.to_numeric(df[c], errors="coerce").quantile(ql)), float(pd.to_numeric(df[c], errors="coerce").quantile(qh))) for c in state_cols}
    x_vals = X.apply(pd.to_numeric, errors="coerce").fillna(X.median(numeric_only=True)).to_numpy(dtype=float)
    train_mean = np.nanmean(x_vals, axis=0)
    train_cov = np.cov(x_vals, rowvar=False)
    if np.ndim(train_cov) == 0:
        train_cov = np.array([[float(train_cov)]], dtype=float)
    return DynamicsModel(
        state_models=state_models,
        done_model=done_model,
        state_cols=state_cols,
        action_cols=action_cols,
        x_cols=x_cols,
        train_one_step_df=train,
        state_bounds=state_bounds,
        train_mean=train_mean,
        train_cov=train_cov,
        clip_to_bounds=bool(dcfg.get("clip_to_bounds", False)),
    )
