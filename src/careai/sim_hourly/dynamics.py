from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import Pipeline


@dataclass(frozen=True)
class DynamicsModel:
    state_models: dict[str, Pipeline]
    done_model: Pipeline
    state_cols: list[str]
    action_cols: list[str]

    def predict_next(self, state: np.ndarray, action: np.ndarray) -> tuple[np.ndarray, float]:
        values = np.concatenate([state, action], axis=0).reshape(1, -1)
        x_cols = [f"s_{c}" for c in self.state_cols] + [f"a_{c}" for c in self.action_cols]
        x = pd.DataFrame(values, columns=x_cols)
        next_state = np.array([float(self.state_models[c].predict(x)[0]) for c in self.state_cols], dtype=float)
        done_prob = float(self.done_model.predict_proba(x)[0, 1])
        return next_state, done_prob


def fit_dynamics_model(df: pd.DataFrame, cfg: dict[str, Any], state_cols: list[str], action_cols: list[str]) -> DynamicsModel:
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
    train = pd.DataFrame(rows)
    if train.empty:
        raise ValueError("No valid state transitions for dynamics training.")

    x_cols = [f"s_{c}" for c in state_cols] + [f"a_{c}" for c in action_cols]
    X = train[x_cols]

    alpha = float(cfg.get("dynamics_model", {}).get("alpha", 1.0))
    state_models: dict[str, Pipeline] = {}
    for c in state_cols:
        y = pd.to_numeric(train[f"n_{c}"], errors="coerce")
        mask = y.notna()
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")), ("reg", Ridge(alpha=alpha))])
        pipe.fit(X[mask], y[mask])
        state_models[c] = pipe

    y_done = pd.to_numeric(train["done_next"], errors="coerce").fillna(0).astype(int)
    done_model = Pipeline(
        [
            ("imp", SimpleImputer(strategy="median")),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    done_model.fit(X, y_done)
    return DynamicsModel(state_models=state_models, done_model=done_model, state_cols=state_cols, action_cols=action_cols)
