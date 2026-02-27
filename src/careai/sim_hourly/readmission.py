from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass(frozen=True)
class ReadmissionModel:
    pipe: Pipeline
    feature_cols: list[str]

    def predict_prob(self, features: dict[str, float]) -> float:
        x = pd.DataFrame([{c: features.get(c, np.nan) for c in self.feature_cols}])
        return float(self.pipe.predict_proba(x)[:, 1][0])


def fit_readmission_model(episode_df: pd.DataFrame, cfg: dict[str, Any]) -> ReadmissionModel:
    feature_cols = list(cfg["readmission_model"]["features"])
    missing = [c for c in feature_cols + ["split", "readmit_30d"] if c not in episode_df.columns]
    if missing:
        raise ValueError(f"Episode table missing columns for readmission model: {missing}")
    work = episode_df.copy()
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work["readmit_30d"], errors="coerce").fillna(0).astype(int)
    train = work["split"] == "train"
    if int(train.sum()) == 0:
        train = pd.Series([True] * len(work))
    mcfg = cfg["readmission_model"]
    steps: list[tuple[str, Any]] = [("imp", SimpleImputer(strategy="median"))]
    if bool(mcfg.get("use_scaling", False)):
        steps.append(("scaler", StandardScaler()))
    steps.append(
        (
            "clf",
            LogisticRegression(
                max_iter=int(mcfg.get("max_iter", 2000)),
                C=float(mcfg.get("c", 1.0)),
                random_state=int(mcfg.get("random_state", 42)),
            ),
        )
    )
    pipe = Pipeline(steps)
    pipe.fit(X[train], y[train])
    return ReadmissionModel(pipe=pipe, feature_cols=feature_cols)


def evaluate_readmission_model(model: ReadmissionModel, episode_df: pd.DataFrame, split: str = "valid") -> dict[str, float | int | None]:
    work = episode_df.copy()
    if "split" not in work.columns or "readmit_30d" not in work.columns:
        return {"n": 0, "auroc": None, "auprc": None, "brier": None}
    mask = work["split"] == split
    if int(mask.sum()) == 0:
        return {"n": 0, "auroc": None, "auprc": None, "brier": None}
    X = work.loc[mask, model.feature_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(work.loc[mask, "readmit_30d"], errors="coerce").fillna(0).astype(int).to_numpy()
    probs = model.pipe.predict_proba(X)[:, 1]
    if len(np.unique(y)) < 2:
        auroc = None
        auprc = float(average_precision_score(y, probs))
    else:
        auroc = float(roc_auc_score(y, probs))
        auprc = float(average_precision_score(y, probs))
    brier = float(brier_score_loss(y, probs))
    return {"n": int(mask.sum()), "auroc": auroc, "auprc": auprc, "brier": brier}


def summarize_trajectory_for_readmit(states: list[np.ndarray], actions: list[np.ndarray], state_cols: list[str]) -> dict[str, float]:
    arr = np.asarray(states, dtype=float)
    acts = np.asarray(actions, dtype=float) if actions else np.zeros((0, 3), dtype=float)
    out: dict[str, float] = {"episode_hours": float(len(states))}

    for i, c in enumerate(state_cols):
        vals = arr[:, i] if arr.size else np.array([np.nan])
        out[f"last_{c}"] = float(vals[-1])
        out[f"mean_{c}"] = float(np.nanmean(vals))
        if c == "s_t_sofa":
            out["max_s_t_sofa"] = float(np.nanmax(vals))
            out["delta_s_t_sofa"] = float(vals[-1] - vals[0]) if len(vals) else 0.0
        if c == "s_t_mbp":
            out["min_s_t_mbp"] = float(np.nanmin(vals))
            out["delta_s_t_mbp"] = float(vals[-1] - vals[0]) if len(vals) else 0.0

    if len(acts):
        out["frac_vaso"] = float(np.mean(acts[:, 0] > 0.5))
        out["frac_vent"] = float(np.mean(acts[:, 1] > 0.5))
        out["frac_crrt"] = float(np.mean(acts[:, 2] > 0.5))
        out["any_vaso"] = float(np.any(acts[:, 0] > 0.5))
        out["any_vent"] = float(np.any(acts[:, 1] > 0.5))
        out["any_crrt"] = float(np.any(acts[:, 2] > 0.5))
    else:
        out.update(
            {
                "frac_vaso": 0.0,
                "frac_vent": 0.0,
                "frac_crrt": 0.0,
                "any_vaso": 0.0,
                "any_vent": 0.0,
                "any_crrt": 0.0,
            }
        )
    return out
