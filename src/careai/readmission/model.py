"""Baseline readmission model."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline


def train_readmission_baseline(df: pd.DataFrame, cfg: dict[str, Any]) -> dict[str, Any]:
    feature_cols = list(cfg["features"]["include"])
    work = df.copy()
    missing = [c for c in feature_cols + ["split", "readmit_30d", "episode_id", "patient_id", "index_hadm_id"] if c not in work.columns]
    if missing:
        raise ValueError(f"Episode table missing required columns: {missing}")

    y = pd.to_numeric(work["readmit_30d"], errors="coerce").fillna(0).astype(int)
    X = work[feature_cols].apply(pd.to_numeric, errors="coerce")

    train_idx = work["split"] == "train"
    valid_idx = work["split"] == "valid"
    test_idx = work["split"] == "test"

    model_cfg = cfg["model"]
    clf = LogisticRegression(
        max_iter=int(model_cfg.get("max_iter", 1000)),
        C=float(model_cfg.get("c", 1.0)),
        random_state=int(model_cfg.get("random_state", 42)),
    )
    pipe = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", clf)])
    pipe.fit(X[train_idx], y[train_idx])

    def eval_split(mask: pd.Series) -> dict[str, Any]:
        if int(mask.sum()) == 0:
            return {"n": 0, "auroc": None, "auprc": None, "brier": None, "prevalence": None}
        probs = pipe.predict_proba(X[mask])[:, 1]
        yy = y[mask].to_numpy()
        if len(np.unique(yy)) == 1:
            auroc = None
            auprc = float(average_precision_score(yy, probs))
        else:
            auroc = float(roc_auc_score(yy, probs))
            auprc = float(average_precision_score(yy, probs))
        brier = float(brier_score_loss(yy, probs))
        return {"n": int(mask.sum()), "auroc": auroc, "auprc": auprc, "brier": brier, "prevalence": float(yy.mean())}

    valid_metrics = eval_split(valid_idx)
    test_metrics = eval_split(test_idx)

    pred_valid = work.loc[valid_idx, ["episode_id", "patient_id", "index_hadm_id", "split", "readmit_30d"]].copy()
    pred_test = work.loc[test_idx, ["episode_id", "patient_id", "index_hadm_id", "split", "readmit_30d"]].copy()
    if len(pred_valid):
        pred_valid["pred_readmit_prob"] = pipe.predict_proba(X[valid_idx])[:, 1]
    else:
        pred_valid["pred_readmit_prob"] = []
    if len(pred_test):
        pred_test["pred_readmit_prob"] = pipe.predict_proba(X[test_idx])[:, 1]
    else:
        pred_test["pred_readmit_prob"] = []

    return {
        "metrics": {"valid": valid_metrics, "test": test_metrics},
        "pred_valid": pred_valid,
        "pred_test": pred_test,
        "feature_cols": feature_cols,
    }
