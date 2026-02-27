from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


def _ece_binary(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    n = len(y_true)
    if n == 0:
        return 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        idx = (y_prob >= lo) & (y_prob < hi if i < n_bins - 1 else y_prob <= hi)
        if np.any(idx):
            acc = float(np.mean(y_true[idx]))
            conf = float(np.mean(y_prob[idx]))
            total += abs(acc - conf) * (float(np.sum(idx)) / float(n))
    return float(total)


def realism_metrics(trajectories: pd.DataFrame, state_cols: list[str], bounds: dict[str, tuple[float, float]]) -> dict[str, Any]:
    feat = {}
    out_counts = 0
    total_counts = 0
    for c in state_cols:
        arr = pd.to_numeric(trajectories[c], errors="coerce").to_numpy()
        lo, hi = bounds[c]
        valid = np.isfinite(arr)
        if np.sum(valid) == 0:
            feat[c] = 0.0
            continue
        out = ((arr[valid] < lo) | (arr[valid] > hi)).astype(int)
        rate = float(np.mean(out))
        feat[c] = rate
        out_counts += int(np.sum(out))
        total_counts += int(np.sum(valid))
    global_rate = float(out_counts / total_counts) if total_counts else 0.0
    return {"global_out_of_range_rate": global_rate, "per_feature_out_of_range_rate": feat}


def ood_metrics(trajectories: pd.DataFrame, state_cols: list[str], train_states: pd.DataFrame, threshold_quantile: float = 0.995) -> dict[str, float]:
    x_train = train_states[state_cols].apply(pd.to_numeric, errors="coerce").fillna(train_states[state_cols].median(numeric_only=True)).to_numpy(dtype=float)
    x_sim = trajectories[state_cols].apply(pd.to_numeric, errors="coerce").fillna(train_states[state_cols].median(numeric_only=True)).to_numpy(dtype=float)
    mu = np.mean(x_train, axis=0)
    cov = np.cov(x_train, rowvar=False)
    if np.ndim(cov) == 0:
        cov = np.array([[float(cov)]], dtype=float)
    cov = cov + np.eye(cov.shape[0]) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    d_train = np.sqrt(np.einsum("ij,jk,ik->i", x_train - mu, inv_cov, x_train - mu))
    d_sim = np.sqrt(np.einsum("ij,jk,ik->i", x_sim - mu, inv_cov, x_sim - mu))
    thr = float(np.quantile(d_train, threshold_quantile))
    exceed = float(np.mean(d_sim > thr)) if len(d_sim) else 0.0
    return {"ood_threshold": thr, "ood_exceedance_rate": exceed, "mean_mahalanobis_sim": float(np.mean(d_sim)) if len(d_sim) else 0.0}


def done_metrics(dynamics_model: Any, done_threshold: float, policy_metrics: pd.DataFrame) -> dict[str, float]:
    train = dynamics_model.train_one_step_df.copy()
    x = train[dynamics_model.x_cols].apply(pd.to_numeric, errors="coerce")
    y = pd.to_numeric(train["done_next"], errors="coerce").fillna(0).astype(int).to_numpy()
    probs = dynamics_model.done_model.predict_proba(x)[:, 1]
    brier = float(np.mean((y - probs) ** 2))
    ece = _ece_binary(y, probs, n_bins=10)
    term_rate = float(policy_metrics["termination_rate"].mean()) if len(policy_metrics) else 0.0
    return {
        "done_prob_mean": float(np.mean(probs)),
        "done_brier": brier,
        "done_ece": ece,
        "termination_rate_mean": term_rate,
        "done_threshold": float(done_threshold),
    }


def rank_stability(seed_policy_frames: list[pd.DataFrame]) -> dict[str, Any]:
    if len(seed_policy_frames) < 2:
        return {"mean_spearman": 1.0, "top_policy_agreement": 1}
    from scipy.stats import spearmanr

    maps = []
    tops = []
    for df in seed_policy_frames:
        ordered = df.sort_values("mean_pred_readmit_risk", ascending=True)["policy"].tolist()
        tops.append(ordered[0] if ordered else None)
        maps.append({p: i for i, p in enumerate(ordered)})
    policies = sorted({p for m in maps for p in m})
    scores = []
    for i in range(len(maps)):
        for j in range(i + 1, len(maps)):
            a = np.array([maps[i].get(p, len(policies)) for p in policies], dtype=float)
            b = np.array([maps[j].get(p, len(policies)) for p in policies], dtype=float)
            rho = spearmanr(a, b).correlation
            scores.append(float(1.0 if np.isnan(rho) else rho))
    top_agreement = max(tops.count(t) for t in set(tops) if t is not None) if tops else 0
    return {"mean_spearman": float(np.mean(scores)) if scores else 1.0, "top_policy_agreement": int(top_agreement)}


@dataclass(frozen=True)
class GateResult:
    status: str
    rules_passed: list[str]
    rules_failed: list[str]
    trust_score: float


def evaluate_gate(metrics: dict[str, Any], thresholds: dict[str, float], baseline_readmit: dict[str, Any] | None = None) -> GateResult:
    passed: list[str] = []
    failed: list[str] = []

    def check(name: str, ok: bool) -> None:
        (passed if ok else failed).append(name)

    realism = metrics["realism"]
    ood = metrics["ood"]
    done = metrics["done"]
    readm = metrics["readmission_valid"]
    stab = metrics["stability"]
    baseline_brier = None if baseline_readmit is None else baseline_readmit.get("brier")
    baseline_auroc = None if baseline_readmit is None else baseline_readmit.get("auroc")

    check("global_out_of_range", realism["global_out_of_range_rate"] <= float(thresholds["global_out_of_range_max"]))
    check(
        "per_feature_out_of_range",
        all(v <= float(thresholds["per_feature_out_of_range_max"]) for v in realism["per_feature_out_of_range_rate"].values()),
    )
    check("ood_exceedance", ood["ood_exceedance_rate"] <= float(thresholds["ood_exceedance_max"]))
    check("done_mean_min", done["done_prob_mean"] >= float(thresholds["done_mean_min"]))
    check("done_mean_max", done["done_prob_mean"] <= float(thresholds["done_mean_max"]))
    check("termination_rate_min", done["termination_rate_mean"] >= float(thresholds["termination_rate_min"]))
    check("termination_rate_max", done["termination_rate_mean"] <= float(thresholds["termination_rate_max"]))
    check("done_ece", done["done_ece"] <= float(thresholds["done_ece_max"]))
    if baseline_brier is None or readm["brier"] is None:
        check("readmit_brier_delta", True)
    else:
        check("readmit_brier_delta", (readm["brier"] - float(baseline_brier)) <= float(thresholds["readmit_brier_delta_max"]))
    if baseline_auroc is None or readm["auroc"] is None:
        check("readmit_auroc_delta", True)
    else:
        check("readmit_auroc_delta", (readm["auroc"] - float(baseline_auroc)) >= float(thresholds["readmit_auroc_delta_min"]))
    check("rank_consistency", stab["mean_spearman"] >= float(thresholds["rank_consistency_min"]))
    check("top_policy_agreement", stab["top_policy_agreement"] >= int(thresholds["top_policy_agreement_min"]))

    trust_score = (
        0.30 * float(realism["global_out_of_range_rate"])
        + 0.25 * float(ood["ood_exceedance_rate"])
        + 0.20 * float(done["done_ece"])
        + 0.15 * float(readm.get("ece", 0.0) or 0.0)
        + 0.10 * (1.0 - float(stab["mean_spearman"]))
    )
    return GateResult(status="pass" if not failed else "fail", rules_passed=passed, rules_failed=failed, trust_score=float(trust_score))

