from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .data import prepare_hourly_data
from .dynamics import fit_dynamics_model
from .policies import build_policies
from .qa_trust import done_metrics, evaluate_gate, ood_metrics, rank_stability, realism_metrics
from .readmission import evaluate_readmission_model, fit_readmission_model
from .rollout import RolloutOutputs, rollout_policies
from .variant_search import apply_variant


@dataclass(frozen=True)
class VariantRunResult:
    variant: str
    metrics: dict[str, Any]
    policy_metrics: pd.DataFrame
    trajectories: pd.DataFrame
    gate: Any


def _hash_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(1024 * 1024)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def build_data_manifest(paths: list[Path]) -> dict[str, Any]:
    items = []
    for p in paths:
        if not p.exists():
            continue
        st = p.stat()
        items.append(
            {
                "path": str(p.resolve()),
                "size_bytes": int(st.st_size),
                "modified_utc": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).isoformat(),
                "sha256": _hash_file(p),
            }
        )
    return {"generated_at_utc": datetime.now(timezone.utc).isoformat(), "files": items}


def evaluate_variant(
    variant: str,
    base_cfg: dict[str, Any],
    transitions: pd.DataFrame,
    episodes: pd.DataFrame,
    thresholds: dict[str, float],
    baseline_readmit_valid: dict[str, Any] | None = None,
) -> VariantRunResult:
    cfg = apply_variant(base_cfg, variant)
    hourly = prepare_hourly_data(transitions, cfg)
    dynamics = fit_dynamics_model(hourly.train, cfg, hourly.state_cols, hourly.action_cols)
    readmit = fit_readmission_model(episodes, cfg)
    readmit_valid = evaluate_readmission_model(readmit, episodes, split="valid")
    readmit_test = evaluate_readmission_model(readmit, episodes, split="test")

    observed_actions = hourly.train[hourly.action_cols].dropna().to_numpy(dtype=float)
    policies = [p for p in build_policies(observed_actions=observed_actions) if p.name in set(cfg["actions"]["policies"])]
    seeds = list(cfg.get("simulation", {}).get("seeds", [int(cfg["simulation"]["seed"])]))

    all_metrics = []
    seed_outputs: list[RolloutOutputs] = []
    for seed in seeds:
        out = rollout_policies(
            start_states=hourly.eval_starts,
            dynamics=dynamics,
            readmit=readmit,
            policies=policies,
            max_steps=int(cfg["simulation"]["max_steps"]),
            n_rollouts=int(cfg["simulation"]["n_rollouts_per_policy"]),
            done_threshold=float(cfg["simulation"]["done_threshold"]),
            seed=int(seed),
        )
        seed_outputs.append(out)
        pm = out.policy_metrics.copy()
        pm["seed"] = int(seed)
        all_metrics.append(pm)
    policy_metrics_all = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    policy_metrics = (
        policy_metrics_all.drop(columns=["seed"])
        .groupby("policy", as_index=False)
        .agg(
            n_rollouts=("n_rollouts", "sum"),
            mean_pred_readmit_risk=("mean_pred_readmit_risk", "mean"),
            std_pred_readmit_risk=("mean_pred_readmit_risk", "std"),
            mean_rollout_hours=("mean_rollout_hours", "mean"),
            termination_rate=("termination_rate", "mean"),
        )
    )
    trajectories = pd.concat([o.trajectories.assign(seed=int(s)) for o, s in zip(seed_outputs, seeds)], ignore_index=True)

    realism = realism_metrics(trajectories, hourly.state_cols, dynamics.state_bounds)
    ood = ood_metrics(trajectories, hourly.state_cols, hourly.train[hourly.state_cols])
    done = done_metrics(dynamics, float(cfg["simulation"]["done_threshold"]), policy_metrics)
    # readmission ECE from valid predictions
    if readmit_valid["n"] and readmit_valid["n"] > 0:
        m = episodes["split"] == "valid"
        y = pd.to_numeric(episodes.loc[m, "readmit_30d"], errors="coerce").fillna(0).astype(int).to_numpy()
        X = episodes.loc[m, readmit.feature_cols].apply(pd.to_numeric, errors="coerce")
        p = readmit.pipe.predict_proba(X)[:, 1]
        # reusing simple ece approximation
        from .qa_trust import _ece_binary

        readmit_valid["ece"] = float(_ece_binary(y, p, n_bins=10))
    else:
        readmit_valid["ece"] = 0.0
    stability = rank_stability([o.policy_metrics for o in seed_outputs])

    metric_payload: dict[str, Any] = {
        "variant": variant,
        "realism": realism,
        "ood": ood,
        "done": done,
        "readmission_valid": readmit_valid,
        "readmission_test": readmit_test,
        "stability": stability,
        "n_train_rows": int(len(hourly.train)),
        "n_eval_starts": int(len(hourly.eval_starts)),
    }
    gate = evaluate_gate(metric_payload, thresholds=thresholds, baseline_readmit=baseline_readmit_valid)
    metric_payload["gate"] = {"status": gate.status, "rules_passed": gate.rules_passed, "rules_failed": gate.rules_failed, "trust_score": gate.trust_score}
    return VariantRunResult(
        variant=variant,
        metrics=metric_payload,
        policy_metrics=policy_metrics.sort_values("mean_pred_readmit_risk", ascending=True).reset_index(drop=True),
        trajectories=trajectories,
        gate=gate,
    )
