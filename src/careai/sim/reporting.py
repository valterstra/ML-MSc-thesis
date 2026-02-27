"""Reporting helpers for Sim v1."""

from __future__ import annotations

from typing import Any

import pandas as pd


def summary_payload(
    cfg: dict[str, Any],
    metrics_df: pd.DataFrame,
    weighting: dict[str, Any] | None = None,
) -> dict[str, Any]:
    best = metrics_df.iloc[0].to_dict() if len(metrics_df) else {}
    payload = {
        "sim_type": "one_step_contextual_bandit_v1",
        "n_policies": int(len(metrics_df)),
        "n_episodes_per_policy": int(cfg["simulation"]["n_episodes"]),
        "reward": {
            "formula": "reward = -readmitted_30d - action_cost",
            "cost_low": float(cfg["reward"]["cost_low"]),
            "cost_high": float(cfg["reward"]["cost_high"]),
        },
        "best_policy_by_mean_reward": best,
        "policies": metrics_df.to_dict(orient="records"),
    }
    if weighting is not None:
        payload["weighting"] = weighting
    return payload


def summary_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Sim v1 Summary",
        "",
        f"- sim_type: {payload['sim_type']}",
        f"- n_policies: {payload['n_policies']}",
        f"- n_episodes_per_policy: {payload['n_episodes_per_policy']}",
        "",
        "## Reward",
        f"- formula: {payload['reward']['formula']}",
        f"- cost_low: {payload['reward']['cost_low']}",
        f"- cost_high: {payload['reward']['cost_high']}",
        "",
        f"- weighting_mode: {payload.get('weighting', {}).get('mode', 'none')}",
        "",
        "## Best policy by mean reward",
    ]
    best = payload.get("best_policy_by_mean_reward", {})
    for k, v in best.items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    if "weighting" in payload:
        lines.append("## Weighting")
        for k, v in payload.get("weighting", {}).items():
            lines.append(f"- {k}: {v}")
        lines.append("")
    lines.append("## Policy metrics")
    for row in payload.get("policies", []):
        lines.append(
            f"- {row['policy']}: mean_reward={row['mean_reward']:.6f}, "
            f"readmit_rate={row['readmit_rate']:.6f}, "
            f"high_support_rate={row['high_support_rate']:.6f}"
        )
    return "\n".join(lines) + "\n"
