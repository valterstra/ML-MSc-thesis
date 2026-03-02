from __future__ import annotations

from typing import Any

import pandas as pd


def build_summary(cfg: dict[str, Any], metrics: pd.DataFrame, n_train_rows: int, n_start_rows: int) -> dict[str, Any]:
    best = None
    if len(metrics):
        best = metrics.sort_values("mean_pred_readmit_risk", ascending=True).iloc[0]["policy"]
    return {
        "n_train_rows": int(n_train_rows),
        "n_start_rows": int(n_start_rows),
        "max_steps": int(cfg["simulation"]["max_steps"]),
        "n_rollouts_per_policy": int(cfg["simulation"]["n_rollouts_per_policy"]),
        "best_policy_lowest_risk": best,
        "policy_metrics": metrics.to_dict(orient="records"),
    }


def summary_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Hourly Simulator Summary", ""]
    lines.append(f"- train rows: {payload['n_train_rows']}")
    lines.append(f"- start rows: {payload['n_start_rows']}")
    lines.append(f"- max steps: {payload['max_steps']}")
    lines.append(f"- rollouts per policy: {payload['n_rollouts_per_policy']}")
    lines.append(f"- best policy (lowest predicted readmit risk): {payload['best_policy_lowest_risk']}")
    lines.append("")
    lines.append("## Policy Metrics")
    for row in payload["policy_metrics"]:
        lines.append(
            f"- `{row['policy']}` risk={row['mean_pred_readmit_risk']:.4f}, "
            f"len={row['mean_rollout_hours']:.2f}, term={row['termination_rate']:.3f}"
        )
    return "\n".join(lines) + "\n"
