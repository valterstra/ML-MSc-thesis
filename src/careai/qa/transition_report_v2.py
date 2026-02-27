"""QA summary for transition v2 multi-step dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_transition_v2_qa(df: pd.DataFrame) -> dict[str, Any]:
    episode_len = df.groupby("episode_id").size() if "episode_id" in df.columns else pd.Series(dtype=int)
    step_action = {}
    step_y = {}
    if "episode_step" in df.columns and "a_t" in df.columns:
        for step, g in df.groupby("episode_step"):
            step_action[str(int(step))] = g["a_t"].value_counts(dropna=False).to_dict()
    if "episode_step" in df.columns and "y_t1" in df.columns:
        for step, g in df.groupby("episode_step"):
            step_y[str(int(step))] = float(g["y_t1"].mean())
    next_cols = [c for c in ["s_t1_length", "s_t1_acuity", "s_t1_comorbidity", "s_t1_lace"] if c in df.columns]
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
        "episodes": int(df["episode_id"].nunique()) if "episode_id" in df.columns else 0,
        "mean_episode_length": float(episode_len.mean()) if len(episode_len) else 0.0,
        "episode_length_distribution": episode_len.value_counts().sort_index().to_dict() if len(episode_len) else {},
        "terminal_rate": float((df["done"] == 1).mean()) if "done" in df.columns else None,
        "non_terminal_rate": float((df["done"] == 0).mean()) if "done" in df.columns else None,
        "y_t1_rate": float(df["y_t1"].mean()) if "y_t1" in df.columns else None,
        "action_counts": df["a_t"].value_counts(dropna=False).to_dict() if "a_t" in df.columns else {},
        "action_by_step": step_action,
        "y_t1_by_step": step_y,
        "next_state_missingness": df[next_cols].isna().mean().to_dict() if next_cols else {},
        "missingness_top10": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transition v2 QA Summary",
        "",
        f"- Rows: {report['rows']}",
        f"- Patients: {report['patients']}",
        f"- Episodes: {report['episodes']}",
        f"- Mean episode length: {report['mean_episode_length']:.4f}",
        f"- y_t1 rate: {report.get('y_t1_rate')}",
        f"- terminal_rate: {report.get('terminal_rate')}",
        f"- non_terminal_rate: {report.get('non_terminal_rate')}",
        "",
        "## Episode length distribution",
    ]
    for k, v in report.get("episode_length_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Action counts")
    for k, v in report.get("action_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## y_t1 by step")
    for k, v in report.get("y_t1_by_step", {}).items():
        lines.append(f"- step {k}: {v:.6f}")
    lines.append("")
    lines.append("## Next-state missingness")
    for k, v in report.get("next_state_missingness", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Missingness top 10")
    for k, v in report.get("missingness_top10", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    return "\n".join(lines) + "\n"

