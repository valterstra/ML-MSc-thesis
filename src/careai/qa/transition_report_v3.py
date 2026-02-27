"""QA summary for transition v3 hourly dataset."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_transition_v3_qa(df: pd.DataFrame) -> dict[str, Any]:
    episode_len = df.groupby("episode_id").size() if "episode_id" in df.columns else pd.Series(dtype=int)
    action_cols = [c for c in ["a_t_vaso", "a_t_vent", "a_t_crrt"] if c in df.columns]
    state_cols = [c for c in df.columns if c.startswith("s_t_")]
    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
        "episodes": int(df["episode_id"].nunique()) if "episode_id" in df.columns else 0,
        "mean_episode_length": float(episode_len.mean()) if len(episode_len) else 0.0,
        "y_t1_rate": float(df["y_t1"].mean()) if "y_t1" in df.columns else None,
        "action_counts": df["a_t"].value_counts(dropna=False).to_dict() if "a_t" in df.columns else {},
        "action_component_rates": {c: float(df[c].mean()) for c in action_cols},
        "state_missingness": df[state_cols].isna().mean().to_dict() if state_cols else {},
        "sofa_missing_for_label": float(df["s_t1_sofa"].isna().mean()) if "s_t1_sofa" in df.columns else None,
        "episode_length_distribution": episode_len.value_counts().sort_index().to_dict() if len(episode_len) else {},
        "missingness_top10": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transition v3 Hourly QA Summary",
        "",
        f"- Rows: {report['rows']}",
        f"- Patients: {report['patients']}",
        f"- Episodes: {report['episodes']}",
        f"- Mean episode length: {report['mean_episode_length']:.4f}",
        f"- y_t1 rate: {report.get('y_t1_rate')}",
        f"- SOFA missing rate for next-step label: {report.get('sofa_missing_for_label')}",
        "",
        "## Action counts",
    ]
    for k, v in report.get("action_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Action component rates")
    for k, v in report.get("action_component_rates", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Episode length distribution")
    for k, v in report.get("episode_length_distribution", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## State missingness")
    for k, v in report.get("state_missingness", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Missingness top 10")
    for k, v in report.get("missingness_top10", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    return "\n".join(lines) + "\n"

