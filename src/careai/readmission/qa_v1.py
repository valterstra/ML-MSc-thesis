"""QA utilities for readmission episode table v1."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_episode_qa_v1(df: pd.DataFrame) -> dict[str, Any]:
    out = {
        "rows": int(len(df)),
        "episodes": int(df["episode_id"].nunique()) if "episode_id" in df.columns else 0,
        "patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
        "readmit_30d_rate": float(df["readmit_30d"].mean()) if "readmit_30d" in df.columns and len(df) else None,
        "split_counts": df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {},
        "readmit_by_split": df.groupby("split")["readmit_30d"].mean().to_dict() if {"split", "readmit_30d"}.issubset(df.columns) else {},
        "missingness_top20": df.isna().mean().sort_values(ascending=False).head(20).to_dict(),
        "duplicate_episode_ids": int(df["episode_id"].duplicated().sum()) if "episode_id" in df.columns else None,
    }
    out["ok"] = bool(out["duplicate_episode_ids"] == 0)
    return out


def qa_markdown_v1(report: dict[str, Any]) -> str:
    lines = [
        "# Readmission Episode Table v1 QA",
        "",
        f"- OK: {report.get('ok')}",
        f"- Rows: {report.get('rows')}",
        f"- Episodes: {report.get('episodes')}",
        f"- Patients: {report.get('patients')}",
        f"- readmit_30d rate: {report.get('readmit_30d_rate')}",
        "",
        "## Split counts",
    ]
    for k, v in report.get("split_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## readmit_30d by split")
    for k, v in report.get("readmit_by_split", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Missingness top 20")
    for k, v in report.get("missingness_top20", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    return "\n".join(lines) + "\n"

