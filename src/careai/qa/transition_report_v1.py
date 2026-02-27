"""Transition dataset QA summary."""

from __future__ import annotations

from typing import Any

import pandas as pd


def generate_transition_qa(df: pd.DataFrame) -> dict[str, Any]:
    readmit_rate = float(df["within_30d_next_admit"].mean()) if len(df) else 0.0
    non_null_delta = df["delta_days_to_next_admit"].dropna()
    delta_summary = {
        "count_non_null": int(non_null_delta.shape[0]),
        "p50": float(non_null_delta.quantile(0.5)) if len(non_null_delta) else None,
        "p90": float(non_null_delta.quantile(0.9)) if len(non_null_delta) else None,
        "max": float(non_null_delta.max()) if len(non_null_delta) else None,
    }
    action_counts = df["a_t"].value_counts(dropna=False).to_dict() if "a_t" in df.columns else {}
    total = max(len(df), 1)
    action_rates = {k: float(v) / float(total) for k, v in action_counts.items()}
    if "a_t" in df.columns and "within_30d_next_admit" in df.columns:
        readmit_by_action = (
            df.groupby("a_t", dropna=False)["within_30d_next_admit"]
            .mean()
            .sort_index()
            .to_dict()
        )
    else:
        readmit_by_action = {}

    split_action_matrix = {}
    if "split" in df.columns and "a_t" in df.columns:
        ct = pd.crosstab(df["split"], df["a_t"], dropna=False)
        split_action_matrix = {str(idx): {str(c): int(ct.loc[idx, c]) for c in ct.columns} for idx in ct.index}

    return {
        "rows": int(len(df)),
        "patients": int(df["patient_id"].nunique()) if "patient_id" in df.columns else 0,
        "unique_index_hadm_id": int(df["index_hadm_id"].nunique()) if "index_hadm_id" in df.columns else 0,
        "readmit_30d_rate": readmit_rate,
        "split_counts": df["split"].value_counts(dropna=False).to_dict() if "split" in df.columns else {},
        "action_counts": action_counts,
        "action_rates": action_rates,
        "readmit_rate_by_action": {k: float(v) for k, v in readmit_by_action.items()},
        "split_action_matrix": split_action_matrix,
        "unknown_action_rate": float((df["a_t"] == "A_UNKNOWN").mean()) if "a_t" in df.columns else None,
        "terminal_action_rate": float((df["a_t"] == "A_TERMINAL").mean()) if "a_t" in df.columns else None,
        "missingness_top10": df.isna().mean().sort_values(ascending=False).head(10).to_dict(),
        "delta_days_summary": delta_summary,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Transition v1 QA Summary",
        "",
        f"- Rows: {report['rows']}",
        f"- Patients: {report['patients']}",
        f"- Unique index admissions: {report['unique_index_hadm_id']}",
        f"- 30d readmission rate: {report['readmit_30d_rate']:.4f}",
        "",
        "## Split counts",
    ]
    split_counts = report.get("split_counts", {})
    for k, v in split_counts.items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Action counts")
    for k, v in report.get("action_counts", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Action rates")
    for k, v in report.get("action_rates", {}).items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Readmit rate by action")
    for k, v in report.get("readmit_rate_by_action", {}).items():
        lines.append(f"- {k}: {v:.4f}")
    lines.append("")
    lines.append("## Split x Action")
    for split, action_map in report.get("split_action_matrix", {}).items():
        lines.append(f"- {split}: {action_map}")
    lines.append("")
    lines.append(f"- unknown_action_rate: {report.get('unknown_action_rate')}")
    lines.append(f"- terminal_action_rate: {report.get('terminal_action_rate')}")
    lines.append("")
    lines.append("## Delta-days summary")
    delta = report.get("delta_days_summary", {})
    for key in ("count_non_null", "p50", "p90", "max"):
        lines.append(f"- {key}: {delta.get(key)}")
    lines.append("")
    lines.append("## Missingness top 10")
    for k, v in report.get("missingness_top10", {}).items():
        lines.append(f"- {k}: {v:.4f}")
    return "\n".join(lines) + "\n"
