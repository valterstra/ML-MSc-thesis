"""Reporting helpers for the readmission model."""

from __future__ import annotations

from typing import Any


def model_summary_markdown(summary: dict[str, Any]) -> str:
    vm = summary["metrics"]["valid"]
    tm = summary["metrics"]["test"]
    lines = [
        "# Readmission Model Summary",
        "",
        "## Validation",
        f"- n: {vm['n']}",
        f"- prevalence: {vm['prevalence']}",
        f"- AUROC: {vm['auroc']}",
        f"- AUPRC: {vm['auprc']}",
        f"- Brier: {vm['brier']}",
        "",
        "## Test",
        f"- n: {tm['n']}",
        f"- prevalence: {tm['prevalence']}",
        f"- AUROC: {tm['auroc']}",
        f"- AUPRC: {tm['auprc']}",
        f"- Brier: {tm['brier']}",
    ]
    return "\n".join(lines) + "\n"
