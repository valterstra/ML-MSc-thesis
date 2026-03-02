from __future__ import annotations

from typing import Any


def trust_summary_markdown(payload: dict[str, Any]) -> str:
    lines = ["# Hourly Simulator Hardening", ""]
    lines.append(f"- run_id: `{payload['run_id']}`")
    lines.append(f"- gate status: `{payload['gate_decision']['status']}`")
    lines.append(f"- selected variant: `{payload['selected_variant']}`")
    lines.append("")
    lines.append("## Gate Rules")
    for r in payload["gate_decision"]["rules_passed"]:
        lines.append(f"- PASS: `{r}`")
    for r in payload["gate_decision"]["rules_failed"]:
        lines.append(f"- FAIL: `{r}`")
    lines.append("")
    lines.append("## Variants")
    for v in payload["variant_table"]:
        lines.append(
            f"- `{v['variant']}` status={v['status']} trust_score={v['trust_score']:.4f} "
            f"realism={v['realism']:.4f} ood={v['ood']:.4f} done_ece={v['done_ece']:.4f}"
        )
    return "\n".join(lines) + "\n"
