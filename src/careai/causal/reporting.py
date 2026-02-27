"""Reporting helpers for causal baseline outputs."""

from __future__ import annotations

from typing import Any


def summary_markdown(payload: dict[str, Any]) -> str:
    est = payload["estimates"]
    ci = payload["bootstrap_ci"]
    diag = payload["diagnostics"]
    lines = [
        "# Causal Effect v1 Summary",
        "",
        "## Setup",
        f"- Estimand: {payload['analysis']['estimand']}",
        f"- Actions compared: {payload['analysis']['treatment_low_label']} vs {payload['analysis']['treatment_high_label']}",
        f"- N rows analyzed: {payload['n_rows']}",
        "",
        "## Crude",
        f"- risk_treated: {est['crude_mu1']:.6f}",
        f"- risk_control: {est['crude_mu0']:.6f}",
        f"- risk_difference: {est['crude_rd']:.6f}",
        f"- risk_ratio: {est['crude_rr']:.6f}",
        "",
        "## IPW",
        f"- risk_treated: {est['ipw_mu1']:.6f}",
        f"- risk_control: {est['ipw_mu0']:.6f}",
        f"- risk_difference: {est['ipw_rd']:.6f}",
        f"- risk_ratio: {est['ipw_rr']:.6f}",
        f"- RD 95% CI: [{ci['ipw_rd'][0]:.6f}, {ci['ipw_rd'][1]:.6f}]",
        f"- RR 95% CI: [{ci['ipw_rr'][0]:.6f}, {ci['ipw_rr'][1]:.6f}]",
        "",
        "## AIPW",
        f"- risk_treated: {est['aipw_mu1']:.6f}",
        f"- risk_control: {est['aipw_mu0']:.6f}",
        f"- risk_difference: {est['aipw_rd']:.6f}",
        f"- risk_ratio: {est['aipw_rr']:.6f}",
        f"- RD 95% CI: [{ci['aipw_rd'][0]:.6f}, {ci['aipw_rd'][1]:.6f}]",
        f"- RR 95% CI: [{ci['aipw_rr'][0]:.6f}, {ci['aipw_rr'][1]:.6f}]",
        "",
        "## Diagnostics",
        f"- propensity_extreme_rate: {diag['propensity_extreme_rate_ps_lt_002_or_gt_098']:.6f}",
        f"- balance_pass_rate_abs_smd_lt_01: {diag['balance_pass_rate_abs_smd_lt_01']:.6f}",
        f"- balance_median_abs_smd_before: {diag['balance_median_abs_smd_before']:.6f}",
        f"- balance_median_abs_smd_after: {diag['balance_median_abs_smd_after']:.6f}",
    ]
    return "\n".join(lines) + "\n"

