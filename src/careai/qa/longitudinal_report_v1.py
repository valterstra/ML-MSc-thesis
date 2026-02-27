"""QA report generation for longitudinal v1 artifacts."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _is_prefix_mask(mask: np.ndarray) -> bool:
    seen_zero = False
    for v in mask.tolist():
        if v == 0:
            seen_zero = True
        elif seen_zero:
            return False
    return True


def generate_longitudinal_v1_qa(
    long_df: pd.DataFrame,
    tensors: dict[str, np.ndarray],
    episode_index_df: pd.DataFrame,
    expected_source_rows: int | None = None,
) -> dict[str, Any]:
    x_state = tensors["X_state"]
    m_valid = tensors["M_valid"]
    a_action = tensors["A_action"]
    y_next = tensors["Y_next"]
    d_done = tensors["D_done"]
    split_arr = tensors["split"]

    errors: list[str] = []

    if expected_source_rows is not None and int(len(long_df)) != int(expected_source_rows):
        errors.append(f"row_conservation_failed long_rows={len(long_df)} source_rows={expected_source_rows}")

    step_integrity_failed = 0
    for _, g in long_df.groupby("episode_id", sort=False):
        g2 = g.sort_values("t", kind="stable")
        expected = list(range(len(g2)))
        if g2["t"].tolist() != expected:
            step_integrity_failed += 1
    if step_integrity_failed > 0:
        errors.append(f"episode_step_integrity_failed episodes={step_integrity_failed}")

    mask_integrity_failed = 0
    for i in range(m_valid.shape[0]):
        if not _is_prefix_mask(m_valid[i].astype(int)):
            mask_integrity_failed += 1
    if mask_integrity_failed > 0:
        errors.append(f"mask_prefix_integrity_failed episodes={mask_integrity_failed}")

    length_map = (
        long_df.groupby("episode_id", sort=False).size().rename("long_len").to_dict()
        if "episode_id" in long_df.columns
        else {}
    )
    length_mismatch = 0
    for _, row in episode_index_df.iterrows():
        ep = str(row["episode_id"])
        l_idx = int(row["length"])
        l_long = int(length_map.get(ep, -1))
        if l_idx != l_long:
            length_mismatch += 1
    if length_mismatch > 0:
        errors.append(f"episode_length_mismatch rows={length_mismatch}")

    split_consistency_failed = 0
    if "split" in episode_index_df.columns:
        for _, row in episode_index_df.iterrows():
            split_name = str(row["split"])
            ep = str(row["episode_id"])
            unique_splits = long_df.loc[long_df["episode_id"].astype(str) == ep, "split"].astype(str).unique().tolist()
            if len(unique_splits) != 1 or unique_splits[0] != split_name:
                split_consistency_failed += 1
    if split_consistency_failed > 0:
        errors.append(f"split_consistency_failed episodes={split_consistency_failed}")

    if split_arr.shape[0] != episode_index_df.shape[0]:
        errors.append("split_array_length_mismatch")

    valid_steps = int(m_valid.sum())
    if int(len(long_df)) != valid_steps:
        errors.append(f"valid_steps_mismatch valid_steps={valid_steps} long_rows={len(long_df)}")

    feature_cols = [c for c in long_df.columns if c.startswith("s_t_")]
    missingness: dict[str, float] = {}
    if feature_cols:
        missingness = {c: float(long_df[c].isna().mean()) for c in feature_cols}

    return {
        "ok": len(errors) == 0,
        "errors": errors,
        "rows_long": int(len(long_df)),
        "rows_source_expected": int(expected_source_rows) if expected_source_rows is not None else None,
        "n_episodes": int(episode_index_df.shape[0]),
        "tensor_shapes": {
            "X_state": list(x_state.shape),
            "M_valid": list(m_valid.shape),
            "A_action": list(a_action.shape),
            "Y_next": list(y_next.shape),
            "D_done": list(d_done.shape),
            "split": list(split_arr.shape),
        },
        "mask_integrity_failed": int(mask_integrity_failed),
        "step_integrity_failed": int(step_integrity_failed),
        "episode_length_mismatch": int(length_mismatch),
        "split_consistency_failed": int(split_consistency_failed),
        "valid_steps": valid_steps,
        "missingness_state_numeric": missingness,
    }


def to_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Longitudinal v1 QA Summary",
        "",
        f"- ok: {report.get('ok')}",
        f"- rows_long: {report.get('rows_long')}",
        f"- rows_source_expected: {report.get('rows_source_expected')}",
        f"- n_episodes: {report.get('n_episodes')}",
        f"- valid_steps: {report.get('valid_steps')}",
        "",
        "## Integrity Counters",
        f"- mask_integrity_failed: {report.get('mask_integrity_failed')}",
        f"- step_integrity_failed: {report.get('step_integrity_failed')}",
        f"- episode_length_mismatch: {report.get('episode_length_mismatch')}",
        f"- split_consistency_failed: {report.get('split_consistency_failed')}",
        "",
        "## Tensor Shapes",
    ]
    for k, v in report.get("tensor_shapes", {}).items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Missingness (state numeric)")
    for k, v in report.get("missingness_state_numeric", {}).items():
        lines.append(f"- {k}: {v:.6f}")
    lines.append("")
    lines.append("## Errors")
    for e in report.get("errors", []):
        lines.append(f"- {e}")
    return "\n".join(lines) + "\n"
