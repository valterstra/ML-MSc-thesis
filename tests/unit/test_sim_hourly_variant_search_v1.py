from __future__ import annotations

from careai.sim_hourly.variant_search_v1 import apply_variant


def test_apply_variant_scaled_clip_done_balanced() -> None:
    base = {"dynamics_model": {"alpha": 1.0}, "readmission_model": {"c": 1.0}}
    out = apply_variant(base, "scaled_clip_done_balanced")
    assert out["dynamics_model"]["use_scaling"] is True
    assert out["dynamics_model"]["clip_to_bounds"] is True
    assert out["dynamics_model"]["done_class_weight"] == "balanced"
    assert out["readmission_model"]["use_scaling"] is True

