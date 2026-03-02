from __future__ import annotations

from copy import deepcopy
from typing import Any


def apply_variant(base_cfg: dict[str, Any], variant: str) -> dict[str, Any]:
    cfg = deepcopy(base_cfg)
    cfg.setdefault("dynamics_model", {})
    cfg.setdefault("readmission_model", {})
    if variant == "baseline":
        return cfg
    if variant == "scaled":
        cfg["dynamics_model"]["use_scaling"] = True
        cfg["readmission_model"]["use_scaling"] = True
        return cfg
    if variant == "scaled_clip":
        cfg["dynamics_model"]["use_scaling"] = True
        cfg["readmission_model"]["use_scaling"] = True
        cfg["dynamics_model"]["clip_to_bounds"] = True
        return cfg
    if variant == "scaled_done_balanced":
        cfg["dynamics_model"]["use_scaling"] = True
        cfg["readmission_model"]["use_scaling"] = True
        cfg["dynamics_model"]["done_class_weight"] = "balanced"
        return cfg
    if variant == "scaled_clip_done_balanced":
        cfg["dynamics_model"]["use_scaling"] = True
        cfg["readmission_model"]["use_scaling"] = True
        cfg["dynamics_model"]["clip_to_bounds"] = True
        cfg["dynamics_model"]["done_class_weight"] = "balanced"
        return cfg
    if variant == "scaled_clip_done_balanced_stronger_reg":
        cfg["dynamics_model"]["use_scaling"] = True
        cfg["readmission_model"]["use_scaling"] = True
        cfg["dynamics_model"]["clip_to_bounds"] = True
        cfg["dynamics_model"]["done_class_weight"] = "balanced"
        cfg["dynamics_model"]["alpha"] = float(cfg["dynamics_model"].get("alpha", 1.0)) * 10.0
        cfg["dynamics_model"]["done_c"] = 0.1
        cfg["readmission_model"]["c"] = float(cfg["readmission_model"].get("c", 1.0)) * 0.1
        return cfg
    raise ValueError(f"Unknown variant: {variant}")

