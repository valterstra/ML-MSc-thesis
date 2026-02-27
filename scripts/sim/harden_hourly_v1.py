from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.sim_hourly.hardening_runner_v1 import build_data_manifest, evaluate_variant
from careai.sim_hourly.reporting_trust_v1 import trust_summary_markdown


def _now_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def main() -> int:
    parser = argparse.ArgumentParser(description="Autonomous hardening runner for hourly simulator v1.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "sim_hourly_hardening_v1.yaml"))
    parser.add_argument("--transitions-input", default=None)
    parser.add_argument("--episode-input", default=None)
    parser.add_argument("--max-iterations", type=int, default=20)
    parser.add_argument("--full-validation", action="store_true")
    parser.add_argument("--stop-on-pass", action="store_true")
    args = parser.parse_args()

    hcfg_path = Path(args.config).resolve()
    hcfg: dict[str, Any] = load_yaml(hcfg_path)

    base_cfg_rel = str(hcfg["base_sim_config"])
    base_cfg_path = resolve_from_config(hcfg_path, base_cfg_rel)
    if not base_cfg_path.exists():
        candidates = [
            (ROOT / base_cfg_rel).resolve(),
            (ROOT / "configs" / Path(base_cfg_rel).name).resolve(),
        ]
        for c in candidates:
            if c.exists():
                base_cfg_path = c
                break
    base_cfg: dict[str, Any] = load_yaml(base_cfg_path)
    # inject hardening simulation overrides into base config
    base_cfg.setdefault("simulation", {})
    for k in ("max_steps", "n_rollouts_per_policy", "done_threshold", "seeds"):
        if k in hcfg.get("simulation", {}):
            base_cfg["simulation"][k] = hcfg["simulation"][k]

    transitions_path = resolve_from_config(hcfg_path, hcfg["input"]["transitions_path"])
    episode_path = resolve_from_config(hcfg_path, hcfg["input"]["episode_table_path"])
    if args.transitions_input:
        transitions_path = Path(args.transitions_input).resolve()
    if args.episode_input:
        episode_path = Path(args.episode_input).resolve()

    run_id = _now_id() if hcfg["output"].get("run_id_mode", "timestamp") == "timestamp" else "run"
    out_root = resolve_from_config(hcfg_path, hcfg["output"]["root_dir"])
    out_dir = out_root / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    transitions = pd.read_csv(transitions_path)
    episodes = pd.read_csv(episode_path)

    variants = list(hcfg["training"]["variant_grid"])[: int(args.max_iterations)]
    thresholds = dict(hcfg["qa"]["thresholds"])
    results = []
    progress_lines = [f"# Hardening Run {run_id}", "", f"- started_utc: {datetime.now(timezone.utc).isoformat()}", f"- variants_planned: {variants}"]

    baseline_res = evaluate_variant("baseline", base_cfg, transitions, episodes, thresholds, baseline_readmit_valid=None)
    baseline_readmit = baseline_res.metrics["readmission_valid"]
    results.append(baseline_res)
    progress_lines.append(f"- baseline status: {baseline_res.gate.status}, trust_score={baseline_res.gate.trust_score:.4f}")

    for v in variants:
        if v == "baseline":
            continue
        res = evaluate_variant(v, base_cfg, transitions, episodes, thresholds, baseline_readmit_valid=baseline_readmit)
        results.append(res)
        progress_lines.append(f"- {v}: status={res.gate.status}, trust_score={res.gate.trust_score:.4f}")
        if args.stop_on_pass and res.gate.status == "pass":
            break

    passing = [r for r in results if r.gate.status == "pass"]
    if passing:
        selected = sorted(passing, key=lambda r: r.gate.trust_score)[0]
    else:
        selected = sorted(results, key=lambda r: r.gate.trust_score)[0]

    variant_rows = []
    for r in results:
        variant_rows.append(
            {
                "variant": r.variant,
                "status": r.gate.status,
                "trust_score": r.gate.trust_score,
                "realism": r.metrics["realism"]["global_out_of_range_rate"],
                "ood": r.metrics["ood"]["ood_exceedance_rate"],
                "done_ece": r.metrics["done"]["done_ece"],
            }
        )
    variant_df = pd.DataFrame(variant_rows).sort_values(["status", "trust_score"], ascending=[False, True]).reset_index(drop=True)

    gate_decision = {
        "status": selected.gate.status,
        "run_id": run_id,
        "selected_variant": selected.variant,
        "rules_passed": selected.gate.rules_passed,
        "rules_failed": selected.gate.rules_failed,
        "trust_score": selected.gate.trust_score,
        "next_actions": "promote_to_full_validation" if selected.gate.status == "pass" else "tune_more_variants",
    }
    summary_payload = {
        "run_id": run_id,
        "config_path": str(hcfg_path),
        "base_config_path": str(base_cfg_path),
        "transitions_path": str(transitions_path),
        "episode_path": str(episode_path),
        "selected_variant": selected.variant,
        "gate_decision": gate_decision,
        "variant_table": variant_rows,
    }

    (out_dir / "baseline_snapshot.json").write_text(json.dumps(baseline_res.metrics, indent=2), encoding="utf-8")
    variant_df.to_csv(out_dir / "variant_results.csv", index=False)
    selected.policy_metrics.to_csv(out_dir / "policy_metrics_with_trust.csv", index=False)
    selected.trajectories.to_csv(out_dir / "counterfactual_with_trust.csv", index=False)
    (out_dir / "selected_variant.json").write_text(json.dumps({"variant": selected.variant, "metrics": selected.metrics}, indent=2), encoding="utf-8")
    (out_dir / "trust_qa_summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    (out_dir / "trust_qa_summary.md").write_text(trust_summary_markdown(summary_payload), encoding="utf-8")
    (out_dir / "gate_decision.json").write_text(json.dumps(gate_decision, indent=2), encoding="utf-8")
    progress_lines.append(f"- selected_variant: {selected.variant}")
    progress_lines.append(f"- final_status: {selected.gate.status}")
    (out_dir / "progress_log.md").write_text("\n".join(progress_lines) + "\n", encoding="utf-8")
    manifest = build_data_manifest([transitions_path, episode_path, out_dir / "variant_results.csv", out_dir / "trust_qa_summary.json"])
    (out_dir / "data_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(json.dumps(summary_payload, indent=2))
    print(f"Wrote run artifacts: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
