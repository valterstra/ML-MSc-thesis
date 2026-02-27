from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.contracts.validators_v2 import validate_transition_v2_contract
from careai.io.load_inputs import load_stage02, load_yaml, resolve_from_config
from careai.io.write_outputs import write_csv, write_json
from careai.qa.transition_report_v2 import generate_transition_v2_qa, to_markdown
from careai.transitions.build_transition_v2_multi import build_transitions_v2
from careai.transitions.sampling import subject_level_sample
from careai.transitions.split import assign_subject_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build transition dataset v2 multi-step.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "transition_v2_multi.yaml"))
    parser.add_argument("--sample-frac", type=float, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    stage02_path = resolve_from_config(config_path, cfg["input"]["stage02_path"])
    full_out = resolve_from_config(config_path, cfg["output"]["transitions_full_path"])
    sample_out = resolve_from_config(config_path, cfg["output"]["transitions_sample_path"])
    manifest_path = resolve_from_config(config_path, cfg["output"]["manifest_path"])
    report_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])

    sample_frac = float(args.sample_frac) if args.sample_frac is not None else float(cfg["sampling"]["fraction"])
    sample_seed = int(cfg["sampling"]["seed"])

    source = load_stage02(stage02_path)

    full_cfg = dict(cfg)
    full_cfg["sample_tag"] = "full"
    full_df = build_transitions_v2(source, full_cfg)
    full_df = assign_subject_splits(
        full_df,
        train=float(cfg["split"]["train"]),
        valid=float(cfg["split"]["valid"]),
        test=float(cfg["split"]["test"]),
        seed=int(cfg["split"]["seed"]),
    )

    sample_source = subject_level_sample(source, fraction=sample_frac, seed=sample_seed)
    sample_cfg = dict(cfg)
    sample_cfg["sample_tag"] = f"sample_{int(round(sample_frac * 100))}pct"
    sample_df = build_transitions_v2(sample_source, sample_cfg)
    sample_df = assign_subject_splits(
        sample_df,
        train=float(cfg["split"]["train"]),
        valid=float(cfg["split"]["valid"]),
        test=float(cfg["split"]["test"]),
        seed=int(cfg["split"]["seed"]),
    )

    full_report = validate_transition_v2_contract(full_df)
    sample_report = validate_transition_v2_contract(sample_df)
    if not full_report["ok"] or not sample_report["ok"]:
        print(json.dumps({"full": full_report, "sample": sample_report}, indent=2))
        return 1

    write_csv(full_df, full_out)
    write_csv(sample_df, sample_out)

    qa = generate_transition_v2_qa(sample_df)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(qa, report_dir / "qa_summary.json")
    (report_dir / "qa_summary.md").write_text(to_markdown(qa), encoding="utf-8")

    manifest = {
        "config_path": str(config_path),
        "input_stage02_path": str(stage02_path),
        "outputs": {
            "full": str(full_out),
            "sample": str(sample_out),
            "qa_json": str(report_dir / "qa_summary.json"),
            "qa_md": str(report_dir / "qa_summary.md"),
        },
        "sampling": {"fraction": sample_frac, "seed": sample_seed},
        "action": cfg.get("action", {}),
        "schema": cfg.get("schema", {}),
        "validation": {"full": full_report, "sample": sample_report},
        "counts": {"source_rows": int(len(source)), "full_rows": int(len(full_df)), "sample_rows": int(len(sample_df))},
    }
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


