from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.contracts.validators_v3 import validate_transition_v3_contract
from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.io.write_outputs import write_csv, write_json
from careai.mimiccode.concept_runner import run_mimiccode_concepts
from careai.mimiccode.extract_hourly_v1 import extract_hourly_from_postgres
from careai.qa.transition_report_v3 import generate_transition_v3_qa, to_markdown
from careai.transitions.build_transition_v3_hourly import build_transitions_v3_hourly
from careai.transitions.sampling import subject_level_sample
from careai.transitions.split import assign_subject_splits


def main() -> int:
    parser = argparse.ArgumentParser(description="Build transition dataset v3 from mimic-code hourly concepts.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "transition_v3_mimiccode_hourly.yaml"))
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--skip-concept-build", action="store_true")
    parser.add_argument("--limit-stays", type=int, default=None)
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    out_full = resolve_from_config(config_path, cfg["output"]["transitions_full_path"])
    out_sample = resolve_from_config(config_path, cfg["output"]["transitions_sample_path"])
    manifest_path = resolve_from_config(config_path, cfg["output"]["manifest_path"])
    report_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    mimiccode_repo = resolve_from_config(config_path, cfg["input"]["mimiccode_repo_path"])
    sql_file = ROOT / "src" / "careai" / "mimiccode" / "sql" / "hourly_transition_v1.sql"

    sample_frac = float(args.sample_frac) if args.sample_frac is not None else float(cfg["sampling"]["fraction"])
    sample_seed = int(cfg["sampling"]["seed"])

    executed_sql: list[str] = []
    if not args.skip_concept_build:
        executed_sql = run_mimiccode_concepts(cfg, mimiccode_repo)

    extracted = extract_hourly_from_postgres(cfg, sql_file, limit_stays=args.limit_stays)

    full_cfg = dict(cfg)
    full_cfg["sample_tag"] = "full"
    full_df = build_transitions_v3_hourly(extracted, full_cfg)
    full_df = assign_subject_splits(
        full_df,
        train=float(cfg["split"]["train"]),
        valid=float(cfg["split"]["valid"]),
        test=float(cfg["split"]["test"]),
        seed=int(cfg["split"]["seed"]),
    )

    sample_source = subject_level_sample(extracted.rename(columns={"subject_id": "subject_id"}), fraction=sample_frac, seed=sample_seed)
    sample_cfg = dict(cfg)
    sample_cfg["sample_tag"] = f"sample_{int(round(sample_frac * 100))}pct"
    sample_df = build_transitions_v3_hourly(sample_source, sample_cfg)
    sample_df = assign_subject_splits(
        sample_df,
        train=float(cfg["split"]["train"]),
        valid=float(cfg["split"]["valid"]),
        test=float(cfg["split"]["test"]),
        seed=int(cfg["split"]["seed"]),
    )

    full_report = validate_transition_v3_contract(full_df, sofa_jump_threshold=int(cfg["outcome"]["sofa_jump_threshold"]))
    sample_report = validate_transition_v3_contract(sample_df, sofa_jump_threshold=int(cfg["outcome"]["sofa_jump_threshold"]))
    if not full_report["ok"] or not sample_report["ok"]:
        print(json.dumps({"full": full_report, "sample": sample_report}, indent=2))
        return 1

    write_csv(full_df, out_full)
    write_csv(sample_df, out_sample)

    qa = generate_transition_v3_qa(sample_df)
    report_dir.mkdir(parents=True, exist_ok=True)
    write_json(qa, report_dir / "qa_summary.json")
    (report_dir / "qa_summary.md").write_text(to_markdown(qa), encoding="utf-8")

    manifest = {
        "config_path": str(config_path),
        "mimiccode_repo_path": str(mimiccode_repo),
        "sql_file": str(sql_file),
        "executed_sql": executed_sql,
        "outputs": {
            "full": str(out_full),
            "sample": str(out_sample),
            "qa_json": str(report_dir / "qa_summary.json"),
            "qa_md": str(report_dir / "qa_summary.md"),
        },
        "sampling": {"fraction": sample_frac, "seed": sample_seed},
        "schema": cfg.get("schema", {}),
        "validation": {"full": full_report, "sample": sample_report},
        "counts": {
            "extracted_rows": int(len(extracted)),
            "full_rows": int(len(full_df)),
            "sample_rows": int(len(sample_df)),
        },
    }
    write_json(manifest, manifest_path)
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
