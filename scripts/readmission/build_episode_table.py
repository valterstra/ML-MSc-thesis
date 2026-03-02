from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.io.write_outputs import write_csv, write_json
from careai.readmission.episode_builder import build_episode_table
from careai.readmission.label_from_admissions import load_readmission_labels_from_db
from careai.readmission.qa import generate_episode_qa, qa_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the episode-level readmission table from hourly transitions.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "readmission.yaml"))
    parser.add_argument("--input", default=None, help="Optional override for transitions path.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    in_path = resolve_from_config(config_path, cfg["input"]["transitions_path"])
    if args.input:
        in_path = Path(args.input).resolve()

    out_table = resolve_from_config(config_path, cfg["output"]["episode_table_path"])
    out_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg["output"]["prefix"])

    trans = pd.read_csv(in_path)
    labels = load_readmission_labels_from_db(cfg, trans)
    epi = build_episode_table(trans, labels)

    qa = generate_episode_qa(epi)
    write_csv(epi, out_table)
    write_json(qa, out_dir / f"{prefix}_qa_summary.json")
    (out_dir / f"{prefix}_qa_summary.md").write_text(qa_markdown(qa), encoding="utf-8")

    summary = {
        "config_path": str(config_path),
        "input_path": str(in_path),
        "episode_table_path": str(out_table),
        "qa_path": str(out_dir / f"{prefix}_qa_summary.json"),
        "rows_in": int(len(trans)),
        "rows_out": int(len(epi)),
        "qa_ok": bool(qa.get("ok", False)),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
