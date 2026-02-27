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
from careai.readmission.qa_v1 import generate_episode_qa_v1, qa_markdown_v1


def main() -> int:
    parser = argparse.ArgumentParser(description="QA for episode readmission table v1.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "readmission_head_v1.yaml"))
    parser.add_argument("--input", default=None, help="Optional override for episode table path.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)
    in_path = resolve_from_config(config_path, cfg["output"]["episode_table_path"])
    if args.input:
        in_path = Path(args.input).resolve()

    out_dir = resolve_from_config(config_path, cfg["output"]["report_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(cfg["output"]["prefix"])

    df = pd.read_csv(in_path)
    qa = generate_episode_qa_v1(df)
    (out_dir / f"{prefix}_qa_summary.json").write_text(json.dumps(qa, indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_qa_summary.md").write_text(qa_markdown_v1(qa), encoding="utf-8")
    print(json.dumps(qa, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

