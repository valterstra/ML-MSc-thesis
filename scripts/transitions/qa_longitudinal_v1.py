from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.io.write_outputs import write_json
from careai.qa.longitudinal_report_v1 import generate_longitudinal_v1_qa, to_markdown


def _load_tensors(tensor_dir: Path) -> dict[str, np.ndarray]:
    return {
        "X_state": np.load(tensor_dir / "X_state.npy"),
        "M_valid": np.load(tensor_dir / "M_valid.npy"),
        "A_action": np.load(tensor_dir / "A_action.npy"),
        "Y_next": np.load(tensor_dir / "Y_next.npy"),
        "D_done": np.load(tensor_dir / "D_done.npy"),
        "split": np.load(tensor_dir / "split.npy"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate QA for longitudinal v1 artifacts.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "longitudinal_v1.yaml"))
    parser.add_argument("--long", default=None, help="Optional override for long table CSV.")
    parser.add_argument("--tensor-dir", default=None, help="Optional override for tensor directory.")
    parser.add_argument("--source", default=None, help="Optional source transitions_v2 CSV for row conservation.")
    parser.add_argument("--output-dir", default=None, help="Optional QA report output directory.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    long_path = resolve_from_config(config_path, cfg["output"]["long_table_path"])
    tensor_dir = resolve_from_config(config_path, cfg["output"]["tensor_dir"])
    source_path = resolve_from_config(config_path, cfg["input"]["transitions_v2_path"])
    out_dir = resolve_from_config(config_path, cfg["output"]["qa_report_dir"])

    if args.long:
        long_path = Path(args.long).resolve()
    if args.tensor_dir:
        tensor_dir = Path(args.tensor_dir).resolve()
    if args.source:
        source_path = Path(args.source).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()

    long_df = pd.read_csv(long_path)
    tensors = _load_tensors(tensor_dir)
    episode_index_df = pd.read_csv(tensor_dir / "episode_index.csv")
    source_rows = int(len(pd.read_csv(source_path))) if source_path.exists() else None

    report = generate_longitudinal_v1_qa(
        long_df=long_df,
        tensors=tensors,
        episode_index_df=episode_index_df,
        expected_source_rows=source_rows,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_json(report, out_dir / "qa_summary.json")
    (out_dir / "qa_summary.md").write_text(to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

