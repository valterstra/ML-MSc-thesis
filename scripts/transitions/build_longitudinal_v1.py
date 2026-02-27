from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.io.load_inputs import load_yaml, resolve_from_config
from careai.io.write_outputs import write_csv, write_json
from careai.transitions.build_longitudinal_v1 import build_longitudinal_from_transitions_v2


def _save_tensors(tensors: dict[str, np.ndarray], tensor_dir: Path) -> None:
    tensor_dir.mkdir(parents=True, exist_ok=True)
    np.save(tensor_dir / "X_state.npy", tensors["X_state"])
    np.save(tensor_dir / "M_valid.npy", tensors["M_valid"])
    np.save(tensor_dir / "A_action.npy", tensors["A_action"])
    np.save(tensor_dir / "Y_next.npy", tensors["Y_next"])
    np.save(tensor_dir / "D_done.npy", tensors["D_done"])
    np.save(tensor_dir / "split.npy", tensors["split"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Build longitudinal v1 long table + tensor package from transition v2.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "longitudinal_v1.yaml"))
    parser.add_argument("--input", default=None, help="Optional override for transitions_v2 CSV path.")
    parser.add_argument("--output-dir", default=None, help="Optional directory override for long/tensor outputs.")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    cfg = load_yaml(config_path)

    in_path = resolve_from_config(config_path, cfg["input"]["transitions_v2_path"])
    long_out = resolve_from_config(config_path, cfg["output"]["long_table_path"])
    tensor_dir = resolve_from_config(config_path, cfg["output"]["tensor_dir"])
    qa_report_dir = resolve_from_config(config_path, cfg["output"]["qa_report_dir"])
    if args.input:
        in_path = Path(args.input).resolve()
    if args.output_dir:
        out_dir = Path(args.output_dir).resolve()
        long_out = out_dir / "longitudinal_v1_long.csv"
        tensor_dir = out_dir / "longitudinal_v1_tensor"

    df = pd.read_csv(in_path)
    artifacts = build_longitudinal_from_transitions_v2(df, cfg)
    build_ts = datetime.now(timezone.utc).isoformat()

    write_csv(artifacts.long_df, long_out)
    _save_tensors(artifacts.tensors, tensor_dir)
    write_csv(artifacts.episode_index_df, tensor_dir / "episode_index.csv")

    metadata = dict(artifacts.metadata)
    metadata["source_transitions_v2_path"] = str(in_path)
    metadata["build_timestamp_utc"] = build_ts
    metadata["paths"] = {
        "long_table": str(long_out),
        "tensor_dir": str(tensor_dir),
        "qa_report_dir": str(qa_report_dir),
    }
    write_json(metadata, tensor_dir / "metadata.json")

    payload = {
        "ok": True,
        "source_rows": int(len(df)),
        "long_rows": int(len(artifacts.long_df)),
        "n_episodes": int(artifacts.episode_index_df.shape[0]),
        "long_table_path": str(long_out),
        "tensor_dir": str(tensor_dir),
        "qa_report_dir": str(qa_report_dir),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

