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
from careai.readmission.model_v1 import train_readmission_baseline
from careai.readmission.reporting_v1 import model_summary_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Train readmission head v1 baseline.")
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
    result = train_readmission_baseline(df, cfg)

    summary = {
        "config_path": str(config_path),
        "input_path": str(in_path),
        "metrics": result["metrics"],
        "n_features": len(result["feature_cols"]),
        "features": result["feature_cols"],
    }
    (out_dir / f"{prefix}_model_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out_dir / f"{prefix}_model_summary.md").write_text(model_summary_markdown(summary), encoding="utf-8")
    result["pred_valid"].to_csv(out_dir / f"{prefix}_predictions_valid.csv", index=False)
    result["pred_test"].to_csv(out_dir / f"{prefix}_predictions_test.csv", index=False)

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

