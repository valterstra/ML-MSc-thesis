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

from careai.qa.transition_report_v2 import generate_transition_v2_qa, to_markdown


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate transition v2 QA report.")
    parser.add_argument("--input", required=True, help="Path to transitions CSV.")
    parser.add_argument("--output-dir", default=str(ROOT / "reports" / "transition_v2_multi"))
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)
    report = generate_transition_v2_qa(df)
    (out_dir / "qa_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (out_dir / "qa_summary.md").write_text(to_markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


