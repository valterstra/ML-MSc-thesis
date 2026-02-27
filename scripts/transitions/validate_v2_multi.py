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

from careai.contracts.validators_v2 import validate_transition_v2_contract


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate transition dataset v2 multi-step contract.")
    parser.add_argument("--input", required=True, help="Path to transitions CSV.")
    args = parser.parse_args()

    in_path = Path(args.input).resolve()
    df = pd.read_csv(in_path)
    report = validate_transition_v2_contract(df)
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())


