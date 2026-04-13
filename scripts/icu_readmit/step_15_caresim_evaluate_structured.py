"""Colab-friendly wrapper for structured CARE-Sim evaluation.

Runs step_12a_caresim_evaluate.py with:
- --model-dir redirected to *_structured
- --report-dir redirected to *_structured
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_structured_suffix(path_str: str) -> str:
    return path_str if path_str.endswith("_structured") else f"{path_str}_structured"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "step_12a_caresim_evaluate.py"

    args = list(sys.argv[1:])

    if "--model-dir" in args:
        idx = args.index("--model-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_structured_suffix(args[idx + 1])
    else:
        args.extend(["--model-dir", "models/icu_readmit/caresim_structured"])

    if "--report-dir" in args:
        idx = args.index("--report-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_structured_suffix(args[idx + 1])
    else:
        args.extend(["--report-dir", "reports/icu_readmit/caresim_structured"])

    cmd = [sys.executable, str(target), *args]
    print("Running structured CARE-Sim evaluation:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
