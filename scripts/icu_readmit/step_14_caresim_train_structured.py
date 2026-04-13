"""Colab-friendly wrapper for structured CARE-Sim training.

Runs step_14_caresim_train.py with:
- static confounders frozen as conditioning-only context
- elapsed bloc/time feature enabled
- random training windows from full stays
- save-dir redirected to a separate *_structured directory

This avoids overwriting the existing baseline or causal CARE-Sim runs.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_structured_suffix(path_str: str) -> str:
    return path_str if path_str.endswith("_structured") else f"{path_str}_structured"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "step_14_caresim_train.py"

    args = list(sys.argv[1:])

    if "--save-dir" in args:
        idx = args.index("--save-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_structured_suffix(args[idx + 1])
    else:
        args.extend(["--save-dir", "models/icu_readmit/caresim_structured"])

    defaults = [
        "--freeze-static-context",
        "--use-time-feature",
        "--train-window-mode", "random",
        "--val-window-mode", "last",
    ]
    for item in defaults:
        if item not in args:
            args.append(item)

    cmd = [sys.executable, str(target), *args]
    print("Running structured CARE-Sim training:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
