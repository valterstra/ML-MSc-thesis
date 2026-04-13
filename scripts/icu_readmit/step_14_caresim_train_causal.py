"""Colab-friendly wrapper for causal CARE-Sim training.

Runs step_14_caresim_train.py with:
- --causal-constraints enabled
- --save-dir redirected to a separate *_causal directory

This avoids overwriting the existing baseline CARE-Sim run.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_causal_suffix(path_str: str) -> str:
    return path_str if path_str.endswith("_causal") else f"{path_str}_causal"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "step_14_caresim_train.py"

    args = list(sys.argv[1:])

    if "--save-dir" in args:
        idx = args.index("--save-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_causal_suffix(args[idx + 1])
    else:
        args.extend(["--save-dir", "models/icu_readmit/caresim_causal"])

    if "--causal-constraints" not in args:
        args.append("--causal-constraints")

    cmd = [sys.executable, str(target), *args]
    print("Running causal CARE-Sim training:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
