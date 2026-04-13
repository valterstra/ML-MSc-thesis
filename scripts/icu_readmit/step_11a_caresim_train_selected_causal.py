"""Colab-friendly wrapper for causal selected-set CARE-Sim training.

Runs step_14_caresim_train_selected.py with:
- the selected-set 9-state / 5-action schema
- no reward head
- FCI causal constraints enabled
- save-dir redirected to a separate *_selected_causal directory
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_selected_causal_suffix(path_str: str) -> str:
    return path_str if path_str.endswith("_selected_causal") else f"{path_str}_selected_causal"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "step_14_caresim_train_selected.py"

    args = list(sys.argv[1:])

    if "--save-dir" in args:
        idx = args.index("--save-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_selected_causal_suffix(args[idx + 1])
    else:
        args.extend(["--save-dir", "models/icu_readmit/caresim_selected_causal"])

    if "--causal-constraints" not in args:
        args.append("--causal-constraints")

    cmd = [sys.executable, str(target), *args]
    print("Running causal selected-set CARE-Sim training:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
