"""Colab-friendly wrapper for selected-set CARE-Sim training.

Runs step_14_caresim_train.py with:
- the new Step 10a selected dataset schema (9 states, 5 actions)
- no reward head
- static confounders frozen as conditioning-only context
- elapsed bloc/time feature enabled
- random training windows from full stays
- save-dir redirected to a separate *_selected directory
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_selected_suffix(path_str: str) -> str:
    return path_str if path_str.endswith("_selected") or path_str.endswith("_selected_causal") else f"{path_str}_selected"


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "step_14_caresim_train.py"

    args = list(sys.argv[1:])

    if "--save-dir" in args:
        idx = args.index("--save-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_selected_suffix(args[idx + 1])
    else:
        args.extend(["--save-dir", "models/icu_readmit/caresim_selected"])

    defaults = [
        "--state-dim", "9",
        "--action-dim", "5",
        "--freeze-static-context",
        "--use-time-feature",
        "--train-window-mode", "random",
        "--val-window-mode", "last",
        "--no-predict-reward",
    ]

    idx = 0
    while idx < len(defaults):
        item = defaults[idx]
        if item.startswith("--") and idx + 1 < len(defaults) and not defaults[idx + 1].startswith("--"):
            if item not in args:
                args.extend([item, defaults[idx + 1]])
            idx += 2
        else:
            if item not in args:
                args.append(item)
            idx += 1

    cmd = [sys.executable, str(target), *args]
    print("Running selected-set CARE-Sim training:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
