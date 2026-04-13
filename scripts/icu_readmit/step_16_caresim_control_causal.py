"""Colab-friendly wrapper for causal CARE-Sim control runs.

Runs step_13a_caresim_control.py with causal-specific default directories:
- model-dir -> *_causal
- control-model-dir -> *_causal
- report-dir -> *_causal
- ddqn-path -> causal control checkpoint
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _ensure_causal_suffix(path_str: str) -> str:
    return path_str if path_str.endswith("_causal") else f"{path_str}_causal"


def _replace_ddqn_path(path_str: str) -> str:
    if "caresim_control_causal" in path_str:
        return path_str
    return path_str.replace("caresim_control", "caresim_control_causal")


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    target = script_dir / "step_13a_caresim_control.py"

    args = list(sys.argv[1:])
    mode = args[0] if args else None

    if "--model-dir" in args:
        idx = args.index("--model-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_causal_suffix(args[idx + 1])
    else:
        args.extend(["--model-dir", "models/icu_readmit/caresim_causal"])

    if "--control-model-dir" in args:
        idx = args.index("--control-model-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_causal_suffix(args[idx + 1])
    else:
        args.extend(["--control-model-dir", "models/icu_readmit/caresim_control_causal"])

    if "--report-dir" in args:
        idx = args.index("--report-dir")
        if idx + 1 < len(args):
            args[idx + 1] = _ensure_causal_suffix(args[idx + 1])
    else:
        args.extend(["--report-dir", "reports/icu_readmit/caresim_control_causal"])

    if mode == "eval":
        if "--ddqn-path" in args:
            idx = args.index("--ddqn-path")
            if idx + 1 < len(args):
                args[idx + 1] = _replace_ddqn_path(args[idx + 1])
        else:
            args.extend(["--ddqn-path", "models/icu_readmit/caresim_control_causal/ddqn_model.pt"])

    cmd = [sys.executable, str(target), *args]
    print("Running causal CARE-Sim control:")
    print(" ".join(cmd))
    return subprocess.call(cmd)


if __name__ == "__main__":
    raise SystemExit(main())
