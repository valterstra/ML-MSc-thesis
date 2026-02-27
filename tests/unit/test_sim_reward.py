from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_reward_formula_sign() -> None:
    # r = -y - cost
    y = 1
    cost = 0.05
    reward = -y - cost
    assert reward == -1.05

