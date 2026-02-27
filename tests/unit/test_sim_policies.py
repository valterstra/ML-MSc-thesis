from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from careai.sim.policies import AlwaysHighPolicy, AlwaysLowPolicy, ObservedPolicy


def test_policy_actions() -> None:
    state = pd.Series({"a_t": "A_HIGH_SUPPORT"})
    assert AlwaysLowPolicy().choose_action(state, model=None) == "A_LOW_SUPPORT"
    assert AlwaysHighPolicy().choose_action(state, model=None) == "A_HIGH_SUPPORT"
    assert ObservedPolicy().choose_action(state, model=None) == "A_HIGH_SUPPORT"

