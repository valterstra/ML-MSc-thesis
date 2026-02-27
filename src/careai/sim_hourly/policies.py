from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class HourlyPolicy(Protocol):
    name: str

    def choose_action(self, rng: np.random.Generator) -> np.ndarray:
        ...


@dataclass(frozen=True)
class FixedPolicy:
    name: str
    action: np.ndarray

    def choose_action(self, rng: np.random.Generator) -> np.ndarray:
        _ = rng
        return self.action.copy()


@dataclass(frozen=True)
class ObservedRandomPolicy:
    name: str
    observed_actions: np.ndarray

    def choose_action(self, rng: np.random.Generator) -> np.ndarray:
        idx = int(rng.integers(0, len(self.observed_actions)))
        return self.observed_actions[idx].copy()


def build_policies(observed_actions: np.ndarray) -> list[HourlyPolicy]:
    return [
        ObservedRandomPolicy("observed_random", observed_actions=observed_actions),
        FixedPolicy("no_treatment", np.array([0, 0, 0], dtype=float)),
        FixedPolicy("always_vaso", np.array([1, 0, 0], dtype=float)),
        FixedPolicy("always_vent", np.array([0, 1, 0], dtype=float)),
        FixedPolicy("always_crrt", np.array([0, 0, 1], dtype=float)),
        FixedPolicy("all_treatments", np.array([1, 1, 1], dtype=float)),
    ]

