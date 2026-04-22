"""
DAG-aware temporal transformer world model for ICU patient trajectory simulation.

This package adapts DAG-aware variable-token attention to the selected ICU
readmission simulator track by unrolling the graph over time. Dynamic state and
action scalars become node-time tokens; a fixed attention mask enforces:

  - no future-to-past information flow
  - static confounders as context-only tokens
  - direct action-to-next-state access only for discovered causal edges

The external interface mirrors the existing CARE-Sim stack:
  - train an ensemble of models
  - load an ensemble from disk
  - simulate trajectories through reset/step/rollout
"""

from .model import DAGAwareTemporalWorldModel
from .ensemble import DAGAwareEnsemble
from .simulator import DAGAwareEnvironment

__all__ = [
    "DAGAwareTemporalWorldModel",
    "DAGAwareEnsemble",
    "DAGAwareEnvironment",
]
