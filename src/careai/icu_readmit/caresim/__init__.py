"""
CARE-Sim: GPT-style transformer world model for ICU patient trajectory simulation.

Architecture (OTTO-style with clinical adaptations):
  - Causal GPT-2 transformer as world model
  - Continuous state embedding (no discretization)
  - Ensemble of N independent models for uncertainty quantification
  - Predicts (next_state, reward, terminal) from history of (state, action) pairs

Reference: OTTO (Zhao et al., KDD 2025, arXiv:2404.10393) adapted for:
  - Binary drug actions (vs continuous action perturbation)
  - Clinical EHR tabular state features
  - Ensemble uncertainty for out-of-distribution detection
"""
from .model import CareSimGPT
from .ensemble import CareSimEnsemble
from .dataset import ICUSequenceDataset, collate_sequences
from .simulator import CareSimEnvironment

__all__ = [
    "CareSimGPT",
    "CareSimEnsemble",
    "ICUSequenceDataset",
    "collate_sequences",
    "CareSimEnvironment",
]
