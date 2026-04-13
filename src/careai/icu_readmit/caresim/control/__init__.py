"""Control utilities built on top of CARE-Sim.

Step 16 uses the trained CARE-Sim ensemble as an environment backend for:
  - short-horizon action planning over the 16 binary drug combinations
  - simulator-based DDQN policy training and evaluation
"""

from .actions import ACTION_GRID, decode_action_id, encode_action_vec
from .ddqn import DQNConfig, train_ddqn
from .evaluation import evaluate_policies, load_seed_episodes
from .observation import ObservationBuilder
from .planner import PlannerConfig, planner_action

__all__ = [
    "ACTION_GRID",
    "DQNConfig",
    "ObservationBuilder",
    "PlannerConfig",
    "decode_action_id",
    "encode_action_vec",
    "evaluate_policies",
    "load_seed_episodes",
    "planner_action",
    "train_ddqn",
]
