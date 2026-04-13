"""
Neural network architectures for ICU readmission RL.

DuelingDQN is identical to the sepsis pipeline but parameterised for
n_actions=32 (2^5 binary drug combinations) and n_input=51 (broad state).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Dueling Double DQN with batch normalisation.

    Architecture:
      Input -> FC(hidden) -> BN -> LeakyReLU
             -> FC(hidden) -> BN -> LeakyReLU
             -> split:
                  Advantage: FC(64) -> FC(n_actions)
                  Value:     FC(64) -> FC(1)
      Q = V + (A - mean(A))

    Args:
        n_input:     state dimension (51 for broad, 15 for narrow)
        n_actions:   action space size (32 = 2^5 binary drug combos)
        hidden:      hidden layer width (default 128)
        leaky_slope: LeakyReLU negative slope (default 0.01)
    """

    def __init__(self, n_input, n_actions=32, hidden=128, leaky_slope=0.01):
        super().__init__()
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.adv_fc  = nn.Linear(hidden, 64)
        self.adv_out = nn.Linear(64, n_actions)

        self.val_fc  = nn.Linear(hidden, 64)
        self.val_out = nn.Linear(64, 1)

        self.leaky_slope = leaky_slope

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), self.leaky_slope)
        x = F.leaky_relu(self.bn2(self.fc2(x)), self.leaky_slope)

        adv = F.leaky_relu(self.adv_fc(x), self.leaky_slope)
        adv = self.adv_out(adv)

        val = F.leaky_relu(self.val_fc(x), self.leaky_slope)
        val = self.val_out(val)

        return val + adv - adv.mean(dim=1, keepdim=True)


class PhysicianPolicy(nn.Module):
    """Supervised policy network: pi(a|s).

    Architecture mirrors sepsis pipeline:
      FC(64) -> BN -> ReLU -> FC(64) -> BN -> ReLU -> FC(n_actions) -> Softmax

    Args:
        n_input:   state dimension (51 for broad state)
        n_actions: action space size (32 = 2^5 binary drug combos)
        hidden:    hidden layer width (default 64)
    """

    def __init__(self, n_input, n_actions=32, hidden=64):
        super().__init__()
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, n_actions)

    def forward(self, x):
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x)  # raw logits; use CrossEntropyLoss

    def predict_proba(self, x):
        """Return action probabilities (n, n_actions)."""
        with torch.no_grad():
            return F.softmax(self.forward(x), dim=1)


class RewardEstimator(nn.Module):
    """Reward function approximator: R(s, a).

    Action is represented as a 5-bit binary vector decoded from the
    integer action index (vasopressor, ivfluid, antibiotic, sedation, diuretic).

    Args:
        n_state:          state dimension (51)
        n_action_features: number of action features (5 binary drug flags)
        hidden:           hidden layer width (default 128)
    """

    def __init__(self, n_state, n_action_features=5, hidden=128):
        super().__init__()
        n_input = n_state + n_action_features
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x).squeeze(-1)


class EnvModel(nn.Module):
    """Transition dynamics model: P(s'|s, a).

    Predicts state DELTA given current state and 5-bit binary action vector.
    Final next state = current state + predicted delta.

    Args:
        n_state:          state dimension (51)
        n_action_features: number of action features (5 binary drug flags)
        hidden:           hidden layer width (default 500, per Raghu 2018)
    """

    def __init__(self, n_state, n_action_features=5, hidden=500):
        super().__init__()
        n_input = n_state + n_action_features
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, n_state)

    def forward(self, state, action):
        """Predict next-state delta.

        Args:
            state:  (batch, n_state)
            action: (batch, 5) binary drug flags
        Returns:
            delta:  (batch, n_state)
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x)
