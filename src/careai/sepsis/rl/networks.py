"""
PyTorch neural network architectures for sepsis RL.
Ported from sepsisrl TensorFlow 1.x code to PyTorch.

Architectures:
  - DuelingDQN: Double Dueling DQN with batch norm
  - SparseAutoencoder: dimensionality reduction (48 -> 200)
  - PhysicianPolicy: supervised policy classifier
  - RewardEstimator: reward function approximator
  - EnvModel: transition dynamics model
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DuelingDQN(nn.Module):
    """Dueling Double DQN with batch normalization.

    Architecture (from sepsisrl/continuous/q_network.ipynb):
      Input -> FC(hidden) -> BN -> LeakyReLU -> FC(hidden) -> BN -> LeakyReLU
      -> split into:
         Advantage: FC(64) -> FC(n_actions)
         Value:     FC(64) -> FC(1)
      Q = V + (A - mean(A))
    """

    def __init__(self, n_input, n_actions=25, hidden=128, leaky_slope=0.01):
        super().__init__()
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)

        # Advantage stream
        self.adv_fc = nn.Linear(hidden, 64)
        self.adv_out = nn.Linear(64, n_actions)

        # Value stream
        self.val_fc = nn.Linear(hidden, 64)
        self.val_out = nn.Linear(64, 1)

        self.leaky_slope = leaky_slope

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.fc1(x)), self.leaky_slope)
        x = F.leaky_relu(self.bn2(self.fc2(x)), self.leaky_slope)

        adv = F.leaky_relu(self.adv_fc(x), self.leaky_slope)
        adv = self.adv_out(adv)

        val = F.leaky_relu(self.val_fc(x), self.leaky_slope)
        val = self.val_out(val)

        # Dueling: Q = V + (A - mean(A))
        q = val + adv - adv.mean(dim=1, keepdim=True)
        return q


class SparseAutoencoder(nn.Module):
    """Sparse autoencoder for state compression.

    Architecture (from sepsisrl/continuous/autoencoder.ipynb):
      Encoder: n_input -> n_hidden (sigmoid)
      Decoder: n_hidden -> n_input (sigmoid)
      Loss: MSE + KL divergence sparsity penalty
    """

    def __init__(self, n_input, n_hidden=200):
        super().__init__()
        self.encoder = nn.Linear(n_input, n_hidden)
        self.decoder = nn.Linear(n_hidden, n_input)

    def encode(self, x):
        return torch.sigmoid(self.encoder(x))

    def decode(self, z):
        return torch.sigmoid(self.decoder(z))

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class PhysicianPolicy(nn.Module):
    """Supervised policy network: pi(a|s).

    Architecture (from sepsisrl/eval/physician_policy_tf.ipynb):
      FC(64) -> BN -> ReLU -> FC(64) -> BN -> ReLU -> FC(n_actions) -> Softmax
    """

    def __init__(self, n_input, n_actions=25, hidden=64):
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
        """Return action probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            return F.softmax(logits, dim=1)


class EnvModel(nn.Module):
    """Transition dynamics model: P(s'|s,a).

    Architecture (from sepsisrl/eval/env_model_regression_for_eval.ipynb):
      Concatenate [state, action/4] -> FC(500) -> BN -> ReLU -> FC(500) -> BN -> ReLU
      -> FC(n_state) (predicts delta, final = delta + current_state)
    """

    def __init__(self, n_state, n_action_features=2, hidden=500):
        super().__init__()
        n_input = n_state + n_action_features
        self.fc1 = nn.Linear(n_input, hidden)
        self.bn1 = nn.BatchNorm1d(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.bn2 = nn.BatchNorm1d(hidden)
        self.out = nn.Linear(hidden, n_state)

    def forward(self, state, action):
        """Predict next state delta.

        Args:
            state: (batch, n_state) current state
            action: (batch, 2) [iv_level/4, vaso_level/4] normalized
        Returns:
            delta: (batch, n_state) predicted state change
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.fc2(x)))
        return self.out(x)


class RewardEstimator(nn.Module):
    """Reward function approximator: R(s, a).

    Simple 2-layer network for reward estimation.
    """

    def __init__(self, n_state, n_action_features=2, hidden=128):
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
