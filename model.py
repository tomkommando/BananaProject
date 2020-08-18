# This script is a modified version of Udacity Deep Reinforcement Learning DQN exercise.

import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed=0, fc1_out_size=64, fc2_out_size=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.fc1 = nn.Linear(state_size, fc1_out_size)
        self.fc2 = nn.Linear(fc1_out_size, fc2_out_size)
        self.fc3 = nn.Linear(fc2_out_size, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = self.fc1(state)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        return x
