import torch
from torch import nn
import torch.nn.functional as F


class FeedForwardNN(nn.Module):
    def __init__(self, state_size):
        super(FeedForwardNN, self).__init__()
        self.layer1 = nn.Linear(state_size, 3)
        self.layer2 = nn.Linear(3, 9)
        self.layer3 = nn.Linear(9, 18)
        self.layer4 = nn.Linear(18, 9)
        self.layer5 = nn.Linear(9, 1)

    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        state = F.relu(self.layer3(state))
        state = F.relu(self.layer4(state))
        state = self.layer5(state)
        return state


class CriticNN(nn.Module):
    def __init__(self, state_size):
        super(CriticNN, self).__init__()
        self.layer1 = nn.Linear(state_size, 3)
        self.layer2 = nn.Linear(3, 9)
        self.layer3 = nn.Linear(9, 18)
        self.layer4 = nn.Linear(18, 9)
        self.layer5 = nn.Linear(9, 1)

    def forward(self, state):
        state = F.relu(self.layer1(state))
        state = F.relu(self.layer2(state))
        state = F.relu(self.layer3(state))
        state = F.relu(self.layer4(state))
        state = self.layer5(state).squeeze().max()
        return state
