import numpy as np
import torch
from torch import nn


class GeneralizedLinearModel(nn.Module):
    def __init__(self, input_channels: int, times: np.ndarray):
        super(GeneralizedLinearModel, self).__init__()
        self.fc1 = nn.Linear(input_channels, 1)
        self.fc2 = nn.Linear(input_channels, 1)
        self.fc3 = nn.Linear(input_channels, 1)
        self.fc4 = nn.Linear(input_channels, 1)
        self.fc5 = nn.Linear(input_channels, 1)
        self.times = torch.Tensor(times)

    def forward(self, x):
        c1 = torch.exp(self.fc1(x))
        inverse_tau1 = torch.exp(self.fc2(x))
        alpha1 = torch.exp(self.fc3(x))
        c2 = torch.exp(self.fc4(x))
        alpha2 = torch.exp(self.fc5(x))
        out = c1 * torch.pow(self.times, alpha1) * torch.exp(-self.times * inverse_tau1)
        out += c2 * torch.pow(self.times, alpha2) * (1 - torch.exp(-self.times * inverse_tau1))
        return out
