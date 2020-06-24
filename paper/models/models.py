import torch
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from backpack.core.layers import Flatten


class LeNetMadry(nn.Module):

    def __init__(self, binary=False):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 5, 1, padding=2)
        self.conv2 = nn.Conv2d(32, 64, 5, 1, padding=2)
        self.flatten = Flatten()
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 1 if binary else 10)

    def forward(self, x):
        x = self.features(x)
        x = self.fc2(x)
        return x

    def features(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        # x = x.view(-1, 7*7*64)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return x
