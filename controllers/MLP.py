import torch
import torch.nn as nn
import numpy as np

from config import device
from assistive_functions import to_tensor


class MLP(nn.Module):
    def __init__(self, dim_out = 1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(11, 10),
            nn.Sigmoid(),
            nn.Linear(10, 10),
            nn.Sigmoid(),
            nn.Linear(10, dim_out),
        )


    def forward(self, x):
        return  self.mlp(x)
