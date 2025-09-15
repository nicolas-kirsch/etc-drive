import torch
import torch.nn as nn
import numpy as np

from config import device
from assistive_functions import to_tensor


class MLP(nn.Module):
    def __init__(self, dim_out = 1):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(9, 6),
            nn.ReLU(),
            nn.Linear(6, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, dim_out),
            nn.Tanh(),
        )


    def forward(self, x):
        return  self.mlp(x)


class RNNModel(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=10, output_dim=1):
        super().__init__()
        self.rnn = nn.RNN(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()

    def forward(self, x):
        # Assuming x is of shape (batch_size, seq_length, input_dim)
        rnn_out, _ = self.rnn(x)
        # Take the output from the last time step
        last_time_step_output = rnn_out[:, -1, :]
        return self.activation(self.fc(last_time_step_output))*2
    