import os, sys


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)

import sys, os, logging, torch,time
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import scipy.io
import torch
import torch.nn as nn
import torch.optim as optim


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)


from config import device
from controllers import PerfBoostController
from arg_parser import argument_parser
from assistive_functions import to_tensor
from plants import ConverterDataset, Converter, ClassifierDataset
from assistive_functions import WrapLogger
from loss_functions import ConverterLoss
import torch.nn.functional as F



# ----- parse and set experiment arguments -----
args = argument_parser()
# Instantiate and load dataset
ds = ClassifierDataset(random_seed=0, horizon=1200, h=1/2000.0, n_phases_lost=0)

train_data_x, train_data_y = ds._generate_data(num_samples=args.num_rollouts)
test_data_x, test_data_y = ds._generate_data(num_samples=args.num_rollouts)

print(train_data_x.shape, train_data_y.shape, train_data_y.shape)
# Pick a sample


# Map class indices to human-readable labels
class_labels = ['NoFault', 'AB', 'BC', 'AC', 'A', 'B', 'C', 'ABC']




class DynamicalClassifier(nn.Module):
    def __init__(self, n_features, hidden_dim, n_classes, rnn_type='LSTM', num_layers=1):
        super().__init__()
        self.rnn_type = rnn_type
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size=n_features, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size=n_features, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True)
        else:
            raise ValueError("rnn_type must be 'LSTM' or 'GRU'")
        
        self.fc = nn.Linear(hidden_dim, n_classes)
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        # x: (batch, time, n_features)
        rnn_out, _ = self.rnn(x)  # (batch, time, hidden_dim)
        logits = self.fc(rnn_out)  # (batch, time, n_classes)
        probs = self.softmax(logits)  # (batch, time, n_classes)
        return probs

# ---------------- Training ---------------- #

# Hyperparameters
n_features = train_data_x.shape[2]
n_classes = train_data_y.shape[2]
hidden_dim = 64
num_layers = 1
lr = 1e-3
epochs = 500

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = DynamicalClassifier(n_features, hidden_dim, n_classes, 'LSTM', num_layers).to(device)
criterion = nn.CrossEntropyLoss()  # expects labels as integers
optimizer = optim.Adam(model.parameters(), lr=lr)

## Convert Y (one-hot) to integer labels
Y_int = torch.argmax(torch.tensor(train_data_y), dim=-1)  # (batch, time)

X_tensor = torch.tensor(train_data_x, dtype=torch.float32).to(device)
Y_tensor = Y_int.to(device)

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    
    probs = model(X_tensor)  # (batch, time, n_classes)
    # CrossEntropyLoss expects (batch*time, n_classes) and (batch*time,)
    loss = criterion(probs.view(-1, n_classes), Y_tensor.view(-1))
    
    loss.backward()
    optimizer.step()

    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")


y_tilde = model(test_data_x).detach().cpu()
y_pred = torch.argmax(y_tilde, dim=-1).cpu().numpy()
y_true = torch.argmax(torch.tensor(test_data_y), dim=-1).cpu().numpy()

plt.plot(y_tilde[0,:,1].numpy(), label='Predicted', alpha=0.7)
plt.plot(y_true[0,:], label='True', alpha=0.7)
plt.legend()
plt.show()