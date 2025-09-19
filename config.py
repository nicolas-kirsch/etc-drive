import torch, os, sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(1, BASE_DIR)

if torch.cuda.is_available():
    print('[INFO] GPU available')
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

elif torch.backends.mps.is_available():
    print('[INFO] MPS available')
    device = torch.device("mps")
else:
    print('[INFO] running on CPU')
    device = torch.device("cpu")

print(f'[INFO] Running on device: {device}')
