#!/usr/bin/env python3


import sys, os, logging, torch,time
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

import scipy.io
import torch


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)


from config import device
from controllers import PerfBoostController
from arg_parser import argument_parser
from assistive_functions import to_tensor
from plants import ConverterDataset, Converter
from assistive_functions import WrapLogger
from loss_functions import ConverterLoss
import torch.nn.functional as F


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'Converter', 'saved_results')
save_folder = os.path.join(save_path, '1_phases_lost/perf_boost_10_23_11_32_45')
params = torch.load(os.path.join(save_folder, 'best_controller_params_epoch_700.pth'),weights_only=False)



# ----- parse and set experiment arguments -----
args = argument_parser()
# msg = print_args(args)    # TODO
# logger.info(msg)
x_min = torch.Tensor([70]).to(device)
x0 = torch.Tensor([[5000],[1000],[1000],[1000],[1000],[1000],[1000],[1000]]).to(device)
x0 = torch.zeros((1,1,8))
xref = torch.Tensor([[80]]).to(device)

Wref = 125.66
Vref = 5000
Qref = 0 
h = 2.5e-4 

yref = torch.Tensor([Vref,Qref]).to(device)

M = 43645
D = 1e-4
C = 0.0040
G = 1e-4
l1 = 3.5e-04
l2 = 5.4154e-04
l = l1+l2
l = 9.6544e-04
r = 0.08
r = 1.9211e-02
w = 2 * np.pi * 50

Lg = np.array([[l, 0], [0, l]])
Z = np.array([[r, 0], [0, r]])

# Path to this file's directory
here = os.path.dirname(os.path.abspath(__file__))

# Gains file relative to this directory
mat_path = os.path.join(here, "gains_small.mat")


# Load the .mat file
data = scipy.io.loadmat(mat_path)

# Remove MATLAB metadata (optional cleanup)
data = {k: v for k, v in data.items() if not k.startswith('__')}

# Convert each to PyTorch tensor
tensors = {k: torch.tensor(v, dtype=torch.float32) for k, v in data.items()}
torch.set_printoptions(precision=10)

# Access individual tensors
Ared = tensors['A_1rb']
Bred = tensors['B_1rb'].T.unsqueeze(0)
Cred = tensors['C_2rb'].T.unsqueeze(0)
Ered = tensors['E_1rb'].T.unsqueeze(0)



m = M
d = D
c = C
g = G

wref = Wref
vref = Vref
qref = Qref

lg = Lg
z = Z
l = l

imax = 2222

base_values = None

# ------------ 1. Dataset ------------
dataset = ConverterDataset(
    random_seed=args.random_seed, horizon=args.horizon, h=h,n_phases_lost=args.phase_loss
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=args.num_rollouts)
train_data, test_data = train_data.to(device), test_data.to(device)

train_data = train_data[:,:args.train_horizon,:]
test_data = test_data[:,:args.horizon,:]
# ------------ 4. Loss ------------
R = 0
Q_Q = 0
Q_vdc = 1
alpha_v_max = 0

#Size of the minimization 
loss_fn = ConverterLoss(
    R=R,Q_Q=Q_Q,Q_vdc=Q_vdc, x_min=x_min,yref=yref,imax = imax, alpha_v_max=alpha_v_max  
)


offset = 0.2 + torch.rand(test_data.shape[0])*0.6
offset = torch.full_like(offset,0)
data_third = int(np.floor(test_data.shape[0]/3))
data_sev = int(np.floor(test_data.shape[0]/7))
vg = 3150 
for t in range(test_data.shape[1]):
    theta = w * t * h
    
    # Three-phase voltages (remove phase a → set v_a = 0)
    va = vg * np.cos(theta)
    vb = vg * np.cos(theta - 2*np.pi/3)
    vc = vg * np.cos(theta + 2*np.pi/3)

    if t  > 300 and t < 2240:
        va_loss = offset * va
        vb_loss = offset * vb
        vc_loss = offset * vc
    elif t > 80060 and t < 9660:
        va_loss = 0.4 * va
        vb_loss = 0.4 * vb
        vc_loss = 0.4 * vc
    else: 
        va_loss =  torch.full_like(offset,va)
        vb_loss =  torch.full_like(offset,vb)
        vc_loss =  torch.full_like(offset,vc)


    

    """v_alpha = (2/3) * (va - 0.5*vb - 0.5*vc_loss)
    v_beta  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)"""

    v_alpha = (2/3) * (va - 0.5*vb - 0.5*vc_loss)
    v_beta  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)

    test_data[:,t,4+2] = v_alpha
    test_data[:,t,4+3] = v_beta



sys  = Converter(x0,wref,vref,qref,h,m,d,c,g,lg,z,l,base_values,Ared,Bred,Cred,Ered)

print(sys.u_init.shape)

# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x0, output_init=sys.u_init,d_init=sys.d,
    dim_internal=args.dim_internal, dim_nl=args.l,
    vdc_ref = vref,Q_ref=qref,batch_size=args.num_rollouts,
    initialization_std=args.cont_init_std,
    output_amplification=20,contraction_rate_lb=1
).to(device)

ctl.set_parameters_as_vector(params["REN"])
ctl.set_MLP_parameters(params["MLP"])



x_log_base,us,u_PB,d_mech = sys.rollout(ctl,test_data, no_PB=True)

"""with torch.no_grad():
    x_log_test, u_log_test, pb,_ = sys.rollout(
        controller=ctl, data=test_data, no_PB=False
    )"""

x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]
ig_log_base = x_log_base[1,:,1:3]
w_log_base = x_log_base[1,:,6:7]

"""x_log_test = x_log_test.cpu().detach()
vdc_log_test = x_log_test[1,:,0:1]
ig_log_test = x_log_test[1,:,1:3]
w_log_test = x_log_test[1,:,6:7]"""



# Plot 1: X profile over the horizon
#axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test[0],label = f"rPB")
plt.plot(np.array(range(vdc_log_base.shape[0]))*h, w_log_base, color="#FF7F0E", label="Base")
plt.title("Vdc profile over the horizon")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend(loc='center right')
plt.grid()
plt.show()