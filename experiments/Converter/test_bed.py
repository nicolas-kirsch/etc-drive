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
save_folder = os.path.join(save_path, '0_phases_lost/perf_boost_09_22_12_30_59')

logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)


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
#offset = torch.full_like(offset,0.2)
data_third = int(np.floor(test_data.shape[0]/3))
data_sev = int(np.floor(test_data.shape[0]/7))
vg = 3150 
for t in range(test_data.shape[1]):
    theta = w * t * h
    
    # Three-phase voltages (remove phase a → set v_a = 0)
    va = vg * np.cos(theta)
    vb = vg * np.cos(theta - 2*np.pi/3)
    vc = vg * np.cos(theta + 2*np.pi/3)

    if t  > 1000 and t < 3000:
        """va = 0.3 * vg * np.cos(theta)
        vb = 0.3 * vg * np.cos(theta - 2*np.pi/3)
        vc = vg * np.cos(theta + 2*np.pi/3)"""
        va_loss = offset * va
        vb_loss = offset * vb
        vc_loss = offset * vc
    elif t > 40000 and t < 5500:
        """va = 0.6 * vg * np.cos(theta)
        vb = 0.6 * vg * np.cos(theta - 2*np.pi/3)
        vc = vg * np.cos(theta + 2*np.pi/3)"""
        va_loss = offset * va
        vb_loss = offset * vb
        vc_loss = offset * vc
    else: 
        va_loss =  torch.full_like(offset,va)
        vb_loss =  torch.full_like(offset,vb)
        vc_loss =  torch.full_like(offset,vc)

    va =  torch.full_like(offset,va)
    vb =  torch.full_like(offset,vb)
    vc =  torch.full_like(offset,vc)

    


    v_alpha = (2/3) * (va - 0.5*vb - 0.5*vc)
    v_beta  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc)

    v_alpha_ab = (2/3) * (va_loss - 0.5*vb_loss - 0.5*vc)
    v_beta_ab  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc)

    v_alpha_bc = (2/3) * (va - 0.5*vb_loss - 0.5*vc_loss)
    v_beta_bc  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc_loss)

    v_alpha_ac = (2/3) * (va_loss - 0.5*vb - 0.5*vc_loss)
    v_beta_ac  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)


    v_alpha_a = (2/3) * (va_loss - 0.5*vb - 0.5*vc)
    v_beta_a  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc)

    v_alpha_b = (2/3) * (va - 0.5*vb_loss - 0.5*vc)
    v_beta_b  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc)

    v_alpha_c = (2/3) * (va - 0.5*vb - 0.5*vc_loss)
    v_beta_c  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)

    v_alpha_c = (2/3) * (va - 0.5*vb - 0.5*vc_loss)
    v_beta_c  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)

    v_alpha_abc = (2/3) * (va_loss - 0.5*vb_loss - 0.5*vc_loss)
    v_beta_abc  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc_loss)

    """test_data[:data_third,t,4+2] = v_alpha_ab[:data_third]
    test_data[:data_third,t,4+3] = v_beta_ab[:data_third]

    test_data[data_third:data_third*2,t,4+2] = v_alpha_bc[data_third:data_third*2]
    test_data[data_third:data_third*2,t,4+3] = v_beta_bc[data_third:data_third*2]

    test_data[data_third*2:,t,4+2] = v_alpha_ac[data_third*2:]
    test_data[data_third*2:,t,4+3] = v_beta_ac[data_third*2:]"""

    test_data[:data_sev,t,4+2] = v_alpha_ab[:data_sev]
    test_data[:data_sev,t,4+3] = v_beta_ab[:data_sev]

    test_data[data_sev:data_sev*2,t,4+2] = v_alpha_bc[data_sev:data_sev*2]
    test_data[data_sev:data_sev*2,t,4+3] = v_beta_bc[data_sev:data_sev*2]

    test_data[data_sev*2:data_sev*3,t,4+2] = v_alpha_ac[data_sev*2:data_sev*3]
    test_data[data_sev*2:data_sev*3,t,4+3] = v_beta_ac[data_sev*2:data_sev*3]

    test_data[data_sev*3:data_sev*4,t,4+2] = v_alpha_a[data_sev*3:data_sev*4]
    test_data[data_sev*3:data_sev*4,t,4+3] = v_beta_a[data_sev*3:data_sev*4]

    test_data[data_sev*4:data_sev*5,t,4+2] = v_alpha_b[data_sev*4:data_sev*5]
    test_data[data_sev*4:data_sev*5,t,4+3] = v_beta_b[data_sev*4:data_sev*5]

    test_data[data_sev*5:data_sev*6,t,4+2] = v_alpha_c[data_sev*5:data_sev*6]
    test_data[data_sev*5:data_sev*6,t,4+3] = v_beta_c[data_sev*5:data_sev*6]

    test_data[data_sev*6:,t,4+2] = v_alpha_abc[data_sev*6:]
    test_data[data_sev*6:,t,4+3] = v_beta_abc[data_sev*6:]



sys  = Converter(x0,wref,vref,qref,h,m,d,c,g,lg,z,l,base_values,Ared,Bred,Cred,Ered)

print(sys.u_init.shape)

# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x0, output_init=sys.u_init,d_init=sys.d,
    dim_internal=args.dim_internal, dim_nl=args.l,
    vdc_ref = vref,Q_ref=qref,batch_size=args.batch_size,
    initialization_std=args.cont_init_std,
    output_amplification=20,contraction_rate_lb=1
).to(device)

params = torch.load(os.path.join(save_folder, 'best_controller_params.pth'),weights_only=False)
ctl.set_parameters_as_vector(params["REN"])
ctl.set_MLP_parameters(params["MLP"])



x_log_base,us,u_PB,d_mech = sys.rollout(ctl,test_data, no_PB=True)

with torch.no_grad():
    x_log_test, u_log_test, pb,_ = sys.rollout(
        controller=ctl, data=test_data, no_PB=False
    )

    test_loss = loss_fn.forward(x_log_test, u_log_test)[0][0].item()
    base_loss = loss_fn.forward(x_log_base, u_log_test)[0][0].item()
    print(f"\n TEST loss: {test_loss:.2f}")  ##
    print(f"\n BASE loss: {base_loss:.2f}")  ##

x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]
ig_log_base = x_log_base[1,:,1:3]

x_log_test = x_log_test.cpu().detach()
vdc_log_test = x_log_test[1,:,0:1]


"""x_log_base_ab = x_log_base[:data_third,:,:]
vdc_log_base_ab = x_log_base_ab[1,:,0:1]
x_log_base_bc = x_log_base[data_third:data_third*2,:,:]
vdc_log_base_bc = x_log_base_bc[1,:,0:1]
x_log_base_ac = x_log_base[data_third*2:,:,:]
vdc_log_base_ac = x_log_base_ac[1,:,0:1]


x_log_test_ab = x_log_test[:data_third,:,:]
vdc_log_test_ab = x_log_test_ab[1,:,0:1]
x_log_test_bc = x_log_test[data_third:data_third*2,:,:]
vdc_log_test_bc = x_log_test_bc[1,:,0:1]
x_log_test_ac = x_log_test[data_third*2:,:,:]
vdc_log_test_ac = x_log_test_ac[1,:,0:1]"""

x_log_base_ab = x_log_base[:data_sev,:,:]
vdc_log_base_ab = x_log_base_ab[4,:,0:1]
x_log_base_bc = x_log_base[data_sev:data_sev*2,:,:]
vdc_log_base_bc = x_log_base_bc[4,:,0:1]
x_log_base_ac = x_log_base[data_sev*2:data_sev*3,:,:]
vdc_log_base_ac = x_log_base_ac[4,:,0:1]

x_log_base_a = x_log_base[data_sev*3:data_sev*4,:,:]
vdc_log_base_a = x_log_base_a[1,:,0:1]
x_log_base_b = x_log_base[data_sev*4:data_sev*5,:,:]
vdc_log_base_b = x_log_base_b[1,:,0:1]
x_log_base_c = x_log_base[data_sev*5:data_sev*6,:,:]
vdc_log_base_c = x_log_base_c[1,:,0:1]
x_log_base_abc = x_log_base[data_sev*6:,:,:]
vdc_log_base_abc = x_log_base_abc[1,:,0:1]


x_log_test_ab = x_log_test[:data_sev,:,:]
vdc_log_test_ab = x_log_test_ab[1,:,0:1]
x_log_test_bc = x_log_test[data_sev:data_sev*2,:,:]
vdc_log_test_bc = x_log_test_bc[1,:,0:1]
x_log_test_ac = x_log_test[data_sev*2:data_sev*3,:,:]
vdc_log_test_ac = x_log_test_ac[1,:,0:1]

x_log_test_a = x_log_test[data_sev*3:data_sev*4,:,:]
vdc_log_test_a = x_log_test_a[1,:,0:1]
x_log_test_b = x_log_test[data_sev*4:data_sev*5,:,:]
vdc_log_test_b = x_log_test_b[1,:,0:1]
x_log_test_c = x_log_test[data_sev*5:data_sev*6,:,:]
vdc_log_test_c = x_log_test_c[1,:,0:1]
x_log_test_abc = x_log_test[data_sev*6:,:,:]
vdc_log_test_abc = x_log_test_abc[1,:,0:1]

test_loss_ab = loss_fn.forward(x_log_test_ab, u_log_test)[0][0].item()
test_loss_bc = loss_fn.forward(x_log_test_bc, u_log_test)[0][0].item()
test_loss_ac = loss_fn.forward(x_log_test_ac, u_log_test)[0][0].item()
test_loss_a = loss_fn.forward(x_log_test_a, u_log_test)[0][0].item()
test_loss_b = loss_fn.forward(x_log_test_b, u_log_test)[0][0].item()
test_loss_c = loss_fn.forward(x_log_test_c, u_log_test)[0][0].item()
test_loss_abc = loss_fn.forward(x_log_test_abc, u_log_test)[0][0].item()


base_loss_ab = loss_fn.forward(x_log_base_ab, u_log_test)[0][0].item()
base_loss_bc = loss_fn.forward(x_log_base_bc, u_log_test)[0][0].item()
base_loss_ac = loss_fn.forward(x_log_base_ac, u_log_test)[0][0].item()
base_loss_a = loss_fn.forward(x_log_base_a, u_log_test)[0][0].item()
base_loss_b = loss_fn.forward(x_log_base_b, u_log_test)[0][0].item()
base_loss_c = loss_fn.forward(x_log_base_c, u_log_test)[0][0].item()
base_loss_abc = loss_fn.forward(x_log_base_abc, u_log_test)[0][0].item()


print(f"\n TEST loss AB: {test_loss_ab:.2f}")  ##

print(f"\n BASE loss AB: {base_loss_ab:.2f}")  ##

print(f"\n TEST loss BC: {test_loss_bc:.2f}")  ##

print(f"\n BASE loss BC: {base_loss_bc:.2f}")  ##

print(f"\n TEST loss AC: {test_loss_ac:.2f}")  ##

print(f"\n BASE loss AC: {base_loss_ac:.2f}")  ##

print(f"\n TEST loss A: {test_loss_a:.2f}")  ##

print(f"\n BASE loss A: {base_loss_a:.2f}")  ##

print(f"\n TEST loss B: {test_loss_b:.2f}")  ##
print(f"\n BASE loss B: {base_loss_b:.2f}")  ##
print(f"\n TEST loss C: {test_loss_c:.2f}")  ##
print(f"\n BASE loss C: {base_loss_c:.2f}")  ##
print(f"\n TEST loss ABC: {test_loss_abc:.2f}")  ##
print(f"\n BASE loss ABC: {base_loss_abc:.2f}")  ##



# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(4, 2, figsize=(13, 8))

# Plot 1: X profile over the horizon
#axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test[0],label = f"rPB")
axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_ab,color = "#FF7F0E", label = "Base")
axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_ab, label = "PB")
axs[0, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[0, 0].axhline(y=5130, linestyle="--", color="teal")
axs[0, 0].set_title("Vdc profile over the horizon - Loss of phases A and B")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Voltage (V)")
axs[0, 0].legend(loc='center right')
axs[0, 0].grid()

axs[1, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_bc,color = "#FF7F0E", label = "Base")
axs[1, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_bc, label = "PB")
axs[1, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[1, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[1, 0].axhline(y=5130, linestyle="--", color="teal")
axs[1, 0].set_title("Vdc profile over the horizon - Loss of phases B and C")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Voltage (V)")
axs[1, 0].legend(loc='center right')
axs[1, 0].grid()

axs[2, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_ac,color = "#FF7F0E", label = "Base")
axs[2, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_ac, label = "PB")
axs[2, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[2, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[2, 0].axhline(y=5130, linestyle="--", color="teal")
axs[2, 0].set_title("Vdc profile over the horizon - Loss of phases A and C")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("Voltage (V)")
axs[2, 0].legend(loc='center right')
axs[2, 0].grid()

axs[0, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_a,color = "#FF7F0E", label = "Base")
axs[0, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_a, label = "PB")
axs[0, 1].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 1].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[0, 1].axhline(y=5130, linestyle="--", color="teal")
axs[0, 1].set_title("Vdc profile over the horizon - Loss of phase A")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Voltage (V)")
axs[0, 1].legend(loc='center right')
axs[0, 1].grid()

axs[1, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_b,color = "#FF7F0E", label = "Base")
axs[1, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_b, label = "PB")
axs[1, 1].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[1, 1].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[1, 1].axhline(y=5130, linestyle="--", color="teal")
axs[1, 1].set_title("Vdc profile over the horizon - Loss of phase B")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Voltage (V)")
axs[1, 1].legend(loc='center right')
axs[1, 1].grid()

axs[2, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_c,color = "#FF7F0E", label = "Base")
axs[2, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_c, label = "PB")
axs[2, 1].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[2, 1].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[2, 1].axhline(y=5130, linestyle="--", color="teal")
axs[2, 1].set_title("Vdc profile over the horizon - Loss of phase C")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Voltage (V)")
axs[2, 1].legend(loc='center right')
axs[2, 1].grid()

axs[3, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base_abc,color = "#FF7F0E", label = "Base")
axs[3, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test_abc, label = "PB")
axs[3, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[3, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Limits")
axs[3, 0].axhline(y=5130, linestyle="--", color="teal")
axs[3, 0].set_title("Vdc profile over the horizon - Loss of all three phases")
axs[3, 0].set_xlabel("Time (s)")
axs[3, 0].set_ylabel("Voltage (V)")
axs[3, 0].legend(loc='center right')
axs[3, 0].grid()
axs[3, 1].axis('off')  # Turn off the unused subplot (bottom-right)
# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'Loss of phase A and B', fontsize=17)
plt.show()
