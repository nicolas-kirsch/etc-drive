#!/usr/bin/env python3


import sys, os, logging, torch,time
from datetime import datetime
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt



BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(BASE_DIR)
sys.path.insert(1, BASE_DIR)


from config import device
from controllers import PerfBoostController, PerfBoostControllerAbs
from arg_parser import argument_parser
from assistive_functions import to_tensor
from plants import ConverterAbs,ConverterDataset, Converter, ConverterDatasetAbs
from assistive_functions import WrapLogger
from loss_functions import ConverterLoss
import torch.nn.functional as F


# ----- SET UP LOGGER -----
now = datetime.now().strftime("%m_%d_%H_%M_%S")
save_path = os.path.join(BASE_DIR, 'experiments', 'DHN', 'saved_results',"log")
save_folder = os.path.join(save_path, 'perf_boost_'+now)
os.makedirs(save_folder)
logging.basicConfig(filename=os.path.join(save_folder, 'log'), format='%(asctime)s %(message)s', filemode='w')
logger = logging.getLogger('perf_boost_')
logger.setLevel(logging.DEBUG)
logger = WrapLogger(logger)



# ----- parse and set experiment arguments -----
args = argument_parser()
# msg = print_args(args)    # TODO
# logger.info(msg)
x_min = torch.Tensor([70]).to(device)
x0 = torch.Tensor([[5000],[1000],[1000],[1000],[1000],[1000]]).to(device)
xref = torch.Tensor([[80]]).to(device)

Wref = 80
Vref = 5000
Qref = 8e+6 
h = 1e-4

yref = torch.Tensor([Vref,Qref]).to(device)

M = 23e3
D = 1e-4
C = 1e-2
G = 1e-5

l = 2 * 1.567e-3
r = 2e-3
w = 2 * np.pi * 50

Lg = np.array([[l, 0], [0, l]])
Z = np.array([[r, -w * l], [w * l, r]])

V_base = 5000
P_base = 8e+6
w_base = 80

I_base = P_base/V_base
Z_base = V_base**2/P_base
C_base = P_base/(V_base**2)
tau_base = P_base/w_base
M_base = P_base/(w_base**2)

base_values = {
    'V_base': V_base,
    'P_base': P_base,
    'w_base': w_base,
    'I_base': I_base,
    'Z_base': Z_base,
    'C_base': C_base,
    'tau_base': tau_base,
    'M_base': M_base
}

#### Convert to per unit system ####
m = M/M_base
d = D/M_base 
c = C/C_base
g = G/C_base

wref = Wref/w_base
vref = Vref/V_base
qref = Qref/P_base

lg = Lg/Z_base
z = Z/Z_base
l = l/Z_base



# ------------ 1. Dataset ------------
dataset = ConverterDataset(
    random_seed=args.random_seed, horizon=args.horizon
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=3)
train_data, test_data = train_data.to(device), test_data.to(device)
train_data = train_data[:,:800,:]
print(train_data.shape)
test_data = test_data[:,:12000,:]

# ------------ 1. Dataset ------------
datasetAbs = ConverterDatasetAbs(
    random_seed=args.random_seed, horizon=args.horizon
)
# divide to train and test
train_dataAbs, test_dataAbs = datasetAbs.get_data(num_train_samples=args.num_rollouts, num_test_samples=3)
train_dataAbs, test_dataAbs = train_dataAbs.to(device), test_dataAbs.to(device)
train_dataAbs = train_dataAbs[:,:800,:]
print(train_dataAbs.shape)
test_dataAbs = test_dataAbs[:,:12000,:]


# ------------ 2. Plant ------------
sys  = Converter(x0,wref,vref,Qref,h,m,d,c,g,lg,z,l,base_values)


# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x0, output_init=sys.u_init,d_mech_init=sys.d_mech,
    dim_internal=args.dim_internal, dim_nl=args.l,
    vdc_ref = vref,Q_ref=Qref,batch_size=args.batch_size,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)

best_params = ctl.get_parameters_as_vector()
zero_array = np.zeros_like(best_params)
ctl.set_parameters_as_vector(zero_array)


x_log_base,us,u_PB,d_mech = sys.rollout(ctl,test_data)


x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]*V_base
ig_log_base = x_log_base[1,:,1:3]*I_base

w_log_base = x_log_base[1,:,6]*w_base
vgj = to_tensor(np.array([0, -3000]))
Q_log_base = vgj[0]*ig_log_base[:,0] + vgj[1]*ig_log_base[:,1]

Q_ref_base = sys.Q_ref_e.cpu().detach().numpy()*P_base

u_log_base = us.cpu().detach()


d_mech = d_mech.cpu().detach().numpy()*P_base
print(x_log_base[0,-1,:])
print(torch.isnan(x_log_base[0,-1,1]).any().item())

# ------------ 2. Plant ------------
sysAbs  = ConverterAbs(x0,wref,vref,Qref,h,m,d,c,g,lg,z,l,base_values)


# ------------ 3. Controller ------------
ctlAbs = PerfBoostControllerAbs(
    noiseless_forward=sysAbs.noiseless_forward,
    input_init=sysAbs.x0, output_init=sysAbs.u_init,d_mech_init=sysAbs.d_mech,
    dim_internal=args.dim_internal, dim_nl=args.l,
    vdc_ref = vref,Q_ref=Qref,batch_size=args.batch_size,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)

# Repeat for Abs variables
best_params_abs = ctlAbs.get_parameters_as_vector()
zero_array_abs = np.zeros_like(best_params_abs)
ctlAbs.set_parameters_as_vector(zero_array_abs)

x_log_abs, us_abs, u_PB_abs, d_mech_abs = sysAbs.rollout(ctlAbs, test_dataAbs)

x_log_abs = x_log_abs.cpu().detach()
vdc_log_abs = x_log_abs[1, :, 0:1] * V_base
ig_log_abs = x_log_abs[1, :, 1:3] * I_base

w_log_abs = x_log_abs[1, :, 6] * w_base
Q_log_abs = vgj[0] * ig_log_abs[:, 0] + vgj[1] * ig_log_abs[:, 1]

Q_ref_abs = sysAbs.Q_ref_e.cpu().detach().numpy() * P_base

u_log_abs = us_abs.cpu().detach()

d_mech_abs = d_mech_abs.cpu().detach().numpy() * P_base




# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(4, 2, figsize=(13, 9))

# Plot 1: X profile over the horizon
#axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, vdc_log_test[0],label = f"rPB")
axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, vdc_log_base,color = "#FF7F0E", label = "Base Controller")
axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, vdc_log_abs,color = "#2CA02C", label = "Abs Controller", linestyle = "--")
axs[0, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Acceptable range")
axs[0, 0].axhline(y=5130, linestyle="--", color="teal")
axs[0, 0].set_title("Vdc profile over the horizon")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Voltage (V)")
axs[0,0].legend(loc='center right')
axs[0, 0].grid()

axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_base,color = "#FF7F0E", label = "Base Controller")
axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_abs,color = "#2CA02C", label = "Abs Controller", linestyle = "--")
axs[0, 1].axhline(y=80, linestyle="--", color="red", label=r"$w*$")
axs[0, 1].set_title("W profile over the horizon")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Angular velocity (W)")
axs[0, 1].set_ylim(79, 81)
axs[0,1].legend()
axs[0, 1].grid()

# Plot 2: DXref profile over the horizon

axs[3, 1].plot(np.array(range(test_data.shape[1]))*h, Q_log_base[:],label = f"Q")
axs[3, 1].plot(np.array(range(test_data.shape[1]))*h, Q_ref_base[0,:],label = r"$Q^*$")
axs[3, 1].plot(np.array(range(test_data.shape[1]))*h,[Qref]*len(np.array(range(test_data.shape[1]))),linestyle = "--", color="red", label=r"Asked Q")
axs[3, 1].set_title("Evolution of Q over the horizon ")
axs[3, 1].set_xlabel("Time (s)")
axs[3, 1].set_ylabel(r"$Q$")
axs[3, 1].legend(loc='center right')
axs[3, 1].grid()

# Plot 3: U profile over the horizon
axs[2, 1].plot(np.array(range(test_data.shape[1]))*h, u_log_base[0,:,:])
axs[2, 1].set_title("Mg profile over the horizon")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("mg")
axs[2, 1].grid()

mg_norm = torch.norm(u_log_base, dim=-1, keepdim=True)
mg_norm_abs = torch.norm(u_log_abs, dim=-1, keepdim=True)

axs[2, 0].plot(np.array(range(mg_norm.shape[1]))*h, mg_norm[0,:,:],label = r"$||M_g||$")
axs[2, 0].plot(np.array(range(mg_norm_abs.shape[1]))*h, mg_norm_abs[0,:,:],label = r"$||M_g||_{Abs}$", linestyle = "--")
axs[2, 0].plot(np.array(range(test_data.shape[1]))*h,[1/np.sqrt(2)]*len(np.array(range(test_data.shape[1]))),linestyle = "--", label = r"$||M_g||_{max}$")
axs[2, 0].set_title("Mg Norm profile over the horizon")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("||Mg||")
axs[2, 0].legend()
axs[2, 0].grid()


# Plot 1: X profile over the horizon
for i in range(test_data.shape[0]): 
    #axs[1, 0].plot(np.array(range(test_data.shape[1]))*h, ig_log_test[i])
    axs[1, 0].plot(np.array(range(test_data.shape[1]))*h, ig_log_base)
                   #linestyle = "--",color = "#FF7F0E", label = "Base")
axs[1, 0].set_title("I profile over the horizon")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("I")
axs[1,0].legend()
axs[1, 0].grid()




#axs[2, 0].plot(np.array(range(epoch+1)),loss_log)

axs[3, 0].plot(np.array(range(test_data.shape[1]))*h, d_mech[0])
axs[3, 0].plot(np.array(range(test_data.shape[1]))*h,[1e+5*80]*len(np.array(range(test_data.shape[1]))),linestyle = "--",)
axs[3, 0].set_title("Evolution of P_mech over the horizon ")
axs[3, 0].set_xlabel("Time (s)")
axs[3, 0].set_ylabel(r"$P_{mech}$")
axs[3, 0].grid()

i_norm = torch.norm(ig_log_base, dim=-1, keepdim=True) 
i_norm_abs = torch.norm(ig_log_abs, dim=-1, keepdim=True)
print(i_norm.shape)

axs[1, 1].plot(np.array(range(i_norm.shape[0]))*h, i_norm,label = r"$||I||$")
axs[1, 1].plot(np.array(range(i_norm_abs.shape[0]))*h, i_norm_abs,label = r"$||I||_{Abs}$",linestyle = "--")
axs[1, 1].plot(np.array(range(test_data.shape[1]))*h,[2800]*len(np.array(range(test_data.shape[1]))),linestyle = "--",label = r"$||I||_{max}$")
axs[1, 1].set_title("I Norm profile over the horizon")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("||I||")
axs[1, 1].legend()
axs[1, 1].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'System evolution', fontsize=17)
plt.show()






