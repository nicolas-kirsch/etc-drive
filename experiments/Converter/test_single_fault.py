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
    elif t > 8060 and t < 9660:
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
iff_base = sys.h/sys.C*sys.iff_traj[:,:,:].cpu().detach()
mgig_base = sys.mgo
with torch.no_grad():
    x_log_test, u_log_test, pb,_ = sys.rollout(
        controller=ctl, data=test_data, no_PB=False
    )

    """test_loss = loss_fn.forward(x_log_test, u_log_test)[0][0].item()
    base_loss = loss_fn.forward(x_log_base, u_log_test)[0][0].item()"""
    """print(f"\n TEST loss: {test_loss:.2f}")  ##
    print(f"\n BASE loss: {base_loss:.2f}")  ##"""

x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]
ig_log_base = x_log_base[1,:,1:3]

x_log_test = x_log_test.cpu().detach()
vdc_log_test = x_log_test[1,:,0:1]




# --- Compute FFT along time dimension ---

beginning = 300
end = 2240
duration = (end - beginning) # duration in seconds

# --- compute mean (DC) and detrend ---
mean_val = vdc_log_base[beginning:end,:].mean()
x_detrended = vdc_log_base[beginning:end,:].reshape(1, duration, 1) - mean_val

X = torch.fft.rfft(x_detrended, dim=1)       # only positive frequencies (real-valued input)
amplitude = torch.abs(X) / duration * 2   # normalize amplitude (factor 2 for single-sided)
freqs = torch.fft.rfftfreq(duration, d=h)

mean_val = vdc_log_test[beginning:end,:].mean()
x_detrended = vdc_log_test[beginning:end,:].reshape(1, duration, 1) - mean_val

X_pb = torch.fft.rfft(x_detrended, dim=1)       # only positive frequencies (real-valued input)
amplitude_pb = torch.abs(X_pb) / duration * 2   # normalize amplitude (factor 2 for single-sided)
freqs_pb = torch.fft.rfftfreq(duration, d=h)

ig_log_test_c = x_log_test[1,:,1:3]
mean_val = ig_log_test_c[beginning:end,0].mean()
x_detrended = ig_log_test_c[beginning:end,0].reshape(1, duration, 1) - mean_val

X_ig_test_a = torch.fft.rfft(x_detrended, dim=1)       # only positive frequencies (real-valued input)
amplitude_ig_test_a = torch.abs(X_ig_test_a) / duration * 2   # normalize amplitude (factor 2 for single-sided)
freq_ig_test_a = torch.fft.rfftfreq(duration, d=h)

mean_val = ig_log_test_c[beginning:end,1].mean()
x_detrended = ig_log_test_c[beginning:end,1].reshape(1, duration, 1) - mean_val

X_ig_test_b = torch.fft.rfft(x_detrended, dim=1)       # only positive frequencies (real-valued input)
amplitude_ig_test_b = torch.abs(X_ig_test_b) / duration * 2   # normalize amplitude (factor 2 for single-sided)
freq_ig_test_b = torch.fft.rfftfreq(duration, d=h)

ig_log_base_ab = x_log_base[1,:,1:3]
mean_val = ig_log_base_ab[beginning:end,0].mean()
x_detrended = ig_log_base_ab[beginning:end,0].reshape(1, duration, 1) - mean_val

X_ig_base_a = torch.fft.rfft(x_detrended, dim=1)       # only positive frequencies (real-valued input)
amplitude_ig_base_a = torch.abs(X_ig_base_a) / duration * 2   # normalize amplitude (factor 2 for single-sided)
freq_ig_base_a = torch.fft.rfftfreq(duration, d=h)

mean_val = ig_log_base_ab[beginning:end,1].mean()
x_detrended = ig_log_base_ab[beginning:end,1].reshape(1, duration, 1) - mean_val

X_ig_base_b = torch.fft.rfft(x_detrended, dim=1)       # only positive frequencies (real-valued input)
amplitude_ig_base_b = torch.abs(X_ig_base_b) / duration * 2   # normalize amplitude (factor 2 for single-sided)
freq_ig_base_b = torch.fft.rfftfreq(duration, d=h)


amp_no_dc = amplitude.clone()
#amp_no_dc[:,0,:] = 0.0


# --- get indices of top 5 amplitudes ---
topk = torch.topk(amp_no_dc, k=5,dim=1)
print(topk)
# --- print results ---
print("Top 5 frequency peaks base:")
for idx, val in zip(topk.indices[0], topk.values[0]):

    print(f"{freqs[idx].item():8.3f} Hz   amplitude = {val.item():.3f}")

amp_no_dc = amplitude_pb.clone()
#amp_no_dc[:,0,:] = 0.0
topk = torch.topk(amp_no_dc, k=5,dim=1)

# Get top 5 peaks by amplitude
# --- print results ---
print("Top 5 frequency peaks rPB:")
for idx, val in zip(topk.indices[0], topk.values[0]):
    print(f"{freqs[idx].item():8.3f} Hz   amplitude = {val.item():.3f}")

amp_no_dc = amplitude_ig_base_b.clone()
#amp_no_dc[:,0,:] = 0.0
topk = torch.topk(amp_no_dc, k=5,dim=1)

# Get top 5 peaks by amplitude
# --- print results ---
print("Top 5 frequency peaks base beta:")
for idx, val in zip(topk.indices[0], topk.values[0]):
    print(f"{freq_ig_base_b[idx].item():8.3f} Hz   amplitude = {val.item():.3f}")

amp_no_dc = amplitude_ig_base_a.clone()
#amp_no_dc[:,0,:] = 0.0
topk = torch.topk(amp_no_dc, k=5,dim=1)

# Get top 5 peaks by amplitude
# --- print results ---
print("Top 5 frequency peaks base alpha:")
for idx, val in zip(topk.indices[0], topk.values[0]):
    print(f"{freq_ig_base_a[idx].item():8.3f} Hz   amplitude = {val.item():.3f}")

amp_no_dc = amplitude_ig_test_b.clone()
#amp_no_dc[:,0,:] = 0.0
topk = torch.topk(amp_no_dc, k=5,dim=1)

# Get top 5 peaks by amplitude
# --- print results ---
print("Top 5 frequency peaks rPB beta:")
for idx, val in zip(topk.indices[0], topk.values[0]):
    print(f"{freq_ig_test_b[idx].item():8.3f} Hz   amplitude = {val.item():.3f}")

amp_no_dc = amplitude_ig_test_a.clone()
#amp_no_dc[:,0,:] = 0.0
topk = torch.topk(amp_no_dc, k=5,dim=1)

# Get top 5 peaks by amplitude
# --- print results ---
print("Top 5 frequency peaks rPB alpha:")
for idx, val in zip(topk.indices[0], topk.values[0]):
    print(f"{freq_ig_test_a[idx].item():8.3f} Hz   amplitude = {val.item():.3f}")

print(x_log_base[0,-1,:])
print("idc",sys.lpf_idc)
print("iff",sys.lpf_iff[0])
print("iff_C",sys.lpf_iff_c[0])
print("vdc",sys.lpf_vdcerr[0])
print("vdcerr",sys.lpf_vdcerr_c[0])




"""# --- Plot ---
plt.figure(figsize=(8, 4))
plt.plot(freq_ig_test_b, amplitude_ig_test_b.squeeze(),label="Base controller with rPB",linewidth=10)
plt.plot(freq_ig_base_b, amplitude_ig_base_b.squeeze(),label="Base controller only", linestyle='--',linewidth=10)
#plt.title("Spectral analysis of the beta coordinate of the grid current with and without rPB")
plt.xlabel("Frequency [Hz]",fontsize=23)
plt.ylabel("Amplitude",fontsize=23)
plt.xlim(0, 800)
plt.legend(fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)

plt.figure(figsize=(8, 4))
plt.plot(freq_ig_test_a, amplitude_ig_test_a.squeeze(),label="Base controller with rPB",linewidth=10)
plt.plot(freq_ig_base_a, amplitude_ig_base_a.squeeze(),label="Base controller only", linestyle='--',linewidth=10)
#plt.title("Spectral analysis of the alpha coordinate of the grid current with and without rPB")
plt.xlabel("Frequency [Hz]",fontsize=23)
plt.ylabel("Amplitude",fontsize=23)
plt.xlim(0, 800)
plt.legend(fontsize=25)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.grid(True)"""

"""plt.figure(figsize=(8, 4))
plt.plot(freqs_pb, amplitude_pb.squeeze(),label="Base controller with rPB")
plt.plot(freqs, amplitude.squeeze(),label="Base controller only", linestyle='--')
plt.title("Spectral analysis of the DC bus voltage with and without rPB")
plt.xlabel("Frequency [Hz]",fontsize='large')
plt.ylabel("Amplitude",fontsize='large')
plt.xlim(0, 1000)
plt.legend(fontsize='large')
plt.grid(True)"""

plt.show()

Pmaxa = sys.P_max[:,:,:].cpu().detach()
P_max_a = Pmaxa[1,:,0:1]

Pa =d_mech[:,:,:].cpu().detach()
P_a = Pa[1,:,0:1]

ig_ref = sys.ig_ref[:,:,:].cpu().detach()
print(ig_ref.shape)
ig_r = torch.norm(ig_ref[1,:,0:2], dim=-1, keepdim=True) 

tm = sys.tm[:,:,:].cpu().detach()
tau_m = tm[1,:,0:1]

x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]
ig_log_base = x_log_base[1,:,1:3]
w_log_base = x_log_base[1,:,6]


x_log_test = x_log_test.cpu().detach()
vdc_log_test = x_log_test[1,:,0:1]
ig_log_test = x_log_test[1,:,1:3]
w_log_test = x_log_test[1,:,6]

u = u_PB.cpu().detach()

mg_norm = torch.norm(us, dim=-1, keepdim=True) 
mg_norm_test = torch.norm(u_log_test, dim=-1, keepdim=True) 


mg_norm = mg_norm[0,:,:]
mg_norm_test = mg_norm_test[0,:,:]
"""plt.figure(figsize=(12, 4))
plt.plot(np.array(range(mg_norm_test.shape[0]))*h, mg_norm_test,label = "Base controller with rPB",linewidth=4)    
plt.plot(np.array(range(mg_norm.shape[0]))*h, mg_norm,color = "#FF7F0E",label = "Base controller only",linewidth=4)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Time (s)",fontsize=30)
plt.ylabel(r"$||m_g||$",fontsize=30)
plt.grid()
plt.legend(fontsize=30)
plt.show()"""



"""plt.figure(figsize=(12, 4))
plt.plot(np.array(range(vdc_log_base.shape[0]))*h, w_log_test, label="Base controller with rPB",linewidth=5)
plt.plot(np.array(range(vdc_log_base.shape[0]))*h, w_log_base, color="#FF7F0E", label="Base controller only",linestyle = "--",linewidth=5)    
plt.axhline(y=125.66, linestyle="--", color="teal", label=r"$w^{ref}$",linewidth=3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Angular velocity (rad/s)",fontsize=20)
plt.grid()
plt.legend(fontsize=20)
plt.show()

t_max_a = P_max_a/w_log_base[0]
t_max_a = torch.clamp(t_max_a, max=46691*1.05)    
plt.figure(figsize=(12, 4))
plt.plot(np.array(range(test_data.shape[1]))*h, tau_m,label = f"Control torque", linewidth=3)
plt.plot(np.array(range(test_data.shape[1]))*h, t_max_a,label = f"Maximum possible torque",linestyle = "--", color= "red",alpha=0.9,linewidth=3)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.xlabel("Time (s)",fontsize=20)
plt.ylabel("Torque (Nm)",fontsize=20)
plt.legend(fontsize=20)
plt.grid()
plt.show()"""

# --- Plot ---
"""plt.figure(figsize=(12, 4))
plt.plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base, color="#FF7F0E", label="Base controller only")
plt.plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test, label="Base controller with rPB")
plt.axhline(y=5000, linestyle="--", color="teal", label=r"$v_{dc}^{ref}$",linewidth=5)
plt.axhline(y=4870, linestyle="--", color="red", label=r"Acceptable range",linewidth=5)
plt.axhline(y=5130, linestyle="--", color="red",linewidth=5)
#plt.title("Vdc profile over the horizon")
plt.xlabel("Time (s)",fontsize=25)
plt.ylabel("Voltage (V)", fontsize=25)
plt.legend(fontsize=23)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.grid(True)
plt.show()"""

i_norm = torch.norm(ig_log_base, dim=-1, keepdim=True) 
i_norm_test = torch.norm(ig_log_test, dim=-1, keepdim=True) 

dot_base = sys.h/sys.C*torch.bmm(us,x_log_base[:,:,1:3].transpose(1, 2))   # result shape: (B, T, 1)
dot = sys.h/sys.C*torch.bmm(u_log_test,x_log_test[:,:,1:3].transpose(1, 2))   # result shape: (B, T, 1)
mgig = sys.mgo

iff = sys.h/sys.C*sys.iff_traj[:,:,:].cpu().detach()


"""# --- Plot ---
plt.figure(figsize=(12, 4))
plt.plot(np.array(range(i_norm_test.shape[0]))*h,mgig_base[1,:,:],label="Base controller",)
plt.plot(np.array(range(i_norm_test.shape[0]))*h, mgig[1,:,:],label="Base controller with rPB",)
#plt.plot(np.array(range(i_norm_test.shape[0]))*h, -iff_base[1,:,:],label="Base controller",)
#plt.plot(np.array(range(i_norm_test.shape[0]))*h, -iff[1,:,:],label="Base controller with rPB",)
#plt.plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",linestyle = "--",label="Base controller only")
#plt.plot(np.array(range(i_norm.shape[0]))*h,[imax]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",color="red",label = r"$||I_g||_{max}$",linewidth=5)
#plt.title("I Norm profile over the horizon")
plt.xlabel("Time (s)",fontsize=25)
#plt.ylabel(r"$||I_{g}||$ (A)",fontsize=25)
plt.grid()
plt.xlim(0.06,0.12)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)

plt.figure(figsize=(12, 4))
plt.plot(np.array(range(i_norm_test.shape[0]))*h, ig_log_base,label=["ab","bb"],)
plt.plot(np.array(range(i_norm_test.shape[0]))*h, ig_log_test,linestyle="--",label=["a","b"])

#plt.plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",linestyle = "--",label="Base controller only")
#plt.plot(np.array(range(i_norm.shape[0]))*h,[imax]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",color="red",label = r"$||I_g||_{max}$",linewidth=5)
#plt.title("I Norm profile over the horizon")
plt.xlabel("Time (s)",fontsize=25)
#plt.ylabel(r"$||I_{g}||$ (A)",fontsize=25)
plt.grid()
plt.xlim(0.06,0.12)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)

plt.figure(figsize=(12, 4))
plt.plot(np.array(range(i_norm_test.shape[0]))*h, test_data[1,:,6],label="Base controller",)
plt.plot(np.array(range(i_norm_test.shape[0]))*h, test_data[1,:,7],label="Base controller with rPB",)

#plt.plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",linestyle = "--",label="Base controller only")
#plt.plot(np.array(range(i_norm.shape[0]))*h,[imax]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",color="red",label = r"$||I_g||_{max}$",linewidth=5)
#plt.title("I Norm profile over the horizon")
plt.xlabel("Time (s)",fontsize=25)
#plt.ylabel(r"$||I_{g}||$ (A)",fontsize=25)
plt.grid()
plt.xlim(0.06,0.12)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)


plt.figure(figsize=(12, 4))
plt.plot(np.array(range(i_norm_test.shape[0]))*h, vdc_log_base,label="Base controller",)
plt.plot(np.array(range(i_norm_test.shape[0]))*h, vdc_log_test,label="Base controller with rPB",)

#plt.plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",linestyle = "--",label="Base controller only")
#plt.plot(np.array(range(i_norm.shape[0]))*h,[imax]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",color="red",label = r"$||I_g||_{max}$",linewidth=5)
#plt.title("I Norm profile over the horizon")
plt.xlabel("Time (s)",fontsize=25)
#plt.ylabel(r"$||I_{g}||$ (A)",fontsize=25)
plt.grid()
plt.xlim(0.06,0.12)
plt.xticks(fontsize=22)
plt.yticks(fontsize=22)
plt.legend(fontsize=15)"""

"""plt.figure(figsize=(12, 4))

plt.axhline(y=1/np.sqrt(2), linestyle="--", color="red", label=r"$||m_g||_{max}$",linewidth=4)
plt.plot(np.array(range(mg_norm_test.shape[0]))*h, mg_norm_test,label = "Base controller with rPB",linewidth=4)    
plt.plot(np.array(range(mg_norm.shape[0]))*h, mg_norm,color = "#FF7F0E",label = "Base controller only",linewidth=4)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
plt.xlabel("Time (s)",fontsize=30)
plt.ylabel(r"$||m_g||$",fontsize=30)
plt.grid()
plt.xlim(0.03,0.12)

plt.legend(fontsize=25)

plt.show()"""

# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(6, 2, figsize=(16, 8))
axs[0, 0].set_title("Base controller only", fontsize=18)
axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, w_log_base,color="#FF7F0E", label = r"$w$")
axs[0, 0].axhline(y=125.66, linestyle="--", color="limegreen", label=r"$w^{ref}$",linewidth=2)
#axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("rad/s")
#axs[0, 0].set_ylim(120, 130)
axs[0, 0].set_xticklabels([])

axs[0,0].legend(loc='center right')
axs[0, 0].grid()


axs[0, 1].set_title("Base controller with rPB", fontsize=18)
axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_test, label = r"$w$")
axs[0, 1].axhline(y=125.66, linestyle="--", color="limegreen", label=r"$w^{ref}$",linewidth=2)
#axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("rad/s")
#axs[0, 1].set_ylim(120, 130)
axs[0, 1].set_xticklabels([])

axs[0,1].legend(loc='center right')
axs[0, 1].grid()

tau_m[0,:] = tau_m[1,:]
# Plot 3: U profile over the horizon
axs[1, 0].plot(np.array(range(test_data.shape[1]))*h, tau_m,color="#FF7F0E",label = r"$\tau_{m}$")
axs[1, 0].plot(np.array(range(i_norm.shape[0]))*h,[46691*0.95]*len(np.array(range(i_norm.shape[0]))),color = "darkorchid",label=r"$\tau_{l}$",linewidth=2)

#axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("Nm")
axs[1, 0].legend()
axs[1, 0].grid()
axs[1, 0].set_xticklabels([])

axs[1, 1].plot(np.array(range(test_data.shape[1]))*h, tau_m,label = r"$\tau_{m}$")
axs[1, 1].plot(np.array(range(i_norm.shape[0]))*h,[46691*0.95]*len(np.array(range(i_norm.shape[0]))),color = "darkorchid",label=r"$\tau_{l}$",linewidth=2)

#axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Nm")
axs[1, 1].legend()
axs[1, 1].grid()
axs[1, 1].set_xticklabels([])


mg_norm_test[0,:] = mg_norm_test[1,:]
mg_norm[0,:] = mg_norm[1,:]
axs[2, 0].axhline(y=1/np.sqrt(2), linestyle="--", color="red", label=r"$||m_g||_{max}$",linewidth=2)
axs[2, 0].plot(np.array(range(mg_norm.shape[0]))*h, mg_norm,color="#FF7F0E",label = r"$||m_g||$")

#axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylim(0,0.8)
axs[2, 0].legend(loc='center right')
axs[2, 0].set_xticklabels([])
axs[2, 0].grid()

axs[2, 1].axhline(y=1/np.sqrt(2), linestyle="--", color="red", label=r"$||m_g||_{max}$",linewidth=2)
axs[2, 1].plot(np.array(range(mg_norm_test.shape[0]))*h, mg_norm_test,label = r"$||m_g||$")
#axs[2, 1].set_xlabel("Time (s)")
#axs[2, 1].set_ylabel("||Mg||")
axs[2, 1].set_ylim(0,0.8)
axs[2, 1].legend(loc='center right')
axs[2, 1].set_xticklabels([])
axs[2, 1].grid()

axs[3, 0].plot(np.array(range(i_norm.shape[0]))*h, i_norm,color="#FF7F0E",label = r"$||i_g||$")
axs[3, 0].axhline(y=imax,linestyle = "--",color="red",label = r"$||i_g||_{max}$",linewidth=2)
#axs[3, 0].plot(np.array(range(i_norm.shape[0]))*h, ig_r,label = r"$||i_g||$")
#axs[3, 0].set_xlabel("Time (s)")
axs[3, 0].set_ylabel("A")
axs[3, 0].legend(loc='center right')
axs[3, 0].set_ylim(1250, 2250)
axs[3, 0].set_xticklabels([])
axs[3, 0].grid()


axs[3, 1].plot(np.array(range(i_norm_test.shape[0]))*h, i_norm_test,label = r"$||i_g||$")
axs[3, 1].axhline(y=imax,linestyle = "--",color="red",label = r"$||i_g||_{max}$",linewidth=2)

#axs[3, 1].set_xlabel("Time (s)")
axs[3, 1].set_ylabel("A")
axs[3, 1].legend(loc='center right')
axs[3, 1].set_ylim(1250, 2250)
axs[3, 1].grid()
axs[3, 1].set_xticklabels([])


axs[4, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base,color="#FF7F0E", label = r"$v_{dc}$")
#axs[4, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test, label = "PB")
axs[4, 0].axhline(y=5000, linestyle="--", color="limegreen", label=r"$v_{dc}^{ref}$",linewidth=2)
axs[4, 0].axhline(y=4870, linestyle="--", color="red", label=r"$[v_{dc}^{min},v_{dc}^{max}]$",linewidth=2)
axs[4, 0].axhline(y=5130, linestyle="--", color="red",linewidth=2)
#axs[4, 0].set_xlabel("Time (s)")
axs[4, 0].set_ylim(4750, 5150)
axs[4, 0].set_ylabel("V")
axs[4,0].legend(loc ='center right')
axs[4, 0].grid()
axs[4, 0].set_xticklabels([])


axs[4, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test,label = r"$v_{dc}$")
#axs[4, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test, label = "PB")
axs[4, 1].axhline(y=5000, linestyle="--", color="limegreen", label=r"$v_{dc}^{ref}$",linewidth=2)
axs[4, 1].axhline(y=4870, linestyle="--", color="red", label=r"$[v_{dc}^{min},v_{dc}^{max}]$",linewidth=2)
axs[4, 1].axhline(y=5130, linestyle="--", color="red",linewidth=2)
#axs[4, 1].set_xlabel("Time (s)")
axs[4, 1].set_ylim(4750, 5150)
axs[4, 1].set_ylabel("V")
axs[4, 1].legend(loc ='center right')
axs[4, 1].grid()
axs[4, 1].set_xticklabels([])

vgj = torch.tensor(np.array([0, -3000]))
Q_log_base = test_data[0,:,6]*ig_log_base[:,1] - test_data[0,:,7]*ig_log_base[:,0]




Q_log_test = test_data[0,:,6]*ig_log_test[:,1] - test_data[0,:,7]*ig_log_test[:,0]

axs[5, 0].plot(np.array(range(test_data.shape[1]))*h, Q_log_base[:],color="#FF7F0E",label = f"Q")
axs[5, 0].axhline(y=Qref,linestyle = "--", color="limegreen", label=r"$Q^{ref}$",linewidth=2)

axs[5, 0].set_xlabel("Time (s)",fontsize=14)
axs[5, 0].set_ylabel(r"$VAR$")
axs[5, 0].legend(loc='center right')
axs[5, 0].set_ylim(-2.4e+6, 2.8e+6)

axs[5, 0].grid()

axs[5, 1].plot(np.array(range(test_data.shape[1]))*h, Q_log_test[:],label = f"Q")
axs[5, 1].set_xlabel("Time (s)",fontsize=14)
axs[5, 1].set_ylabel(r"$VAR$")
axs[5, 1].axhline(y=Qref,linestyle = "--", color="limegreen", label=r"$Q^{ref}$",linewidth=2)

axs[5, 1].legend(loc='center right')
axs[5, 1].set_ylim(-2.4e+6, 2.8e+6)
axs[5, 1].grid()


plt.setp([ax.yaxis.label for ax in axs.flat], fontsize=14)
plt.setp([ax.get_legend().get_texts() 
          for ax in axs.flat if ax.get_legend() is not None], 
         fontsize=14)
for ax in axs.flat:
    ax.set_xlim(left=0)

plt.show()


"""# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(4, 2, figsize=(13, 9))




# Plot 1: X profile over the horizon
axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test,label = f"rPB")
axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base,color = "#FF7F0E", label = "Base")
#axs[0, 0].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test, label = "PB")
axs[0, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Acceptable range")
axs[0, 0].axhline(y=5130, linestyle="--", color="teal")
axs[0, 0].set_title("Vdc profile over the horizon")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Voltage (V)")
axs[0,0].legend(loc='center right')
axs[0, 0].grid()




axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_base,color = "#FF7F0E", label = "Base Controller")
axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_test, label = "rPB Controller")

#axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_abs,color = "#2CA02C", label = "Abs Controller", linestyle = "--")
axs[0, 1].axhline(y=125.66, linestyle="--", color="red", label=r"$w*$")
axs[0, 1].set_title("W profile over the horizon")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Angular velocity (rad/s)")
#axs[0, 1].set_ylim(120, 130)
axs[0,1].legend()
axs[0, 1].grid()

# Plot 2: DXref profile over the horizon


# Plot 3: U profile over the horizon
axs[2, 1].plot(np.array(range(test_data.shape[1]))*h, tau_m)
axs[2, 1].set_title(r"$\tau_{m}$ profile over the horizon")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("Torque (Nm)")
axs[2, 1].grid()



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


# Plot 1: X profile over the horizon
for i in range(test_data.shape[0]): 
    #axs[1, 0].plot(np.array(range(test_data.shape[1]))*h, ig_log_test[i])
    axs[2, 0].plot(np.array(range(test_data.shape[1]))*h, ig_log_test)
                   #linestyle = "--",color = "#FF7F0E", label = "Base")
axs[2, 0].set_title("I profile over the horizon")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("I")
axs[2,0].legend()
axs[2, 0].grid()



#axs[2, 0].plot(np.array(range(epoch+1)),loss_log)



i_norm = torch.norm(ig_log_base, dim=-1, keepdim=True) 
i_norm_test = torch.norm(ig_log_test, dim=-1, keepdim=True) 



axs[1, 1].plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",label = r"$||I||$")
axs[1, 1].plot(np.array(range(i_norm_test.shape[0]))*h, i_norm_test,label = r"$||I||_{PB}$")
axs[1, 1].plot(np.array(range(i_norm.shape[0]))*h,[imax]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",label = r"$||I||_{max}$")
axs[1, 1].set_title("I Norm profile over the horizon")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("||I||")
axs[1, 1].legend(loc='center right')
axs[1, 1].grid()

print(i_norm.shape)
i_norm = i_norm[:3000,:]
i_norm_test = i_norm_test[:3000,:]

# Plot 3: U profile over the horizon
axs[3, 0].plot(np.array(range(test_data.shape[1]))*h, u_log_test[0,:,:])
axs[3, 0].set_title("Mg profile over the horizon")
axs[3, 0].set_xlabel("Time (s)")
axs[3, 0].set_ylabel("mg")
axs[3, 0].grid()

mg_norm = torch.norm(us, dim=-1, keepdim=True) 
mg_norm_test = torch.norm(u_log_test, dim=-1, keepdim=True) 


mg_norm = mg_norm[0,:,:]
mg_norm_test = mg_norm_test[0,:,:]

axs[3, 1].plot(np.array(range(mg_norm.shape[0]))*h, pb[1,:,0],color = "#FF7F0E",label = r"$Mg_0$")
axs[3, 1].plot(np.array(range(mg_norm_test.shape[0]))*h, pb[1,:,1],label = r"$Mg_1$")
axs[3, 1].set_title("PB output")
axs[3, 1].set_xlabel("Time (s)")
axs[3, 1].set_ylabel("||Mg||")
axs[3, 1].legend(loc='center right')
axs[3, 1].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'System evolution', fontsize=17)
plt.show()"""


