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
imax = 2800/I_base

yref = torch.Tensor([vref,qref]).to(device)


# ------------ 1. Dataset ------------
dataset = ConverterDataset(
    random_seed=args.random_seed, horizon=args.horizon
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=3)
train_data, test_data = train_data.to(device), test_data.to(device)
train_data = train_data[:,:3000,:]
print(train_data.shape)
test_data = test_data[:,:12000,:]


# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False) 
## might be slower is batch size too big, shuffle was True, and batch_size was args.batch_size (or smthng)


# ------------ 2. Plant ------------
sys  = Converter(x0,wref,vref,Qref,h,m,d,c,g,lg,z,l,base_values)

# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x0, output_init=sys.u_init,d_mech_init=sys.d_mech,
    dim_internal=args.dim_internal, dim_nl=args.l,
    vdc_ref = Vref,Q_ref=Qref,batch_size=args.batch_size,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)


x_log_base,us,u_PB,d_mech = sys.rollout(ctl,test_data)
Q_ref_base = sys.Q_ref_e.cpu().detach().numpy()

# ------------ 4. Loss ------------
R = 0
Q_Q = 0
Q_vdc = 1e+5
alpha_i_max = 0

msg = 'R: %.2f --- Q_Q: %.2f --- Q_vdc: %.2f --- alpha_i_max: %.2f' % (R, Q_Q, Q_vdc, alpha_i_max)

logger.info(msg)

# ------------ 4. Loss ------------
#Size of the minimization 
loss_fn = ConverterLoss(
    R=R,Q_Q=Q_Q,Q_vdc=Q_vdc, x_min=x_min,yref=yref,imax = imax, alpha_i_max=alpha_i_max  
)

# ------------ 5. Optimizer ------------
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
valid_data = train_data      # use the entire train data for validation



loss_log = []


# ------------ 6. Training ------------
logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e20

for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()

        # simulate over horizon steps
        x_log, u_log, _ ,_= sys.rollout(controller=ctl, data=train_data_batch)

        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log)
        loss.backward()
        optimizer.step()




    loss_log.append(loss.cpu().detach().item())

    # print info
    if epoch%args.log_epoch == 0:
        print('---------------')
        msg = 'Epoch: %i --- TRAIN LOSS : %.2f'% (epoch, loss)
        #msg +='--- Loss xh : %.2f ---  loss ul: %.2f---  loss uh: %.2f'% (loss_xh, loss_ul, loss_uh)
        if args.return_best:
            # rollout the current controller on the valid data
            with torch.no_grad():
                x_log_valid, u_log_valid, _ ,_= sys.rollout(
                    controller=ctl, data=valid_data
                )
                if epoch == 0: 
                    x_log_test = x_log_valid.cpu()
                    u_log_test = u_log_valid.cpu()


                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid)

            msg += ' ---||---  TESTING LOSS: %.2f ' % (
                loss_valid)
            # compare with the best valid loss
            if loss_valid.item()<best_valid_loss:
                x_log_valid_best = x_log_valid          ##
                u_log_valid_best = u_log_valid
                best_valid_loss = loss_valid.item()
                best_params = ctl.get_parameters_as_vector()  # record state dict if best on valid
                msg += ' (*** best so far ***)'
        logger.info(msg)


# set to best seen during training
if args.return_best:
    ctl.set_parameters_as_vector(best_params)

with torch.no_grad():
    x_log_test, u_log_test, pb,_ = sys.rollout(
        controller=ctl, data=test_data, no_PB=False
    )
    test_loss = loss_fn.forward(x_log_test, u_log_test)[0][0].item()
    print(f"\n TEST loss: {test_loss:.2f}")  ##


x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]*V_base
vdc_log_base_surge = x_log_base[0,:,0:1]*V_base
ig_log_base = x_log_base[1,:,1:3]*I_base

x_log_test = x_log_test.cpu().detach()
vdc_log_test = x_log_test[1,:,0:1]*V_base
vdc_log_test_surge = x_log_test[0,:,0:1]*V_base

ig_log_test = x_log_test[1,:,1:3]*I_base

w_log_base = x_log_base[1,:,6]*w_base
vgj = to_tensor(np.array([0, -3000]))
Q_log_base = vgj[0]*ig_log_base[:,0] + vgj[1]*ig_log_base[:,1]

w_log_test = x_log_test[1,:,6]*w_base
vgj = to_tensor(np.array([0, -3000]))
Q_log_test = vgj[0]*ig_log_test[:,0] + vgj[1]*ig_log_test[:,1]



pb = pb.cpu()
pb_ref = pb[:,:,0:1]+Qref
d_ierr = pb[:,:,1:2]+Vref


u_log_base = us.cpu().detach()

d_mech = d_mech.cpu().detach().numpy()*P_base
print(x_log_base[0,-1,:])
print(torch.isnan(x_log_base[0,-1,1]).any().item())

# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(4, 2, figsize=(13, 9))

# Plot 1: X profile over the horizon
#axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, vdc_log_test[0],label = f"rPB")
axs[0, 0].plot(np.array(range(vdc_log_base_surge.shape[0]))*h, vdc_log_base_surge,color = "#FF7F0E", label = "Base Controller")
axs[0, 0].plot(np.array(range(vdc_log_base_surge.shape[0]))*h, vdc_log_test_surge, label = "PB Controller")
axs[0, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Acceptable range")
axs[0, 0].axhline(y=5130, linestyle="--", color="teal")
axs[0, 0].set_title("Vdc profile over the horizon")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Voltage (V)")
axs[0,0].legend(loc='center right')
axs[0, 0].grid()


axs[0, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_base,color = "#FF7F0E", label = "Base Controller")
axs[0, 1].plot(np.array(range(vdc_log_base.shape[0]))*h, vdc_log_test, label = "PB Controller")
axs[0, 1].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 1].axhline(y=4870, linestyle="--", color="teal", label=r"Acceptable range")
axs[0, 1].axhline(y=5130, linestyle="--", color="teal")
axs[0, 1].set_title("Vdc profile over the horizon")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Voltage (V)")
axs[0, 1].legend(loc='center right')
axs[0, 1].grid()

"""axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_base,color = "#FF7F0E", label = "Base Controller")
axs[0, 1].plot(np.array(range(test_data.shape[1]))*h, w_log_test, label = "PB Controller")
axs[0, 1].axhline(y=80, linestyle="--", color="red", label=r"$w*$")
axs[0, 1].set_title("W profile over the horizon")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("Angular velocity (W)")
axs[0, 1].set_ylim(79, 81)
axs[0,1].legend()
axs[0, 1].grid()"""

# Plot 2: DXref profile over the horizon

"""axs[3, 1].plot(np.array(range(test_data.shape[1]))*h, Q_log_base[:],label = f"Q")
axs[3, 1].plot(np.array(range(test_data.shape[1]))*h, Q_ref_base[0,:],label = r"$Q^*$")
axs[3, 1].plot(np.array(range(test_data.shape[1]))*h,[Qref]*len(np.array(range(test_data.shape[1]))),linestyle = "--", color="red", label=r"Asked Q")
axs[3, 1].set_title("Evolution of Q over the horizon ")
axs[3, 1].set_xlabel("Time (s)")
axs[3, 1].set_ylabel(r"$Q$")
axs[3, 1].legend(loc='center right')
axs[3, 1].grid()
"""
# Plot 3: U profile over the horizon
axs[2, 1].plot(np.array(range(test_data.shape[1]))*h, u_log_base[0,:,:])
axs[2, 1].set_title("Mg profile over the horizon")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel("mg")
axs[2, 1].grid()

mg_norm = torch.norm(u_log_base, dim=-1, keepdim=True) 
mg_norm_test = torch.norm(u_log_test, dim=-1, keepdim=True) 


axs[2, 0].plot(np.array(range(mg_norm.shape[1]))*h, mg_norm[0],color = "#FF7F0E",label = r"$||M_g||$")
axs[2, 0].plot(np.array(range(mg_norm_test.shape[1]))*h, mg_norm_test[0],label = r"$||M_g||_{PB}$")
axs[2, 0].plot(np.array(range(test_data.shape[1]))*h,[1/np.sqrt(2)]*len(np.array(range(test_data.shape[1]))),linestyle = "--", label = r"$||M_g||_{max}$")
axs[2, 0].set_title("Mg Norm profile over the horizon")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel("||Mg||")
axs[2, 0].legend(loc ='center right')
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

"""axs[3, 0].plot(np.array(range(test_data.shape[1]))*h, d_mech[0])
axs[3, 0].plot(np.array(range(test_data.shape[1]))*h,[1e+5*80]*len(np.array(range(test_data.shape[1]))),linestyle = "--",)
axs[3, 0].set_title("Evolution of P_mech over the horizon ")
axs[3, 0].set_xlabel("Time (s)")
axs[3, 0].set_ylabel(r"$P_{mech}$")
axs[3, 0].grid()"""

i_norm = torch.norm(ig_log_base, dim=-1, keepdim=True) 
i_norm_test = torch.norm(ig_log_test, dim=-1, keepdim=True) 



axs[1, 1].plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",label = r"$||I||$")
axs[1, 1].plot(np.array(range(i_norm_test.shape[0]))*h, i_norm_test,label = r"$||I||_{PB}$")
axs[1, 1].plot(np.array(range(i_norm.shape[0]))*h,[2800]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",label = r"$||I||_{max}$")
axs[1, 1].set_title("I Norm profile over the horizon")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("||I||")
axs[1, 1].legend(loc='center right')
axs[1, 1].grid()

print(i_norm.shape)
i_norm = i_norm[:3000,:]
i_norm_test = i_norm_test[:3000,:]

axs[3, 0].plot(np.array(range(i_norm.shape[0]))*h, i_norm,color = "#FF7F0E",label = r"$||I||$")
axs[3, 0].plot(np.array(range(i_norm_test.shape[0]))*h, i_norm_test,label = r"$||I||_{PB}$")
axs[3, 0].plot(np.array(range(i_norm.shape[0]))*h,[2800]*len(np.array(range(i_norm.shape[0]))),linestyle = "--",label = r"$||I||_{max}$")
axs[3, 0].set_title("I Norm profile over the horizon - zoomed")
axs[3, 0].set_xlabel("Time (s)")
axs[3, 0].set_ylabel("||I||")
axs[3, 0].legend(loc='center right')
axs[3, 0].grid()


mg_norm = mg_norm[0,:3000,:]
mg_norm_test = mg_norm_test[0,:3000,:]

axs[3, 1].plot(np.array(range(mg_norm.shape[0]))*h, mg_norm,color = "#FF7F0E",label = r"$||Mg||$")
axs[3, 1].plot(np.array(range(mg_norm_test.shape[0]))*h, mg_norm_test,label = r"$||Mg||_{PB}$")
axs[3, 1].plot(np.array(range(mg_norm.shape[0]))*h,[1/np.sqrt(2)]*len(np.array(range(mg_norm.shape[0]))),linestyle = "--",label = r"$||Mg||_{max}$")
axs[3, 1].set_title("Mg Norm profile over the horizon - zoomed")
axs[3, 1].set_xlabel("Time (s)")
axs[3, 1].set_ylabel("||Mg||")
axs[3, 1].legend(loc='center right')
axs[3, 1].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'System evolution', fontsize=17)
plt.savefig(os.path.join(save_folder, 'system_evolution.png'), dpi=300, bbox_inches='tight')
plt.show()






