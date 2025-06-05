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

wref = 80
vref = 5000
Qref = 8e+6 
h = 1e-4

yref = torch.Tensor([vref,Qref]).to(device)

M = 23e3
D = 1e-4
C = 1e-2
G = 1e-5

l = 2 * 1.567e-3
r = 2e-3
w = 2 * np.pi * 50


# ------------ 1. Dataset ------------
dataset = ConverterDataset(
    random_seed=args.random_seed, horizon=args.horizon
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=3)
train_data, test_data = train_data.to(device), test_data.to(device)
train_data = train_data[:,:800,:]
print(train_data.shape)
test_data = test_data[:,:3000,:]

# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False) ## might be slower is batch size too big, shuffle was True, and batch_size was args.batch_size (or smthng)



# ------------ 2. Plant ------------
sys  = Converter(x0,wref,vref,Qref,h,M,D,C,G,l,r,w)


# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x0, output_init=sys.u_init,d_mech_init=sys.d_mech,
    dim_internal=args.dim_internal, dim_nl=args.l,
    vdc_ref = vref,Q_ref=Qref,batch_size=args.batch_size,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)


R = 0
Q_Q = 0
Q_vdc = 10
alpha_i_max = 0

# ------------ 4. Loss ------------
#Size of the minimization 
loss_fn = ConverterLoss(
    R=R,Q_Q=Q_Q,Q_vdc=Q_vdc, x_min=x_min,yref=yref,alpha_i_max=alpha_i_max  
)


x_log_base,us,u_PB = sys.rollout(ctl,test_data)


x_log_base = x_log_base.cpu().detach()
vdc_log_base = x_log_base[1,:,0:1]
ig_log_base = x_log_base[1,:,1:3]

"""plt.figure(figsize=(6, 3))
plt.plot(np.array(range(test_data.shape[1]))*h, vdc_log_base,  color="#FF7F0E",label="Base Controller")
plt.axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
plt.xlabel("Time (s)")
plt.ylabel("Voltage (V)")
plt.legend()
plt.grid()
plt.savefig("vdc_base.png", bbox_inches='tight')
"""

# ------------ 5. Optimizer ------------
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
valid_data = train_data      # use the entire train data for validation


loss_log = []


# ------------ 6. Training ------------
logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e10

for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()

        # simulate over horizon steps
        x_log, u_log, _ = sys.rollout(controller=ctl, data=train_data_batch)
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
                x_log_valid, u_log_valid, _ = sys.rollout(
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

"""print("REN output biases")
print(ctl.c_ren.b_y)"""

with torch.no_grad():
    x_log_test, u_log_test, pb = sys.rollout(
        controller=ctl, data=test_data, no_PB=False
    )
    test_loss = loss_fn.forward(x_log_test, u_log_test)[0][0].item()
    print(f"\n TEST loss: {test_loss:.2f}")  ##

x_log_test = x_log_test.cpu()
u_log_test = u_log_test.cpu()
pb = pb.cpu()
pb_ref = pb[:,:,0:1]+Qref
d_ierr = pb[:,:,1:2]+vref

vdc_log_test = x_log_test[:,:,0:1]
ig_log_test = x_log_test[:,:,1:3]
integrator = x_log_test[:,:,3:6]
inte =  x_log_test[:,:,7:8]
print(integrator[0,-1,:])
print(inte[0,-1,:])
w_log_test = x_log_test[:,:,6]
vgj = to_tensor(np.array([0, -3000]))
Q_log_test = vgj[0]*ig_log_test[:,:,0] + vgj[1]*ig_log_test[:,:,1]
print(Q_log_test.shape)
# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(3, 2, figsize=(13, 9))

# Plot 1: X profile over the horizon
#axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, vdc_log_test[0],label = f"rPB")
axs[0, 0].plot(np.array(range(test_data.shape[1]))*h, vdc_log_base,color = "#FF7F0E", label = "Base Controller")
axs[0, 0].axhline(y=5000, linestyle="--", color="red", label=r"$v_{dc}^*$")
axs[0, 0].axhline(y=4870, linestyle="--", color="teal", label=r"Acceptable range")
axs[0, 0].axhline(y=5130, linestyle="--", color="teal")
axs[0, 0].set_title("Vdc profile over the horizon")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("Voltage (V)")
axs[0,0].legend()
axs[0, 0].grid()


# Plot 2: DXref profile over the horizon
for i in range(test_data.shape[0]): 
    axs[2, 1].plot(np.array(range(test_data.shape[1]))*h, Q_log_test[i,:])
axs[2, 1].set_title("Evolution of Q over the horizon ")
axs[2, 1].set_xlabel("Time (s)")
axs[2, 1].set_ylabel(r"$Q^*$")
axs[2, 1].grid()

# Plot 3: U profile over the horizon
for i in range(test_data.shape[0]): 
    axs[1, 1].plot(np.array(range(test_data.shape[1]))*h, u_log_test[i])
axs[1, 1].set_title("Mg profile over the horizon")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("mg")
axs[1, 1].grid()

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
for i in range(test_data.shape[0]): 
    axs[2, 0].plot(np.array(range(test_data.shape[1]))*h, d_ierr[i])
axs[2, 0].set_title("Evolution of d_ierr over the horizon ")
axs[2, 0].set_xlabel("Time (s)")
axs[2, 0].set_ylabel(r"$v_{dc}^*$")
axs[2, 0].grid()

i_norm = torch.norm(ig_log_test, dim=-1, keepdim=True) 
print(i_norm.shape)

axs[0, 1].plot(np.array(range(i_norm.shape[1]))*h, i_norm[0])
axs[0, 1].plot(np.array(range(test_data.shape[1]))*h,[2800]*len(np.array(range(test_data.shape[1]))),linestyle = "--")
axs[0, 1].set_title("I Norm profile over the horizon")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("||I||")
axs[0, 1].legend()
axs[0, 1].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'System evolution', fontsize=17)
plt.show()





