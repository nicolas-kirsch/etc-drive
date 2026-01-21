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
from plants import MechanicalDataset, MechanicalSystem
from assistive_functions import WrapLogger
from loss_functions import MechanicalLoss

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
x0 = torch.Tensor([[80],[0]]).to(device)

xref = torch.Tensor([[80]]).to(device)

if args.lower_stiff:
    h = 1e-2
    M = 1
    D = 1e-4
else: 
    h = 1e-4
    M = 23e+3
    D = 1e-4

# ------------ 1. Dataset ------------
dataset = MechanicalDataset(
    random_seed=args.random_seed, horizon=args.horizon
)

# divide to train and test
train_data, test_data = dataset.get_data(num_train_samples=args.num_rollouts, num_test_samples=10)
train_data, test_data = train_data.to(device), test_data.to(device)



# batch the data
train_dataloader = DataLoader(train_data, batch_size=args.num_rollouts, shuffle=False) ## might be slower is batch size too big, shuffle was True, and batch_size was args.batch_size (or smthng)


# ------------ 2. System ------------

sys = MechanicalSystem(x0,xref,h,M,D).to(device)


# ------------ 3. Controller ------------
ctl = PerfBoostController(
    noiseless_forward=sys.noiseless_forward,
    input_init=sys.x0, output_init=sys.u_init,
    dim_internal=args.dim_internal, dim_nl=args.l,
    xref = xref,batch_size=args.batch_size,
    initialization_std=args.cont_init_std,
    output_amplification=20,
).to(device)

# ------------ 2.bis Base rollout ------------
with torch.no_grad():
    x_log_base, u_log_base, pb_base = sys.rollout(
        controller=ctl, data=test_data, no_PB=True
    )
x_log_base = x_log_base.cpu()[:,:,0:1]



# ------------ 4. Loss ------------
#Size of the minimization 
loss_fn = MechanicalLoss(
    R=0,Q=10, x_min=x_min,xref=xref,alpha_min=30  
)

# ------------ 5. Optimizer ------------
optimizer = torch.optim.Adam(ctl.parameters(), lr=args.lr)
valid_data = train_data      # use the entire train data for validation





# ------------ 6. Training ------------
logger.info('\n------------ Begin training ------------')
best_valid_loss = 1e6

for epoch in range(1+args.epochs):
    # iterate over all data batches
    for train_data_batch in train_dataloader:
        optimizer.zero_grad()

        # simulate over horizon steps
        x_log, u_log, _ = sys.rollout(controller=ctl, data=train_data_batch)
        x_log = x_log[:,:,0:1]
        # loss of this rollout
        loss = loss_fn.forward(x_log, u_log)

        loss.backward()
        optimizer.step()

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
                x_log_valid = x_log_valid[:,:,0:1]
                if epoch == 0: 

                    x_log_test = x_log_valid.cpu()
                    u_log_test = u_log_valid.cpu()

                # loss of the valid data
                loss_valid = loss_fn.forward(x_log_valid, u_log_valid)

            msg += ' ---||---  VALIDATION LOSS: %.2f ' % (
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
    x_log_test, u_log_test, pb = sys.rollout(
        controller=ctl, data=test_data, no_PB=False
    )
    xlt = x_log_test[:,:,0:1]
    test_loss = loss_fn.forward(xlt, u_log_test)[0][0].item()
    print(f"\n TEST loss: {test_loss:.2f}")  ##

x_log_test = x_log_test.cpu()
u_log_test = u_log_test.cpu()
pb = pb.cpu()

v_log_test = x_log_test[:,:,1:2]
x_log_test = x_log_test[:,:,0:1]

# Create a figure with a 2x2 grid of subplots
fig, axs = plt.subplots(2, 2, figsize=(13, 9))

# Plot 1: X profile over the horizon
for i in range(test_data.shape[0]): 
    axs[0, 0].plot(np.array(range(test_data.shape[1]))*0.01, x_log_test[i])
axs[0, 0].plot(np.array(range(test_data.shape[1]))*0.01, x_log_base[1],linestyle = "--",color = "#FF7F0E", label = "Base")
axs[0, 0].set_title("X profile over the horizon")
axs[0, 0].set_xlabel("Time (s)")
axs[0, 0].set_ylabel("W")
axs[0,0].legend()
axs[0, 0].grid()


# Plot 2: DXref profile over the horizon
for i in range(test_data.shape[0]): 
    axs[0, 1].plot(np.array(range(test_data.shape[1]))*0.01, pb[i])
axs[0, 1].set_title("DXref profile over the horizon")
axs[0, 1].set_xlabel("Time (s)")
axs[0, 1].set_ylabel("W")
axs[0, 1].grid()

# Plot 3: U profile over the horizon
for i in range(test_data.shape[0]): 
    axs[1, 1].plot(np.array(range(test_data.shape[1]))*0.01, u_log_test[i])
axs[1, 1].set_title("U profile over the horizon")
axs[1, 1].set_xlabel("Time (s)")
axs[1, 1].set_ylabel("Tau_m")
axs[1, 1].grid()

# Plot 3: U profile over the horizon
for i in range(test_data.shape[0]): 
    axs[1, 0].plot(np.array(range(test_data.shape[1]))*0.01, v_log_test[i])
axs[1, 0].set_title("V profile over the horizon")
axs[1, 0].set_xlabel("Time (s)")
axs[1, 0].set_ylabel("V")
axs[1, 0].grid()

# Adjust layout to prevent overlap
plt.tight_layout()
plt.subplots_adjust(top=0.9)  # Adjust the top space to make room for the suptitle

plt.suptitle(f'System evolution', fontsize=17)

# plt.savefig("testing_no_load_no_PB.png")
plt.show()
