import path
import sys
import torch.nn.functional as F

# directory reach
directory = path.Path(__file__).absolute()
 
# setting path
sys.path.append(directory.parent.parent)

import torch
from config import device
from assistive_functions import to_tensor



class ConverterLoss():
    def __init__(self, R,Q_Q,Q_vdc,x_min, yref,imax,alpha_i_max=None ):

        self.vmat =(torch.tensor([3000,0])@torch.tensor([[0,-1],[1,0]])).float()

        self.imax = imax
        self.alpha_i_max = alpha_i_max
        
        self.xmin = to_tensor(x_min).to(device)
        self.yref = yref
        self.Q_Q = to_tensor(Q_Q)
        self.Q_vdc = to_tensor(Q_vdc)
        self.R =  to_tensor(R)
        if isinstance(self.R, torch.Tensor):     # cast to device if is not a scalar
            self.R = self.R.to(device)
        assert (not hasattr(self.R, "__len__")) or len(self.R.shape) == 2

    def forward(self, xs, us):
        """
        Compute loss.

        Args:
            - xs: tensor of shape (S, T, state_dim)
            - us: tensor of shape (S, T, in_dim)

        Return:
            - loss of shape (1, 1).
        """

        # batch
        x_batch = xs.reshape(*xs.shape,1)
        u_batch = us.reshape(*us.shape, 1)
        Q_batch = F.linear(xs[:,:,1:3],self.vmat).reshape((xs.shape[0],xs.shape[1],1,1))

        # Extract the last two elements along the state_dim axis
        i_g = xs[:,:, 1:3]  # Shape: (S, T, 2)

        # Compute the 2-norm along the last dimension (state_dim)
        i_norm = torch.norm(i_g, dim=-1, keepdim=True)  # Shape: (S, T, 1)
        i_norm_batch = i_norm.reshape(*i_norm.shape, 1)
        # Should print (S, T, 1)

        uTRu = self.R * torch.matmul(
                u_batch.transpose(-1, -2),
                u_batch
            )   # shape = (S, T, 1, 1)
        
        e_Q = Q_batch-self.yref[1]
        eTQe = self.Q_Q * torch.matmul(
                e_Q.transpose(-1, -2),
                e_Q
            )   # shape = (S, T, 1, 1)
        
        e_vdc = x_batch[:,:,0:1,:]-self.yref[0]

        evTQev = self.Q_vdc * torch.matmul(
                e_vdc.transpose(-1, -2),
                e_vdc
            )   # shape = (S, T, 1, 1)
        

        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1] 
        loss_eQ = torch.sum(eTQe, 1) / x_batch.shape[1]
        loss_ev = torch.sum(evTQev, 1) / x_batch.shape[1]


        # lower bound on temperature loss
        if self.alpha_i_max is None:
            loss_imax = 0
        else:
            loss_imax = self.alpha_i_max * self.f_upper_bound_i(i_norm_batch) # shape = (S, 1, 1)


        loss_val = loss_eQ + loss_ev + loss_imax 
        loss_val = torch.sum(loss_val, 0)/xs.shape[0] 
        return loss_val   

        

    def f_upper_bound_i(self, norm_i):

        delta = norm_i  - self.imax

        loss_bound = torch.relu(delta)**2 
        loss_xl = loss_bound.sum(1)/loss_bound.shape[1]
        return loss_xl


