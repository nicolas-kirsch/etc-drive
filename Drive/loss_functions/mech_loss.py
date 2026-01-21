import path
import sys
 
# directory reach
directory = path.Path(__file__).absolute()
 
# setting path
sys.path.append(directory.parent.parent)

import torch
from config import device
from assistive_functions import to_tensor



class MechanicalLoss():
    def __init__(self, R,Q,x_min, xref,alpha_min=None ):
        
        self.alpha_min = alpha_min
        
        self.xmin = to_tensor(x_min).to(device)
        self.xref = xref

        self.Q = to_tensor(Q)
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

        uTRu = self.R * torch.matmul(
                u_batch.transpose(-1, -2),
                u_batch
            )   # shape = (S, T, 1, 1)
        e = x_batch-self.xref

        eTQe = self.Q * torch.matmul(
                e.transpose(-1, -2),
                e
            )   # shape = (S, T, 1, 1)
        

        loss_u = torch.sum(uTRu, 1) / x_batch.shape[1] 
        loss_e = torch.sum(eTQe, 1) / x_batch.shape[1]


        # lower bound on temperature loss
        if self.alpha_min is None:
            loss_xl = 0
        else:
            loss_xl = self.alpha_min * self.f_lower_bound_x(x_batch) # shape = (S, 1, 1)

        loss_val = loss_u + loss_e + loss_xl 
        loss_val = torch.sum(loss_val, 0)/xs.shape[0] 
        return loss_val   

        

    def f_lower_bound_x(self, x_batch):

        delta = self.xmin - x_batch  

        loss_bound = torch.relu(delta) 
        loss_xl = loss_bound.sum(1)/loss_bound.shape[1]
        return loss_xl


