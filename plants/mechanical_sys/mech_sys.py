import os,sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(1, BASE_DIR)


from assistive_functions import to_tensor
from config import device
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt

class MechanicalSystem(torch.nn.Module):
    def __init__(self,x0,xref,h, M, D,u_init = None):

        super().__init__()

        # Parameters
        self.h = h
        self.M = M
        self.D = D


        A = np.array([[(1-self.h*D/M)]])
        B = np.array([[self.h/M]])
        Bd = np.array([[self.h/M],[0]])
         

        omega_m = (2 * np.pi * 5.55) / 6
        zeta_m = np.sqrt(2)
        
        # Gain calculations
        KI_m = np.array([[omega_m**2 * M]])
        KP_m = np.array([[2 * zeta_m * omega_m * M]])



        # Exteral values
        self.tau_l = 1e+5
        

        #Initial conditions 
        self.x0 = x0
        
        #Reference values
        self.xref = xref

        self.A, self.B, self.KI_m,self.KP_m, self.Bd = to_tensor(A),to_tensor(B),to_tensor(KI_m),to_tensor(KP_m), to_tensor(Bd)
  
        self.state_dim = self.A.shape[0]
        self.in_dim = self.B.shape[1]

        self.u_init = torch.full((1, int(self.x0.shape[1])),float(0)).to(device) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)


    def base_controller(self,x: torch.Tensor,v: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """

            u_base = -F.linear(x,self.KP_m) - F.linear(v,self.KI_m)       # one is transient an other is SS
            
            
            return u_base
    
    def noiseless_forward(self, x: torch.Tensor,u: torch.Tensor, no_PB=False):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state without the noise.
        """

        x = x.view(-1, 1, 2)

        v = x[:,0:1,1:2]
        x = x[:,0:1,0:1]
     



        x = x.view(-1, 1, self.state_dim)
        v = v.view(-1, 1, self.state_dim)

        u_PB = u.view(-1, 1, self.in_dim)

        if no_PB:
            ref = self.xref
        else:  
            ref = self.xref + u_PB

        error = x-ref

        

        u_base = self.base_controller(x,v) 
   

        u_cont = u_base

        #Saturate u
        #u_cont = torch.where(u_cont<0,0,torch.where(u_cont>self.umax,self.umax,u_cont))

        x_ = F.linear(x,self.A)+ F.linear(u_cont,self.B)
        v_ = v+self.h*error
        #Save base controller input
        self.u_cont = u_cont

        f = torch.cat((x_,v_),2)

        return f
    
    def forward(self, x, u,d,no_PB):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """
        d = d.view(-1, 1, self.state_dim)

        return self.noiseless_forward(x,u,no_PB) - F.linear(d,self.Bd)


    
    def rollout(self, controller,data: torch.Tensor, no_PB = False):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """


        controller.reset()



        x = data[:,0:1,:]
        v = torch.zeros(data[:,0:1,:].shape)
        
        xs = torch.cat((x,v),2)
        u_PB = controller.forward(xs[:, 0:1, :])
        us = torch.full(u_PB.shape,self.u_cont.item()).to(device)


        #us = torch.full(dxref.shape,self.u_cont.item()).to(device)
        #us = to_tensor(np.array([0])).to(device)
        for t in range(1, data.shape[1]):
            xs = torch.cat(
                (
                    xs,
                    self.forward(xs[:, t-1:t, :],u_PB[:, t-1:t, :],data[:, t:t+1, :],no_PB)
                    ),
                1
            )

            u_PB = torch.cat(
                (u_PB, controller.forward(xs[:, t:t+1, :])),
                1
            )

            us = torch.cat(
                (us, self.u_cont),
                1
            )
        controller.reset()


        #dxref = dxref     ##
        return xs, us,u_PB
        
    def rollout_no_pb(self, controller,data: torch.Tensor, no_PB = False):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """

        self.v = to_tensor(np.array([0]))



        xs = data[:,0:1,:]
        
        u_PB = controller.forward(xs[:, 0:1, :])
        us = torch.full(u_PB.shape,self.u_cont.item()).to(device)


        #us = torch.full(dxref.shape,self.u_cont.item()).to(device)
        #us = to_tensor(np.array([0])).to(device)
        for t in range(1, data.shape[1]):
            xs = torch.cat(
                (
                    xs,
                    self.forward(xs[:, t-1:t, :],u_PB[:, t-1:t, :],data[:, t:t+1, :],no_PB)
                    ),
                1
            )

            u_PB = torch.cat(
                (u_PB, controller.forward(xs[:, t:t+1, :])),
                1
            )

            us = torch.cat(
                (us, self.u_cont),
                1
            )
        controller.reset()


        #dxref = dxref     ##
        return xs, us,u_PB

        

"""# Instantiate the class
mecha = MechanicalSystem()
x = to_tensor(np.array([70])) 
data = torch.zeros(1,50000,1)
data[:,0,0] = 80
w = to_tensor(np.array([1e+5])) 

xs,us = mecha.rollout(data)

plt.plot(xs[0,:,:])
plt.show()"""
