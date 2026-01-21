import os,sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(1, BASE_DIR)


from assistive_functions import to_tensor
from config import device
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt


torch.set_printoptions(precision=8)

class Converter(torch.nn.Module):
    def __init__(self,wref,vref,Qref,h, M, D,C,G,l,r,w,u_init = None):

        super().__init__()

        # Parameters
        self.h = h

        self.M = M
        self.D = D

        self.C = C
        self.G = G


        self.Lg = np.array([[l, 0], [0, l]])
        self.Z = np.array([[r, -w * l], [w * l, r]])

        self.Lg_inv = np.linalg.inv(self.Lg)


        Am = np.array([[(1-self.h*D/M)]])
        Bm = np.array([[self.h/M]])
        Bdm = np.array([[self.h/M],[0]])
         

        omega_m = (2 * np.pi * 5.55) / 6
        zeta_m = np.sqrt(2)
        
        # Gain calculations
        KI_m = np.array([[omega_m**2 * M]])
        KP_m = np.array([[2 * zeta_m * omega_m * M]])

        # Additional parameters for electrical system
        omega_d = 2 * np.pi * 140
        zeta_d = 1.2
                
        # Gain calculations for electrical system
        KI_g = omega_d**2 * l
        KP_g = 2 * zeta_d * omega_d * l
    
         

        # Additional parameters for DC system
        omega_dc = 2 * np.pi * 2
        zeta_dc = 2
        
        
        # Gain calculations for DC system
        KI_dc = omega_dc**2 * C
        KP_dc = 2 * zeta_dc * omega_dc * C

        #KI_dc = 1.579136704174297
        #KP_dc = 0.502654824574367
        # Exteral values
        tau_l = 1e+5
        self.vg = np.array([3000, 0])

        self.v_mat = to_tensor(np.linalg.inv(np.array([[self.vg[0], self.vg[1]], 
                               [self.vg[1], -self.vg[0]]])))
        
        
        #Reference values
        self.wref = to_tensor(wref)
        self.Qref = to_tensor(Qref)
        self.vref = to_tensor(vref)

        self.Lg_inv = to_tensor(self.Lg_inv)
        self.Z = to_tensor(self.Z)

        self.Am, self.Bm, self.KI_m,self.KP_m, self.Bdm = to_tensor(Am),to_tensor(Bm),to_tensor(KI_m),to_tensor(KP_m), to_tensor(Bdm)
        self.KI_g,self.KP_g,self.KI_dc,self.KP_dc = to_tensor(KI_g),to_tensor(KP_g),to_tensor(KI_dc),to_tensor(KP_dc)

        self.vg = to_tensor(self.vg)
        #Matrix for disturbance

        self.state_dim = 4
        self.in_dim = 1

        self.d_coef = torch.zeros(3,3)
        self.dw_coef = 1/M
        self.d_coef[1:3,1:3] = -self.Lg_inv


        #self.u_init = torch.full((1, int(self.x0.shape[1])),float(0)).to(device) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)


    def PI_m(self,w_e: torch.Tensor,w_v: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """

            PI_m = -F.linear(w_e,self.KP_m) - F.linear(w_v,self.KI_m)       # one is transient an other is SS
            
            
            return PI_m
    
    def PI_g(self,ig_e: torch.Tensor,ig_v: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """


            PI_g = ig_e*self.KP_g + ig_v*self.KI_g       # one is transient an other is SS
            
            
            return PI_g
    
    def PI_dc(self,v_dc_e: torch.Tensor,v_dc_v: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """

            PI_dc = -v_dc_e*self.KP_dc - v_dc_v*self.KI_dc       # one is transient an other is SS
            """print("Prop")
            print(-v_dc_e*self.KP_dc)
            
            print("inte")
            print(-v_dc_v*self.KI_dc)"""

            return PI_dc
    
    def compute_i_ref(self,P_ref: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """



            Qref_tensor = torch.full(P_ref.shape,self.Qref)
            PQ_ref = torch.cat((P_ref,Qref_tensor),2)

            i_ref = F.linear(PQ_ref,self.v_mat)
            
            
            return i_ref
    
    def compute_d_mech(self,x: torch.Tensor,dw):
        x = x.view(-1, 1, 2)

     
        w = x[:,0:1,0:1]
        w_v = x[:,0:1,1:2]
        w_e = w-self.wref

        tau_m = self.PI_m(w_e,w_v) 

        d_mech = tau_m*w


        w_ = F.linear(w,self.Am) + F.linear(tau_m,self.Bm) - dw
        w_v_ = w_v+self.h*w_e

        return d_mech,w_,w_v_
    
    def noiseless_forward(self, x: torch.Tensor,u: torch.Tensor,d_mech: torch.Tensor, no_PB=False):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state without the noise.
        """

        x = x.view(-1, 1, 3*2)
     
        #w = x[:,0:1,0:1]
        #w_v = x[:,0:1,self.state_dim:self.state_dim+1]

        v_dc = x[:,0:1,0:1]
        v_dc_v = x[:,0:1,self.state_dim-1:self.state_dim-1+1]

        ig = x[:,0:1,1:3]
        ig_v = x[:,0:1,self.state_dim-1+1:self.state_dim-1+3]

 


        

        

        v_dc_e = v_dc - self.vref
        i_err = self.PI_dc(v_dc_e,v_dc_v)

        #Mechanical disturbance effect
        i_ff = d_mech/v_dc


        i_dc = i_err+i_ff

        P_ref = i_dc*self.vref

        i_ref = self.compute_i_ref(P_ref)



        v_ff = self.vg - F.linear(i_ref,self.Z)
   
        i_e = ig-i_ref

        us = self.PI_g(i_e,ig_v) + v_ff

        mg = us/v_dc



        """print("us")
        print(us)
        
        print("vff")
        print(v_ff)

        print("irefZ")
        print(F.linear(i_ref,self.Z))


        print("PIg")
        print(self.PI_g(i_e,ig_v))


        print("Vx")
        print(mg*v_dc)"""

        #Saturate u
        #u_cont = torch.where(u_cont<0,0,torch.where(u_cont>self.umax,self.umax,u_cont))

        #w_ = F.linear(w,self.Am) + F.linear(tau_m,self.Bm)

        v_dc_ = (1-self.h*self.G/self.C)*v_dc - self.h/self.C*i_ff+ self.h/self.C*torch.bmm(mg,ig.transpose(1, 2))

        ig_ = F.linear(ig,torch.eye(2)-self.h*self.Lg_inv@self.Z) - self.h*F.linear(mg*v_dc,self.Lg_inv)



        #w_v_ = w_v+self.h*w_e

        v_dc_v_ = v_dc_v + self.h*v_dc_e

        ig_v_ = ig_v + self.h*i_e


        #Save base controller input
        self.u_cont = v_ff

        f = torch.cat((v_dc_,ig_,v_dc_v_,ig_v_),2)
        #f = torch.cat((w_,w_v_),2)
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
        d_x = d.view(-1, 1, self.state_dim)
        
        d_w = self.h*self.dw_coef*d_x[:,:,0:1]


        d_eta = self.h*F.linear(d_x[:,:,1:],self.d_coef)
        d_v = torch.zeros(d_eta.shape)
        d = torch.cat((d_eta,d_v),2)


        d_mech,w_,w_v_ = self.compute_d_mech(x[:,:,6:8],d_w) 


        eta_ = self.noiseless_forward(x[:,:,0:6],u,d_mech,no_PB) - d
        self.u_cont = d_mech
        return  torch.cat((eta_,w_,w_v_),2)


    
    def rollout(self,data: torch.Tensor, no_PB = False):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """


        #controller.reset()



        x = data[:,0:1,:]
        v = torch.zeros(data[:,0:1,:].shape)
        
        xs = torch.cat((x[:,0:1,1:4],v[:,0:1,1:4],x[:,0:1,0:1],v[:,0:1,0:1]),2)
        #u_PB = controller.forward(xs[:, 0:1, :])
        u_PB = torch.zeros(1,1,1)
        us = torch.full(u_PB.shape,0).to(device)

        #us = torch.full(dxref.shape,self.u_cont.item()).to(device)
        #us = to_tensor(np.array([0])).to(device)
        for t in range(1, data.shape[1]):
            #print(t)
            xs = torch.cat(
                (
                    xs,
                    self.forward(xs[:, t-1:t, :],u_PB[:, t-1:t, :],data[:, t:t+1, :],no_PB)
                    ),
                1
            )

            u_PB = torch.cat(
                (u_PB, torch.zeros(1,1,1)),
                1
            )

            us = torch.cat(
                (us, self.u_cont),
                1
            )


        #dxref = dxref     ##
        return xs, us,u_PB
        
    


wref = 80
vref = 5000
Qref = 8e+6
h = 1e-4

M = 23e3
D = 1e-4
C = 1e-2
G = 1e-5

l = 2 * 1.567e-3
r = 2e-3
w = 2 * np.pi * 50

conv = Converter(wref,vref,Qref,h,M,D,C,G,l,r,w)
data = torch.zeros(1,30000,4)
data[:,0,0] = 80
data[:,0,1] = 5000
data[:,0,2] = 0
data[:,0,3] = 0

data[:,1:,0] = 1e+5
data[:,1:,2] = 3000


xs,us,u_PB = conv.rollout(data)

plt.figure()
plt.plot(xs[0,:,6], label = "W")
plt.plot([80]*xs.shape[1],"--", label = "W ref")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()
plt.title("Evolution of W")



plt.figure()
plt.plot(xs[0,:,1], label = "Ig D")
plt.plot(xs[0,:,2], label = "Ig Q",color = "lightseagreen")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()
plt.title("Evolution of Ig")


plt.figure()
plt.plot(xs[0,:,0], label = "V DC")
#plt.plot([5000]*xs.shape[1],"--", label = "V DC ref")
plt.xlabel("Time")
plt.ylabel("State")
plt.legend()
plt.title("Evolution of V DC")


plt.show()
