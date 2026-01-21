import os,sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(1, BASE_DIR)


from assistive_functions import to_tensor
from config import device
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt



class ConverterAbs(torch.nn.Module):
    def __init__(self,x0,wref,vref,Qref,h, M, D,C,G,lg,z,l,base_values,u_init = None):

        super().__init__()

        # Parameters
        self.h = h

        self.M = M
        self.D = D

        self.C = C
        self.G = G



        self.Lg = lg

        self.Z = z



        self.Lg_inv = np.linalg.inv(self.Lg)



        self.d_mech = torch.tensor([[8e+6]])/base_values["P_base"]



        self.x0 = x0



        self.base_values = base_values

        self.igmax = 2800/base_values["I_base"]

        self.tau_max = 1e+5/base_values["tau_base"]

        self.wmax = 80/base_values["w_base"]







        # Exteral values

        self.vg = np.array([[3000, 0]])/base_values["V_base"]

        self.d_mech = torch.tensor([[8e+6]])/base_values["P_base"]




        self.mg_max = 1/np.sqrt(2)


        Am = np.array([[(1-self.h*D/M)]])
        Bm = np.array([[self.h/M]])
        Bdm = np.array([[self.h/M],[0]])



        omega_m = (2 * np.pi * 5.55) / 6
        zeta_m = 1
        
        # Gain calculations
        KI_m = np.array([[omega_m**2 * M]])
        KP_m = np.array([[2 * zeta_m * omega_m * M]])

        # Additional parameters for electrical system
        omega_d = 2 * np.pi * 140
        zeta_d = 1
                
        # Gain calculations for electrical system
        KI_g = omega_d**2 * l
        KP_g = 2 * zeta_d * omega_d * l
    
         

        # Additional parameters for DC system
        omega_dc = 2 * np.pi * 2
        zeta_dc = 1
        
        
        # Gain calculations for DC system
        KI_dc = omega_dc**2 * C
        KP_dc = 2 * zeta_dc * omega_dc * C


        self.v_mat = to_tensor(np.linalg.inv(np.array([[self.vg[0,0], self.vg[0,1]], 
                               [self.vg[0,1], -self.vg[0,0]]])))
        
        
        #Reference values
        self.wref = to_tensor(wref)
        self.Qref = to_tensor(Qref)
        self.original_Qref = self.Qref

        self.vref = to_tensor(vref)

        self.Lg_inv = to_tensor(self.Lg_inv)
        self.Z = to_tensor(self.Z)

        self.Am, self.Bm, self.KI_m,self.KP_m, self.Bdm = to_tensor(Am),to_tensor(Bm),to_tensor(KI_m),to_tensor(KP_m), to_tensor(Bdm)
        self.KI_g,self.KP_g,self.KI_dc,self.KP_dc = to_tensor(KI_g),to_tensor(KP_g),to_tensor(KI_dc),to_tensor(KP_dc)

        self.vg = to_tensor(self.vg)
        self.original_vg = self.vg.clone()

        #Matrix for disturbance

        self.state_dim = 4
        self.in_dim = 1

        self.d_coef = torch.zeros(3,3)
        self.dw_coef = 1/M
        self.d_coef[1:3,1:3] = -self.Lg_inv


        self.u_init = torch.full((1, int(self.x0.shape[1])),float(0)).to(device) if u_init is None else u_init.reshape(1, -1)   # shape = (1, in_dim)


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


            PI_g = ig_e*self.KP_g + ig_v*self.KI_g       
            
            
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

            PI_dc = -v_dc_e*self.KP_dc - v_dc_v*self.KI_dc      


            return PI_dc
    
    def compute_i_ref(self,P_ref: torch.Tensor,u_PB:torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """

            
            # Compute the 2-norm along the last dimension
            vg_norm = torch.linalg.norm(self.vg, ord=2, dim=-1, keepdim=True)  # Shape: (1, 1)
            
            mask_Q = self.Qref**2<vg_norm**2*self.igmax**2-P_ref**2

            self.Qref = torch.where(mask_Q,self.Qref,torch.sqrt(torch.maximum(vg_norm**2*self.igmax**2-P_ref**2,torch.zeros(P_ref.shape).to(device))))

            Q_ref_PB = self.Qref+u_PB

            PQ_ref = torch.cat((P_ref,Q_ref_PB),2)

            i_ref = F.linear(PQ_ref,self.v_mat)

            # Compute the 2-norm along the last dimension
            i_norm = torch.linalg.norm(i_ref, ord=2, dim=-1, keepdim=True)  # Shape: (b, 1, 1)


            # Identify vectors with norm > 0.7
            mask = i_norm > self.igmax

            # Scale vectors where norm > 0.7
            scaled_tensor = i_ref * (self.igmax / i_norm)

            # Replace only the vectors exceeding the threshold
            i_ref = torch.where(mask, scaled_tensor, i_ref)
            return i_ref
    
    def compute_d_mech(self,x: torch.Tensor,dw):
        x = x.view(-1, 1, 2)

     
        w = x[:,0:1,0:1]
        w_v = x[:,0:1,1:2]
        w_e = w-self.wref

        tau_m = self.PI_m(w_e,w_v) 
        


        tau_m = torch.clamp(tau_m, -self.tau_max, self.tau_max)

        self.tau_m = tau_m
        d_mech = tau_m*w


        w_ = F.linear(w,self.Am) + F.linear(tau_m,self.Bm) - dw
        w_v_ = w_v+self.h*w_e

        return d_mech,w_,w_v_
    
    def noiseless_forward(self, x: torch.Tensor,u_PB: torch.Tensor,d_mech: torch.Tensor,vg:torch.Tensor, no_PB=False):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state without the noise.
        """

        x = x.view(-1, 1, 3*2)
        u_PB = u_PB.view(-1, 1, 2)
        d_Q = u_PB[:,:,0:1]
        d_i = u_PB[:,:,1:2]
        #w = x[:,0:1,0:1]
        #w_v = x[:,0:1,self.state_dim:self.state_dim+1]

        v_dc = x[:,0:1,0:1]
        v_dc_v = x[:,0:1,self.state_dim-1:self.state_dim-1+1]

        ig = x[:,0:1,1:3]
        ig_v = x[:,0:1,self.state_dim-1+1:self.state_dim-1+3]

        

        v_dc_e = v_dc - (self.vref)
        i_err = self.PI_dc(v_dc_e,v_dc_v)

        #Mechanical disturbance effect
        i_ff = d_mech/v_dc


        i_dc = i_err+i_ff

        P_ref = i_dc*self.vref
        vg_norm = torch.linalg.norm(self.vg, ord=2, dim=-1, keepdim=True)  # Shape: (1, 1)
        P_max = torch.minimum(vg_norm*self.igmax,torch.tensor(self.tau_max*self.wmax))[0,0].item()
        

        P_ref = torch.clamp(P_ref, -P_max, P_max)

        i_ref = self.compute_i_ref(P_ref,d_Q)



        v_ff = vg - F.linear(i_ref,self.Z)
   
        i_e = ig-(i_ref)

        us = self.PI_g(i_e,ig_v) + v_ff

        mg = us/v_dc

        # Compute the 2-norm along the last dimension
        norm = torch.linalg.norm(mg, ord=2, dim=-1, keepdim=True)  # Shape: (b, 1, 1)

        # Define the threshold
        threshold = 0.7

        # Identify vectors with norm > 0.7
        mask = norm > threshold

        # Scale vectors where norm > 0.7
        scaled_tensor = mg * (threshold / norm)

        # Replace only the vectors exceeding the threshold
        mg = torch.where(mask, scaled_tensor, mg)


        v_dc_ = (1-self.h*self.G/self.C)*v_dc - self.h/self.C*i_ff+ self.h/self.C*torch.bmm(mg,ig.transpose(1, 2))

        ig_ = F.linear(ig,torch.eye(2)-self.h*self.Lg_inv@self.Z) - self.h*F.linear(mg*v_dc,self.Lg_inv) + self.h*F.linear(self.vg,self.Lg_inv)


        #w_v_ = w_v+self.h*w_e

        v_dc_v_ = v_dc_v + self.h*v_dc_e

        ig_v_ = ig_v + self.h*i_e


        #Save base controller input
        self.u_cont = mg

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
        eta_ = self.noiseless_forward(x[:,:,0:6],u,d_mech,self.vg,no_PB) - d
        self.d_mech = d_mech
        return  torch.cat((eta_,w_,w_v_),2)


    
    def rollout(self,controller,data: torch.Tensor, no_PB = False):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """


        controller.reset()

        self.Qref = self.original_Qref 


        x = data[:,0:1,:]
        v = torch.zeros(data[:,0:1,:].shape)

        v[:,0:1,:] = to_tensor(np.array([-9.8744e-02/self.base_values["w_base"],
                                          -1.0916e+00/self.base_values["V_base"],
                                          5.2917e-06/self.base_values["I_base"]
                                           ,4.6380e-07/self.base_values["I_base"]]))
        
        xs = torch.cat((x[:,0:1,1:4],v[:,0:1,1:4],x[:,0:1,0:1],v[:,0:1,0:1]),2)

        d_mech = torch.zeros(xs.shape[0],1,1)
        
        self.Q_ref_e = torch.zeros(xs.shape[0],1,1)
        vg = self.vg.detach().clone()
        u_PB = controller.forward(xs[:, 0:1, :6],d_mech[:, 0:1, :],vg,init = True)
        u_cont = torch.zeros(xs.shape[0],1,2)
        for t in range(1, data.shape[1]):

            if t>=1 and t<2000:
                w = 2*np.pi*50
            
                self.vg[:,0] = np.sqrt(2/3)*self.original_vg[0,0]*(1-(1/2)*np.cos(2*w*t*self.h))
                self.vg[:,1] = np.sqrt(2/3)*self.original_vg[0,0]*(1/2)*np.sin(2*w*t*self.h)
            else:
                self.vg = self.original_vg.clone()
            if t < 3:
                print(xs[0,t-1,:])
            xs = torch.cat(
                (
                    xs,
                    self.forward(xs[:, t-1:t, :],u_PB[:, t-1:t, :],data[:, t:t+1, :],no_PB)
                    ),
                1
            )

            
            d_mech = torch.cat(
                (d_mech, self.d_mech),
                1
            )
            
            self.Q_ref_e = torch.cat(
                (self.Q_ref_e, self.Qref),
                1
            )

            vg = torch.cat(
                (vg, self.vg),
                0
            )


            u_PB = torch.cat(
                (u_PB, controller.forward(xs[:, t:t+1, :6],d_mech[:, t:t+1, :],vg[t:t+1,:])),
                1
            )
            
            u_cont = torch.cat(
                (u_cont, self.u_cont),
                1
            )

             

            


        controller.reset()

        #dxref = dxref     ##
        return xs, u_cont,u_PB,d_mech
        
    


"""wref = 80
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
data = torch.zeros(1,3000,4)
data[:,0,0] = 80
data[:,0,1] = 5000
data[:,0,2] = 0
data[:,0,3] = 0

data[:,1:,0] = 1e+5
data[:,1:,2] = 0


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
"""