import os,sys, time
import scipy.io
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sys.path.insert(1, BASE_DIR)


from assistive_functions import to_tensor
from config import device
import torch.nn.functional as F
import numpy as np
import torch
import matplotlib.pyplot as plt
import cvxpy as cp


class Converter(torch.nn.Module):
    def __init__(self,x0,wref,vref,Qref,h, M, D,C,G,lg,z,l,base_values,Ared, Bred, Cred,Ered,u_init = None):
        super().__init__()
        self.test = False
        # Parameters
        self.h = h
        self.M = M
        self.D = D
        Jtot = 23364

        self.Ared = Ared
        self.Bred = Bred
        self.Ered = Ered
        self.Cred = Cred
        #self.Cred = torch.eye(self.Cred.shape[-1])

        self.C = C
        self.G = G


        self.Lg = lg
        self.Z = z
        self.l = l

        l1 = 3.5e-04
        l2 = 5.4154e-04
        l = l1+l2
        
        self.Lg_inv = np.linalg.inv(self.Lg)
        print(torch.eye(2)-self.h*self.Lg_inv@self.Z)

        self.x0 = x0

        """self.d_mech = torch.tensor([[8e+6]])/base_values["P_base"]


        self.P_base = base_values["P_base"]
        self.tau_base = base_values["tau_base"]

        self.base_values = base_values
        self.igmax = 2222/base_values["I_base"]
        self.tau_max = 46691/base_values["tau_base"]
        self.wmax = 125.66/base_values["w_base"]

        # Exteral values
        self.vg = np.array([[3150, 0]])/base_values["V_base"]
        """

        self.d_mech = torch.tensor([[8e+6]])
        self.d = torch.zeros((1,1,4))  # shape = (batch_size, 1, state_dim*2)

        self.igmax = 2222
        self.tau_max = 46691
        self.wmax = 125.66
        
        self.kappa2 = 400
        self.kappa1 = 0.1

        # Exteral values
        self.vg = np.array([[3150, 0]])

        
        self.mg_max = 1/np.sqrt(2)


        Am = np.array([[(1-self.h*D/Jtot)]])
        Bm = np.array([[self.h/Jtot]])
        Bdm = np.array([[self.h/Jtot],[0]])



        #omega_m = (2 * np.pi * 5.55) / 6
        omega_m = 0.9968
        zeta_m = 1.4
        
        # Gain calculations
        KI_m = np.array([[omega_m**2 * Jtot/(1.5**2)]])
        KP_m = np.array([[2*0.8/1.5 * zeta_m * omega_m * Jtot]])
        print("KI_m",KI_m)
        print("KP_m",KP_m)
        # Additional parameters for electrical system
        #omega_d = 2 * np.pi * 140
        omega_d = 816.8141
        zeta_d = 1
                
        # Gain calculations for electrical system
        KI_g = - omega_d**2 * l
        KP_g = -2 * zeta_d * omega_d * l
    
        self.Gamma1 = 0
        self.Gamm2 = 0

        # Additional parameters for DC system
        #omega_dc = 2 * np.pi * 2
        omega_dc = 17.4432
        zeta_dc = 1.6180
        
        
        # Gain calculations for DC system
        KI_dc = -omega_dc**2 * C
        KP_dc = -2 * zeta_dc * omega_dc * C


        self.v_mat = to_tensor(np.linalg.inv(np.array([[self.vg[0,0], self.vg[0,1]], 
                               [self.vg[0,1], -self.vg[0,0]]]))).to(device)
        
        
        #Reference values
        self.wref = to_tensor(wref)
        self.Qext = to_tensor(Qref)
        print("eeee")
        print(self.Qext)
        self.original_Qref = self.Qext

        self.vref = to_tensor(np.array([vref])).to(device)

        self.Lg_inv = to_tensor(self.Lg_inv).to(device)
        self.Z = to_tensor(self.Z).to(device)

        self.Am, self.Bm, self.KI_m,self.KP_m, self.Bdm = to_tensor(Am),to_tensor(Bm),to_tensor(KI_m),to_tensor(KP_m), to_tensor(Bdm)
        self.KI_g,self.KP_g,self.KI_dc,self.KP_dc = to_tensor(KI_g),to_tensor(KP_g),to_tensor(KI_dc),to_tensor(KP_dc)

        """        self.vg = to_tensor(self.vg)
        self.vg = torch.zeros(1,1,2)
        self.vg[:,:,0] = 3150/base_values["V_base"]
        self.vg[:,:,1] = 0
        self.original_vg = self.vg.clone()

        self.vg_nom = 3150/base_values["V_base"]  # Nominal grid voltage in pu


        print(2*self.vg_nom*self.igmax*base_values["P_base"])"""

        self.vg = to_tensor(self.vg)
        self.vg = torch.zeros(1,1,2)
        self.vg[:,:,0] = 3150
        self.vg[:,:,1] = 0
        self.original_vg = self.vg.clone()

        self.vg_nom = 3150 # Nominal grid voltage in pu
        self.vdc_nom = 5000

        self.sat_integrator = 2*self.vdc_nom/(omega_d**2*(l)*np.sqrt(2))  # Saturation limit for the integrator state

        print(self.sat_integrator)
        #Matrix for disturbance

        self.state_dim = 4
        self.in_dim = 1

        self.d_coef = torch.zeros(3,3)
        self.dw_coef = 1/M
        self.d_coef[1:3,1:3] = -self.Lg_inv

        self.J = to_tensor(np.array([[0, -1], [1, 0]]))
        self.J_inv = to_tensor(np.array([[1, 0], [0, -1]]))
        
        self.u_init = torch.full((1, int(self.x0.shape[1]),2),float(0)).to(device) if u_init is None else u_init.reshape(1, -1,2)   # shape = (1, in_dim)

        #Integrator state
        self.ig_v = torch.zeros((1, 1, 2)).to(device)  # shape = (batch_size, 1, 2)
        self.idc = 0
        self.w_v = 0

        #LPFs
        self.lpf_iff = 0
        self.lpf_vdcerr = 0

        self.lpf_iff_c = 0
        self.lpf_vdcerr_c = 0

        self.w_aug = torch.zeros((1, 1, 2)).to(device)  # shape = (batch_size, 1, state_dim*2)
        self.w_aug[:,:,1:] = 125.66/self.Cred[0,0,0]
        self.w_aug_c = torch.zeros((1, 1, 2)).to(device)
        self.w_aug_c[:,:,1:] = 125.66/self.Cred[0,0,0]

    def PI_m(self,w_e: torch.Tensor, w_v: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """

            w_v = w_v + self.h*w_e
            w_v = torch.clamp(w_v, min=-self.tau_max, max=self.tau_max)  # Clamp the integrator state
            PI_m = -F.linear(w_e,self.KP_m) - F.linear(w_v,self.KI_m)       # one is transient an other is SS
            
            
            return PI_m, w_v
    
    def PI_g(self,ig_e: torch.Tensor,ig_v,v_ff: torch.Tensor, vg_angle: torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """
            ig_e_dq = self.to_dq(vg_angle,ig_e)

            
            
            ig_v_ab = ig_v + self.h*ig_e_dq  # Update the integrator state in alpha-beta coordinates
             
            """ig_v = self.to_ab(vg_angle, ig_v)
            ig_v = torch.clamp(ig_v, min=-self.sat_integrator, max=self.sat_integrator)  # Clamp the integrator state
            PI_g = -ig_e*self.KP_g - ig_v*self.KI_g       
            us = PI_g + v_ff"""

            ig_v = self.to_ab(vg_angle, ig_v)
            
            ig_v_ab = torch.clamp(ig_v_ab, min=-self.sat_integrator, max=self.sat_integrator)  # Clamp the integrator state
            PI_g = -ig_e*self.KP_g - ig_v*self.KI_g       
            us = PI_g + v_ff

            return us,ig_v_ab
    
    def PI_dc(self,v_dc_e: torch.Tensor,idc,iff: torch.Tensor, idc_max: torch.Tensor,in_controller=False):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """
            if in_controller: 
                vdce_lpf, lpf_vdcerr = self.low_pass_filter(v_dc_e, self.lpf_vdcerr_c, 45)
                iff_lpf, lpf_iff = self.low_pass_filter(iff, self.lpf_iff_c, 450)


                PI_dc = vdce_lpf*self.KP_dc + self.lpf_vdcerr_c*self.KI_dc 

            else: 
                vdce_lpf, lpf_vdcerr = self.low_pass_filter(v_dc_e, self.lpf_vdcerr, 45)
                iff_lpf, lpf_iff = self.low_pass_filter(iff, self.lpf_iff, 450)


                PI_dc = vdce_lpf*self.KP_dc + self.lpf_vdcerr*self.KI_dc 

            i_dc = PI_dc + iff_lpf

            # Clamp the output to the maximum DC current
            
            idc = idc + i_dc * self.h  # Update the integrator state
            

            idc_max = torch.clamp(idc_max, min=torch.zeros_like(i_dc), max=torch.full_like(idc_max, 1.2*self.igmax*self.vg_nom/self.vref.item()))  # Ensure idc_max is not greater than the maximum DC current
            idc = torch.clamp(idc, min=-idc_max, max=idc_max)
            P_dc = idc * self.vref  # Calculate the DC power 
            if in_controller:
                self.lpf_vdcerr_c = lpf_vdcerr
                self.lpf_iff_c = lpf_iff  
            else:
                self.lpf_vdcerr = lpf_vdcerr  # Update the low-pass filter state
                self.lpf_iff = lpf_iff
            return P_dc,idc

    def to_dq(self, vg_angle: torch.Tensor, signal: torch.Tensor):
            """
            Convert a signal from ab to dq coordinates.

            Args:
                - vg (torch.Tensor): Grid voltage at t. shape = (batch_size, 1, 2) serves as reference for angles
                - signal (torch.Tensor): alpha beta signal at t. shape = (batch_size, 1, 2)

            Returns:
                dq coordinates of the signal.
            """

            #Get cos and sin theta from vg
              # Shape: (batch_size, 1, 2)
            
            #Clarke tranformation
            vg_angle_tensor = torch.zeros(vg_angle.shape[0], 1, 2, 2).to(device)
            vg_angle_tensor[:, 0, 0, 0] = vg_angle[:, 0, 0]
            vg_angle_tensor[:, 0, 1, 0] = vg_angle[:, 0, 1]
            vg_angle_tensor[:, 0, 0, 1] = -vg_angle[:, 0, 1]  # Using the J matrix for transformation
            vg_angle_tensor[:, 0, 1, 1] = vg_angle[:, 0, 0]  # Using the J matrix for transformationx

            vg_angle_tensor = vg_angle_tensor.squeeze(1)  # Remove the singleton dimension

            # Convert signal to dq coordinates
            out_dq = torch.matmul(signal, vg_angle_tensor)
             # Assuming similar operation for current

            return out_dq
    
    def to_ab(self, vg_angle: torch.Tensor, signal: torch.Tensor):
            """
            Convert a signal from dq to ab coordinates.

            Args:
                - vg (torch.Tensor): Grid voltage at t. shape = (batch_size, 1, 2) serves as reference for angles
                - signal (torch.Tensor): dqa signal at t. shape = (batch_size, 1, 2)

            Returns:
                alpha beta coordinates of the signal.
            """
            vg_angle = vg_angle.clone()  # Clone to avoid modifying the original tensor
            vg_angle[:,:,1] = -vg_angle[:,:,1]  # Negate the second column to match the ab coordinates

            out_ab = self.to_dq(vg_angle, signal)  # Use the to_dq method for transformation
            
             # Assuming similar operation for current

            return out_ab
        
    def gen_igs_G1(self, vg, P_ref, Q_ref):
        vg_norm = torch.linalg.norm(vg, ord=2, dim=-1, keepdim=True)**2  # Shape: (batch_size, 1, 1)
        vg_norm_safe = torch.maximum(vg_norm, torch.full_like(vg_norm, 1))  # Avoid division by zero


        PQ_ref = torch.cat((P_ref, Q_ref), 2)
        a = PQ_ref.clone().roll(-1, 2)
        a[:,:,0:1] = -a[:,:,0:1]

        i_ref_presat = torch.cat((
            torch.matmul(PQ_ref, vg.transpose(1, 2)),
            torch.matmul(a, vg.transpose(1, 2))
        ), dim=2)  # Shape: (batch_size, 1, 2)
        
        i_ref_presat = (i_ref_presat / vg_norm_safe)# Normalize by the grid voltage norm


        i_ref = torch.clamp(i_ref_presat, min=-8*self.igmax, max=8*self.igmax)
        igs = i_ref * (self.igmax / torch.linalg.norm(i_ref, ord=2, dim=-1, keepdim=True)).clamp(max=1)

        i_norm = torch.linalg.norm(i_ref_presat, ord=2, dim=-1, keepdim=True)
        i_norm_sat = torch.minimum(i_norm, torch.full_like(i_norm, self.igmax))
        Gamma_1 = i_norm - i_norm_sat
        return igs, Gamma_1

    def gen_mg_G2(self,ms: torch.Tensor):

        ms_norm = torch.linalg.norm(ms, ord=2, dim=-1, keepdim=True)  # Shape: (batch_size, 1, 1)

        ms_lin_norm_sat = torch.minimum(ms_norm, torch.full_like(ms_norm, self.mg_max).to(device))
        ms_norm_sat = 1/25*torch.log(1/(torch.exp(-25*ms_norm)+torch.exp(-25*torch.tensor([self.mg_max*0.99]))))  # Smooth saturation function

        Gamma_2 = ms_norm - ms_norm_sat

        # Scale the vectors where norm > 0.7
        #mg = ms * (self.mg_max / ms_norm) 
        atan = torch.atan2(ms[:,:,1], ms[:,:,0]).reshape(-1, 1, 1)  # Get the angle in radians
        mg = torch.cat((torch.cos(atan), torch.sin(atan)),-1) * ms_lin_norm_sat


        return mg, Gamma_2
    


    def low_pass_filter(self, x: torch.Tensor, y: torch.Tensor, cutoff_freq):
        
        y_dot = (x-y)*2*np.pi*cutoff_freq
        y = y + self.h*y_dot

        return y_dot, y
    

    
    def update_qs(self, Q_ext, Gamma_1, Gamma_2, Qmm):

        Q_ext_clamped = torch.clamp(Q_ext, min=torch.full_like(Q_ext,-self.igmax*self.vg_nom), max=torch.full_like(Q_ext,self.igmax*self.vg_nom))

        Qmm_sat = torch.clamp(Qmm, min=-2*self.igmax*self.vg_nom, max=2*self.igmax*self.vg_nom)

        self.Qs = self.Qs + self.h*(-(self.Qs-Q_ext_clamped)*2*np.pi*5 - self.Qs*Gamma_1*self.kappa1 - (self.Qs-Qmm_sat)*Gamma_2*self.kappa2)

    
    def compute_i_ref(self,P_ref: torch.Tensor,u_PB:torch.Tensor,vg:torch.Tensor):
            """
            output of the base controller

            Args:
                - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
                - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
                - Kr (torch.Tensor): Gain of the base controller.

            Returns:
                next state without the noise.
            """
 

            """vg_norm = torch.linalg.norm(vg, ord=2, dim=-1, keepdim=True)  # Shape: (1, 1)


            mask_Q = self.Qref**2<vg_norm**2*self.igmax**2-P_ref**2
            self.Qref = torch.where(mask_Q,self.Qref,torch.sqrt(torch.maximum(vg_norm**2*self.igmax**2-P_ref**2,torch.zeros(P_ref.shape).to(device))))
            Q_ref_PB = self.Qref"""
            Q_ref_PB = self.Qs.clone().to(device)
            P_ref = P_ref.to(device)
            PQ_ref = torch.cat((P_ref,Q_ref_PB),2)

            i_ref_presat = F.linear(PQ_ref,self.v_mat)

            """# Compute the 2-norm along the last dimension
            i_norm = torch.linalg.norm(i_ref, ord=2, dim=-1, keepdim=True)  # Shape: (b, 1, 1)

            # Identify vectors with norm > 0.7
            mask = i_norm > self.igmax
            # Scale vectors where norm > 0.7
            scaled_tensor = i_ref * (self.igmax / i_norm)
            # Replace only the vectors exceeding the threshold
            i_ref = torch.where(mask, scaled_tensor, i_ref)"""

            i_ref = i_ref_presat * (self.igmax / torch.linalg.norm(i_ref_presat, ord=2, dim=-1, keepdim=True)).clamp(max=1)




            return i_ref, i_ref_presat
    
    def update_Q(self,mg,i_ref,vg):
        vg_norm = torch.linalg.norm(vg, ord=2, dim=-1, keepdim=True)  # Shape: (1, 1)
        Z_norm = torch.linalg.norm(self.Z, ord=2)
        Qmm = 2*np.pi*50*self.l*vg_norm**2/Z_norm**2

        mg_norm = torch.linalg.norm(mg, ord=2, dim=-1, keepdim=True)
        mg_max_tensor = torch.full_like(mg_norm, self.mg_max)
        Gamma2 = mg_norm - torch.minimum(mg_norm, mg_max_tensor)


        i_norm = torch.linalg.norm(i_ref, ord=2, dim=-1, keepdim=True)
        ig_max_tensor = torch.full_like(i_norm, self.igmax)
        Gamma1 = i_norm - torch.minimum(i_norm, ig_max_tensor)

        new_Qs  =  self.Qs + self.h*(-2*np.pi*5*(self.Qs-self.Qref) - self.kappa1*Gamma1*self.Qs - self.kappa2*Gamma2*(self.Qs-Qmm))


        return new_Qs

    def compute_tau_m(self, w:torch.Tensor,w_v, P_max: torch.Tensor):
        tau_max = P_max/torch.abs(w)

        w_e = w - self.wref 
        tau_m,w_v = self.PI_m(w_e,w_v) 
        tau_max = torch.minimum(tau_max, torch.full_like(tau_max, self.tau_max*1.05))

        tau_m = torch.clamp(tau_m, -tau_max, tau_max)
        return tau_m, w_v
    
    def mech_dynamics(self,w_aug,tau_m,tau_l):
        
        mech_input =  torch.matmul(tau_l, self.Ered) + torch.matmul(tau_m, self.Bred) 
        #print(mech_imput)
        
        """print("w_aug")
        print(w_aug)"""
        mech_input2 = torch.matmul(w_aug, self.Ared.T)# Compute the mechanical input
        """print("mech_input2")
        print(mech_input2)"""
        w_aug_ = w_aug + self.h * (mech_input+mech_input2)  # Update the augmented state with the mechanical input

        xv = torch.matmul(w_aug_, self.Cred)  # Extract the mechanical speed from the augmented state
        w = xv[:,:,1:2]  # Extract the mechanical speed from the augmented state
        return w_aug_, w
    
    def compute_P_mech(self,x: torch.Tensor,tau_l: torch.Tensor, P_max: torch.Tensor, in_controller=False):
        x = x.view(-1, 1, 2)

     
        w = x[:,0:1,0:1]
        w_v = x[:,0:1,1:2]
        tau_m, w_v = self.compute_tau_m(w, w_v,P_max)  # Compute the mechanical disturbance and dynamics
        self.tau_m = tau_m
        P_mech = tau_m*w
        if in_controller:
            self.w_aug_c, w_ = self.mech_dynamics(self.w_aug_c,tau_m,tau_l)  # Update the augmented state with the mechanical input
        else:
            self.w_aug, w_ = self.mech_dynamics(self.w_aug,tau_m,tau_l)  # Update the augmented state with the mechanical input
        #w_ = F.linear(w,self.Am) + F.linear(tau_m,self.Bm) + F.linear(tau_l,self.Bm)


        return P_mech,w_,w_v
    
    def gen_mg(self,v_dc,idc,ig,ig_v,P_mech,vg: torch.Tensor, d_mg: torch.Tensor=0,in_controller=False, no_PB=False):

        vg_norm = torch.linalg.norm(vg, ord=2, dim=-1, keepdim=True)  # Shape: (1, 1)


        P_max = 6.6/7*vg_norm*self.igmax  # Maximum power limit based on grid voltage and current limits

        P_mech_sat = torch.clamp(P_mech, min=-P_max, max=P_max)
        iff = P_mech_sat / v_dc

        vdc_e = v_dc - self.vref  # Compute the DC voltage error
        idc_max = P_max/v_dc  # Maximum DC current limit based on power and voltage reference
        
        P_dc,idc_ = self.PI_dc(vdc_e,idc, iff, idc_max,in_controller=in_controller)  # Compute the DC power reference

        igs, Gamma1 = self.gen_igs_G1(vg, P_dc, self.Qs)

        ig_e = ig - igs  # Compute the error between actual and reference current
        vff = vg - F.linear(igs, self.Z)  # Feedforward voltage based on reference current

        vg_angle = vg/ vg_norm

        us,ig_v = self.PI_g(ig_e,ig_v, vff, vg_angle)  # Compute the control input based on current error and feedforward voltage
        print(us.device)
        if no_PB:
            mg_in = us / v_dc  # Convert control input to mechanical generator input
            
        else:
 
            mg_in = us / v_dc + d_mg

        print("v_dc",v_dc.device)
        print("mg_in",mg_in.device)

        #mg_in = us / v_dc 

        mg, Gamma2 = self.gen_mg_G2(mg_in)  # Generate mechanical generator input and compute saturation error

        """Qmm = 2*np.pi*50*self.l*vg_norm**2/torch.linalg.norm(self.Z, ord=2)**2

        Qext = torch.full_like(self.Qs, self.Qref)  # Reference reactive power"""
        #self.update_qs(Qext, Gamma1, Gamma2, Qmm)


        self.u_cont = mg

        return mg,ig_v,idc_
         
    def noiseless_forward(self, x: torch.Tensor,u_PB: torch.Tensor,d:torch.Tensor, no_PB=False,in_controller=False):
        """
        forward of the plant without the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)

        Returns:
            next state without the noise.
        """
        x = x.view(-1, 1, self.state_dim*2)
        #u_PB = u_PB.view(-1, 1, 2)
        d_mg = u_PB[:,:,0:2]
        #d_i = u_PB[:,:,1:2]
        #w = x[:,0:1,0:1]
        #w_v = x[:,0:1,self.state_dim:self.state_dim+1]
        v_dc = x[:,0:1,0:1]
        ig = x[:,0:1,1:3]
        idc = x[:,0:1,3:4]  
        ig_v = x[:,0:1,4:6]

        #Disturbances
        tau_l = d[:,0:1,0:1]  # Mechanical disturbance (torque)
        vg = d[:,0:1,2:4]  # Grid voltage disturbance (alpha-beta coordinates)


        vg_norm = torch.linalg.norm(vg, ord=2, dim=-1, keepdim=True)  # Shape: (1, 1)

        P_max = 6.6/7*vg_norm*self.igmax  # Maximum power limit based on grid voltage and current limits

        #Mechanical dynamics
        P_mech, w_, w_v_ = self.compute_P_mech(x[:,:,6:8],tau_l, P_max, in_controller)  # Compute mechanical disturbance and dynamics
        self.d_mech = P_mech.clone()
        P_mech_sat = torch.clamp(P_mech, min=-P_max, max=P_max)
        iff = P_mech_sat / v_dc
 
        mg,ig_v_,idc_ = self.gen_mg(v_dc,idc, ig,ig_v, P_mech, vg, d_mg, in_controller=in_controller, no_PB=no_PB) 

        #mg = vg/self.vdc_nom
        #mg = torch.zeros_like(mg)  # Set mechanical generator input to zero for testing purposes

        """P_mech_sat = torch.clamp(P_mech, min=-P_max, max=P_max)  # Saturate mechanical disturbance
          # Compute the feedforward current based on saturated mechanical disturbance


        vdc_e = v_dc - self.vref  # Compute the DC voltage error
        idc_max = P_max/v_dc  # Maximum DC current limit based on power and voltage reference
        
        P_dc = self.PI_dc(vdc_e, iff, idc_max)  # Compute the DC power reference
        igs, Gamma1 = self.gen_igs_G1(vg, P_dc, self.Qs)

        ig_e = ig - igs  # Compute the error between actual and reference current
        vff = vg - F.linear(igs, self.Z)  # Feedforward voltage based on reference current

        vg_angle = vg/ vg_norm
        
        us = self.PI_g(ig_e, vff, vg_angle)  # Compute the control input based on current error and feedforward voltage
        
        mg_in = us / v_dc  # Convert control input to mechanical generator input

        mg, Gamma2 = self.gen_mg_G2(mg_in)  # Generate mechanical generator input and compute saturation error

        Qmm = 2*np.pi*50*self.l*vg_norm**2/torch.linalg.norm(self.Z, ord=2)**2

        Qext = torch.full_like(self.Qs, self.Qref)  # Reference reactive power
        #self.update_qs(Qext, Gamma1, Gamma2, Qmm)
        """
        v_dc_ = (1-self.h*self.G/self.C)*v_dc - self.h/self.C*iff+ self.h/self.C*torch.bmm(mg,ig.transpose(1, 2))

        ig_ = F.linear(ig,torch.eye(2)-self.h*self.Lg_inv@self.Z) - self.h*F.linear(
            mg*v_dc,self.Lg_inv) + self.h*F.linear(vg,self.Lg_inv)

        #ig_ = ig_ = F.linear(ig,torch.eye(2)-self.h*self.Lg_inv@self.Z) + self.h*F.linear(vg,self.Lg_inv)


        self.u_cont = mg


        f = torch.cat((v_dc_,ig_,idc_,ig_v_,w_,w_v_),2)
        #f = torch.cat((w_,w_v_),2)
        return f
    
    def forward(self, x, u,w,d,no_PB):
        """
        forward of the plant with the process noise.

        Args:
            - x (torch.Tensor): plant's state at t. shape = (batch_size, 1, state_dim)
            - u (torch.Tensor): plant's input at t. shape = (batch_size, 1, in_dim)
            - w (torch.Tensor): process noise at t. shape = (batch_size, 1, state_dim)

        Returns:
            next state.
        """

        #Get the l2 (w) and non l2 (d) disturbances
        """dist = d.view(-1, 1, self.state_dim*2)
        w = dist[:,:,:self.state_dim]
        d = dist[:,:,self.state_dim:]"""

        v = torch.zeros_like(w).to(device)
        w = torch.cat((w[:,:,1:4],v[:,:,1:4],w[:,:,0:1],v[:,:,0:1]),2).to(device) # Add zero disturbance for the second half of the state vector


        #Take a step
        eta_ = self.noiseless_forward(x,u,d,no_PB) + w



        """d_x = d.view(-1, 1, self.state_dim)
        d_vg = d_x[:,:,2:4]

        d_w = self.h*self.dw_coef*d_x[:,:,0:1]
        d_x = torch.zeros_like(d_x)

        d_eta = self.h*F.linear(d_x[:,:,1:],self.d_coef)
        d_v = torch.zeros(d_eta.shape)

        d = torch.cat((d_eta,d_v),2)
        print(d)
        #eta_ = self.noiseless_forward(x[:,:,0:6],u,d_w,d_vg,no_PB) - d
        
        d_mech,w_,w_v_ = self.compute_d_mech(x[:,:,6:8],d_w) 
        eta_ = self.noiseless_forward(x[:,:,0:6],u,d_mech,d_vg,no_PB) - d
        self.d_mech = d_mech"""
        return  eta_


    
    def rollout(self,controller,data: torch.Tensor, no_PB = False):
        """
        rollout with state-feedback controller

        Args:
            - controller: state-feedback controller
            - data (torch.Tensor): batch of disturbance samples, with shape (batch_size, T, state_dim)
        """

        controller.reset()

        self.Qref = self.original_Qref 
        w = data[:,:,:self.state_dim]
        d = data[:,:,self.state_dim:]
        x = w[:,0:1,:].to(device)
        
        self.Qs = torch.full((x.shape[0],1,1),self.original_Qref).to(device)
        self.ig_v = torch.zeros((1, 1, 2)).to(device)  # shape = (batch_size, 1, 2)
        self.idc = 0
        self.w_v = 0

        #LPFs
        self.lpf_iff = 0
        self.lpf_vdcerr = 0

        #LPFs
        self.lpf_iff_c = 0
        self.lpf_vdcerr_c = 0

        self.w_aug = torch.zeros((1, 1, 2)).to(device)  # shape = (batch_size, 1, state_dim*2)
        self.w_aug[:,:,1:] = 125.66/self.Cred[0,0,0]
        self.w_aug_c = torch.zeros((1, 1, 2)).to(device)
        self.w_aug_c[:,:,1:] = 125.66/self.Cred[0,0,0]

        v = torch.zeros(x.shape).to(device)
        """v[:,0:1,:] = to_tensor(np.array([-2.0373e-04*125.6,
                                          -1.9188e-04*3150, -1.9305e-08*2222,1.6000e-09*2222]))"""
        xs = torch.cat((x[:,0:1,1:4],v[:,0:1,1:4],x[:,0:1,0:1],v[:,0:1,0:1]),2)
        
        init = torch.tensor([ 4.9999536133e+03,  8.2062780762e+02, -6.4584533691e+01,
         5.1859454346e+02,  1.6475370154e-02, -4.1930428147e-01,
         1.2552771759e+02, -1.5503498316e+00]).to(device)
        
        xs[:,0,:] = init
        self.lpf_iff = 517.1087646484
        self.lpf_idc = -0.0468370356

        self.lpf_iff_c = 517.108764648
        self.lpf_vdcerr_c = -0.0468370356
        
        d_mech = torch.zeros(xs.shape[0],1,1)
        
        self.Q_ref_e = torch.zeros(xs.shape[0],1,1)

        vg = self.vg.repeat(xs.shape[0],1,1)
        u_PB = controller.forward(xs[:, 0:1, :],d[:, 0:1, :],vg,init = True)
        #u_PB = torch.zeros(xs.shape[0],1,2)  # Set u_PB to zero for testing purposes
        uuu = u_PB.clone().detach()
        u_cont = torch.zeros(xs.shape[0],1,2)


        for t in range(1, data.shape[1]):
            """
                        if t < 4:
                print(xs[0,t-1,:])
                if t>=1 and t<2000:
                w = 2*np.pi*50
            
                self.vg[:,0] = np.sqrt(2/3)*self.original_vg[0,0]*(1-(1/2)*np.cos(2*w*t*self.h))
                self.vg[:,1] = np.sqrt(2/3)*self.original_vg[0,0]*(1/2)*np.sin(2*w*t*self.h)
            else:
                self.vg = self.original_vg.clone()
        
            """
            if t == 16000:
                print(xs[0,t-1,:])

                print(self.lpf_iff)
                print(self.lpf_vdcerr)


            xs = torch.cat(
                (
                    xs,
                    self.forward(xs[:, t-1:t, :],u_PB[:, t-1:t, :],w[:, t:t+1, :],d[:, t-1:t, :],no_PB)
                    ),
                1
            )


            d_mech = torch.cat(
                (d_mech, self.d_mech),
                1
            )
            
            

            vg = torch.cat(
                (vg, data[:,t:t+1,self.state_dim+2:self.state_dim+4]),
                1
            )

            u_PB = torch.cat(
                (u_PB, controller.forward(xs[:, t:t+1, :],d[:, t:t+1, :],data[:,t:t+1,self.state_dim+2:self.state_dim+4], no_PB=no_PB)),
                1
            )
            """u_PB = torch.cat(
                (u_PB, torch.zeros_like(uuu)),
                1
            )"""

            u_cont = torch.cat(
                (u_cont, self.u_cont),
                1
            )

             

            


        controller.reset()

        #dxref = dxref     ##
        return xs, u_cont,u_PB,d_mech
        
 
"""x_min = torch.Tensor([70]).to(device)
x0 = torch.Tensor([[5000],[1000],[1000],[1000],[1000],[1000]]).to(device)
xref = torch.Tensor([[80]]).to(device)

Wref = 125.66
Vref = 5000
Qref = 0 
h = 2.5e-4

yref = torch.Tensor([Vref,Qref]).to(device)

M = 4364.5
D = 1e-4
C = 0.0040
G = 1e-5
l1 = 3.5e-04
l2 = 5.4154e-04
l = l1+l2
l = 3.5897e-3
r = 0.08
r = 4.4797e-2
w = 2 * np.pi * 50

Lg = np.array([[l, 0], [0, l]])
Z = np.array([[r, 0], [0, r]])

# Load the .mat file
data = scipy.io.loadmat('gains_small.mat')

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
print(Cred)
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



conv = Converter(x0,wref,vref,qref,h,m,d,c,g,lg,z,l,base_values,Ared,Bred,Cred,Ered)

a = torch.zeros(1,1,2)
b = torch.zeros(1,1,2)
c = torch.zeros(1,1,2)

idc = torch.zeros(1,1,1)
ig_v = torch.zeros(1,1,2)

vdc = torch.zeros(1,1,1)
vdc[:,:,0] = 5000

ig = torch.zeros(1,1,2)

P_mech = torch.zeros(1,1,1)
P_mech[:,:,0] = 1e+6

vg = torch.zeros(1,1,2)
vg[:,:,0] = 3150

wref = torch.zeros(1,1,1)
wref[:,:,0] = 125.66

a[:,:,0] = 3000
b[:,:,0] = 1e+6
c[:,:,0] = 1

a[:,:,1] = 0
b[:,:,1] = 0
c[:,:,1] = -8
u = torch.zeros(1,1,1)
v = torch.zeros(1,1,2)
conv.Qs = torch.zeros(1,1,1)
x = torch.zeros(1,1,2)
w_v = torch.zeros(1,1,1)
w = torch.zeros(1,1,1)
tau_l = torch.zeros(1,1,1)
tau_m = torch.zeros(1,1,1)
P_max = torch.zeros(1,1,1)
w_aug = torch.zeros(1,1,2)
w_aug[:,:,1:] = 125.66/Cred[0,0,0]
print(w_aug[:,:,1:])
w[:,:,0] = 125
P_max[:,:,0] = 4e+6

tau_l[:,:,0] = -2e+4

tau_m[:,:,0] = 2e+4

x[:,:,0] = 125

for i in range(90000):
    #vdc[:,:,0] = 5000 + i*conv.h
    vg[:,:,0] = np.cos(2*np.pi*5*i*conv.h)*10000
    vg[:,:,1] = np.sin(2*np.pi*5*i*conv.h)*10000


    
    #us, ig_v = conv.PI_g(c,ig_v,a,vg_angle)
    #print(ig_v)
    #us,ig_v,idc = conv.gen_mg(vdc,idc,ig,ig_v,P_mech,vg, no_PB=True)

    #w_aug,w = conv.mech_dynamics(w_aug,tau_m,tau_l)  # Update the augmented state with the mechanical input
    #u = torch.cat((u,w),dim = 1)
    

    P_mech,w_,w_v = conv.compute_P_mech(x,tau_l,P_max)
    x = torch.cat((w_, w_v), dim=2)
    u = torch.cat((u,w_),dim = 1)



plt.figure()
plt.plot(np.array(range(u.shape[1]-1))*h,u[0,1:,0].detach().numpy(), label = r"$w$")
plt.xlabel("Time (s)")
plt.ylabel(r"$w$")
plt.ylim(124.6, 126.8)
plt.grid(True, which='both', axis='both', linestyle='--', linewidth=0.5)
plt.xticks(np.arange(0, u.shape[1]*h, 2))
plt.yticks(np.arange(124.6, 126.8, 0.2))
plt.legend()
plt.show()"""

