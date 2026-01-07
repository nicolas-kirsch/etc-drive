import torch
import torch.nn as nn
import numpy as np
import time
from config import device
from .contractive_ren import ContractiveREN, REN
from .MLP import MLP, RNNModel
from assistive_functions import to_tensor
import torch.nn.functional as F
from collections import OrderedDict


class PerfBoostController(nn.Module):
    """
    Performance boosting controller, following the paper:
        "Learning to Boost the Performance of Stable Nonlinear Systems".
    Implements a state-feedback controller with stability guarantees.
    NOTE: When used in closed-loop, the controller input is the measured state
        of the plant and the controller output is the input to the plant.
        This controller has a memory for the last input ("self.last_input") and
        the last output ("self.last_output").
    """
    def __init__(
        self, noiseless_forward, input_init: torch.Tensor, output_init: torch.Tensor,d_init: torch.Tensor,
        # acyclic REN properties
        dim_internal: int, dim_nl: int, vdc_ref: int,Q_ref: int,batch_size: int,
        initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0,
        ren_internal_state_init=None,
        # misc
        output_amplification: float=20,
    ):
        """
         Args:
            noiseless_forward: system dynamics without process noise. can be TV.
            input_init (torch.Tensor): initial input to the controller.
            output_init (torch.Tensor): initial output from the controller before anything is calculated.
            output_amplification (float): TODO
            * the following are the same as AcyclicREN args:
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block of REN.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
            ren_internal_state_init (torch.Tensor, optional): initial state of the REN. Defaults to 0 when None.
        """
        super().__init__()

        self.output_amplification = output_amplification
        # set initial conditions
        self.input_init = input_init
        self.output_init = output_init.reshape(1, -1,2)
        self.d_init = d_init
        self.vg_init = torch.zeros_like(self.d_init)
        self.v_beta_init = torch.zeros((batch_size,1,1)).to(device)
        print("PB")
        print(self.v_beta_init.shape)
        print(self.output_init.shape)   

        # set dimensions (keep only x and not v for dim_in)
        self.dim_in = self.input_init.shape[-1]
        self.dim_out = self.output_init.shape[-1]
        self.dim_out = 2
        dim_in_ren = self.dim_in+3+1+2 # +1 for vdc_ref
        self.Vbase = 5000
        self.wbase = 126.66
        self.ibase = 2222.2
        self.vdc_ref = vdc_ref
        self.Qref = Q_ref
        self.batch_size = batch_size
        self.vmat =(torch.tensor([0.6,0]).float()@torch.tensor([[0,-1],[1,0]]).float()).float().to(device)

        self.base = torch.tensor([self.Vbase,self.ibase,self.ibase]).float().to(device)
        self.base_w = torch.tensor([self.Vbase,self.ibase,self.ibase,
                               self.Vbase,self.ibase,self.ibase,self.wbase,self.wbase]).float().to(device)

        # define the REN
        self.c_ren = ContractiveREN(
            dim_in=dim_in_ren, dim_out=self.dim_out, dim_internal=dim_internal,
            dim_nl=dim_nl, initialization_std=initialization_std,
            internal_state_init=ren_internal_state_init,
            posdef_tol=posdef_tol, contraction_rate_lb=contraction_rate_lb
        ).to(device)

        self.h = 2.5e-4
        self.period_step = int(1/(50*self.h))
        
        self.last_dq = torch.zeros((2,1,2)).to(device)
        self.last_dq[:,:,0] = 3150
        self.last_dq[:,:,1] = 0
        self.v_norm_nom = 3150

        self.last_disturbance_t_init = torch.full((self.batch_size,1,1),-1000)

        self.threshold = 9e-3

        self.MLP = MLP(dim_out = self.dim_out)
        """self.RNN = RNNModel(input_dim=9, hidden_dim=10, output_dim=self.dim_out)
        #self.c_ren = REN(self.dim_in,self.dim_out,dim_internal,dim_nl,gamma=300)

        # Use default config if none provided, and customize based on parameters
        self.config = SSMConfig()

        self.emme = DeepSSM(self.dim_in, self.dim_out, self.config).to(device)"""

        # define the system dynamics without process noise
        self.noiseless_forward = noiseless_forward

        self.reset()

    def reset(self):
        """
        set time to 0 and reset to initial state.
        """
        self.t = 0  # time
        self.last_input = self.input_init.detach().clone()
        self.last_output = self.output_init.detach().clone()
        self.last_d = self.d_init.detach().clone()
        self.last_vg = self.vg_init.detach().clone()
        self.c_ren.x = self.c_ren.init_x 
        self.last_disturbance_t = self.last_disturbance_t_init   # reset the REN state to the initial value
        self.v_beta_prev = self.v_beta_init.detach().clone()
        self.v_alpha_prev = torch.full((self.batch_size,1,1),3150).to(device)
        self.sign_change_happened = torch.zeros((self.batch_size,1,1),device=device).bool()
        #self.emme.reset()

    def forward(self, input_t: torch.Tensor,d: torch.Tensor,vg:torch.Tensor,init = False, no_PB = False):
        """
        Forward pass of the controller.

        Args:
            input_t (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).
            NOTE: when used in closed-loop, "input_t" is the measured states.

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        
        if init:
            w_ = input_t
        else:

            # apply noiseless forward to get noise less input (noise less state of the plant)
            u_noiseless = self.noiseless_forward(
                x=self.last_input,  # last input to the controller is the last state of the plant
                u_PB=self.last_output,
                d = self.last_d,
                in_controller = True,
                no_PB = no_PB
                  # last output of the controller is the last input to the plant
            )  # shape = (self.batch_size, 1, self.dim_in)
            
            # reconstruct the noise
            w_ = input_t - u_noiseless # shape = (self.batch_size, 1, self.dim_in)
            #w_[:,:,0:1] = w_[:,:,0:1]/self.wbase
            #w_[:,:,1:4] = w_[:,:,1:4]/self.Vbase  # shift the first element to be the error from the reference
        # reconstruct the noise
        theta = 2*np.pi*50*self.t*self.h

        v_norm = torch.norm(vg[:,:,0:2], p=2, dim=-1)
        v_norm = v_norm.unsqueeze(-1)    # shape: (N, M, 1)

        v_norm_dist = v_norm - self.v_norm_nom
        v_norm_dist = torch.where(torch.norm(v_norm_dist)>self.threshold, v_norm, torch.zeros_like(v_norm_dist))/self.Vbase
        v_norm_dist = v_norm_dist

        vg_pu = vg / self.Vbase

        v_beta = vg[:,:,1:2]
        v_alpha = vg[:,:,0:1]
        is_disturbance = torch.zeros((vg.shape[0],1,1),device=device)
        is_ldisturbance = torch.zeros((vg.shape[0],1,1),device=device)

        """if torch.norm(v_dq-self.last_dq) > 1e-3:
            is_disturbance += 1
            self.last_dq = v_dq"""
        


                # Check for new disturbance
        in_disturbance = (torch.norm(v_norm - self.v_norm_nom) > self.threshold)
  
        self.last_disturbance_t = torch.where(in_disturbance,torch.full_like(self.last_disturbance_t,int(self.t)),self.last_disturbance_t)
        self.sign_change_happened = torch.where(in_disturbance,torch.zeros_like(self.sign_change_happened).bool(),self.sign_change_happened)
        
        
        # Stay in disturbance mode if last disturbance within past 50 steps

        mask_within_period = (self.t - self.last_disturbance_t <= self.period_step)
        current_step = torch.full_like(self.last_disturbance_t,int(self.t))
        #sign_change = (current_step%40-1==0) 
        #sign_change = (self.v_beta_prev * v_beta <= 0) & () & (~self.sign_change_happened)
        sign_change =(self.v_beta_prev * v_beta <= 0) & (current_step>self.last_disturbance_t) & (~self.sign_change_happened)  & (mask_within_period)

        self.sign_change_happened = torch.where(sign_change,torch.ones_like(self.sign_change_happened).bool(),self.sign_change_happened)
        

        
        # logical update: still disturbance if within period and no zero-crossing yet
        still_active = mask_within_period & (~self.sign_change_happened)
        #still_active = mask_within_period
        """if still_active.any():
            print("Sign change detected at time step:", self.t)"""
           

            
        is_disturbance = torch.where(still_active,torch.ones_like(is_disturbance),is_disturbance)

        """if self.t - self.last_disturbance_t <= 20:
            is_disturbance += 1"""
        w_ = w_/ self.base_w.view(1, 1, -1)

        pu_vals = input_t[:,:,0:3] / self.base.view(1, 1, -1)
        dist_presence = torch.zeros_like(input_t[:,:,0:3])  # shape = (self.batch_size, 1, 1)
        
        dist_presence = torch.clone(pu_vals)*is_disturbance
        vg_pu = vg_pu*is_disturbance
        #Get the current error
        error_vdc = (input_t[:,:,0:1] - self.vdc_ref)  # shape = (self.batch_size, 1, 1)
        
        ren_input = torch.cat((w_,dist_presence,v_norm_dist,vg_pu), dim=2)


        mlp_input = torch.cat((w_[:,:,0:4],d,input_t), dim=2)
        mlp_input = mlp_input.view(input_t.shape[0],1, -1)
   
        # apply REN on disturbance
        output_REN = self.c_ren.forward(ren_input)

        #output_REN = output_REN[:,:,0:1]




        


        # apply MLP on reference plus disturbance 
        output_MLP = self.MLP.forward(mlp_input)
        #output_RNN = self.RNN.forward(mlp_input)
        
        #output = output_MLP*output_REN*self.output_amplification   # shape = (self.batch_size, 1, self.dim_out)
        #output = output[:,:,0:1]
        #output = output.reshape(self.batch_size,1,-1)
        # update internal states
        output = output_REN*output_MLP  # shape = (self.batch_size, 1, self.dim_out)

        #time.sleep(1)
        #output = torch.clamp(output, min=-2, max=2)  # Clamp output between -2 and 2
        self.last_input, self.last_output,self.last_d = input_t, output,d
        self.v_beta_prev = v_beta
        self.v_alpha_prev = v_alpha
        self.t += 1

        #print(output)
        #time.sleep(1)
        return output

    # setters and getters
    def get_parameter_shapes(self):
        return self.c_ren.get_parameter_shapes()

    def get_named_parameters(self):
        return self.c_ren.get_named_parameters()

    def get_parameters_as_vector(self):
        # TODO: implement without numpy
        return np.concatenate([p.detach().clone().cpu().numpy().flatten() for p in self.c_ren.parameters()])

    def get_MLP_parameters(self):
        # TODO: implement without numpy
        params =  self.MLP.state_dict()
        return params
    
    def set_MLP_parameters(self, param_dict):
        self.MLP.load_state_dict(param_dict)


    def set_parameter(self, name, value):
        current_val = getattr(self.c_ren, name)
        with torch.no_grad():
            current_val.copy_(to_tensor(value.reshape(current_val.shape)))
        # Do NOT rewrap with nn.Parameter
        # Do NOT reassign via setattr
        self.c_ren._update_model_param()  # if this recomputes dependent stuff

    def set_parameters(self, param_dict):
        for name, value in param_dict.items():

            self.set_parameter(name, value)

    def set_parameters_as_vector(self, value):
        idx = 0
        for name, shape in self.get_parameter_shapes().items():
            if len(shape) == 1:
                dim = shape
            elif len(shape) == 2:
                dim = shape[0]*shape[1]
            else:
                raise NotImplementedError
            idx_next = idx + dim
            # select indx
            if len(value.shape) == 1:
                value_tmp = value[idx:idx_next]
            elif len(value.shape) == 2:
                value_tmp = value[:, idx:idx_next]
            else:
                raise AssertionError
            # set
            with torch.no_grad():

                self.set_parameter(name, value_tmp.reshape(shape))
            idx = idx_next
        assert idx_next == value.shape[-1]

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

