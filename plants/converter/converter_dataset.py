import torch
import numpy as np
import random
from plants import CostumDataset
import matplotlib.pyplot as plt

class ConverterDataset(CostumDataset):
    def __init__(self, random_seed, horizon, h, phase_loss=2):
        # experiment and file names
        exp_name = 'Converter'
        file_name = 'data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'
        self.h = h
        self.phase_loss = phase_loss

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

    def _generate_data(self, num_samples):

        # generate data
        n_data_total = num_samples
        n_states = 4
        n_w = n_states
        #Double state dim: First for l2 dist (init) then for const dist like vg and tau_l
        d = torch.zeros(n_data_total,self.horizon,n_w*2) 
        e = torch.zeros(n_data_total,self.horizon,2)
        d_norm = torch.zeros(n_data_total,self.horizon,1)

        h = self.h
        V_base = 5000
        P_base = 8e+6
        w_base = 125.66

        I_base = P_base/V_base

        tau_base = P_base/w_base



        data_x0 = torch.tensor([1.0000e+0*w_base,1.0000e+00*V_base,0,  0])
        #data_x0 = torch.tensor([0,0,0,  0])
        d[:, 0, :n_w] = data_x0
        d[:,1:,n_w] = -2e+4

        ###Add the vg data###
        vg = 3150  # vg in pu
        w = 2*np.pi*50

        # Parameters
        batch_size, T = n_data_total, self.horizon

        offset = 0.2 + torch.rand(d.shape[0])*0.8
        offset[:10] = 0.2
        """for t in range(d.shape[1]):
            theta = w * t * h
            
            # base cosines (batch-independent)
            cos_a = np.cos(theta)
            cos_b = np.cos(theta - 2*np.pi/3)
            cos_c = np.cos(theta + 2*np.pi/3)

            # expand to batch
            cos_a = torch.full((d.shape[0],), cos_a)
            cos_b = torch.full((d.shape[0],), cos_b)
            cos_c = torch.full((d.shape[0],), cos_c)

            # random offset per batch
            if 600 < t < 1000:

                #offset[0:10] = 0
                #offset[-1] = 0.5
                va = offset * vg * cos_a
                vb = offset * vg * cos_b
                vc = vg * cos_c
            else:
                va = vg * cos_a
                vb = vg * cos_b
                vc = vg * cos_c"""



        # Random offset for each batch
        offset = 0.2 + torch.rand(batch_size) * 0.7
        offset[:10] = 0.2  # force first profiles to min

        # Create base cosines (T, 3)
        t = torch.arange(T) * h
        theta = w * t
        cos = torch.stack([
            torch.cos(theta),
            torch.cos(theta - 2*np.pi/3),
            torch.cos(theta + 2*np.pi/3)
        ], dim=1)  # shape (T, 3)

        # Expand to (batch, T, 3)
        cos_batch = cos.unsqueeze(0).repeat(batch_size, 1, 1)

        # ---- Scaling factors ----
        # Each batch needs [1, offset, offset] but shuffled
        scales = []
        for i in range(batch_size):
            if self.phase_loss == 1:
                base = torch.tensor([1.0, 1.0, offset[i].item()])
            elif self.phase_loss == 2:
                base = torch.tensor([offset[i].item(), offset[i].item(),1])
            elif self.phase_loss == 3:
                base = torch.tensor([offset[i].item(), offset[i].item(), offset[i].item()])
            #scales.append(base[torch.randperm(3)])  # shuffle per batch
        
            scales.append(base)  # shuffle per batch
        scales = torch.stack(scales, dim=0)  # (batch, 3)
        # Make it time dependent (apply only between 600–1000)
        scale_time = torch.ones((T, 3))
        scale_time[600:1000] = 0.0  # "indicator" for drop window
        # Broadcast to (batch, T, 3)
        scale_time = scale_time.unsqueeze(0).repeat(batch_size, 1, 1)

        # Final scaling: 1 outside drop window, shuffled [1, offset, offset] inside
        scale_batch = (1 - scale_time) * scales.unsqueeze(1) + scale_time * 1.0


        # Apply scaling
        amps = vg * cos_batch * scale_batch  # (batch, T, 3)
        va_all, vb_all, vc_all = amps[:,:,0], amps[:,:,1], amps[:,:,2]


        # Clarke transform (batch-wise)
        v_alpha = (2/3) * (va_all - 0.5*vb_all - 0.5*vc_all)
        v_beta  = (2/3) * ((np.sqrt(3)/2)*vb_all - (np.sqrt(3)/2)*vc_all)

        d[:, :, n_w+2] = v_alpha
        d[:, :, n_w+3] = v_beta
        return d
