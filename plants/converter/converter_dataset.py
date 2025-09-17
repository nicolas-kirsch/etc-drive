import torch
import numpy as np
import random,time
from plants import CostumDataset
import matplotlib.pyplot as plt

class ConverterDataset(CostumDataset):
    def __init__(self, random_seed, horizon, h):
        # experiment and file names
        exp_name = 'Converter'
        file_name = 'data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'
        self.h = h

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

        """data_x0 = torch.tensor([1.0000e+0,1.0000e+00,0,  0])

        d[:, 0, :n_w] = data_x0
        d[:,1:,n_w] = 1e+4/tau_base

        ###Add the vg data###
        vg = 2900/V_base  # vg in pu
        w = 2*np.pi*50
        
        d[:,1:,n_w + 2] = vg

        print(int(np.floor(num_samples/2)))
        for t in range(1,d.shape[1]):
    
            d[:,t,2] = vg*(np.cos(w*t*h))
            d[:,t,3] = vg*(np.sin(w*t*h))
        """

        data_x0 = torch.tensor([1.0000e+0*w_base,1.0000e+00*V_base,0,  0])
        #data_x0 = torch.tensor([0,0,0,  0])
        d[:, 0, :n_w] = data_x0
        d[:,1:,n_w] = -2e+4

        ###Add the vg data###
        vg = 3150  # vg in pu
        w = 2*np.pi*50
        

        print(int(np.floor(num_samples/2)))
        offset = 0.2 + torch.rand(d.shape[0])*0.6
        offset[:10] = 0.2

        tri_offset = torch.ones((d.shape[0],3))
        tri_offset[:,2] = offset
        tri_offset[:,1] = offset

        # shuffle columns independently per row
        for i in range(d.shape[0]):
            tri_offset[i,:] = tri_offset[i, torch.randperm(3)]

        va_full = torch.zeros((d.shape[0],d.shape[1],1))
        vb_full = torch.zeros((d.shape[0],d.shape[1],1))
        vc_full = torch.zeros((d.shape[0],d.shape[1],1))

        for t in range(d.shape[1]):
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
                va = tri_offset[:,0] * vg * cos_a
                vb = tri_offset[:,1] * vg * cos_b
                vc = tri_offset[:,2] * vg * cos_c
            else:
                va = vg * cos_a
                vb = vg * cos_b
                vc = vg * cos_c

            # Clarke transform (batch-wise)
            v_alpha = (2/3) * (va - 0.5*vb - 0.5*vc)
            v_beta  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc)

            va_full[:,t,0] = va
            vb_full[:,t,0] = vb
            vc_full[:,t,0] = vc

            d[:, t, n_w+2] = v_alpha
            d[:, t, n_w+3] = v_beta


        #d = d[torch.randperm(d.size(0))]

        return d

