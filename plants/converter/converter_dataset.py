import torch
import numpy as np
import random,time
from plants import CostumDataset
import matplotlib.pyplot as plt

class ConverterDataset(CostumDataset):
    def __init__(self, random_seed, horizon, h,n_phases_lost=2):
        # experiment and file names
        exp_name = 'Converter'
        file_name = 'data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'
        self.h = h
        self.n_phases_lost = n_phases_lost
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
        #offset[:20] = 0.2

        data_third = int(np.floor(d.shape[0]/3))
        tri_offset = torch.ones((d.shape[0],3))

        if self.n_phases_lost == 1:
            tri_offset[:,0] = offset
        elif self.n_phases_lost == 2:
            print("Quaqua")
            tri_offset[:,2] = offset
            tri_offset[:,1] = offset
        elif self.n_phases_lost == 3:
            tri_offset[:,2] = offset
            tri_offset[:,1] = offset
            tri_offset[:,0] = offset
        else: 
            print("Here here ")
            tri_offset[:data_third,2] = offset[:data_third]
            tri_offset[:data_third,1] = offset[:data_third]
            tri_offset[:data_third,0] = offset[:data_third]

            tri_offset[data_third:data_third*2,1] = offset[data_third:data_third*2]
            tri_offset[data_third:data_third*2,0] = offset[data_third:data_third*2]

            tri_offset[data_third*2:,0] = offset[data_third*2:]




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


        # Create a figure with a 2x2 grid of subplots
        """       fig, ax = plt.subplots(9, 1, figsize=(13, 9))

        ax[0].plot(va_full[0])
        ax[0].plot(vb_full[0])
        ax[0].plot(vc_full[0])

        ax[1].plot(va_full[3])
        ax[1].plot(vb_full[3])
        ax[1].plot(vc_full[3])

        ax[2].plot(va_full[15])
        ax[2].plot(vb_full[15])
        ax[2].plot(vc_full[15])

        ax[3].plot(va_full[33])
        ax[3].plot(vb_full[33])
        ax[3].plot(vc_full[33])

        ax[4].plot(va_full[38])
        ax[4].plot(vb_full[38])
        ax[4].plot(vc_full[38])

        ax[5].plot(va_full[25])
        ax[5].plot(vb_full[25])
        ax[5].plot(vc_full[25])

        ax[6].plot(va_full[44])
        ax[6].plot(vb_full[44])
        ax[6].plot(vc_full[44])

        ax[7].plot(va_full[55])
        ax[7].plot(vb_full[55])
        ax[7].plot(vc_full[55])

        ax[8].plot(va_full[59])
        ax[8].plot(vb_full[59])
        ax[8].plot(vc_full[59])"""


        plt.show()
        #d = d[torch.randperm(d.size(0))]

        return d

