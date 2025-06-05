import torch
import numpy as np
import random
from plants import CostumDataset


class ConverterDataset(CostumDataset):
    def __init__(self, random_seed, horizon):
        # experiment and file names
        exp_name = 'Converter'
        file_name = 'data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

    def _generate_data(self, num_samples):
        # generate data
        n_data_total = num_samples
        n_states = 4
        n_w = n_states

        d = torch.zeros(n_data_total,self.horizon,n_w) 

        h = 1e-4
        V_base = 5000
        P_base = 8e+6
        w_base = 80

        I_base = P_base/V_base

        tau_base = P_base/w_base

        data_x0 = torch.tensor([80/w_base,5000/V_base,2.1803e+03/I_base,-1.3628e+03/I_base])

        d[:, 0, :] = data_x0
        d[:,1:,0] = 3e+4/tau_base

        ###Add the vg data###
        vg = 3000/V_base  # vg in pu
        w = 2*np.pi*50
        
        d[:,1:,2] = vg

        print(int(np.floor(num_samples/2)))
        for t in range(1,2000):
            d[:int(np.floor(num_samples/2)),t,2] = np.sqrt(2/3)*vg*(1-(1/2)*np.cos(2*w*t*h))
            d[:int(np.floor(num_samples/2)),t,3] = np.sqrt(2/3)*vg*(1/2)*np.sin(2*w*t*h)
            
            d[int(np.floor(num_samples/2)):,t,2] = 3300/V_base
            d[int(np.floor(num_samples/2)):,t,3] = 0
            
        d = d[torch.randperm(d.size(0))]
        return d

