import torch
import random
from plants import CostumDataset


class MechanicalDataset(CostumDataset):
    def __init__(self, random_seed, horizon):
        # experiment and file names
        exp_name = 'mechanical'
        file_name = 'data_T'+str(horizon)+'_RS'+str(random_seed)+'.pkl'

        super().__init__(random_seed=random_seed, horizon=horizon, exp_name=exp_name, file_name=file_name)

    def _generate_data(self, num_samples):
        # generate data
        n_data_total = num_samples
        n_states = 1
        n_w = n_states

        d = torch.zeros(n_data_total,self.horizon,n_w) 

        data_x0 = (20 + 80*torch.rand(n_data_total, n_states))

        for i in range(n_data_total):
            d[i][0] = data_x0[i]

        return d
