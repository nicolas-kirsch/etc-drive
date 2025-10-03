import torch, os, pickle
import numpy as np
from torch.utils.data import Dataset
from config import BASE_DIR
from assistive_functions import to_tensor


class ClassifierDataset(Dataset):
    """
    Dataset for converter fault classification.
    Generates:
        - d       : (num_samples, horizon, features)   input signals
        - y_cat   : (num_samples, horizon, n_classes) one-hot fault class
        - y_scale : (num_samples, horizon, 3)         scale regression labels
    """
    def __init__(self, random_seed, horizon, h, n_phases_lost=2):
        self.random_seed = random_seed
        self.horizon = horizon
        self.h = h
        self.n_phases_lost = n_phases_lost
        self._data = None
        torch.manual_seed(self.random_seed)

        # file name and path
        exp_name = "Converter"
        file_name = f"data_T{horizon}_RS{random_seed}.pkl"
        file_path = os.path.join(BASE_DIR, 'experiments', exp_name, 'saved_results')
        if not os.path.exists(file_path):
            os.makedirs(file_path)
        self.file_name = os.path.join(file_path, file_name)

    # ------------ label helper ------------
    def _fault_to_class(self, mask):
        a, b, c = mask
        code = (1 if a else 0) + (2 if b else 0) + (4 if c else 0)
        mapping = {
            0: 0,  # NoFault
            1: 1,  # A
            2: 2,  # B
            3: 4,  # AB
            4: 3,  # C
            5: 6,  # AC
            6: 5,  # BC
            7: 7   # ABC
        }
        return mapping[code]

    # ------------ main data generator ------------
    def _generate_data(self, num_samples):
        n_states = 4
        d = torch.zeros(num_samples, self.horizon, n_states * 2)
        e = torch.zeros(num_samples, self.horizon, 2)
        d_norm = torch.zeros(num_samples, self.horizon, 1)

        V_base = 5000
        P_base = 8e6
        w_base = 125.66
        I_base = P_base / V_base
        tau_base = P_base / w_base

        # init state
        data_x0 = torch.tensor([1.0000 * w_base, 1.0000 * V_base, 0, 0])
        d[:, 0, :n_states] = data_x0
        d[:, 1:, n_states] = -2e4

        # grid params
        vg = 3150
        w = 2 * np.pi * 50

        # fault offsets
        offset = 0.2 + torch.rand(num_samples) * 0.6
        offset = torch.full_like(offset,0.2)
        data_third = int(np.floor(num_samples / 3))
        tri_offset = torch.ones((num_samples, 3))

        
        if self.n_phases_lost == 1:
            tri_offset[:, 0] = offset
        elif self.n_phases_lost == 2:
            tri_offset[:, 1] = offset
            tri_offset[:, 2] = offset
        elif self.n_phases_lost == 3:
            tri_offset[:, :] = offset[:, None]
        else:
            tri_offset[:data_third, :] = offset[:data_third, None]
            tri_offset[data_third:data_third * 2, :2] = offset[data_third:data_third * 2, None]
            tri_offset[data_third * 2:, 0] = offset[data_third * 2:]

        # shuffle per row
        for i in range(num_samples):
            tri_offset[i, :] = tri_offset[i, torch.randperm(3)]

        va_full = torch.zeros((num_samples, self.horizon, 1))
        vb_full = torch.zeros((num_samples, self.horizon, 1))
        vc_full = torch.zeros((num_samples, self.horizon, 1))

        # labels
        n_classes = 8
        y_cat = torch.zeros(num_samples, self.horizon, n_classes)
        #y_scale = torch.ones(num_samples, self.horizon, 3)

        data_sev = int(np.floor(d.shape[0]/7))

        for t in range(self.horizon):
            theta = w * t * self.h
            
            # Three-phase voltages (remove phase a → set v_a = 0)
            va = vg * np.cos(theta)
            vb = vg * np.cos(theta - 2*np.pi/3)
            vc = vg * np.cos(theta + 2*np.pi/3)

            if t  > 200 and t < 600:
                """va = 0.3 * vg * np.cos(theta)
                vb = 0.3 * vg * np.cos(theta - 2*np.pi/3)
                vc = vg * np.cos(theta + 2*np.pi/3)"""
                va_loss = offset * va
                vb_loss = offset * vb
                vc_loss = offset * vc
            elif t > 40000 and t < 5500:
                """va = 0.6 * vg * np.cos(theta)
                vb = 0.6 * vg * np.cos(theta - 2*np.pi/3)
                vc = vg * np.cos(theta + 2*np.pi/3)"""
                va_loss = offset * va
                vb_loss = offset * vb
                vc_loss = offset * vc
            else: 
                va_loss =  torch.full_like(offset,va)
                vb_loss =  torch.full_like(offset,vb)
                vc_loss =  torch.full_like(offset,vc)

            va =  torch.full_like(offset,va)
            vb =  torch.full_like(offset,vb)
            vc =  torch.full_like(offset,vc)

            v_alpha = (2/3) * (va - 0.5*vb - 0.5*vc)
            v_beta  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc)

            v_alpha_ab = (2/3) * (va_loss - 0.5*vb_loss - 0.5*vc)
            v_beta_ab  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc)

            v_alpha_bc = (2/3) * (va - 0.5*vb_loss - 0.5*vc_loss)
            v_beta_bc  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc_loss)

            v_alpha_ac = (2/3) * (va_loss - 0.5*vb - 0.5*vc_loss)
            v_beta_ac  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)


            v_alpha_a = (2/3) * (va_loss - 0.5*vb - 0.5*vc)
            v_beta_a  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc)

            v_alpha_b = (2/3) * (va - 0.5*vb_loss - 0.5*vc)
            v_beta_b  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc)

            v_alpha_c = (2/3) * (va - 0.5*vb - 0.5*vc_loss)
            v_beta_c  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)

            v_alpha_c = (2/3) * (va - 0.5*vb - 0.5*vc_loss)
            v_beta_c  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc_loss)

            v_alpha_abc = (2/3) * (va_loss - 0.5*vb_loss - 0.5*vc_loss)
            v_beta_abc  = (2/3) * ((np.sqrt(3)/2)*vb_loss - (np.sqrt(3)/2)*vc_loss)


            d[:data_sev,t,4+2] = v_alpha_ab[:data_sev]
            d[:data_sev,t,4+3] = v_beta_ab[:data_sev]

            d[data_sev:data_sev*2,t,4+2] = v_alpha_bc[data_sev:data_sev*2]
            d[data_sev:data_sev*2,t,4+3] = v_beta_bc[data_sev:data_sev*2]

            d[data_sev*2:data_sev*3,t,4+2] = v_alpha_ac[data_sev*2:data_sev*3]
            d[data_sev*2:data_sev*3,t,4+3] = v_beta_ac[data_sev*2:data_sev*3]

            d[data_sev*3:data_sev*4,t,4+2] = v_alpha_a[data_sev*3:data_sev*4]
            d[data_sev*3:data_sev*4,t,4+3] = v_beta_a[data_sev*3:data_sev*4]

            d[data_sev*4:data_sev*5,t,4+2] = v_alpha_b[data_sev*4:data_sev*5]
            d[data_sev*4:data_sev*5,t,4+3] = v_beta_b[data_sev*4:data_sev*5]

            d[data_sev*5:data_sev*6,t,4+2] = v_alpha_c[data_sev*5:data_sev*6]
            d[data_sev*5:data_sev*6,t,4+3] = v_beta_c[data_sev*5:data_sev*6]

            d[data_sev*6:,t,4+2] = v_alpha_abc[data_sev*6:]
            d[data_sev*6:,t,4+3] = v_beta_abc[data_sev*6:]
        y_cat[:, :200, 0] = 1
        y_cat[:, 600:, 0] = 1

        y_cat[:data_sev, 200:600, 1] = 1
        y_cat[data_sev:data_sev*2, 200:600, 2] = 1
        y_cat[data_sev*2:data_sev*3, 200:600, 3] = 1
        y_cat[data_sev*3:data_sev*4, 200:600, 4] = 1
        y_cat[data_sev*4:data_sev*5, 200:600, 5] = 1
        y_cat[data_sev*5:data_sev*6, 200:600, 6] = 1
        y_cat[data_sev*6:, 200:600, 7] = 1
        return d, y_cat
