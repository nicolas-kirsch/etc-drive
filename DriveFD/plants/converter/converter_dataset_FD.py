import torch
import numpy as np
import random,time
from plants import CostumDataset
import matplotlib.pyplot as plt

class ConverterDatasetFD(CostumDataset):
    def __init__(self, random_seed, horizon, h,n_phases_lost=2):
        # experiment and file names
        exp_name = 'Converter_wFD'
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
        Tmax = 46691
        I_base = P_base/V_base

        tau_base = P_base/w_base

       
        data_x0 = torch.tensor([1.0000e+0*w_base,1.0000e+00*V_base,0,  0])
        #data_x0 = torch.tensor([0,0,0,  0])
        d[:, 0, :n_w] = data_x0
        d[:,1:,n_w] = -0.95*Tmax

        ###Add the vg data###
        vg = 3150  # vg in pu
        w = 2*np.pi*50
        

        offset = torch.rand(d.shape[0])*0.5 + 0.5  # random values in [0.5, 1.0]
        offset[:] = 0
        # Shuffle the offset tensor
        offset = offset[torch.randperm(offset.size(0))]

        data_third = int(np.floor(d.shape[0]/3))
        tri_offset = torch.ones((d.shape[0],3))

        if self.n_phases_lost == 1:
            tri_offset[:,1] = offset
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

        t_start = 300
        one_phase_in_timesteps = 80
        length_fault = 1860

        # Randomize fault start times per sample
        #delta_t_start = torch.randint(0, one_phase_in_timesteps, (d.shape[0],))
        fault_starts = torch.full((d.shape[0],), t_start) 

        delta_t_end = torch.randint(0, one_phase_in_timesteps, (d.shape[0],))
        #delta_t_end[:] = 10  # First 20 samples have fixed fault duration
        fault_ends = fault_starts + length_fault + delta_t_end
        #fault_ends[:20] = t_start + length_fault  # First 20 samples have fixed fault end time
        
        N, T = d.shape[0], d.shape[1]

        t_idx = torch.arange(T).unsqueeze(0)  # (1, T)

        in_fault = (t_idx >= fault_starts.unsqueeze(1)) & \
                (t_idx <  fault_ends.unsqueeze(1))   # (N, T)



        fault_type = torch.zeros((d.shape[0], d.shape[1], 1), dtype=torch.long)
        fault_scale = torch.zeros((d.shape[0], d.shape[1], 1))

        # True where phase is faulty
        phase_fault = tri_offset < 1.0        # (N, 3)
        fault_scale_val = (1-tri_offset).max(dim=1).values  # (N, )

        # index: 0=a, 1=b, 2=c
        fault_phase_idx = phase_fault.long().argmax(dim=1)  # (N,)

        # labels: 1,2,3
        fault_phase_label = fault_phase_idx + 1 
        #fault_type[in_fault, 0] = fault_phase_label[:, None].expand(-1, d.shape[1])[in_fault]
        fault_type[in_fault] = (
            fault_phase_label.unsqueeze(1)
            .expand(-1, T)[in_fault]
            .unsqueeze(-1)
        )

        fault_scale[in_fault] = (
            fault_scale_val.unsqueeze(1)
            .expand(-1, T)[in_fault]
            .unsqueeze(-1)
        )

        # time vector
        t = torch.arange(T, device=d.device, dtype=d.dtype) * h  # (T,)

        # angular frequency
        w = 2 * np.pi * 50

        # phase
        theta = w * t  # (T,)

        # sin / cos
        time_enc = torch.zeros((N, T, 2), device=d.device, dtype=d.dtype)
        time_enc[:, :, 0] = torch.sin(theta)
        time_enc[:, :, 1] = torch.cos(theta)


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

            # Determine which samples are currently in fault
            in_fault = (t >= fault_starts) & (t < fault_ends)


            # Compute va, vb, vc with or without fault
            va = torch.where(in_fault, tri_offset[:,0] * vg * cos_a, vg * cos_a)
            vb = torch.where(in_fault, tri_offset[:,1] * vg * cos_b, vg * cos_b)
            vc = torch.where(in_fault, tri_offset[:,2] * vg * cos_c, vg * cos_c)
            

                
            fault_type[:, t, 0] = torch.where(in_fault, fault_phase_label, fault_type[:, t, 0])

            # Clarke transform (batch-wise)
            v_alpha = (2/3) * (va - 0.5*vb - 0.5*vc)
            v_beta  = (2/3) * ((np.sqrt(3)/2)*vb - (np.sqrt(3)/2)*vc)

            va_full[:,t,0] = va
            vb_full[:,t,0] = vb
            vc_full[:,t,0] = vc

            d[:, t, n_w+2] = v_alpha
            d[:, t, n_w+3] = v_beta

        # Choose a few random samples to visualize
        d = torch.cat((d, time_enc, fault_type, fault_scale), dim=2)
        """plt.figure(figsize=(12, 8))
        plt.plot(fault_type[0,:,0], label='v_alpha')
        plt.title(f"Sample 0 — First 80 timesteps of Clarke Transform Voltages")
        plt.ylabel("Voltage [V]")
        plt.legend()
        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.show()
        """

        n_plot = 5
        sample_ids = torch.randint(0, d.shape[0], (n_plot,))

        t = torch.arange(d.shape[1]) * h

        """plt.figure(figsize=(12, 8))


        for i, idx in enumerate(sample_ids):
            plt.subplot(n_plot, 1, i+1)
            plt.plot(range(200,400), va_full[idx,200:400,0], label='va')
            plt.plot(range(200,400), vb_full[idx,200:400,0], label='vb')
            plt.plot(range(200,400), vc_full[idx,200:400,0], label='vc')
            plt.title(f"Sample {idx.item()} — Randomized Fault Start")
            plt.ylabel("Voltage [V]")
            if i == 0:
                plt.legend()
            if i < n_plot - 1:
                plt.xticks([])

        plt.xlabel("Time [s]")
        plt.tight_layout()
        plt.show()"""

        """idx = 2
        plt.subplot(6,1,1)
        plt.plot(range(0,2500), d[idx,:2500,n_w+2], label='va')
        
        plt.subplot(6,1,2)
        plt.plot(range(0,2500), d[idx,:2500,n_w+3], label='vb')
        
        plt.subplot(6,1,3)
        plt.plot(range(0,2500), d[idx,:2500,0], label='vc')
        
        plt.subplot(6,1,4)
        plt.plot(range(0,2500), d[idx,:2500,-2], label='fault type')

        plt.subplot(6,1,5)
        plt.plot(range(0,2500), d[idx,:2500,-1], label='fault scale')
        
        plt.subplot(6,1,6)
        plt.plot(range(0,2500), d[idx,:2500,-4:-2], label='rotor speed')
        plt.show()"""
        # Create a figure with a 2x2 grid of subplots
        


        return d

