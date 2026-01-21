import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ContractiveREN(nn.Module):
    """
    Acyclic contractive recurrent equilibrium network, following the paper:
    "Recurrent equilibrium networks: Flexible dynamic models with guaranteed
    stability and robustness, Revay M et al. ."

    The mathematical model of RENs relies on an implicit layer embedded in a recurrent layer.
    The model is described as,

                    [  E . x_t+1 ]  =  [ F    B_1  B_2   ]   [  x_t ]   +   [  b_x ]
                    [  Λ . v_t   ]  =  [ C_1  D_11  D_12 ]   [  w_t ]   +   [  b_w ]
                    [  y_t       ]  =  [ C_2  D_21  D_22 ]   [  u_t ]   +   [  b_u ]

    where E is an invertible matrix and Λ is a positive-definite diagonal matrix. The model parameters
    are then {E, Λ , F, B_i, C_i, D_ij, b} which form a convex set according to the paper.

    NOTE: REN has input "u", output "y", and internal state "x". When used in closed-loop,
          the REN input "u" would be the noise reconstruction ("w") and the REN output ("y")
          would be the input to the plant. The internal state of the REN ("x") should not be mistaken
          with the internal state of the plant.
    """

    def __init__(
        self, dim_in: int, dim_out: int, dim_internal: int,
        dim_nl: int, internal_state_init = None, initialization_std: float = 0.5,
        posdef_tol: float = 0.001, contraction_rate_lb: float = 1.0
    ):
        """
        Args:
            dim_in (int): Input (u) dimension.
            dim_out (int): Output (y) dimension.
            dim_internal (int): Internal state (x) dimension. This state evolves with contraction properties.
            dim_nl (int): Dimension of the input ("v") and ouput ("w") of the nonlinear static block.
            initialization_std (float, optional): Weight initialization. Set to 0.1 by default.
            internal_state_init (torch.Tensor or None, optional): Initial condition for the internal state. Defaults to 0 when set to None.
            epsilon (float, optional): Positive and negligible scalar to force positive definite matrices.
            contraction_rate_lb (float, optional): Lower bound on the contraction rate. Defaults to 1.
        """
        super().__init__()

        # set dimensions
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_internal = dim_internal
        self.dim_nl = dim_nl

        # set functionalities
        self.contraction_rate_lb = contraction_rate_lb

        # auxiliary elements
        self.epsilon = posdef_tol

        # initialize internal state
        if internal_state_init is None:
            self.x = torch.zeros(1, 1, self.dim_internal)
        else:
            assert isinstance(internal_state_init, torch.Tensor)
            self.x = internal_state_init.reshape(1, 1, self.dim_internal)
        self.register_buffer('init_x', self.x.detach().clone())

        # define matrices shapes
        # auxiliary matrices
        self.X_shape = (2 * self.dim_internal + self.dim_nl, 2 * self.dim_internal + self.dim_nl)
        self.Y_shape = (self.dim_internal, self.dim_internal)
        # nn state dynamics
        self.B2_shape = (self.dim_internal, self.dim_in)
        # nn output
        self.C2_shape = (self.dim_out, self.dim_internal)
        self.D21_shape = (self.dim_out, self.dim_nl)
        self.D22_shape = (self.dim_out, self.dim_in)
        # v signal
        self.D12_shape = (self.dim_nl, self.dim_in)


        #Biases
        self.b_xi_shape = (1,self.dim_internal)      
        self.b_v_shape = (1,self.dim_nl)
        self.b_y_shape = (1,2)

        # define trainble params
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12'] #,'b_y','b_v','b_xi'
        self._init_trainable_params(initialization_std)


        # mask
        self.register_buffer('eye_mask_H', torch.eye(2 * self.dim_internal + self.dim_nl))
        self.register_buffer('eye_mask_w', torch.eye(self.dim_nl))

    def _update_model_param(self):
        """
        Update non-trainable matrices according to the REN formulation to preserve contraction.
        """
        # dependent params
        H = torch.matmul(self.X.T, self.X) + self.epsilon * self.eye_mask_H
        h1, h2, h3 = torch.split(H, [self.dim_internal, self.dim_nl, self.dim_internal], dim=0)
        H11, H12, H13 = torch.split(h1, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H21, H22, _ = torch.split(h2, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        H31, H32, H33 = torch.split(h3, [self.dim_internal, self.dim_nl, self.dim_internal], dim=1)
        P = H33

        # nn state dynamics
        self.F = H31
        self.B1 = H32

        # nn output
        self.E = 0.5 * (H11 + self.contraction_rate_lb * P + self.Y - self.Y.T)

        # v signal for strictly acyclic REN
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21

    def forward(self, u_in):
        """
        Forward pass of REN.

        Args:
            u_in (torch.Tensor): Input with the size of (batch_size, 1, self.dim_in).

        Return:
            y_out (torch.Tensor): Output with (batch_size, 1, self.dim_out).
        """
        # update non-trainable model params
        self._update_model_param()

        batch_size = u_in.shape[0]

        w = torch.zeros(batch_size, 1, self.dim_nl, device=u_in.device)

        # update each row of w using Eq. (8) with a lower triangular D11
        for i in range(self.dim_nl):
            #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u_in, self.D12[i,:])# + self.b_v[:,i]
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.dim_nl)

        # compute next state using Eq. 18
        self.x = F.linear(
            F.linear(self.x, self.F) + F.linear(w, self.B1) + F.linear(u_in, self.B2),#+ self.b_xi,
            self.E.inverse())

        # compute output
        y_out = F.linear(self.x, self.C2) + F.linear(w, self.D21) + F.linear(u_in, self.D22)#+ self.b_y
        return y_out

    # init trainable params
    def _init_trainable_params(self, initialization_std):
        for training_param_name in self.training_param_names:  # name of one of the training params, e.g., X
            # read the defined shapes of the selected training param, e.g., X_shape
            shape = getattr(self, training_param_name + '_shape')
            
            # define the selected param (e.g., self.X) as nn.Parameter
            setattr(self, training_param_name, nn.Parameter((torch.randn(*shape) * initialization_std)))

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict

class REN(nn.Module):
    # ## Implementation of REN model, modified from "Recurrent Equilibrium Networks: Flexible Dynamic Models with
    # Guaranteed Stability and Robustness" by Max Revay et al.
    def __init__(self, m, p, n, l, bias=False, mode="l2stable", gamma=0.3, gammaTrain=False, Q=None, R=None, S=None,
                 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')):
        super().__init__()
        self.m = m  # input dimension
        self.n = n  # state dimension
        print("REN state dimension:", n)
        self.l = l  # dimension of v(t) and w(t)
        self.p = p  # output dimension
        self.mode = mode
        self.device = device
        self.gamma = gamma
        self.gammaTrain = gammaTrain
        if gammaTrain:
            self.sg = nn.Parameter(torch.tensor(gamma))
        # # # # # # # # # IQC specification # # # # # # # # #
        self.Q = Q
        self.R = R
        self.S = S
        # # # # # # # # # Training parameters # # # # # # # # #
        std = 0.01
        # Sparse training matrix parameters
        self.x0 = nn.Parameter((torch.randn(1, n, device=device) * std))
        self.X = nn.Parameter((torch.randn(2 * n + l, 2 * n + l, device=device) * std))
        self.Y = nn.Parameter((torch.randn(n, n, device=device) * std))
        self.Z3 = nn.Parameter(torch.randn(abs(p - m), min(p, m), device=device) * std)
        self.X3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.Y3 = nn.Parameter(torch.randn(min(p, m), min(p, m), device=device) * std)
        self.D12 = nn.Parameter(torch.randn(l, m, device=device))
        #self.D21 = nn.Parameter((torch.randn(p, l, device=device) * std)) #set to zero later to enable
        # computable Networked RENs
        self.B2 = nn.Parameter((torch.randn(n, m, device=device) * std))
        self.C2 = nn.Parameter((torch.randn(p, n, device=device) * std))

        if bias:
            self.bx = nn.Parameter(torch.randn(n, device=device) * std)
            self.bv = nn.Parameter(torch.randn(l, device=device) * std)
            self.bu = nn.Parameter(torch.randn(p, device=device) * std)
        else:
            self.bx = torch.zeros(n, device=device)
            self.bv = torch.zeros(l, device=device)
            self.bu = torch.zeros(p, device=device)
        # # # # # # # # # Non-trainable parameters # # # # # # # # #
        # Auxiliary elements

        self.x = torch.zeros(1, n, device=device)
        self.register_buffer('init_x', self.x.detach().clone())
        self.register_buffer('eye_mask_w', torch.eye(self.l))


        self.epsilon = 0.001
        self.F = torch.zeros(n, n, device=device)
        self.B1 = torch.zeros(n, l, device=device)
        self.E = torch.zeros(n, n, device=device)
        self.Lambda = torch.ones(l, device=device)
        self.C1 = torch.zeros(l, n, device=device)
        self.D11 = torch.zeros(l, l, device=device)
        self.D22 = torch.zeros(p, m, device=device)
        self.P = torch.zeros(n, n, device=device)
        self.P_cal = torch.zeros(n, n, device=device)
        self.D21 = torch.zeros(p, l, device=device)
        self.set_param(gamma)
        self.training_param_names = ['X', 'Y', 'B2', 'C2', 'D21', 'D22', 'D12',"sg"] #,'b_y','b_v','b_xi'


    def set_param(self, gamma=0.3):
        if self.gammaTrain:
            gamma = self.sg**2
        self.gamma = gamma
        n, l, m, p = self.n, self.l, self.m, self.p
        # Updating of Q,S,R with variable gamma if needed
        self.Q, self.R, self.S = self._set_mode(self.mode, gamma, self.Q, self.R, self.S)
        M = F.linear(self.X3.T, self.X3.T) + self.Y3 - self.Y3.T + F.linear(self.Z3.T,
                                                                            self.Z3.T) + self.epsilon * torch.eye(
            min(m, p), device=self.device)
        if p >= m:
            N = torch.vstack((F.linear(torch.eye(m, device=self.device) - M,
                                       torch.inverse(torch.eye(m, device=self.device) + M).T),
                              -2 * F.linear(self.Z3, torch.inverse(torch.eye(m, device=self.device) + M).T)))
        else:
            N = torch.hstack((F.linear(torch.inverse(torch.eye(p, device=self.device) + M),
                                       (torch.eye(p, device=self.device) - M).T),
                              -2 * F.linear(torch.inverse(torch.eye(p, device=self.device) + M), self.Z3)))

        Lq = torch.linalg.cholesky(-self.Q).T
        Lr = torch.linalg.cholesky(self.R - torch.matmul(self.S, torch.matmul(torch.inverse(self.Q), self.S.T))).T
        self.D22 = -torch.matmul(torch.inverse(self.Q), self.S.T) + torch.matmul(torch.inverse(Lq),
                                                                                 torch.matmul(N, Lr))
        # Calculate psi_r:
        R_cal = self.R + torch.matmul(self.S, self.D22) + torch.matmul(self.S, self.D22).T + torch.matmul(self.D22.T,
                                                                                                          torch.matmul(
                                                                                                              self.Q,
                                                                                                              self.D22))
        R_cal_inv = torch.linalg.inv(R_cal)
        C2_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.C2).T
        D21_cal = torch.matmul(torch.matmul(self.D22.T, self.Q) + self.S, self.D21).T - self.D12
        vec_r = torch.cat((C2_cal, D21_cal, self.B2), dim=0)
        psi_r = torch.matmul(vec_r, torch.matmul(R_cal_inv, vec_r.T))
        # Calculate psi_q:
        vec_q = torch.cat((self.C2.T, self.D21.T, torch.zeros(self.n, self.p, device=self.device)), dim=0)
        psi_q = torch.matmul(vec_q, torch.matmul(self.Q, vec_q.T))
        # Create H matrix:
        H = torch.matmul(self.X.T, self.X) + self.epsilon * torch.eye(2 * n + l, device=self.device) + psi_r - psi_q
        h1, h2, h3 = torch.split(H, [n, l, n], dim=0)
        H11, H12, H13 = torch.split(h1, [n, l, n], dim=1)
        H21, H22, _ = torch.split(h2, [n, l, n], dim=1)
        H31, H32, H33 = torch.split(h3, [n, l, n], dim=1)
        self.P_cal = H33
        # NN state dynamics:
        self.F = H31
        self.B1 = H32
        # NN output:
        self.E = 0.5 * (H11 + self.P_cal + self.Y - self.Y.T)
        # v signal:  [Change the following 2 lines if we don't want a strictly acyclic REN!]
        self.Lambda = 0.5 * torch.diag(H22)
        self.D11 = -torch.tril(H22, diagonal=-1)
        self.C1 = -H21
        # Matrix P
        self.P = torch.matmul(self.E.T, torch.matmul(torch.inverse(self.P_cal), self.E))

    def forward(self, u, t):
        x = self.x

        batch_size = u.shape[0]
        w = torch.zeros(batch_size, 1, self.l, device=u.device)



        decay_rate = 0.95
        vec = torch.zeros(self.l, device=self.device)
        epsilon = torch.zeros(self.l, device=self.device)
        if self.l > 0:
            vec[0] = 1

            
            """v = F.linear(x, self.C1[0, :]) + F.linear(u, self.D12[0, :]) + (decay_rate ** t) * self.bv[0]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[0])"""
        for i in range(1, self.l):
            """vec = torch.zeros(self.l, device=self.device)
            vec[i] = 1

            v = F.linear(x, self.C1[i, :]) + F.linear(epsilon, self.D11[i, :]) + F.linear(u, self.D12[i, :]) + (
                    decay_rate ** t) * self.bv[i]
            epsilon = epsilon + vec * torch.tanh(v / self.Lambda[i])
            """
                        #  v is element i of v with dim (batch_size, 1)
            v = F.linear(self.x, self.C1[i, :]) + F.linear(w, self.D11[i, :]) + F.linear(u, self.D12[i,:])# + self.b_v[:,i]
            w = w + (self.eye_mask_w[i, :] * torch.tanh(v / self.Lambda[i])).reshape(batch_size, 1, self.l)

        E_x_ = F.linear(x, self.F) + F.linear(epsilon, self.B1) + F.linear(u, self.B2) + (decay_rate ** t) * self.bx

        x_ = F.linear(E_x_, self.E.inverse())

        y = F.linear(x, self.C2) + F.linear(epsilon, self.D21) + F.linear(u, self.D22) + (decay_rate ** t) * self.bu
        self.x = x_
        return y
    def _set_mode(self, mode, gamma, Q, R, S, eps=1e-4):
        # We set Q to be negative definite. If Q is nsd we set: Q - \epsilon I.
        # I.e. The Q we define here is denoted as \matcal{Q} in REN paper.
        if mode == "l2stable":
            Q = -(1. / gamma) * torch.eye(self.p, device=self.device)
            R = gamma * torch.eye(self.m, device=self.device)
            S = torch.zeros(self.m, self.p, device=self.device)
        elif mode == "input_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing input passivity.")
            Q = torch.zeros(self.p, self.p, device=self.device) - eps * torch.eye(self.p, device=self.device)
            R = -2. * gamma * torch.eye(self.m, device=self.device)
            S = torch.eye(self.p, device=self.device)
        elif mode == "output_p":
            if self.p != self.m:
                raise NameError("Dimensions of u(t) and y(t) need to be the same for enforcing output passivity.")
            Q = -2. * gamma * torch.eye(self.p, device=self.device)
            R = torch.zeros(self.m, self.m, device=self.device)
            S = torch.eye(self.m, device=self.device)
        else:
            print("Using matrices R,Q,S given by user.")
            # Check dimensions:
            if not (len(R.shape) == 2 and R.shape[0] == R.shape[1] and R.shape[0] == self.m):
                raise NameError("The matrix R is not valid. It must be a square matrix of %ix%i." % (self.m, self.m))
            if not (len(Q.shape) == 2 and Q.shape[0] == Q.shape[1] and Q.shape[0] == self.p):
                raise NameError("The matrix Q is not valid. It must be a square matrix of %ix%i." % (self.p, self.p))
            if not (len(S.shape) == 2 and S.shape[0] == self.m and S.shape[1] == self.p):
                raise NameError("The matrix S is not valid. It must be a matrix of %ix%i." % (self.m, self.p))
            # Check R=R':
            if not (R == R.T).prod():
                raise NameError("The matrix R is not valid. It must be symmetric.")
            # Check Q is nsd:
            eigs, _ = torch.linalg.eig(Q)
            if not (eigs.real <= 0).prod():
                print('oh!')
                raise NameError("The matrix Q is not valid. It must be negative semidefinite.")
            if not (eigs.real < 0).prod():
                # We make Q negative definite: (\mathcal{Q} in the REN paper)
                Q = Q - eps * torch.eye(self.p, device=self.device)
        return Q, R, S

    # setters and getters
    def get_parameter_shapes(self):
        param_dict = OrderedDict(
            (name, getattr(self, name).shape) for name in self.training_param_names
        )
        return param_dict

    def get_named_parameters(self):
        param_dict = OrderedDict(
            (name, getattr(self, name)) for name in self.training_param_names
        )
        return param_dict
