import math
import numpy as np
import scipy as sp
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from flows.utils import unconstrained_RQS

"""
Implementation of flows in this file is taken and developed from https://github.com/tonyduan/normalizing-flows.

Thank you to the authors for providing well-documented source code!
"""

# supported non-linearities: note that the function must be invertible
functional_derivatives = {
    torch.tanh: lambda x: 1 - torch.pow(torch.tanh(x), 2),
    F.leaky_relu: lambda x: (x > 0).type(torch.FloatTensor) + \
                            (x < 0).type(torch.FloatTensor) * -0.01,
    F.elu: lambda x: (x > 0).type(torch.FloatTensor) + \
                     (x < 0).type(torch.FloatTensor) * torch.exp(x)
}

class FCNN(nn.Module):
    """
    Simple fully connected neural network.
    """
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.network(x)

class NSF_AR(nn.Module):
    """
    Neural spline flow, auto-regressive.

    [Durkan et al. 2019]
    """
    #the B parameter is just for dim in Euclidean space
    #k is the number of spline segmentation not the number of knots
    def __init__(self, dim, K = 5, B = 5.0, hidden_dim = 8, base_network = FCNN):
        super().__init__()
        self.dim = dim
        self.K = K
        self.B = B
        self.layers = nn.ModuleList()
        self.init_param = nn.Parameter(torch.Tensor(3 * K - 1))
        for i in range(1, dim):
            self.layers += [base_network(i, 3 * K - 1, hidden_dim)]
        self.reset_parameters()

    def reset_parameters(self):
        init.uniform_(self.init_param, - 1 / 2, 1 / 2)

    def forward(self, x: torch.Tensor):
        n, d = x.shape
        z = torch.zeros_like(x)
        log_det = torch.zeros(z.shape[0])
        Ws = torch.zeros((n*d, self.K))
        if x.is_cuda:
            log_det = log_det.cuda()
            Ws = Ws.cuda()
        Hs = torch.zeros_like(Ws)
        # the number of derivatives is smaller than the number of bins by 1
        Ds = torch.zeros_like(Ws)[:,:-1]

        for i in range(d):
            if i == 0:
                init_param = self.init_param.expand(n, 3 * self.K - 1).clone()
                Ws[:n,:], Hs[:n,:], Ds[:n,:] = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                Ws[i*n:i*n+n,:], Hs[i*n:i*n+n,:], Ds[i*n:i*n+n,:] = torch.split(out, self.K, dim = 1)
            # W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            # W, H = 2 * self.B * W, 2 * self.B * H
            # D = F.softplus(D)
        # pull RQS out of for loop and vectorize the RQS computation
        zs, lds = unconstrained_RQS(
                x.transpose(0,1).flatten(), Ws, Hs, Ds, inverse=False, tail_bound=self.B)
            # z[:, i], ld = unconstrained_RQS(
            #     x[:, i], W, H, D, inverse=False, tail_bound=self.B)
            # log_det += ld
        return zs.reshape((n,d)), lds.reshape((n,d)).sum(dim = 1)

    def inverse(self, z):
        x = torch.zeros_like(z)
        log_det = torch.zeros(x.shape[0])
        if z.is_cuda:
            log_det = log_det.cuda()
        for i in range(z.shape[1]):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1).clone()
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            #W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            #W, H = 2 * self.B * W, 2 * self.B * H
            #D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i], W, H, D, inverse = True, tail_bound = self.B)
            log_det += ld
        return x, log_det

    def inverse_given_separator(self, z, x_s):
        x0 = torch.zeros_like(z)
        if x_s is None:
            x = x0
            separator_dim = 0
        else:
            x = torch.cat((x_s, x0), 1)
            separator_dim = x_s.shape[1]
        if z.is_cuda:
            x = x.cuda()
        for i in range(separator_dim, x.shape[1]):
            if i == 0:
                init_param = self.init_param.expand(x.shape[0], 3 * self.K - 1).clone()
                W, H, D = torch.split(init_param, self.K, dim = 1)
            else:
                out = self.layers[i - 1](x[:, :i])
                W, H, D = torch.split(out, self.K, dim = 1)
            #W, H = torch.softmax(W, dim = 1), torch.softmax(H, dim = 1)
            #W, H = 2 * self.B * W, 2 * self.B * H
            #D = F.softplus(D)
            x[:, i], ld = unconstrained_RQS(
                z[:, i-separator_dim], W, H, D, inverse = True, tail_bound = self.B)
        return x[:,separator_dim:]