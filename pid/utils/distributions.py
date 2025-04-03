import warnings
import numpy as np
import math
np.math = math

import torch
import torch.nn as nn
from normflows.distributions.base import BaseDistribution


## Full rank Gaussian distribution
class MultivariateGaussian(BaseDistribution):
    def __init__(self, dim):
        super().__init__()

        self.dim = dim
        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.loc_scale = nn.Parameter(torch.zeros(1, dim))
        self.L = nn.Parameter(torch.eye(dim))
        self.register_buffer('M', torch.tril(torch.ones(dim, dim), diagonal=-1))  # Lower triangular matrix for Cholesky decomposition

    def forward(self, num_samples=1):
        eps = torch.randn(num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device)
        Sig_L = self.M * self.L + torch.diag(torch.exp(self.loc_scale))

        z_ = torch.matmul(eps, Sig_L)  # Sample from the multivariate Gaussian
        z = z_ + self.loc

        Sig = Sig_L @ Sig_L.T # Covariance matrix
        log_p = (
                self.dim / 2 * np.log(2 * np.pi)
                - 0.5 * torch.det(Sig)
                - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )

        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc

        Sig_L = self.M * self.L + torch.diag(torch.exp(self.loc_scale))
        Sig = Sig_L @ Sig_L.T  # Covariance matrix
        log_p = (
                self.dim / 2 * np.log(2 * np.pi)
                - 0.5 * torch.det(Sig)
                - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )
        return log_p


def poisson_dist(lamda, x):
    return (np.exp(-lamda[..., None]) * lamda[..., None] ** x
            / np.array([np.math.factorial(i) for i in x]))


def binomial_dist(n, p):
    x = np.arange(n + 1)
    return np.array([np.math.comb(n, i) * p**i * (1 - p)**(n - i) for i in x])


def mult_poisson_dist(lamda_m, w_x, w_y, lamda_x, lamda_y, D=None):
    """
    XXX: not yet fully general - X and Y have no shared randomness except M.
    """

    d_M = lamda_m.size
    d_X = lamda_x.size
    d_Y = lamda_y.size

    if D is None:
        D = int(np.rint(np.max(lamda_m) * 3))  # Size of domain
    #p = np.ones([D,] * (d_M + d_X + d_Y))

    d = np.arange(D)
    pm = poisson_dist(lamda_m, d)

    # XXX: Dimension-specific code starts here
    assert (d_M == 2 and d_X == 1 and d_Y == 1)

    pmm = pm[[0], :].T * pm[[1], :]
    pmmx = np.zeros((D, D, D))
    pmmy = np.zeros((D, D, D))

    for i in range(D):
        for j in range(D):
            pmmx[i, j, :] = np.convolve(
                np.convolve(poisson_dist(lamda_x, d).squeeze(),
                            binomial_dist(d[i], w_x[0, 0])),
                binomial_dist(d[j], w_x[0, 1])
            )[:D]
            pmmy[i, j, :] = np.convolve(
                np.convolve(poisson_dist(lamda_y, d).squeeze(),
                            binomial_dist(d[i], w_y[0, 0])),
                binomial_dist(d[j], w_y[0, 1])
            )[:D]
    #lamda_eff_x = (w_x[0, 0] * d[:, None]) + (w_x[0, 1] * d) + lamda_x[0]
    #lamda_eff_y = (w_y[0, 0] * d[:, None]) + (w_y[0, 1] * d) + lamda_y[0]
    #pmmx = poisson_dist(lamda_eff_x, d)
    #pmmy = poisson_dist(lamda_eff_y, d)

    p = pmm[:, :, None, None] * pmmx[:, :, :, None] * pmmy[:, :, None, :]
    p_sum = p.sum()
    if p_sum < 0.95:
        warnings.warn('Total probability of truncated distribution < 0.95: p = %g' % p_sum)
    p /= p_sum  # Renormalize
    return p