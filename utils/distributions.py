import warnings
import numpy as np
import math
np.math = math

import torch
import torch.nn as nn
from normflows.distributions.base import BaseDistribution


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


class GaussianBC(BaseDistribution):
    """
    Gaussian distribution resulting from random binary channel
    X = H_x * M + N_x
    Y = H_y * M + N_y

    where M is a random variable with Gaussian distribution, H_x and H_y are
    channel matrices, and N_x and N_y are Gaussian noise variables.
    """

    def __init__(self, dm, dx, dy, sigma=0.1):
        """Constructor

        Args:
          dm, dx, dy: Number of dimensions of the flow variables
          sigma: Noise level
        """
        super().__init__()

        self.dm = dm
        self.dx = dx
        self.dy = dy
        self.dim = dm + dx + dy

        self.loc = nn.Parameter(torch.zeros(1, dm))
        self.log_scale = nn.Parameter(torch.zeros(1, dm))

        self.Hx = nn.Parameter(torch.randn(dm, dx))
        self.Hy = nn.Parameter(torch.randn(dm, dy))
        self.log_sigma_x = nn.Parameter(torch.tensor(np.log(sigma)))
        self.log_sigma_y = nn.Parameter(torch.tensor(np.log(sigma)))

    def _get_sig(self, detach=False):
        sigm = torch.exp(self.log_scale) * torch.eye(self.dm, dtype=self.loc.dtype, device=self.loc.device)
        sigx_m = torch.exp(self.log_sigma_x) * torch.eye(self.dx, dtype=self.loc.dtype, device=self.loc.device)
        sigy_m = torch.exp(self.log_sigma_y) * torch.eye(self.dy, dtype=torch.float, device=self.loc.device)

        block1 = torch.cat([sigm, sigm @ self.Hx, sigm @ self.Hy], dim=1)
        block2 = torch.cat([self.Hx.T @ sigm, self.Hx.T @ sigm @ self.Hx + sigx_m, self.Hx.T @ sigm @ self.Hy], dim=1)
        block3 = torch.cat([self.Hy.T @ sigm, self.Hy.T @ sigm @ self.Hx, self.Hy.T @ sigm @ self.Hy + sigy_m], dim=1)
        Sig = torch.cat([block1, block2, block3], dim=0)

        if detach:
            return Sig.detach().cpu().numpy()
        else:
            return Sig

    def forward(self, num_samples=1):
        eps_m = torch.randn(
            num_samples, self.dm, dtype=self.loc.dtype, device=self.loc.device
        )
        m_ = eps_m * torch.exp(self.log_scale)
        m = m_ + self.loc
        x_ = eps_m @ self.Hx
        x = x_ + self.loc @ self.Hx
        y_ = eps_m @ self.Hy
        y = y_ + self.loc @ self.Hy
        mxy = torch.cat([m, x, y], dim=-1)

        Sig = self._get_sig()
        z_ = torch.cat([m_, x_, y_], dim=-1)
        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(Sig)
            - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )

        return mxy, log_p

    def log_prob(self, z):
        loc_x = self.loc @ self.Hx
        loc_y = self.loc @ self.Hy
        loc = torch.cat([self.loc, loc_x, loc_y], dim=1)
        z_ = z - loc

        Sig = self._get_sig()
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