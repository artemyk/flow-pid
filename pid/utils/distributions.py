import warnings
import numpy as np
import math
np.math = math

import torch
import torch.nn as nn
from normflows.distributions.base import BaseDistribution


class DiagGaussian(BaseDistribution):
    """
    Multivariate Gaussian distribution with diagonal covariance matrix
    """

    def __init__(self, shape, trainable=True):
        """Constructor

        Args:
          shape: Tuple with shape of data, if int shape has one dimension
          trainable: Flag whether to use trainable or fixed parameters
        """
        super().__init__()
        if isinstance(shape, int):
            shape = (shape,)
        if isinstance(shape, list):
            shape = tuple(shape)
        self.shape = shape
        self.n_dim = len(shape)
        self.d = np.prod(shape)
        if trainable:
            self.loc = nn.Parameter(torch.zeros(1, *self.shape))
            self.log_scale = nn.Parameter(torch.zeros(1, *self.shape))
        else:
            self.register_buffer("loc", torch.zeros(1, *self.shape))
            self.register_buffer("log_scale", torch.zeros(1, *self.shape))
        self.temperature = None  # Temperature parameter for annealed sampling

    def forward(self, num_samples=1, context=None):
        eps = torch.randn(
            (num_samples,) + self.shape, dtype=self.loc.dtype, device=self.loc.device
        )
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        z = self.loc + torch.exp(log_scale) * eps
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow(eps, 2), list(range(1, self.n_dim + 1))
        )
        return z, log_p

    def log_prob(self, z, context=None):
        if self.temperature is None:
            log_scale = self.log_scale
        else:
            log_scale = self.log_scale + np.log(self.temperature)
        log_p = -0.5 * self.d * np.log(2 * np.pi) - torch.sum(
            log_scale + 0.5 * torch.pow((z - self.loc) / torch.exp(log_scale), 2),
            list(range(1, self.n_dim + 1)),
        )
        return log_p

    def get_loc(self, detach=True):
        return self.loc.detach() if detach else self.loc

    def get_log_scale(self, detach=True):
        return self.log_scale.detach() if detach else self.log_scale


class MultivariateGaussian(BaseDistribution):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.loc = nn.Parameter(torch.zeros(dim))
        self.L = nn.Parameter(torch.eye(dim))  # Initialize with a small value

    def forward(self, num_samples=1):
        eps = torch.randn(
            num_samples, self.dim, dtype=self.loc.dtype, device=self.loc.device
        )
        z = self.loc + torch.matmul(eps, self.L)

        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(self.L @ self.L.T)
            - 0.5 * torch.sum(eps * torch.matmul(eps, torch.inverse(self.L)), 1)
        )

        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc

        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(self.L @ self.L.T)
            - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(self.L)), 1)
        )

        return log_p

    def get_covariance(self, detach=True):
        cov = self.L @ self.L.T
        return cov.detach() if detach else cov


class GaussianPCA(BaseDistribution):
    """
    Gaussian distribution resulting from linearly mapping a normal distributed latent
    variable describing the "content of the target"
    """

    def __init__(self, dim, latent_dim=None, sigma=0.1):
        """Constructor

        Args:
          dim: Number of dimensions of the flow variables
          latent_dim: Number of dimensions of the latent "content" variable;
                           if None it is set equal to dim
          sigma: Noise level
        """
        super().__init__()

        self.dim = dim
        if latent_dim is None:
            self.latent_dim = dim
        else:
            self.latent_dim = latent_dim

        self.loc = nn.Parameter(torch.zeros(1, dim))
        self.W = nn.Parameter(torch.randn(self.latent_dim, dim))
        self.log_sigma = nn.Parameter(torch.tensor(np.log(sigma)))

    def forward(self, num_samples=1):
        eps = torch.randn(
            num_samples, self.latent_dim, dtype=self.loc.dtype, device=self.loc.device
        )
        z_ = torch.matmul(eps, self.W)
        z = z_ + self.loc

        Sig = torch.matmul(self.W.T, self.W) + torch.exp(
            self.log_sigma * 2
        ) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)

        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(Sig)
            - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )

        return z, log_p

    def log_prob(self, z):
        z_ = z - self.loc

        Sig = torch.matmul(self.W.T, self.W) + torch.exp(
            self.log_sigma * 2
        ) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        log_p = (
            self.dim / 2 * np.log(2 * np.pi)
            - 0.5 * torch.det(Sig)
            - 0.5 * torch.sum(z_ * torch.matmul(z_, torch.inverse(Sig)), 1)
        )

        return log_p

    def get_H(self, detach=True):
        return self.W.T.detach() if detach else self.W.T

    def get_covariance(self, detach=True):
        cov = torch.matmul(self.W.T, self.W) + torch.exp(
            self.log_sigma * 2
        ) * torch.eye(self.dim, dtype=self.loc.dtype, device=self.loc.device)
        return cov.detach() if detach else cov


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