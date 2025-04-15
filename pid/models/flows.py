import numpy as np
import torch
import torch.nn as nn
import normflows as nf

from tqdm import tqdm
import matplotlib.pyplot as plt


def create_flows(n_flows, latent_size, q0, flow_type='RealNVP'):
    flows = norm_flows(n_flows, latent_size, flow_type)
    return nf.NormalizingFlow(
        q0=q0,
        flows=flows,
    )

class CartesianProductFlow(nn.Module):
    def __init__(self, dm, dx, dy, n_flows, encoder=None):
        super(CartesianProductFlow, self).__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy

        self.model_m = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dm),
            flows=norm_flows(n_flows, dm),
        )
        self.model_x = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dx),
            flows=norm_flows(n_flows, dx),
        )
        self.model_y = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dy),
            flows=norm_flows(n_flows, dy),
        )

        self.q_mx = nf.distributions.base.GaussianPCA(dm+dx, dm)
        self.q_my = nf.distributions.base.GaussianPCA(dm+dy, dm)
        self.encoder = encoder

    def forward(self, m, x, y):
        z_x = self.model_x.inverse(x)
        z_y = self.model_y.inverse(y)
        z_m = self.model_m.inverse(m)

        return z_m, z_x, z_y

    def learning_loss(self, m, x, y):
        z_m, log_det_m = self.model_m.inverse_and_log_det(m)
        z_x, log_det_x = self.model_x.inverse_and_log_det(x)
        z_y, log_det_y = self.model_y.inverse_and_log_det(y)

        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)

        log_det = log_det_m * 2 + log_det_x + log_det_y
        log_prob = self.q_mx.log_prob(z_mx) + self.q_my.log_prob(z_my)

        return -(log_prob + log_det).mean()

    def estimate_latent_mean(self, m, x, y):
        z_m, z_x, z_y = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.mean(z_combined, dim=0)

    def stack_mxy(self, m, x, y):
        z_m, z_x, z_y = self.forward(m, x, y)
        z_mxy = torch.cat([z_m, z_x, z_y], dim=-1)
        return z_mxy

    def estimate_latent_cov(self, m, x, y):
        z_m, z_x, z_y = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.cov(z_combined.T)

    def save(self, path):
        """Save state dict of model

        Args:
          path: Path including filename where to save model
        """
        torch.save(self.state_dict(), path)

    def load(self, path):
        """Load model from state dict

        Args:
          path: Path including filename where to load model from
        """
        self.load_state_dict(torch.load(path))



class GaussianFlow(nn.Module):
    def __init__(self, dm, dx, dy, n_flows, encoder=None):
        super(GaussianFlow, self).__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy

        q_m = nf.distributions.base.DiagGaussian(dm)
        q_x = nf.distributions.base.DiagGaussian(dx)
        q_y = nf.distributions.base.GaussianPCA(dy, dm)

        flow_m = norm_flows(n_flows, dm)
        flow_x = norm_flows(n_flows, dx)
        flow_y = norm_flows(n_flows, dy)

        self.model_m = nf.NormalizingFlow(q0=q_m, flows=flow_m)
        self.model_x = nf.NormalizingFlow(q0=q_x, flows=flow_x)
        self.model_y = nf.NormalizingFlow(q0=q_y, flows=flow_y)

        self.encoder = encoder

    def forward(self, m, x, y):
        z_x = self.model_x.inverse(x)
        z_y = self.model_y.inverse(y)
        z_m = self.model_m.inverse(m)

        return z_m, z_x, z_y

    def learning_loss(self, m, x, y):
        # loss_m = self.model_m.forward_kld(m)
        loss_x = self.model_x.forward_kld(x)
        # loss_y = self.model_y.forward_kld(y)

        return loss_x


def norm_flows(n_flows, latent_size, flow_type='RealNVP'):
    if flow_type == 'Planar':
        flows = [nf.flows.Planar((latent_size,)) for k in range(n_flows)]
    elif flow_type == 'Radial':
        flows = [nf.flows.Radial((latent_size,)) for k in range(n_flows)]
    elif flow_type == 'Spline':
        flows = []
        for i in range(n_flows):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, 2, 128)]
            flows += [nf.flows.LULinearPermute(latent_size)]
    elif flow_type == 'RealNVP':
        b = torch.Tensor([1 if i % 2 == 0 else 0 for i in range(latent_size)])
        flows = []
        for i in range(n_flows):
            s = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
            t = nf.nets.MLP([latent_size, 2 * latent_size, latent_size], init_zeros=True)
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
            flows += [nf.flows.ActNorm(latent_size)]
    else:
        raise NotImplementedError

    return flows


def glows(n_flows, n_bottleneck, channels, hidden_channels, input_shape, num_classes):
    # Set up flows, distributions and merge operations
    q0 = []
    merges = []
    flows = []
    for i in range(n_flows):
        flows_ = []
        for j in range(n_bottleneck):
            flows_ += [nf.flows.GlowBlock(channels * 2 ** (n_flows + 1 - i), hidden_channels,
                                          split_mode='channel', scale=True)]
        flows_ += [nf.flows.Squeeze()]
        flows += [flows_]
        if i > 0:
            merges += [nf.flows.Merge()]
            latent_shape = (input_shape[0] * 2 ** (n_flows - i), input_shape[1] // 2 ** (n_flows - i),
                            input_shape[2] // 2 ** (n_flows - i))
        else:
            latent_shape = (input_shape[0] * 2 ** (n_flows + 1), input_shape[1] // 2 ** n_flows,
                            input_shape[2] // 2 ** n_flows)
        q0 += [nf.distributions.ClassCondDiagGaussian(latent_shape, num_classes)]

    return q0, flows, merges