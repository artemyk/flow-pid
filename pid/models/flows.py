import numpy as np
import torch
import torch.nn as nn
import normflows as nf

from tqdm import tqdm
import matplotlib.pyplot as plt


def _create_flow(q0, n_flows, n_bottleneck, flow_type='RealNVP'):
    flows = norm_flows(n_flows, n_bottleneck, flow_type)
    return nf.NormalizingFlow(
        q0=q0(n_bottleneck),
        flows=flows
    )

class CartesianProductFlow(nn.Module):
    def __init__(self, dm, dx, dy, q0, n_flows, encoder=None):
        super().__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy

        self.model_m = _create_flow(q0, n_flows, dm)
        self.model_x = _create_flow(q0, n_flows, dx)
        self.model_y = _create_flow(q0, n_flows, dy)

        self.mean_mx = nn.Parameter(torch.zeros(dm + dy))
        self.L_mx = nn.Parameter(torch.eye(dm + dx))  # Initialize with a small value

        self.mean_my = nn.Parameter(torch.zeros(dm + dy))
        self.L_my = nn.Parameter(torch.eye(dm + dy))  # Initialize with a small value

        self.q_mx = q0(dm + dx)
        self.q_my = q0(dm + dy)
        self.encoder = encoder

    def forward(self, m, x, y):
        z_x, log_det_x = self.model_x.inverse_and_log_det(x)
        z_y, log_det_y = self.model_y.inverse_and_log_det(y)
        z_m, log_det_m = self.model_m.inverse_and_log_det(m)
        log_det = log_det_x + log_det_y + log_det_m*2

        return z_m, z_x, z_y, log_det

    def forward_kld(self, m, x, y):
        z_m, z_x, z_y, log_det = self.forward(m, x, y)
        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)
        log_prob = self.q_mx.log_prob(z_mx) + self.q_my.log_prob(z_my)

        return -(log_prob + log_det).mean()

    def estimate_latent_mean(self, m, x, y):
        z_m, z_x, z_y, log_det = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.mean(z_combined, dim=0)

    def stack_mxy(self, m, x, y):
        z_m, z_x, z_y, _ = self.forward(m, x, y)
        z_mxy = torch.cat([z_m, z_x, z_y], dim=-1)
        return z_mxy

    def estimate_latent_cov(self, m, x, y):
        z_m, z_x, z_y, log_det = self.forward(m, x, y)
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


def norm_flows(n_flows, n_bottleneck, flow_type='RealNVP'):
    if flow_type == 'Planar':
        flows = [nf.flows.Planar((n_bottleneck,)) for k in range(n_flows)]
    elif flow_type == 'Radial':
        flows = [nf.flows.Radial((n_bottleneck,)) for k in range(n_flows)]
    elif flow_type == 'RealNVP':
        b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])
        flows = []
        for i in range(n_flows):
            s = nf.nets.MLP([n_bottleneck, n_bottleneck])
            t = nf.nets.MLP([n_bottleneck, n_bottleneck])
            if i % 2 == 0:
                flows += [nf.flows.MaskedAffineFlow(b, t, s)]
            else:
                flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
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