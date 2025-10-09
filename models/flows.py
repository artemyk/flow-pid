import torch
import torch.nn as nn
import normflows as nf


def create_flows(n_flows, latent_size, q0, flow_type='RealNVP'):
    flows = norm_flows(n_flows, latent_size, flow_type)
    return nf.NormalizingFlow(
        q0=q0,
        flows=flows,
    )


class CartesianProductFlow(nn.Module):
    def __init__(self, dm, dx, dy, n_flows, encoder=None, gamma=1.0):
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

        self.q_mx = nf.distributions.base.GaussianPCA(dm + dx, dm)
        self.q_my = nf.distributions.base.GaussianPCA(dm + dy, dm)

        self.encoder = encoder
        self.gamma = gamma

    def forward(self, m, x, y):
        if self.encoder is not None:
            x, y = self.encoder(x, y)

        z_x = self.model_x.inverse(x)
        z_y = self.model_y.inverse(y)
        z_m = self.model_m.inverse(m)

        return z_m, z_x, z_y

    def learning_loss(self, m, x, y):
        lmi_loss = 0.0
        if self.encoder is not None:
            lmi_loss += self.encoder.learning_loss(x, y, m)
            x, y = self.encoder(x, y)

        z_m, log_det_m = self.model_m.inverse_and_log_det(m)
        z_x, log_det_x = self.model_x.inverse_and_log_det(x)
        z_y, log_det_y = self.model_y.inverse_and_log_det(y)

        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)
        log_prob = self.q_mx.log_prob(z_mx) + self.q_my.log_prob(z_my)
        log_det = log_det_m * 2 + log_det_x + log_det_y

        loss = -torch.mean(log_prob + log_det)
        if self.encoder is not None:
            loss += lmi_loss * self.gamma

        return loss

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
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


class BroadcastChannelFlow(nn.Module):
    def __init__(self, dm, dx, dy, n_flows, encoder=None, gamma=1.0):
        super(BroadcastChannelFlow, self).__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy

        context_encoder_x = nf.nets.MLP([dm, dx * 2], init_zeros=True)
        context_encoder_y = nf.nets.MLP([dm, dy * 2], init_zeros=True)

        self.nf_m = nf.NormalizingFlow(
            q0=nf.distributions.base.DiagGaussian(dm),
            flows=norm_flows(n_flows, dm),
        )
        self.nf_x = nf.NormalizingFlow(
            q0=nf.distributions.base.ConditionalDiagGaussian(dx, context_encoder_x),
            flows=norm_flows(n_flows, dx),
        )
        self.nf_y = nf.NormalizingFlow(
            q0=nf.distributions.base.ConditionalDiagGaussian(dy, context_encoder_y),
            flows=norm_flows(n_flows, dy),
        )

        self.encoder = encoder
        self.gamma = gamma

    def forward(self, m, x, y):
        if self.encoder is not None:
            x, y = self.encoder(x, y)
        z_m = self.nf_m.inverse(m)
        z_x = self.nf_x.inverse(x)
        z_y = self.nf_y.inverse(y)
        return z_m, z_x, z_y

    def learning_loss(self, m, x, y):
        lmi_loss = 0.0
        if self.encoder is not None:
            lmi_loss += self.encoder.learning_loss(x, y, m)
            x, y = self.encoder(x, y)

        z_m, log_det_m = self.nf_m.inverse_and_log_det(m)
        z_x, log_det_x = self.nf_x.inverse_and_log_det(x)
        z_y, log_det_y = self.nf_y.inverse_and_log_det(y)
        log_det = log_det_m + log_det_x + log_det_y
        log_prob = (self.nf_m.q0.log_prob(z_m) + self.nf_x.q0.log_prob(z_x, context=z_m)
                    + self.nf_y.q0.log_prob(z_y, context=z_m))
        loss = -(log_prob + log_det).mean()

        if self.encoder is not None:
            loss += lmi_loss * self.gamma

        return loss

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))


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
            s = nf.nets.MLP([latent_size, latent_size // 2, 64, latent_size], init_zeros=True)
            t = nf.nets.MLP([latent_size, latent_size // 2, 64, latent_size], init_zeros=True)
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


def conditional_flows(n_flows, latent_size, hidden_units, hidden_layers, context_size, flow_type='AutoregressiveNeuralSpline'):
    flows = []
    if flow_type == 'MaskedAffineAutoregressive':
        for i in range(n_flows):
            flows += [nf.flows.MaskedAffineAutoregressive(latent_size, hidden_units,
                                                          context_features=context_size,
                                                          num_blocks=hidden_layers)]
            flows += [nf.flows.LULinearPermute(latent_size)]
    elif flow_type == 'AutoregressiveNeuralSpline':
        for i in range(n_flows):
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,
                                                                     num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]
    elif flow_type == 'CoupledNeuralSpline':
        for i in range(n_flows):
            flows += [nf.flows.CoupledRationalQuadraticSpline(latent_size, hidden_layers, hidden_units,
                                                              num_context_channels=context_size)]
            flows += [nf.flows.LULinearPermute(latent_size)]

    return flows
