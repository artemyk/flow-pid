import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

import normflows as nf
from ..utils.distributions import MultivariateGaussian
from tqdm import tqdm
import matplotlib.pyplot as plt


def norm_flows(n_flows, n_bottleneck, flow_type='Planar'):
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


class CartesianFlow(nn.Module):
    def __init__(self, dm, dx, dy, flows, p=None):
        super().__init__()
        self.dm = dm
        self.dx = dx
        self.dy = dy
        self.dim_mxy = dm + dx + dy

        self.q_mx = MultivariateGaussian(dm+dx)
        self.q_my = MultivariateGaussian(dm+dy)
        self.flows = nn.ModuleList(flows)
        self.p = p

    def forward(self, x):
        """Transforms flow variable x to the latent variable z

        Args:
          x: Batch in the space of the target distribution

        Returns:
          Batch in the latent space
        """
        for i in range(len(self.flows) - 1, -1, -1):
            x, _ = self.flows[i].inverse(x)
        return x

    def forward_kld(self, x):
        log_q = torch.zeros(len(x), device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        z_m, z_x, z_y = z.split([self.dm, self.dx, self.dy], dim=-1)
        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)
        log_q += self.q_mx.log_prob(z_mx) + self.q_my.log_prob(z_my)
        return -torch.mean(log_q)

    def log_prob(self, x):
        """Get log probability for batch

        Args:
          x: Batch

        Returns:
          log probability
        """
        log_q = torch.zeros(len(x), dtype=x.dtype, device=x.device)
        z = x
        for i in range(len(self.flows) - 1, -1, -1):
            z, log_det = self.flows[i].inverse(z)
            log_q += log_det
        z_m, z_x, z_y = z.split([self.dm, self.dx, self.dy], dim=-1)
        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)
        log_q += self.q_mx.log_prob(z_mx) + self.q_my.log_prob(z_my)
        return log_q

    def _standardize_data(self, data):
        # standardize data along columns
        return (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)

    def fit(self, m_data, x_data, y_data, batch_size, epochs, lr, verbose=False):
        m_data, x_data, y_data = self._standardize_data(m_data), self._standardize_data(x_data), self._standardize_data(y_data)

        dataset = TensorDataset(x_data, y_data, m_data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        losses = []
        for epoch in tqdm(range(epochs)):
            epoch_losses = []
            for x_batch, y_batch, m_batch in dataloader:
                optimizer.zero_grad()
                mxy = torch.cat([m_batch, x_batch, y_batch], dim=-1)
                loss = self.forward_kld(mxy)

                loss.backward()
                optimizer.step()
                scheduler.step()

                epoch_losses.append(loss.item())

            avg_loss = sum(epoch_losses) / len(epoch_losses)
            losses.append(avg_loss)

        with torch.no_grad():
            if verbose:
                plt.figure(figsize=(10, 5))
                plt.plot(losses)
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.title('Training Loss')
                plt.show()

            cov = self.estimate_latent_cov(m_data, x_data, y_data)

        return cov, losses

    def estimate_latent_cov(self, m, x, y):
        mxy = torch.cat([m, x, y], dim=-1)
        z_mxy = self.forward(mxy)
        cov = torch.cov(z_mxy.T).detach().cpu().numpy()

        covariance_matrix = np.array(cov)

        if covariance_matrix.shape[0] != covariance_matrix.shape[1]:
            raise ValueError("Covariance matrix must be square")

        if not np.allclose(covariance_matrix, covariance_matrix.T):
            raise ValueError("Covariance matrix must be symmetric")

        std_devs = np.sqrt(np.diag(covariance_matrix))
        outer_std = np.outer(std_devs, std_devs)
        correlation_matrix = covariance_matrix / outer_std
        np.fill_diagonal(correlation_matrix, 1.0)

        return correlation_matrix

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


class CartesianProductFlowVAE(nn.Module):
    def __init__(self, dim_m, dim_x, dim_y, q0=None, flows=None):
        """Constructor of normalizing flow model

        Args:
          prior: Prior distribution of te VAE, i.e. Gaussian
          decoder: Optional decoder
          flows: Flows to transform output of base encoder
          q0: Base Encoder
        """
        super().__init__()
        self.q0 = q0
        self.flows = nn.ModuleList(flows)

    def _load_encoder(self, path):
        self.q0.load_state_dict(torch.load(path, weights_only=True))

    def forward(self, x, num_samples=1):
        """Takes data batch, samples num_samples for each data point from base distribution

        Args:
          x: data batch
          num_samples: number of samples to draw for each data point

        Returns:
          latent variables for each batch and sample, log_q, and log_p
        """
        z, log_q = self.q0(x, num_samples=num_samples)
        # Flatten batch and sample dim
        z = z.view(-1, *z.size()[2:])
        log_q = log_q.view(-1, *log_q.size()[2:])
        for flow in self.flows:
            z, log_det = flow(z)
            log_q -= log_det
        log_p = self.prior.log_prob(z)
        if self.decoder is not None:
            log_p += self.decoder.log_prob(x, z)
        # Separate batch and sample dimension again
        z = z.view(-1, num_samples, *z.size()[1:])
        log_q = log_q.view(-1, num_samples, *log_q.size()[1:])
        log_p = log_p.view(-1, num_samples, *log_p.size()[1:])
        return z, log_q, log_p