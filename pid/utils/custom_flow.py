import torch
import torch.nn as nn
import normflows as nf

class CartesianProductFlow(nn.Module):
    def __init__(self, dim_m, dim_x, dim_y, n_flows=3):
        super().__init__()
        self.dim_m = dim_m
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.dim_total = dim_m + dim_x + dim_y
        
        # Create base distribution
        self.base_m = nf.distributions.base.DiagGaussian(dim_m)
        self.base_x = nf.distributions.base.DiagGaussian(dim_x)
        self.base_y = nf.distributions.base.DiagGaussian(dim_y)


        # Create flows
        flows_x = []
        flows_y = []
        flows_m = []
        
        for _ in range(n_flows):
            flows_x += [nf.flows.AutoregressiveRationalQuadraticSpline(dim_x, 2, 128)]
            flows_x += [nf.flows.LULinearPermute(dim_x)]
            flows_y += [(nf.flows.AutoregressiveRationalQuadraticSpline(dim_y, 2, 128))]
            flows_y += [nf.flows.LULinearPermute(dim_y)]
            flows_m += [(nf.flows.AutoregressiveRationalQuadraticSpline(dim_m, 2, 128))]
            flows_m += [nf.flows.LULinearPermute(dim_m)]
        
        self.model_x = nf.NormalizingFlow(q0=self.base_x, flows=flows_x)
        self.model_y = nf.NormalizingFlow(q0=self.base_y, flows=flows_y)
        self.model_m = nf.NormalizingFlow(q0=self.base_m, flows=flows_m)

        # Learnable parameters for target Gaussian of joint distribution
        # self.mean = nn.Parameter(torch.zeros(self.dim_total))
        # self.L = nn.Parameter(torch.eye(self.dim_total))  # Initialize with a small value

        self.mean_mx = nn.Parameter(torch.zeros(self.dim_m + self.dim_x))
        self.L_mx = nn.Parameter(torch.eye(self.dim_m + self.dim_x))  # Initialize with a small value

        self.mean_my = nn.Parameter(torch.zeros(self.dim_m + self.dim_y))
        self.L_my = nn.Parameter(torch.eye(self.dim_m + self.dim_y))  # Initialize with a small value
        
    # def get_covariance(self):
    #     return self.L @ self.L.T
        
    def forward(self, m, x, y):
        # z_x = self.model_x.forward(x)
        # log_det_x = self.model_x.forward_kld(x)
        # z_y = self.model_y.forward(y)
        # log_det_y = self.model_y.forward_kld(y)
        # z_m = self.model_m.forward(m)
        # log_det_m = self.model_m.forward_kld(m)

        ### This is what I think to compute the forward pass
        ### Reference of normflows package: https://github.com/VincentStimper/normalizing-flows/blob/master/normflows/core.py

        z_x, log_det_x = self.model_x.inverse_and_log_det(x)
        z_y, log_det_y = self.model_y.inverse_and_log_det(y)
        z_m, log_det_m = self.model_m.inverse_and_log_det(m)

        return z_m, z_x, z_y, log_det_x + log_det_y + log_det_m
    
    def estimate_latent_mean(self, m, x, y):
        z_m, z_x, z_y, log_det = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.mean(z_combined, dim=0)

    def estimate_latent_cov(self, m, x, y):
        z_m, z_x, z_y, log_det = self.forward(m, x, y)
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        return torch.cov(z_combined.T)
    
    def compute_loss(self, z_m, z_x, z_y, log_det):
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)

        z_mx = torch.cat([z_m, z_x], dim=-1)
        z_my = torch.cat([z_m, z_y], dim=-1)

        # print(z_m.shape, z_x.shape, z_y.shape, z_mx.shape, z_my.shape)

        # mean_mx = self.mean[:self.dim_m+self.dim_x]
        # mean_my = torch.cat([self.mean[:self.dim_m], self.mean[-self.dim_y:]], dim=0)
        # cov_mx = self.L[:self.dim_m+self.dim_x] @ self.L[:self.dim_m+self.dim_x].T
        # L_my = torch.cat([self.L[:self.dim_m], self.L[-self.dim_y:]], dim=0)
        # cov_my =  L_my @ L_my.T
        
        # target_mx = torch.distributions.MultivariateNormal(
        #     loc = mean_mx,
        #     covariance_matrix=cov_mx
        # )
        # target_my = torch.distributions.MultivariateNormal(
        #     loc=mean_my,
        #     covariance_matrix=cov_my
        # )

        target_mx = torch.distributions.MultivariateNormal(
            loc = self.mean_mx,
            covariance_matrix=self.L_mx @ self.L_mx.T,
        )
        target_my = torch.distributions.MultivariateNormal(
            loc=self.mean_my,
            covariance_matrix=self.L_my @ self.L_my.T
        )

        # log_prob = target.log_prob(z_combined)
        log_prob = target_mx.log_prob(z_mx) + target_my.log_prob(z_my)

        return -(log_prob+ log_det).mean()