import torch
import torch.nn as nn
import normflows as nf

class CartesianProductFlow(nn.Module):
    def __init__(self, dim_m, dim_x, dim_y, n_flows=3):
        super().__init__()
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
        self.mean = nn.Parameter(torch.zeros(self.dim_total))
        self.L = nn.Parameter(torch.eye(self.dim_total) * 0.1)  # Initialize with a small value
        
    def get_covariance(self):
        return self.L @ self.L.T
        
    def forward(self, m,x,y):
        # print(self.model_x.forward(x))
        z_x = self.model_x.forward(x)
        log_det_x = self.model_x.forward_kld(x)
        z_y = self.model_y.forward(y)
        log_det_y = self.model_y.forward_kld(y)
        z_m = self.model_m.forward(m)
        log_det_m = self.model_m.forward_kld(m)
        # print(log_det_x, log_det_y, log_det_m)
        return z_m, z_x, z_y, log_det_x + log_det_y + log_det_m
    
    def compute_loss(self, z_m, z_x, z_y, log_det):
        z_combined = torch.cat([z_m, z_x, z_y], dim=-1)
        target = torch.distributions.MultivariateNormal(
            loc=self.mean,
            covariance_matrix=self.get_covariance()
        )
        log_prob = target.log_prob(z_combined)
        return -(log_prob + log_det).mean()