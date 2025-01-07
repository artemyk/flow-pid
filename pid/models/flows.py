import normflows as nf
from ..utils.distributions import DiagGaussian, GaussianPCA


def gauss_pid_flows(dm, dx, dy, flows):
    base_m = DiagGaussian(dm, trainable=False)
    base_x = GaussianPCA(dim=dx, latent_dim=dm, sigma=1.)
    base_y = GaussianPCA(dim=dy, latent_dim=dm, sigma=1.)

    flow_m = nf.NormalizingFlow(base_m, flows)
    flow_x = nf.NormalizingFlow(base_x, flows)
    flow_y = nf.NormalizingFlow(base_y, flows)

    return flow_m, flow_x, flow_y