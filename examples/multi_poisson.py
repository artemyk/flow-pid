import numpy as np
from tqdm import tqdm
import scipy.linalg as la
import numpy.linalg as npla

import torch
import normflows as nf

import pid
import pid.utils.distributions as dist
from pid.utils.generate import sample_mult_poisson


def get_params():
    lamda_m = np.array([2., 2.])
    lamda_x = np.array([1.,])      # Noise to be added to X: shape (d_X,)
    lamda_y = np.array([1.,])

    w_x1_vals = np.arange(11) / 10
    w_x2 = 0.5
    w_y1 = 0.5
    w_y2 = 0.5

    return lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2


def pid_flows(base, n_flows, latent_dim, flow_type='Spline'):
    flows = []
    for i in range(n_flows):
        if flow_type == 'Spline':
            flows += [nf.flows.AutoregressiveRationalQuadraticSpline(latent_dim, 2, 128)]
            flows += [nf.flows.LULinearPermute(latent_dim)]
        elif flow_type == 'Planar':
            flows += [nf.flows.Planar((latent_dim,))]
        elif flow_type == 'Radial':
            flows += [nf.flows.Radial((latent_dim,))]
        else:
            raise NotImplementedError

    # Construct flow model
    model = nf.NormalizingFlow(base, flows)
    return model


def objective(sig, hx, hy, dm, dx, dy, reg):
    S = (1 + reg) * np.eye(dx) - sig @ sig.T
    B = hx - sig @ hy
    obj = 0.5 / np.log(2) * npla.slogdet(
        np.eye(dm) + hy.T @ hy + B.T @ la.solve(S, B)
    )[1]
    return obj


def main(num_samples=512, flow_iter=4000, n_flows=16, enable_cuda=True):
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2 = get_params()
    dm = lamda_m.size  # Number of dimensions of M
    dx = lamda_x.size  # Number of dimensions of X
    dy = lamda_y.size  # Number of dimensions of Y

    ######### Prepare the pid flows #########
    base_x = dist.GaussianPCA(dim=dx, latent_dim=dm, sigma=1.)
    base_y = dist.GaussianPCA(dim=dy, latent_dim=dm, sigma=1.)
    base_m = dist.DiagGaussian(dm, trainable=False)

    flow_x = pid_flows(base_x, n_flows, dx).to(device)
    flow_y = pid_flows(base_y, n_flows, dy).to(device)
    flow_m = pid_flows(base_m, n_flows, dm).to(device)

    for i, w_x1 in enumerate(w_x1_vals):
        w_x = np.array([[w_x1, w_x2]])  # Binomial thinning weights: shape (dx, dm)
        w_y = np.array([[w_y1, w_y2]])

        m, x, y = sample_mult_poisson(num_samples, lamda_m, w_x, w_y, lamda_x, lamda_y)  # Shape: (d, n)

        ######### Train pid flows #########
        optimizer_x = torch.optim.Adam(flow_x.parameters(), lr=5e-4, weight_decay=1e-5)
        optimizer_y = torch.optim.Adam(flow_y.parameters(), lr=5e-4, weight_decay=1e-5)
        optimizer_m = torch.optim.Adam(flow_m.parameters(), lr=5e-4, weight_decay=1e-5)

        for it in tqdm(range(flow_iter)):
            flow_x.train(), flow_y.train(), flow_m.train()
            optimizer_x.zero_grad(), optimizer_y.zero_grad(), optimizer_m.zero_grad()

            x_ = torch.from_numpy(x.T).float().to(device)
            y_ = torch.from_numpy(y.T).float().to(device)
            m_ = torch.from_numpy(m.T).float().to(device)

            loss_m = flow_m.forward_kld(m_)
            loss_x = flow_x.forward_kld(x_)
            loss_y = flow_y.forward_kld(y_)

            # Do backprop and optimizer step
            if ~(torch.isnan(loss_x) | torch.isinf(loss_x)):
                loss_x.backward()
                optimizer_x.step()
            if ~(torch.isnan(loss_y) | torch.isinf(loss_y)):
                loss_y.backward()
                optimizer_y.step()
            if ~(torch.isnan(loss_m) | torch.isinf(loss_m)):
                loss_m.backward()
                optimizer_m.step()

        ######### Get params #########
        hx = flow_x.q0.get_W(detach=True).cpu().numpy()
        hy = flow_y.q0.get_W(detach=True).cpu().numpy()

        sig, obj, _ = pid.optimizers.thinpid_exact_pid_minimizer(hx, hy, ret_obj=True, reg=1e-7)
        union_info = objective(sig, hx, hy, dm, dx, dy, reg=1e-7)

        print(union_info)


if __name__ == '__main__':
    main(num_samples=512, flow_iter=4000, n_flows=16, enable_cuda=True)



