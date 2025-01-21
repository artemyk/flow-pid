import time
import numpy as np
import numpy.ma as ma
from admUI import admUI_numpy
import torch
import normflows as nf

from pid.models import CartesianFlow
from pid.optimizers import exact_gauss_tilde_pid, mmi_pid, exact_gauss_thin_pid, flow_pid
from pid.utils.estimate import approx_pid_from_cov
from pid.utils.generate import sample_mult_poisson
from pid.utils.distributions import mult_poisson_dist


def get_params():
    lamda_m = np.array([2., 2.])
    lamda_x = np.array([1.])      # Noise to be added to X: shape (d_X,)
    lamda_y = np.array([1.])

    w_x1_vals = np.arange(11) / 10
    w_x2 = 0.5
    w_y1 = 0.5
    w_y2 = 0.5

    return lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2



if __name__ == '__main__':
    enable_cuda = False
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    np.set_printoptions(precision=4, linewidth=200)

    lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2 = get_params()
    dm = lamda_m.size  # Number of dimensions of M
    dx = lamda_x.size
    dy = lamda_y.size

    n = 10000  # Sample size

    log_f_name = './results/multi_poisson_sample' + str(n) + '.csv'
    print("logging at : " + log_f_name)
    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('id,w1,r,ux,uy,si,time\n')

    ## 0: tilde, 1: delta, 2: mmi, 3: flow, 4: gt
    pid_ids = [0, 1, 2]
    pid_bench_names = ['tilde', 'delta', 'mmi']
    pid_benchmarks = [exact_gauss_tilde_pid, approx_pid_from_cov, mmi_pid]

    for i, w_x1 in enumerate(w_x1_vals):
        print(f"\nw_x1: {w_x1}")
        w_x = np.array([[w_x1, w_x2]])  # Binomial thinning weights: shape (dx, dm)
        w_y = np.array([[w_y1, w_y2]])

        m, x, y = sample_mult_poisson(n, lamda_m, w_x, w_y, lamda_x, lamda_y)  # Shape: (d, n)
        mxy = np.vstack((m, x, y))
        cov = np.corrcoef(mxy)  # Shape: (dm+dx+dy, dm+dx+dy)

        ## tilde, delta, mmi pid
        for pid_id, pid_name, pid_defn in zip(pid_ids, pid_bench_names, pid_benchmarks):

            start_time = time.time()
            ret = pid_defn(cov, dm, dx, dy)  # pid benchmarks
            end_time = time.time()
            batch_time = end_time - start_time

            norm = ret[7] + ret[5] + ret[6] + ret[8]
            r, ux, uy, si = ret[5]/norm, ret[6]/norm, ret[7]/norm, ret[8]/norm

            log_f.write('{},{},{},{},{},{},{}\n'.format(pid_id, w_x1, r, ux, uy, si, batch_time))
            log_f.flush()
            print(
                f"pid name: {pid_name}, normalized I_mxy, "
                f"R: {r}, UX: {ux}, UY: {uy}, S: {si}"
            )

        start_time = time.time()
        ret = flow_pid(m.T, x.T, y.T, n_flows=3, n_epochs=250, batch_size=1000, lr=1e-3, verbose=False, device=device)
        end_time = time.time()
        batch_time = end_time - start_time

        norm = ret[7] + ret[5] + ret[6] + ret[8]
        r, ux, uy, si = ret[5] / norm, ret[6] / norm, ret[7] / norm, ret[8] / norm

        log_f.write('{},{},{},{},{},{},{}\n'.format(3, w_x1, r, ux, uy, si, batch_time))
        log_f.flush()
        print(
            f"flow pid, normalized I_mxy, R: {r}, UX: {ux}, UY: {uy}, S: {si}")