import numpy as np
import pandas as pd
from utils.generate import generate_cov_from_config, random_rotation_mxy, merge_covs
from pid import exact_gauss_tilde_pid, exact_gauss_thin_pid, flow_pid


if __name__ == '__main__':
    # Starting from 2-D case
    dm, dx, dy = 2, 2, 2
    theta = 0
    gain_y = np.sqrt(2)
    gains = np.linspace(0.5, 3, 10)
    gains[3] = 0.99  # A gain of 1 is unstable

    num_doubles = 4
    random_rotn = True

    # for simplicity, only print gains[0]
    gain = gains[0]

    for j in range(num_doubles):
        if j == 0:
            dm, dx, dy = 2, 2, 2
            hx = np.array([[gain, 0], [0, 1]])
            hy = np.array([[1, 0], [0, np.sqrt(2)]])
            sigm = np.eye(dm)
            sigx_m = np.eye(dx)
            sigy_m = np.eye(dy)
            sigw = np.zeros((dx, dy))
            cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                            [hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
                            [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])
            cov_old = cov.copy()
        else:
            cov, dm, dx, dy = merge_covs(cov_old, cov_old.copy(), dm, dx, dy)
            cov_old = cov.copy()

        if random_rotn:
            cov = random_rotation_mxy(cov, dm, dx, dy)

        num_samples = 1000  # Number of samples to generate
        mean = np.zeros(sum([dm, dx, dy]))
        samples = np.random.multivariate_normal(mean, cov, num_samples)
        print(f'j: {j}, samples.shape: {samples.shape} \n')

        m = samples[:, :dm]
        x = samples[:, dm:dm + dx]
        y = samples[:, dm + dx:]
        mxy = np.vstack((m.T, x.T, y.T))
        cov = np.corrcoef(mxy)  # Shape: (dm+dx+dy, dm+dx+dy)
        print(f'j: {j}, cov.shape: {cov.shape} \n')

        ret = exact_gauss_tilde_pid(cov, dm, dx, dy)
        print(f'tilde pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')

        ret = exact_gauss_thin_pid(cov, dm, dx, dy)
        print(f'thin pid: {ret[7]}, {ret[5]}, {ret[6]}, {ret[8]} \n')

        dm = dx = dy = 2 ** (j + 1)