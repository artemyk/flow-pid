import numpy as np
import pandas as pd

from pid.optimizers import exact_gauss_tilde_pid, mmi_pid, exact_gauss_thin_pid


if __name__ == '__main__':
    dm, dx, dy = 1, 1, 1

    # -------------------------------------------------------------------------
    # UI_X + RI

    pid_table = pd.DataFrame()

    pid_defn_names = ['thin', 'tilde', 'mmi']
    pid_defns = [exact_gauss_thin_pid, exact_gauss_tilde_pid, mmi_pid]

    config_cols = ['desc', 'id', 'dm', 'dx', 'dy', 'sigma_y__x', 'rho']
    pid_cols = ['imxy', 'uix', 'uiy', 'ri', 'si']

    sigma2_vals = np.r_[0, np.logspace(0, 2, 10)]
    sigm = np.eye(dm)
    hx = np.array([[1]])
    hyx = np.array([[1]])
    hy = hyx @ hx
    sigx_m = 1 * np.eye(dx)
    covx = hx @ sigm @ hx.T + sigx_m

    for i, sigma2 in enumerate(sigma2_vals):
        print(i, end=' ')

        sigy_x = sigma2 * np.eye(dx)
        cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                        [hx @ sigm, covx, covx @ hyx.T],
                        [hy @ sigm, hyx @ covx, hyx @ covx @ hyx.T + sigy_x]])

        cols = [(col, '') for col in config_cols]
        vals = ['uix+ri', i, dm, dx, dy, sigy_x[0, 0], None]

        for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
            ret = pid_defn(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

            if pid_name == 'mmi':
                # Ground truth values are equal to MMI values
                cols.extend([('gt', col) for col in pid_cols])
                vals.extend([imxy, uix, uiy, ri, si])

        row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
        pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    # -------------------------------------------------------------------------
    # UI_X + SI

    sigm = np.eye(dm)
    hx = np.array([[1]])
    hy = np.array([[0]])
    sigx_m = np.eye(dx)
    sigy_m = np.eye(dy)

    rho_vals = np.r_[np.linspace(0, 1, 10, endpoint=False), 0.99]

    for i, rho in enumerate(rho_vals):
        print(i, end=' ')

        sigw = np.array([[rho]])
        cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                        [hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
                        [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])

        cols = [(col, '') for col in config_cols]
        vals = ['uix+si', i, dm, dx, dy, None, rho]

        for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
            ret = pid_defn(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

            if pid_name == 'mmi':
                # Ground truth values are equal to MMI values
                cols.extend([('gt', col) for col in pid_cols])
                vals.extend([imxy, uix, uiy, ri, si])

        row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
        pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    # -------------------------------------------------------------------------
    # RI + SI

    sigm = np.eye(dm)
    hx = np.array([[1]])
    hy = np.array([[1]])
    sigx_m = np.eye(dx)
    sigy_m = np.eye(dy)

    rho_vals = np.r_[np.linspace(0, 1, 10, endpoint=False), 0.99]

    for i, rho in enumerate(rho_vals):
        print(i, end=' ')

        sigw = np.array([[rho]])
        cov = np.block([[sigm, sigm @ hx.T, sigm @ hy.T],
                        [hx @ sigm, hx @ sigm @ hx.T + sigx_m, hx @ sigm @ hy.T + sigw],
                        [hy @ sigm, hy @ sigm @ hx.T + sigw.T, hy @ sigm @ hy.T + sigy_m]])

        cols = [(col, '') for col in config_cols]
        vals = ['ri+si', i, dm, dx, dy, None, rho]

        for pid_name, pid_defn in zip(pid_defn_names, pid_defns):
            ret = pid_defn(cov, dm, dx, dy)
            imxy, uix, uiy, ri, si = ret[2], *ret[-4:]

            cols.extend([(pid_name, col) for col in pid_cols])
            vals.extend([imxy, uix, uiy, ri, si])

            if pid_name == 'mmi':
                # Ground truth values are equal to MMI values
                cols.extend([('gt', col) for col in pid_cols])
                vals.extend([imxy, uix, uiy, ri, si])

        row = pd.DataFrame({col: [val] for col, val in zip(cols, vals)})
        pid_table = pd.concat((pid_table, row), ignore_index=True)

    print()

    pid_table.to_pickle('../results/canonical_exs.pkl.gz')

    print(pid_table)
    assert np.allclose((pid_table['thin'] - pid_table['gt']).dropna().to_numpy(), 0, atol=1e-5)
