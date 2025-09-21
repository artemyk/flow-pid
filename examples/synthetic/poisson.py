import time
import numpy as np
import numpy.ma as ma
from admUI import admUI_numpy
import torch
import normflows as nf

from pid.models import CartesianProductFlow
from pid.optimizers import exact_gauss_tilde_pid, mmi_pid, flow_pid
from pid.utils.estimate import approx_pid_from_cov
from pid.utils.generate import sample_mult_poisson
from pid.utils.distributions import mult_poisson_dist

from torch.utils.data import TensorDataset, DataLoader

MASK_THRESH = 1e-8


def get_params():
    lamda_m = np.array([2., 2.])
    lamda_x = np.array([1.])      # Noise to be added to X: shape (d_X,)
    lamda_y = np.array([1.])

    w_x1_vals = np.arange(11) / 10
    w_x2 = 0.5
    w_y1 = 0.5
    w_y2 = 0.5

    return lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2


def pid(p, qstar, ind=1):
    """
    Computes the unique, redundant, synergistic and total information about M
    present in X_`ind` with respect to the other variables, given the optimal
    `q_star` from Bertschinger's definition.

    Here, `p` is the original joint distribution of all variables, where the
    first dimension of p represents M, and the remaining dimensions represent
    X_i, the variables being used in the decomposition.

    `q_star` may be computed using one of the two functions below,
    `compute_qstar` or `compute_qstar_admui`.

    Note: X_i's are 1-indexed, so X_1 is the first X variable. Equivalently,
          `ind` is the index of X_`ind` in *p*.
    """

    # Roll axes so that the variable of interest is always at index 1
    qstar = np.rollaxis(qstar, ind, 1)
    p = np.rollaxis(p, ind, 1)
    # Mask out near-zero values to avoid divide-by-zero and log errors
    qstar = ma.array(qstar, mask=(abs(qstar) < MASK_THRESH))
    p = ma.array(p, mask=(abs(p) < MASK_THRESH))

    # First, compute the unique information
    axes = (0, 1)
    qrest = ma.sum(qstar, axis=axes, keepdims=True)  # q*(x2)
    qstar_rest = qstar / qrest                       # q*(m, x1 | x2)

    qmrest = ma.sum(qstar, axis=1, keepdims=True)    # q*(m, x2)
    qm_rest = qmrest / qrest                         # q*(m | x2)

    qrestx = ma.sum(qstar, axis=0, keepdims=True)    # q*(x1, x2)
    qx_rest = qrestx / qrest                         # q*(x1 | x2)

    uniq_info = ma.sum(qstar * ma.log2(qstar_rest / (qm_rest * qx_rest)))

    # Then, compute redundant information
    axes = tuple(range(2, qstar.ndim))
    qmx = ma.sum(qstar, axis=axes, keepdims=True)    # q*(m, x1)
    axes = tuple(range(1, qstar.ndim))
    qm = ma.sum(qstar, axis=axes, keepdims=True)     # q*(m)
    axes = list(range(qstar.ndim))
    axes.remove(1)
    axes = tuple(axes)
    qx = ma.sum(qstar, axis=axes, keepdims=True)     # q*(x1)
    Imx = ma.sum(qmx * ma.log2(qmx / (qm * qx)))     # I(M ; X1)
    red_info = Imx - uniq_info

    # Alternate way to compute redundant info from p
    #axes = tuple(range(1, p.ndim))
    #pm = ma.sum(p, axis=axes, keepdims=True)
    #axes = tuple(range(2, p.ndim))
    #pmx = ma.sum(p, axis=axes, keepdims=True)
    #axes = list(range(p.ndim))
    #axes.remove(1)
    #axes = tuple(axes)
    #px = ma.sum(p, axis=axes, keepdims=True)
    #Imx = ma.sum(pmx * ma.log2(pmx / (pm * px)))
    #red_info = Imx - uniq_info

    # Finally, synergistic information
    axes = tuple(range(1, p.ndim))
    pm = ma.sum(p, axis=axes, keepdims=True)         # p(m)
    pall = ma.sum(p, axis=0, keepdims=True)          # p(x1, x2)
    Imall_p = ma.sum(p * ma.log2(p / (pm * pall)))   # I_p(M ; (X1, X2))
    Imall_q = ma.sum(qstar * ma.log2(qstar / (qm * qrestx))) # I_q(M ; (X1, X2))
    syn_info = Imall_p - Imall_q

    return (uniq_info, red_info, syn_info, Imall_p)


def compute_qstar_admui(p, M_dim, lamdas, ind=1, maxiter=250, verbose=False):
    """
    Compute the optimal distribution for the bivariate PID of information about
    M contained between X_ind and the other variables, using the iterative
    algorithm given by Banerjee et al. (ISIT 2018).

    Parameters:
        p - np.ndarry
            Joint distribution of (M, X1, X2, ...), each dimension of p
            represents these variables in order
        M_dim - int
                Should be equal to p.shape[0]
        lamdas - not used
        ind - int
              Index to be used as X1 (remaining indices are treated together
              as X2)
        verbose - not used
    """

    p = np.rollaxis(p, ind, 1)
    newshape = p.shape
    p = p.reshape((M_dim, p.shape[1], -1))  # Reshape into a bivariate PID

    pm = p.sum(axis=(1, 2))
    px_m = p.sum(axis=2).T / pm  # p(x1 | m)
    py_m = p.sum(axis=1).T / pm  # p(y | m) - here, `y` is the same as x2

    qstar = admUI_numpy.computeQUI_numpy(px_m, py_m, pm, maxiter=maxiter)
    # In the above, `maxiter` may need to be tuned based on performance

    qstar = qstar.reshape(newshape)  # Reshape back into the original shape
    qstar = np.rollaxis(qstar, 1, ind+1)  # And place X1 in its original index
    return qstar


def standardize_data(data):
    # standardize data along columns
    return (data - data.mean(dim=0, keepdim=True)) / data.std(dim=0, keepdim=True)


if __name__ == '__main__':
    enable_cuda = True
    device = torch.device('cuda' if torch.cuda.is_available() and enable_cuda else 'cpu')
    print(f"Using device: {device}")

    np.set_printoptions(precision=4, linewidth=200)

    lamda_m, lamda_x, lamda_y, w_x1_vals, w_x2, w_y1, w_y2 = get_params()
    dm = lamda_m.size  # Number of dimensions of M
    dx = lamda_x.size
    dy = lamda_y.size

    n = 1000000  # Sample size
    D = 10  # supports of distribution in each dimension

    log_f_name = '../results/multi_poisson_' + str(n) + '.csv'
    print("logging at : " + log_f_name)
    # logging file
    log_f = open(log_f_name, "w+")
    log_f.write('w1,id,ux,uy,ri,si\n')

    ## 0: flow, 1: tilde, 2: mmi, 3: gt
    pid_ids = [1, 2]
    pid_bench_names = ['tilde', 'delta']
    pid_benchmarks = [exact_gauss_tilde_pid, approx_pid_from_cov, mmi_pid]

    for i, w_x1 in enumerate(w_x1_vals):
        # if not i == 2:
        #     continue  # skip the loop, for step-by-step debugging only

        print(f"\nw_x1: {w_x1}")
        w_x = np.array([[w_x1, w_x2]])  # Binomial thinning weights: shape (dx, dm)
        w_y = np.array([[w_y1, w_y2]])

        m, x, y = sample_mult_poisson(n, lamda_m, w_x, w_y, lamda_x, lamda_y)  # Shape: (d, n)
        mxy = np.vstack((m, x, y))
        cov = np.corrcoef(mxy)  # Shape: (dm+dx+dy, dm+dx+dy)

        ## flow pid
        m_standardized = standardize_data(torch.tensor(m.T, dtype=torch.float32))
        x_standardized = standardize_data(torch.tensor(x.T, dtype=torch.float32))
        y_standardized = standardize_data(torch.tensor(y.T, dtype=torch.float32))
        dataset = TensorDataset(x_standardized, y_standardized, m_standardized)
        dataloader = DataLoader(dataset, batch_size=512, shuffle=True)

        model = CartesianProductFlow(
            dm=dm,
            dx=dx,
            dy=dy,
            n_flows=6,
        ).to(device)

        ret = flow_pid.fit(model, dataloader, n_epochs=5000, lr=5e-4, device=device, verbose=False)
        ux, uy, r, si = ret[5], ret[6], ret[7], ret[8]

        log_f.write('{},{},{},{},{},{}\n'.format(w_x1, 0, ux, uy, r, si))
        log_f.flush()
        print(
            f"pid name: flow-pid, "
            f"R: {r}, UX: {ux}, UY: {uy}, S: {si}"
        )

        ## tilde, delta, mmi pid
        for pid_id, pid_name, pid_defn in zip(pid_ids, pid_bench_names, pid_benchmarks):

            ret = pid_defn(cov, dm, dx, dy)  # pid benchmarks
            ux, uy, r, si = ret[5], ret[6], ret[7], ret[8]

            log_f.write('{},{},{},{},{},{}\n'.format(w_x1, pid_id, ux, uy, r, si))
            log_f.flush()
            print(
                f"pid name: {pid_name}, "
                f"R: {r}, UX: {ux}, UY: {uy}, S: {si}"
            )

        ## ground truth (time-consuming) ##
        gt = True
        if gt:
            p = mult_poisson_dist(lamda_m, w_x, w_y, lamda_x, lamda_y, D=D)
            D = p.shape[0]
            # Compute Bertschinger's PID using the Banerjee et al. package
            p = p.reshape((D ** 2, D, D))
            qstar = compute_qstar_admui(p, p.shape[0], [], maxiter=500)
            uix, ri, si, imxy = pid(p, qstar)  # UI corresponds to X
            uiy = imxy - uix - ri - si

            log_f.write('{},{},{},{},{},{}\n'.format(w_x1, 3, uix, uiy, ri, si))
            log_f.flush()
            print(f"ground truth: R: {ri}, UX: {uix}, UY: {uiy}, S: {si}")

    log_f.close()
    print("done")