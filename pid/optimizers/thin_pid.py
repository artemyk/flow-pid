# Tian Nov. 2024.
# Thin PID core algorithm
# import os
# os.environ['OPENBLAS_NUM_THREADS'] = '1'

import warnings
import numpy as np
import scipy.linalg as la
import numpy.linalg as npla

from ..utils import whiten


# Suppress the specific warning: linAlgWarning: Ill-conditioned matrix since we can just resolve it or approximate it
# warnings.filterwarnings("ignore", category=la.LinAlgWarning)

def objective(sig, G):  # the objective function
    dx, dy = sig.shape
    sig_all = np.block([[np.eye(dx), sig],
                        [sig.T, np.eye(dy)]])
    obj = 0.5 / np.log(2) * (npla.slogdet(G + sig_all)[1] - npla.slogdet(sig_all)[1])
    return obj


def project(sig_temp):  # project the matrix onto the PSD cone, but only need to work with the upper triangular part

    U, S, VT = npla.svd(sig_temp, full_matrices=False)
    S_clamped = np.clip(S, 0,
                        0.99999999999999999)  # instead of clamping to [0,1], we clamp to [1e-10,0.99999999999999999] to avoid numerical issues
    S_clamped_matrix = np.diag(S_clamped)
    sig_modified = U @ S_clamped_matrix @ VT

    return sig_modified


def thinpid_exact_pid_minimizer(hx, hy, ret_obj=False, reg=1e-7, max_iters=20000):
    dx, dT = hx.shape
    dy, dT_ = hy.shape
    if dT != dT_:
        raise ValueError('Incompatible shapes for Hx and Hy')

    if dx < dy:  # Swap if necessary, since we assume dx >= dy
        dx, dy = dy, dx

    # Gradient descent
    eta_sig = 1e-3 * np.ones((dx, dy))
    beta = 0.9  # Factor to increase or decrease LR for Rprop
    alpha = 0.999  # Slow decay of overall learning rate

    stop_threshold = reg
    max_iterations = max_iters  # Maximum number of iterations
    patience = 20  # Num iters with small gradient before stopping (min=1)
    extra_iters = 0  # Num of extra iters after stop criterion is attained

    H = np.concatenate((hx, hy), axis=0)
    G = H @ H.T
    G_11 = G[0:dx, 0:dx]
    G_12 = G[0:dx, dx:dx + dy]
    G_22 = G[dx:dx + dy, dx:dx + dy]

    minima = None
    g_sig_prev = None
    running_obj = []
    i = 1
    extra = 0

    GI_11_inv = npla.inv(G_11 + np.eye(dx))
    GI_22 = G_22 + np.eye(dy)
    sig = np.zeros((dx, dy))

    while True:
        # Evaluate the objective
        obj = objective(sig, G)

        if minima is None or obj < min(running_obj):
            minima = (sig.copy(), obj)

        if len(running_obj) >= patience:
            if extra == 0:
                if (np.abs(np.array(running_obj[-patience:]) - obj) < stop_threshold).all() or i >= max_iterations:
                    if i >= max_iterations:
                        warnings.warn('Exceeded maximum number of iterations. May not have converged.')
                    if extra_iters == 0: break
                    extra += 1
            elif extra > extra_iters:
                break
            else:
                extra += 1

        if np.isnan(obj):
            running_obj.append(np.inf)
        else:
            running_obj.append(obj)
        i += 1

        # Matrix computation intermediate steps to compute the gradient
        safty = 1e-8
        R = (G_12 + sig)
        B = GI_11_inv @ R
        A = GI_22 - R.T @ B
        A1 = np.eye(dy) - sig.T @ sig

        # need to invert the matrices, but the result is occasionally unstable
        try:
            sol_AB = npla.solve(A, B.T)
        except npla.LinAlgError:
            A = A + safty * np.eye(dy)
            sol_AB = npla.solve(A, B.T)
        try:
            sol_AB1 = npla.solve(A1, sig.T)
        except npla.LinAlgError:
            A1 = A1 + safty * np.eye(dy)
            sol_AB = npla.solve(A1, sig.T)

        g_sig = -sol_AB + sol_AB1
        g_sig = g_sig.T
        # take only the signs for Rprop
        g_sig = np.sign(g_sig).astype(int)
        # gradient descent
        sig_plus = sig - alpha ** i * eta_sig * g_sig
        # project sig back onto the PSD cone, but only need to work with the upper triangular part
        sig_proj = project(sig_plus)

        # Learning rate update
        if g_sig_prev is not None:
            sign_changed = - g_sig * g_sig_prev  # -1 if sign did not change, +1 if sign changed
            eta_sig *= beta ** sign_changed

        g_sig_prev = g_sig
        sig[:, :] = sig_proj

    sig, obj = minima

    if ret_obj:
        return sig, obj, i
    return sig


def bias(d, n):
    return sum(np.log(1 - k / n) for k in range(1, d+1)) / np.log(2) / 2


def compute_bias(du, dv, n):
    """
    Compute the bias in the mutual information estimate based on the work of
    Cai et al. (J. Mult. Anal., 2015).

    This value needs to be subtracted from the mutual information estimate to
    recover the unbiased mutual information.
    """
    # Bias of differential entropy estimate
    return bias(du, n) + bias(dv, n) - bias(du + dv, n)


def debias(imxy, bias_):
    """Remove bias while ensuring non-negativity."""

    return np.maximum(imxy - bias_, 0)


def exact_gauss_thin_pid(cov, dm, dx, dy, verbose=False, ret_t_sigt=False,
                          plot=False, unbiased=False, sample_size=None):

    # XXX: Debiasing has not been thoroughly tested.
    # Right now, we assume that the proportion of bias in the union information
    # is the same as the proportion of bias in I(M ; (X, Y)).

    # Regularization
    reg = 1e-7

    if unbiased == True and sample_size is None:
        raise ValueError('Must supply sample_size when requesting unbiased estimates')

    ret = whiten(cov, dm, dx, dy, ret_channel_params=True)
    sig_mxy, hx, hy, hxy, sigxy = ret

    imx = 0.5 * npla.slogdet(np.eye(dm) + hx.T @ hx)[1] / np.log(2)
    imy = 0.5 * npla.slogdet(np.eye(dm) + hy.T @ hy)[1] / np.log(2)
    imxy = 0.5 * npla.slogdet(np.eye(dm) + hxy.T @ la.solve(sigxy + reg * np.eye(*sigxy.shape), hxy))[1] / np.log(2)

    if unbiased:
        imx = debias(imx, compute_bias(dm, dx, sample_size))
        imy = debias(imy, compute_bias(dm, dy, sample_size))
        imxy_debiased = debias(imxy, compute_bias(dm, dx + dy, sample_size))

        # But ensure that the debiased imxy does not go below the debiased imx
        # or the debiased imy, as this will make PID values negative
        imxy_debiased = max(imxy_debiased, imx, imy)
    else:
        imxy_debiased = imxy

    debias_factor = imxy_debiased / imxy

    #sig = exact_tilde_union_info_minimizer(hx, hy, plot=plot)
    sig, obj, _ = thinpid_exact_pid_minimizer(hx, hy, plot=plot, ret_obj=True, reg=reg)
    covxy__m = np.block([[np.eye(dx), sig], [sig.T, np.eye(dy)]])
    #covxy = covxy__m + np.vstack((hx, hy)) @ np.vstack((hx, hy)).T

    #union_info = 0.5 / np.log(2) * npla.slogdet(
    #    np.eye(dm) + hxy.T @ la.solve(covxy__m + 1e-7 * np.eye(*covxy__m.shape), hxy))[1]
    #union_info = obj
    union_info = objective(sig, hx, hy, dm, dx, dy, reg=reg)

    union_info *= debias_factor

    # Union info is lower bounded by max{I(M; X), I(M; Y)} and upper bounded by
    # min{I(M; X) + I(M; Y), I(M; (X, Y))}: imposing this ensures positivity of
    # the PID terms
    union_info = max(union_info, imx, imy)
    union_info = min(union_info, imx + imy, imxy_debiased)

    uix = union_info - imy
    uiy = union_info - imx
    ri = imx + imy - union_info
    si = imxy_debiased - union_info

    #uix = (union_info - imy) * debias_factor
    #uiy = (union_info - imx) * debias_factor
    #ri = (imx + imy - union_info) * debias_factor
    #si = (imxy - union_info) * debias_factor

    # Return union_info and None in place of deficiency values to keep return signature consistent
    ret = (imx, imy, imxy_debiased, union_info, obj, uix, uiy, ri, si)
    if ret_t_sigt:
        ret = (*ret, None, None, None, sig)

    return ret