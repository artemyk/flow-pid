import numpy as np
import pandas as pd
from pid.utils import generate_randomBC
from pid.optimizers.tilde_pid import exact_tilde_union_info_minimizer
from pid.optimizers.thin_pid import exact_thin_pid_minimizer

import matplotlib.pyplot as plt


if __name__ == '__main__':
    suptitlesize = 20
    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12

    step = 100
    start = 100
    max_dX = 1000
    num_runs = 1
    ratio = [0.6, 0.1]

    ## put the above two figures into one figure
    fig, axs = plt.subplots(1, 2, figsize=(6, 5), sharex=False, sharey=False)
    # fig.suptitle(r'Execution Time of G-PID', fontsize=titlesize)

    # Generate dx, dy, dT sequence
    R = np.array([1, ratio[0]])
    sequence = np.arange(start, max_dX + 1, step)
    XY_dim = np.array([seq * R for seq in sequence]).astype(int)  # TXY_dim = sequence * R = [dT, dx, dy]

    dx = XY_dim[:, 0]

    time_tilde_0 = np.array(
        [0.3564, 3.4241, 11.2748, 15.0652, 35.8061, 46.9894, 83.7819, 344.7741])
    time_thin_0 = np.array(
        [0.1847, 1.2717, 3.3798, 5.7316, 13.2286, 20.3649, 47.2469, 67.3551,
         115.1058, 256.9196])

    time_tilde_1 = np.array(
        [0.9929, 2.6590, 5.5786, 10.2216, 14.4565, 21.6659, 29.2177, 35.4743,
         43.6238, 50.8118])
    time_thin_1 = np.array(
        [0.1552, 0.2442, 0.5660, 0.9463, 1.2553, 2.1821, 2.7512, 3.5437,
         3.9378, 5.0572])


    ax = axs[0]
    ax.plot(dx, time_thin_0, marker='s', label='Thin PID', color='C0', linewidth=3)
    ax.plot(dx[:8], time_tilde_0, marker='o', label='Tilde PID', color='C1', linewidth=3)
    ax.set_xlabel('$dX1=dX2$', fontsize=labelsize)
    ax.set_ylabel('Execution Time (s)', fontsize=labelsize)
    # ax.set_title('Execution Time of G-PID', fontsize=titlesize)
    ax.tick_params(axis='x', rotation=45, labelsize=ticksize)
    ax.grid(True)
    ax.legend(fontsize=legendsize)

    ax = axs[1]
    ax.plot(dx, time_thin_1, marker='s', label='Thin PID', color='C0', linewidth=3)
    ax.plot(dx, time_tilde_1, marker='o', label='Tilde PID', color='C1', linewidth=3)
    ax.set_xlabel('$dX1$ ($dX2=100$)', fontsize=labelsize)
    # ax.set_ylabel('Execution Time (s)', fontsize=labelsize)
    # ax.set_title('Execution Time of G-PID', fontsize=titlesize)
    ax.tick_params(axis='x', rotation=45, labelsize=ticksize)
    ax.grid(True)
    # ax.legend(fontsize=legendsize)

    plt.tight_layout()
    plt.savefig('../plots/gpid_runtime_2.pdf')




    # ## get convergence
    # hx, hy = generate_randomBC(2, 500, 500)
    # _, _, _, thin_hist = exact_thin_pid_minimizer(hx, hy, ret_obj=True, reg=1e-8, verbose=True)
    # print(f"thin_hist: {thin_hist}")
    # _, _, _, tilde_hist = exact_tilde_union_info_minimizer(hx, hy, ret_obj=True, reg=1e-8, verbose=True)
    # print(f"tilde_hist: {tilde_hist}")
    #
    # ax = axs[0]
    # ax.plot(np.log(thin_hist), label='Thin PID', color='C0', linewidth=3)
    # ax.plot(np.log(tilde_hist), label='Tilde PID', color='C1', linewidth=3)
    # ax.set_xlabel('Steps', fontsize=labelsize)
    # ax.set_ylabel('Objective', fontsize=labelsize)
    # # ax.set_title('Execution Time of G-PID', fontsize=titlesize)
    # ax.tick_params(axis='x', rotation=45, labelsize=ticksize)
    # ax.grid(True)
    # ax.legend(fontsize=legendsize)
    #
    # hx, hy = generate_randomBC(2, 500, 500)
    # _, _, _, thin_hist = exact_thin_pid_minimizer(hx, hy, ret_obj=True, reg=1e-8, verbose=True)
    # print(f"thin_hist: {thin_hist}")
    # _, _, _, tilde_hist = exact_tilde_union_info_minimizer(hx, hy, ret_obj=True, reg=1e-8, verbose=True)
    # print(f"tilde_hist: {tilde_hist}")
    #
    # ax = axs[0]
    # ax.plot(np.log(thin_hist), label='Thin PID', color='C0', linewidth=3)
    # ax.plot(np.log(tilde_hist), label='Tilde PID', color='C1', linewidth=3)
    # ax.set_xlabel('Steps', fontsize=labelsize)
    # ax.set_ylabel('Objective', fontsize=labelsize)
    # # ax.set_title('Execution Time of G-PID', fontsize=titlesize)
    # ax.tick_params(axis='x', rotation=45, labelsize=ticksize)
    # ax.grid(True)
    # ax.legend(fontsize=legendsize)


# Running config [dx, dy, dT]: 100-100-  2
# Average obj_tilde: 6.763247778, obj_thin: 6.763247785, time_tilde: 0.3564, time_thin: 0.1847

# Running config [dx, dy, dT]: 200-200-  2
# Average obj_tilde: 7.765792798, obj_thin: 7.765792805, time_tilde: 3.4241, time_thin: 1.2717
#
# Running config [dx, dy, dT]: 300-300-  2
# Average obj_tilde: 8.293565780, obj_thin: 8.293565780, time_tilde: 11.2748, time_thin: 3.3798
#
# Running config [dx, dy, dT]: 400-400-  2
# Average obj_tilde: 8.696308755, obj_thin: 8.696308762, time_tilde: 15.0652, time_thin: 5.7316
#
# Running config [dx, dy, dT]: 500-500-  2
# Average obj_tilde: 8.972872262, obj_thin: 8.972872269, time_tilde: 35.8061, time_thin: 13.2286
#
# Running config [dx, dy, dT]: 600-600-  2
# Average obj_tilde: 9.267903961, obj_thin: 9.267903968, time_tilde: 46.9894, time_thin: 20.3649
#
# Running config [dx, dy, dT]: 700-700-  2
# Average obj_tilde: 9.517327697, obj_thin: 9.517327489, time_tilde: 344.7741, time_thin: 57.2469
#
# Running config [dx, dy, dT]: 800-800-  2

