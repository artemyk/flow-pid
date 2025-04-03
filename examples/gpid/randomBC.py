import os
import time
import numpy as np
from pathlib import Path
from scipy.io import savemat
import argparse

from pid.optimizers.tilde_pid import exact_tilde_union_info_minimizer
from pid.optimizers.thin_pid import thinpid_exact_pid_minimizer
from pid.utils import generate_randomBC


def main(step=10, start=10, max_dT=50, num_runs=3, ratio=[0.8, 0.8]):
    R = np.array([1, ratio[0], ratio[1]])
    result_dir = Path(os.path.join(f"../results"))
    result_dir.mkdir(parents=True, exist_ok=True)
    result_file = Path(os.path.join(result_dir, f"testingresult_{ratio[0]}_{ratio[1]}.mat"))

    # Generate dx, dy, dT sequence
    sequence = np.arange(start, max_dT + 1, step)
    TXY_dim = np.array([seq * R for seq in sequence]).astype(int)  # TXY_dim = sequence * R = [dT, dx, dy]

    dtype = [('dT', int), ('dx', int), ('dy', int), ('obj_tilde', float), ('obj_thin', float),
             ('execution_time_tilde', float), ('execution_time_thin', float), ('itr_tilde', int),
             ('itr_thin', int)]
    result_comparison = np.zeros((len(sequence), num_runs), dtype=dtype)  # compare tilde and thin pid

    for i, seq in enumerate(sequence):
        dT, dx, dy = TXY_dim[i]
        print(f"\nRunning config [dT, dx, dy]: {dT:3d}-{dx:3d}-{dy:3d}")

        for j in range(num_runs):
            print(f"\nRun {j+1}/{num_runs}")

            hx, hy = generate_randomBC(dT, dx, dy)
            if dx + dy < 230:  # only run the original algorithm when the size is small, otherwise it takes too long
                start_time1 = time.time()
                sig_tilde, obj_tilde, itr_tilde = exact_tilde_union_info_minimizer(hx, hy, ret_obj=True, reg=1e-8)
                end_time1 = time.time()
            else:
                sig_tilde, obj_tilde, itr_tilde = 0, 0, 0
                end_time1, start_time1 = 0.0, 0.0
                print(f"Skipping tilde pid for {dT:3d}-{dx:3d}-{dy:3d}")

            start_time2 = time.time()
            sig_thin, obj_thin, itr_thin = thinpid_exact_pid_minimizer(hx, hy, ret_obj=True, reg=1e-8)
            end_time2 = time.time()

            execution_time_tilde = end_time1 - start_time1
            execution_time_thin = end_time2 - start_time2
            result_comparison[i, j] = (
            dT, dx, dy, obj_tilde, obj_thin, execution_time_tilde, execution_time_thin, itr_tilde, itr_thin)

            print(f"obj_tilde: {obj_tilde:4.9f}, obj_thin: {obj_thin:4.9f}, time_tilde: {execution_time_tilde:3.4f}, time_thin: {execution_time_thin:3.4f}")

        # save to the record after each setup
        savemat(result_file, {'results': result_comparison})
        # print(f"Complete config {dT:3d}-{dx:3d}-{dy:3d}. Results saved to {result_file}.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--step', type=int, default=10, help='Step size for the sequence')
    parser.add_argument('--start', type=int, default=10, help='Start value for the sequence')
    parser.add_argument('--max_dT', type=int, default=50, help='Maximum value for dT')
    parser.add_argument('--num_runs', type=int, default=3, help='Number of runs')
    parser.add_argument('--ratio', type=float, nargs='+', default=[0.8,0.8], help='Ratio')

    args = parser.parse_args()
    main(args.step, args.start, args.max_dT, args.num_runs, args.ratio)