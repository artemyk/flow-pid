import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == '__main__':
    pid_table = pd.read_csv('./pid_results.csv')

    ri = pid_table['R']
    si = pid_table['S']
    u1 = pid_table['U1']
    u2 = pid_table['U2']
    imxy = ri + si + u1 + u2

    train_sets = ['redundancy', 'uniqueness0', 'uniqueness1', 'synergy', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6']
    test_sets = ['mimic', 'enrico', 'humor12', 'humor01', 'humor02', 'mosei12', 'mosei01', 'mosei02', 'sarcasm01', 'sarcasm02', 'sarcasm12']

    ri_pretrained = np.array(ri[:10] / imxy[:10])
    ri_test = np.array(ri[10:] / imxy[10:])
    si_pretrained = np.array(si[:10] / imxy[:10])
    si_test = np.array(si[10:] / imxy[10:])
    u1_pretrained = np.array(u1[:10] / imxy[:10])
    u1_test = np.array(u1[10:] / imxy[10:])
    u2_pretrained = np.array(u2[:10] / imxy[:10])
    u2_test = np.array(u2[10:] / imxy[10:])

    for i, test_set in enumerate(test_sets):
        ri_diff = np.absolute(np.ones_like(ri_pretrained) * ri_test[i] - ri_pretrained)
        si_diff = np.absolute(np.ones_like(si_pretrained) * si_test[i] - si_pretrained)
        u1_diff = np.absolute(np.ones_like(u1_pretrained) * u1_test[i] - u1_pretrained)
        u2_diff = np.absolute(np.ones_like(u2_pretrained) * u2_test[i] - u2_pretrained)
        pid_score = ri_diff + si_diff + u1_diff + u2_diff
        selected_set = np.argsort(pid_score)[:1][0]

        print(
            f"Test set: {test_set}, PID score: {pid_score}, Selected set: {train_sets[selected_set]}")