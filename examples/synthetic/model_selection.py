import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


if __name__ == '__main__':
    pid_table = pd.read_csv('./data/experiments/results.csv')

    ri = pid_table['R']
    si = pid_table['S']
    u1 = pid_table['U1']
    u2 = pid_table['U2']
    imxy = ri + si + u1 + u2
    acc = pid_table['acc']

    train_sets = ['redundancy', 'uniqueness0', 'uniqueness1','synergy', 'mix1', 'mix2', 'mix3', 'mix4', 'mix5', 'mix6']
    test_sets = ['synthetic1', 'synthetic2', 'synthetic3', 'synthetic4', 'synthetic5']

    models = ['dataset', 'additive', 'agree', 'align', 'elem', 'mi', 'lower', 'recon']

    ri_pretrained = np.array(ri[::8] / imxy[::8])[:10]
    ri_test = np.array(ri[::8] / imxy[::8])[10:]
    si_pretrained = np.array(si[::8] / imxy[::8])[:10]
    si_test = np.array(si[::8] / imxy[::8])[10:]
    u1_pretrained = np.array(u1[::8] / imxy[::8])[:10]
    u1_test = np.array(u1[::8] / imxy[::8])[10:]
    u2_pretrained = np.array(u2[::8] / imxy[::8])[:10]
    u2_test = np.array(u2[::8] / imxy[::8])[10:]

    for i, test_set in enumerate(test_sets):
        ri_diff = np.absolute(np.ones_like(ri_pretrained) * ri_test[i] - ri_pretrained)
        si_diff = np.absolute(np.ones_like(si_pretrained) * si_test[i] - si_pretrained)
        u1_diff = np.absolute(np.ones_like(u1_pretrained) * u1_test[i] - u1_pretrained)
        u2_diff = np.absolute(np.ones_like(u2_pretrained) * u2_test[i] - u2_pretrained)
        pid_score = ri_diff + si_diff + u1_diff + u2_diff
        selected_set = np.argsort(pid_score)[:1][0]

        selected_acc = np.array(acc[selected_set*8:8+selected_set*8])
        selected_models = np.argsort(-selected_acc)[:4]

        test_acc = np.array(acc[80+i*8:88+i*8])
        best_models = np.argsort(-test_acc)[:4]

        print(f"Test set: {test_set}, PID score: {pid_score}, Selected set: {train_sets[selected_set]}, Selected models: {selected_models}, Best models: {best_models}")






