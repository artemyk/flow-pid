import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns


if __name__ == '__main__':
    pid_table = pd.read_csv('../results/multi_poisson_sample1000.csv')

    suptitlesize = 20
    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12

    w1 = pid_table['w1'][::5]
    ri = pid_table['ri']
    si = pid_table['si']
    ux = pid_table['ux']
    uy = pid_table['uy']
    imxy = ri + si + ux + uy

    pid_defns = ['tilde', 'delta', 'mmi', 'thin','gt']
    linestyles = ['-.', '--', ':', '-', '']
    markers = ['', '', '', '', 'o']

    pid_atoms = [imxy, ux, uy, ri, si]
    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))
    axs = axs.flatten()

    # Un-normalized PID values
    ax = axs[0]
    lines = {}  # Dictionary to hold all line handles for legend
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(w1, pid_atom[i::5],
                           color=colors[j], linestyle=linestyles[i],
                           marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line
            if i==0 or i==1 or i==2:
                if j==0:
                    line.remove()

    fig.suptitle(r'PID values (left) and PID values normalized by $I(X_1,X_2;Y)$ (right)'
                 + '\nin a multivariate Poisson spike-count simulation', fontsize=titlesize)
    # ax.set_title(r'PIDs for multivariate Poisson example', fontsize=titlesize)
    ax.set_xlabel(r'Weight from $Y$ to $X_1$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
    ax.set_xticks(w1)
    # ax.set_xticklabels(['%.g' % val for val in rows.w_x1])  # , rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # Normalized PID values
    ax = axs[1]
    lines = {}  # Dictionary to hold all line handles for legend
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(w1, pid_atom[i::5] / imxy[i::5],
                    color=colors[j], linestyle=linestyles[i], marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line
            if j == 0:
                line.remove()

    ax.set_ylim(-0.03, 0.63)
    # ax.set_title(r'PIDs for multivariate Poisson example', fontsize=titlesize)
    ax.set_xlabel(r'Weight from $Y$ to $X_1$', fontsize=labelsize)
    ax.set_ylabel('Normalized partial information', fontsize=labelsize)
    ax.set_xticks(w1)
    # ax.set_xticklabels(['%.g' % val for val in rows.w_x1])  # , rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # Legend
    handles = [lines[(c, '-', '')] for c in colors]
    texts = ['$I(X_1,X_2;Y)$', r'$U_1$', r'$U_2$', r'$R$', r'$S$']   # I(Y\;\!;(X, Y\;\!\;\!))
    color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                             bbox_to_anchor=(1, 0.75), fontsize=legendsize,
                             title='PID component', title_fontsize=labelsize)
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(color_legend)

    handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
    texts = ['Tilde-PID', 'Delta-PID', 'MMI-PID', 'Flow-PID', 'Ground truth']
    defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                            bbox_to_anchor=(1, 0.15), fontsize=legendsize,
                            title='PID definition', title_fontsize=labelsize)

    plt.tight_layout()
    plt.subplots_adjust(left=0.07, right=0.8, bottom=0.12, top=0.85, wspace=0.25)
    plt.savefig('../plots/mult-poisson_10000.pdf')
    # plt.show()

