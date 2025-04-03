#!/usr/bin/env python3

from __future__ import print_function, division

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import seaborn as sns

if __name__ == '__main__':
    pid_table = pd.read_pickle('../results/canonical_exs.pkl.gz')

    pid_defns = ['thin', 'tilde', 'mmi', 'gt']
    linestyles = ['-', '--', ':', '']
    markers = ['', '', '', 'o']

    pid_atoms = ['imxy', 'uix', 'uiy', 'ri', 'si']
    colors = ['k', 'C0', 'C1', 'C2', 'C3']

    suptitlesize = 20
    titlesize = 18
    labelsize = 16
    legendsize = 14
    ticksize = 12

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(15, 4), sharey=True)
    axs = axs.flatten()

    # UI + RI
    ax = axs[0]
    rows = pid_table[pid_table.desc == 'uix+ri']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                    color=colors[j], linestyle=linestyles[i], marker=markers[i])

    ax.set_title(r'Unique and Redundant', fontsize=titlesize)
    ax.set_xlabel(r'Noise in $Y$  given $X$, $\sigma_{Y|X}$', fontsize=labelsize)
    ax.set_ylabel('Partial information (bits)', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in rows.sigma_y__x],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # RI + SI
    ax = axs[1]
    lines = {}  # Dictionary to hold all line handles for legend
    rows = pid_table[pid_table.desc == 'ri+si']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            line = ax.plot(rows['id'], rows[(pid_defn, pid_atom)],
                           color=colors[j], linestyle=linestyles[i],
                           marker=markers[i])[0]
            lines[(colors[j], linestyles[i], markers[i])] = line

    ax.set_title(r'Redundant and Synergistic', fontsize=titlesize)
    ax.set_xlabel(r'Noise correlation, $\rho$', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in rows.rho],
                       rotation=45, rotation_mode='anchor', ha='right')
    ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # UI + SI
    ax = axs[2]
    rows = pid_table[pid_table.desc == 'uix+si']
    for i, pid_defn in enumerate(pid_defns):
        for j, pid_atom in enumerate(pid_atoms):
            ax.plot(rows['id'], rows[(pid_defn, pid_atom)], color=colors[j],
                    linestyle=linestyles[i], marker=markers[i])

    ax.set_title(r'Unique and Synergistic', fontsize=titlesize)
    ax.set_xlabel(r'Noise correlation, $\rho$', fontsize=labelsize)
    ax.set_xticks(rows['id'])
    ax.set_xticklabels(['%.2f' % val for val in rows.rho],
                       rotation=45, rotation_mode='anchor', ha='right')
    # ax.tick_params(axis='both', which='major', labelsize=ticksize)
    ax.grid(True)

    # set y-axis range
    # for ax in axs:
    #     ax.set_ylim(top=1.5)
    #     ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    # fig.suptitle('PIDs for Canonical Gaussian Examples', fontsize=suptitlesize)

    # Legend (tied to last axis)
    handles = [lines[(c, '-', '')] for c in colors]
    texts = [r'$I(M\;\!; (X, Y\;\!\;\!))$', '$U_X$', '$U_Y$', '$R$', '$S$']
    color_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                             bbox_to_anchor=(1, 0.75), fontsize=legendsize,
                             title='PID component', title_fontsize=labelsize)
    # https://matplotlib.org/stable/tutorials/intermediate/legend_guide.html#multiple-legends-on-the-same-axes
    ax.add_artist(color_legend)

    handles = [lines[('k', ls, m)] for (ls, m) in zip(linestyles, markers)]
    texts = ['Thin-PID', '$\sim_G$-PID', 'MMI-PID', 'Ground truth']
    defn_legend = ax.legend(handles, texts, loc='center left', frameon=False,
                            bbox_to_anchor=(1, 0.1), fontsize=legendsize,
                            title='PID definition', title_fontsize=labelsize)

    plt.tight_layout()
    plt.savefig('../plots/canonical-exs.pdf')
    #plt.show()