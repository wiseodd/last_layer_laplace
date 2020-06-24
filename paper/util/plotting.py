import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
import numpy as np
from math import *
from backpack import backpack, extend
from backpack.extensions import KFAC
import seaborn as sns
import matplotlib
from matplotlib import offsetbox
from sklearn.metrics import roc_auc_score
import scipy
import tikzplotlib


matplotlib.rcParams['figure.figsize'] = (5,5)
matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['font.size'] = 16
matplotlib.rcParams['font.family'] = 'serif'
matplotlib.rcParams['lines.linewidth'] = 1.0
plt = matplotlib.pyplot

sns.set_palette('colorblind')
sns.set_context("talk", font_scale=1)


def plot(ents, labels, legend=False, figname=None):
    for ent, label in zip(ents, labels):
        sns.distplot(ent, hist=False, kde_kws=dict(cumulative=True), label=label)

    plt.xlim(0, log(10))
    plt.ylim(0, 1)

    plt.xlabel('Predictive entropy')
    plt.ylabel('Empirical CDF')

    if legend:
        plt.legend()

    if figname is not None:
        plt.savefig(f'./figs/mnist_{figname}.pdf', bbox_inches='tight')

    plt.clf()
    plt.cla()
    plt.close()


def plot_calibration(pys, y_true, M=15, figname=None):
    # Put the confidence into M bins
    _, bins = np.histogram(pys, M, range=(0, 1))

    labels = pys.argmax(1)
    confs = np.max(pys, axis=1)
    conf_idxs = np.digitize(confs, bins)

    # Accuracy and avg. confidence per bin
    accs_bin = []
    confs_bin = []
    nitems_bin = []

    for i in range(M):
        labels_i = labels[conf_idxs == i]
        y_true_i = y_true[conf_idxs == i]
        confs_i = confs[conf_idxs == i]

        acc = np.nan_to_num(np.mean(labels_i == y_true_i), 0)
        conf = np.nan_to_num(np.mean(confs_i), 0)

        accs_bin.append(acc)
        confs_bin.append(conf)
        nitems_bin.append(len(labels_i))

    accs_bin, confs_bin = np.array(accs_bin), np.array(confs_bin)
    nitems_bin = np.array(nitems_bin)

    ECE = np.average(np.abs(confs_bin-accs_bin), weights=nitems_bin/nitems_bin.sum())
    MCE = np.max(np.abs(accs_bin - confs_bin))

    # In percent
    ECE, MCE = ECE*100, MCE*100

    plt.bar(bins[:-1], accs_bin, align='edge', width=bins[1]-bins[0], alpha=0.5)
    plt.plot([0, 1], [0, 1], 'k--', lw=3, alpha=1)
    plt.plot([], label=f'ECE = {ECE:.2f}\nMCE = {MCE:.2f}')
    plt.legend(handletextpad=-0.1, handlelength=0, loc='lower right', fontsize=20)

    plt.xlim(0, 1)
    plt.ylim(0, 1)

    plt.xlabel('Confidence')
    plt.ylabel('Accuracy')

    if figname is not None:
        plt.savefig(f'./figs/mnist_{figname}_calibration.pdf', bbox_inches='tight')
        tikzplotlib.save(f'./figs/mnist_{figname}_calibration.tex')
    else:
        plt.show()

    plt.clf()
    plt.cla()
    plt.close()


def plot_histogram(values, labels, fname=None, show=False):
    for val, label in zip(values, labels):
        plt.hist(val, bins=50, range=(0, 1), alpha=0.4, density=True, label=label)

    plt.legend(loc='upper left')
    plt.xlim(0, 1)
    # plt.ylim(0, 800)

    if fname is not None:
        plt.savefig(f'./figs/{fname}.pdf', bbox_inches='tight')
        tikzplotlib.save(f'./figs/{fname}.tex')

    if show:
        plt.show()

    plt.clf()
