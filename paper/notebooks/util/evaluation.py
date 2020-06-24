import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle as skshuffle


@torch.no_grad()
def predict(dataloader, model, n_samples=1, T=1):
    py = []

    for x, _ in dataloader:
        x = x.cuda()

        py_ = 0
        for _ in range(n_samples):
            f_s = model.forward(x)
            py_ += torch.softmax(f_s/T, 1)
        py_ /= n_samples

        py.append(py_)

    return torch.cat(py, dim=0)


@torch.no_grad()
def predict_logit(dataloader, model):
    logits = []

    for x, _ in dataloader:
        x = x.cuda()
        out = model.forward(x)
        logits.append(out)

    return torch.cat(logits, dim=0)


def get_auroc(py_in, py_out):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return roc_auc_score(labels, examples)


def get_calib(pys, y_true, M=100):
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

    return ECE, MCE


def timing(fun):
    """
    Return the original output(s) and a wall-clock timing in second.
    """
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    torch.cuda.synchronize()

    start.record()
    ret = fun()
    end.record()

    torch.cuda.synchronize()

    return ret, start.elapsed_time(end)/1000
