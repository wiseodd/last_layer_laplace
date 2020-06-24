import torch
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle as skshuffle


@torch.no_grad()
def predict(dataloader, model, n_samples=1, T=1, delta=1, return_targets=False):
    py = []
    targets = []

    for x, y in dataloader:
        x = delta*x.cuda()

        py_ = 0
        for _ in range(n_samples):
            f_s = model.forward(x)
            py_ += torch.softmax(f_s/T, 1)
        py_ /= n_samples

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0)
    else:
        return torch.cat(py, dim=0)


@torch.no_grad()
def predict_logit(dataloader, model):
    logits = []

    for x, _ in dataloader:
        x = x.cuda()
        out = model.forward(x)
        logits.append(out)

    return torch.cat(logits, dim=0)


@torch.no_grad()
def predict_binary(dataloader, model, n_samples=1, T=1, apply_sigm=True, return_targets=False, delta=1):
    py = []
    targets = []

    for x, y in dataloader:
        x = delta * x.cuda()

        f_s = model.forward(x).squeeze()
        py_ = torch.sigmoid(f_s/T) if apply_sigm else f_s/T

        py.append(py_)
        targets.append(y)

    if return_targets:
        return torch.cat(py, dim=0), torch.cat(targets, dim=0).float()
    else:
        return torch.cat(py, dim=0)


def get_confidence(py, binary=False):
    return py.max(1) if not binary else np.maximum(py, 1-py)


def get_auroc(py_in, py_out):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    examples = np.concatenate([py_in.max(1), py_out.max(1)])
    return roc_auc_score(labels, examples)


def get_auroc_binary(py_in, py_out):
    labels = np.zeros(len(py_in)+len(py_out), dtype='int32')
    labels[:len(py_in)] = 1
    conf_in = np.maximum(py_in, 1-py_in)
    conf_out = np.maximum(py_out, 1-py_out)
    examples = np.concatenate([conf_in, conf_out])
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
