import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
from models import resnet_orig as resnet
from models import hendrycks as resnet_oe
from laplace import llla, kfla, dla
import laplace.util as lutil
from pycalib.calibration_methods import TemperatureScaling
from util.evaluation import *
from util.tables import *
import util.dataloaders as dl
from util.misc import *
from util.plotting import plot_histogram
import util.adversarial as adv
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
from tqdm import tqdm, trange
import torch.utils.data as data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--type', help='Pick one \\{"plain", "ACET", "OE"\\}', default='plain')
parser.add_argument('--generate_adv', action='store_true', default=False)
parser.add_argument('--compute_hessian', action='store_true', default=False)
parser.add_argument('--generate_histograms', action='store_true', default=False)
parser.add_argument('--generate_figures', action='store_true', default=False)
parser.add_argument('--generate_confmat', action='store_true', default=False)
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_loader = dl.CIFAR10(train=True, augm_flag=False)
val_loader, test_loader = dl.CIFAR10(train=False, val_size=2000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

test_loader_SVHN = dl.SVHN(train=False)
test_loader_LSUN = dl.LSUN_CR(train=False)

delta = 10 if args.type == 'OE' else 1
ood_loader = dl.UniformNoise('CIFAR10', delta=delta, size=2000)


def load_model():
    if args.type == 'OE':
        model = resnet_oe.ResNet18(dataset='CIFAR10').gpu()
    else:
        model = resnet.ResNet18().cuda()

    model.load_state_dict(torch.load(f'./pretrained_models/CIFAR10_{args.type}.pt'))
    model.eval()

    return model


tab_ood = {'CIFAR10 - CIFAR10': [],
           'CIFAR10 - SVHN': [],
           'CIFAR10 - LSUN': [],
           'CIFAR10 - FarAway': [],
           'CIFAR10 - Adversarial': [],
           'CIFAR10 - FarAwayAdv': []}

tab_cal = {'MAP': ([], []),
           'Temp': ([], []),
           'LLLA': ([], []),
           'DLA': ([], []),
           'KFLA': ([], [])}


delta = 2000


"""
=============================================================================================
MAP
=============================================================================================
"""

print()

model = load_model()
noise_loader = dl.UniformNoise('CIFAR10', size=2000)

test_loader_adv_nonasymp = adv.create_adv_loader(model, noise_loader,
                                f'cifar10_adv_{args.type}',
                                delta=1,
                                epsilon=0.3,
                                load=not args.generate_adv,
                                p='inf')

test_loader_adv_asymp = adv.create_adv_loader(model, noise_loader,
                                f'cifar10_adv_{args.type}',
                                delta=delta,
                                epsilon=0.3,
                                load=not args.generate_adv,
                                p='inf')

if args.generate_adv:
    sys.exit(0)


# In-distribution
py_in, time_pred = timing(lambda: predict(test_loader, model).cpu().numpy())
acc_in = np.mean(np.argmax(py_in, 1) == targets)
conf_in = get_confidence(py_in)
mmc = conf_in.mean()
ece, mce = get_calib(py_in, targets)
save_res_ood(tab_ood['CIFAR10 - CIFAR10'], mmc)
save_res_cal(tab_cal['MAP'], ece, mce)
print(f'[In, MAP] Time: NA/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# ----------------------------------------------------------------------
# Make a binary classification dataset
if args.generate_confmat:
    from sklearn.metrics import confusion_matrix

    C = confusion_matrix(targets, py_in.argmax(1))
    print(C)
    np.fill_diagonal(C, -1)
    max_idx = np.unravel_index(C.argmax(), shape=C.shape)
    max_val = C.max()
    print(f'Most confused index: {max_idx}, value: {max_val}')

    sys.exit()
# ----------------------------------------------------------------------

# Out-distribution - SVHN
py_out = predict(test_loader_SVHN, model).cpu().numpy()
conf_svhn = get_confidence(py_out)
mmc = conf_svhn.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - SVHN'], mmc, auroc)
print(f'[Out-SVHN, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - LSUN
py_out = predict(test_loader_LSUN, model).cpu().numpy()
conf_lsun = get_confidence(py_out)
mmc = conf_lsun.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - LSUN'], mmc, auroc)
print(f'[Out-LSUN, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = predict(noise_loader, model, delta=delta).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAway'], mmc, auroc)
print(f'[Out-FarAway, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = predict(test_loader_adv_nonasymp, model).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')


# Out-distribution - FarAwayAdversarial
py_out = predict(test_loader_adv_asymp, model).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_svhn, conf_farway], ['In - CIFAR10', 'Out - SVHN', 'Out - FarAway'], 'hist_confs_cifar10_map')


"""
=============================================================================================
Temperature scaling
=============================================================================================
"""

model = load_model()
X = predict_logit(val_loader, model).cpu().numpy()
y = torch.cat([y for x, y in val_loader], dim=0).numpy()
T, time = timing(lambda: TemperatureScaling().fit(X, y).T)

print(f'T = {T:.3f}. Done in {time:.1f}s')

# In-distribution
py_in, time_pred = timing(lambda: predict(test_loader, model, T=T).cpu().numpy())
acc_in = np.mean(np.argmax(py_in, 1) == targets)
conf_in = get_confidence(py_in)
mmc = conf_in.mean()
ece, mce = get_calib(py_in, targets)
save_res_ood(tab_ood['CIFAR10 - CIFAR10'], mmc)
save_res_cal(tab_cal['Temp'], ece, mce)
print(f'[In, Temp.] Time: NA/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution - SVHN
py_out = predict(test_loader_SVHN, model, T=T).cpu().numpy()
conf_svhn = get_confidence(py_out)
mmc = conf_svhn.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - SVHN'], mmc, auroc)
print(f'[Out-SVHN, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - LSUN
py_out = predict(test_loader_LSUN, model, T=T).cpu().numpy()
conf_lsun = get_confidence(py_out)
mmc = conf_lsun.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - LSUN'], mmc, auroc)
print(f'[Out-LSUN, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = predict(noise_loader, model, delta=delta, T=T).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAway'], mmc, auroc)
print(f'[Out-FarAway, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = predict(test_loader_adv_nonasymp, model, T=T).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAwayAdversarial
py_out = predict(test_loader_adv_asymp, model, T=T).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_svhn, conf_farway], ['In - CIFAR10', 'Out - SVHN', 'Out - FarAway'], 'hist_confs_cifar10_temp')


"""
=============================================================================================
LLLA
=============================================================================================
"""

model = load_model()

if args.compute_hessian:
    hessians, time_inf = timing(lambda: llla.get_hessian(model, train_loader, mnist=False))

    # interval = torch.logspace(-3, 3, 100).cuda()
    # var0 = llla.gridsearch_var0(model, hessians, val_loader, ood_loader, interval, lam=0.25)

    if args.type == 'plain':
        var0 = torch.tensor(41.3636).float().cuda()
    elif args.type == 'ACET':
        var0 = torch.tensor(1).float().cuda()
    else:
        var0 = torch.tensor(0.2656).float().cuda()

    print(var0)

    M_W, M_b, U, V, B = llla.estimate_variance(var0, hessians)
    np.save(f'./pretrained_models/CIFAR10_{args.type}_llla.npy', [M_W, M_b, U, V, B])
else:
    time_inf = 0
    M_W, M_b, U, V, B = list(np.load(f'./pretrained_models/CIFAR10_{args.type}_llla.npy', allow_pickle=True))

# In-distribution
py_in, time_pred = timing(lambda: llla.predict(test_loader, model, M_W, M_b, U, V, B).cpu().numpy())
acc_in = np.mean(np.argmax(py_in, 1) == targets)
conf_in = get_confidence(py_in)
mmc = conf_in.mean()
ece, mce = get_calib(py_in, targets)
save_res_ood(tab_ood['CIFAR10 - CIFAR10'], mmc)
save_res_cal(tab_cal['LLLA'], ece, mce)
print(f'[In, LLLA] Time: {time_inf:.1f}/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution
py_out = llla.predict(test_loader_SVHN, model, M_W, M_b, U, V, B).cpu().numpy()
conf_svhn = get_confidence(py_out)
mmc = conf_svhn.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - SVHN'], mmc, auroc)
print(f'[Out-SVHN, LLLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - LSUN
py_out = llla.predict(test_loader_LSUN, model, M_W, M_b, U, V, B).cpu().numpy()
conf_lsun = get_confidence(py_out)
mmc = conf_lsun.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - LSUN'], mmc, auroc)
print(f'[Out-LSUN, LLLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = llla.predict(noise_loader, model, M_W, M_b, U, V, B, delta=delta).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAway'], mmc, auroc)
print(f'[Out-FarAway, LLLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = llla.predict(test_loader_adv_nonasymp, model, M_W, M_b, U, V, B).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, LLLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')


# Out-distribution - FarAwayAdversarial
py_out = llla.predict(test_loader_adv_asymp, model, M_W, M_b, U, V, B).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, LLLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_svhn, conf_farway], ['In - CIFAR10', 'Out - SVHN', 'Out - FarAway'], 'hist_confs_cifar10_llla')


if args.generate_figures:
    @torch.no_grad()
    def get_conf(dataloader, delta, model, T=1, laplace=True):
        if laplace:
            py = llla.predict(dataloader, model, M_W, M_b, U, V, B, delta=delta)
        else:
            py = predict(dataloader, model, delta=delta, T=T)

        conf = get_confidence(py.cpu().numpy())

        return conf


    range_max = 30
    deltas = np.arange(0, range_max+0.1, 1)

    confs_map = np.transpose([get_conf(test_loader, delta, model, laplace=False) for delta in deltas])
    confs_temp = np.transpose([get_conf(test_loader, delta, model, T=T, laplace=False) for delta in deltas])
    confs_laplace = np.transpose([get_conf(test_loader, delta, model, laplace=True) for delta in deltas])


    plt.clf()
    cmap = sns.color_palette()


    def plot_conf(confs, name='figs/conf_map_cifar10', ylabel='Conf. (MAP)'):
        mean = confs.mean(0)
        std = confs.std(0)

        plt.fill_between(deltas, np.maximum(0.1, mean-3*std), np.minimum(1, mean+3*std), alpha=0.15)
        plt.plot(deltas, mean, lw=3)

        plt.axhline(0.1, lw=3, ls='--', c='k')

        # plt.xticks(range(0, 6))
        plt.xlim(0, range_max)
        plt.ylim(0, 1.05)
        plt.xlabel(r'$\delta$')
        plt.ylabel(ylabel)

        plt.savefig(f'{name}.pdf', bbox_inches='tight')
        tikzplotlib.save(f'{name}.tex')

        # plt.show()
        plt.clf()


    plot_conf(confs_map, 'figs/conf_map_cifar10_', 'Conf. (MAP)')
    plot_conf(confs_temp, 'figs/conf_temp_cifar10_', 'Conf. (Temp.)')
    plot_conf(confs_laplace, 'figs/conf_laplace_cifar10_', 'Conf. (LLLA)')

    sys.exit(0)


"""
=============================================================================================
DiagLaplace
=============================================================================================
"""

model = load_model()
model_dla = dla.DiagLaplace(model)

if args.compute_hessian:
    _, time_inf = timing(lambda: model_dla.get_hessian(train_loader))

    # interval = torch.logspace(-8, -4, 100)
    # var0 = model_dla.gridsearch_var0(val_loader, ood_loader, interval, n_classes=10, lam=0.25)

    if args.type == 'plain':
        var0 = torch.tensor(3.8390e-05).float().cuda()
    elif args.type == 'ACET':
        var0 = torch.tensor(1.5160e-05).float().cuda()
    else:
        var0 = torch.tensor(1.2328e-07).float().cuda()

    print(var0)

    model_dla.estimate_variance(var0)
    torch.save(model_dla.state_dict(), f'./pretrained_models/CIFAR10_{args.type}_dla.pt')
else:
    time_inf = 0
    model_dla.load_state_dict(torch.load(f'./pretrained_models/CIFAR10_{args.type}_dla.pt'))
    model_dla.eval()

# In-distribution
py_in, time_pred = timing(lambda: lutil.predict(test_loader, model_dla).cpu().numpy())
acc_in = np.mean(np.argmax(py_in, 1) == targets)
conf_in = get_confidence(py_in)
mmc = conf_in.mean()
ece, mce = get_calib(py_in, targets)
save_res_ood(tab_ood['CIFAR10 - CIFAR10'], mmc)
save_res_cal(tab_cal['DLA'], ece, mce)
print(f'[In, DiagLaplace] Time: {time_inf:.1f}/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution
py_out = lutil.predict(test_loader_SVHN, model_dla).cpu().numpy()
conf_svhn = get_confidence(py_out)
mmc = conf_svhn.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - SVHN'], mmc, auroc)
print(f'[Out-SVHN, DiagLaplace] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - LSUN
py_out = lutil.predict(test_loader_LSUN, model_dla).cpu().numpy()
conf_lsun = get_confidence(py_out)
mmc = conf_lsun.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - LSUN'], mmc, auroc)
print(f'[Out-LSUN, DiagLaplace] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = lutil.predict(noise_loader, model_dla, delta=delta).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAway'], mmc, auroc)
print(f'[Out-FarAway, DLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = lutil.predict(test_loader_adv_nonasymp, model_dla).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, DLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAwayAdversarial
py_out = lutil.predict(test_loader_adv_asymp, model_dla).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, DLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_svhn, conf_farway], ['In - CIFAR10', 'Out - SVHN', 'Out - FarAway'], 'hist_confs_cifar10_dla')


"""
=============================================================================================
KFLA
=============================================================================================
"""

model = load_model()
model_kfla = kfla.KFLA(model)

if args.compute_hessian:
    _, time_inf = timing(lambda: model_kfla.get_hessian(train_loader))

    # interval = torch.logspace(-8, -4, 100)
    # var0 = model_kfla.gridsearch_var0(val_loader, ood_loader, interval, n_classes=10, lam=0.25)

    if args.type == 'plain':
        var0 = torch.tensor(5.6570e-05).float().cuda()
    elif args.type == 'ACET':
        var0 = torch.tensor(1.9200e-05).float().cuda()
    else:
        var0 = torch.tensor(3.7649e-07).float().cuda()

    print(var0)

    model_kfla.estimate_variance(var0)
    torch.save(model_kfla.state_dict(), f'./pretrained_models/CIFAR10_{args.type}_kfla.pt')
else:
    time_inf = 0
    model_kfla.load_state_dict(torch.load(f'./pretrained_models/CIFAR10_{args.type}_kfla.pt'))
    model_kfla.eval()

# In-distribution
py_in, time_pred = timing(lambda: lutil.predict(test_loader, model_kfla).cpu().numpy())
acc_in = np.mean(np.argmax(py_in, 1) == targets)
conf_in = get_confidence(py_in)
mmc = conf_in.mean()
ece, mce = get_calib(py_in, targets)
save_res_ood(tab_ood['CIFAR10 - CIFAR10'], mmc)
save_res_cal(tab_cal['KFLA'], ece, mce)
print(f'[In, KFLA] Time: {time_inf:.1f}/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution
py_out = lutil.predict(test_loader_SVHN, model_kfla).cpu().numpy()
conf_svhn = get_confidence(py_out)
mmc = conf_svhn.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - SVHN'], mmc, auroc)
print(f'[Out-SVHN, KFLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - LSUN
py_out = lutil.predict(test_loader_LSUN, model_kfla).cpu().numpy()
conf_lsun = get_confidence(py_out)
mmc = conf_lsun.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - LSUN'], mmc, auroc)
print(f'[Out-LSUN, KFLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = lutil.predict(noise_loader, model_kfla, delta=delta).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAway'], mmc, auroc)
print(f'[Out-FarAway, KLFA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = lutil.predict(test_loader_adv_nonasymp, model_dla).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, KLFA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAwayAdversarial
py_out = lutil.predict(test_loader_adv_asymp, model_dla).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, KLFA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()
print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_svhn, conf_farway], ['In - CIFAR10', 'Out - SVHN', 'Out - FarAway'], 'hist_confs_cifar10_kfla')


"""
===========================================================================================
"""

# Save dict
if not os.path.exists('results/'):
    os.makedirs('results/')

with open(f'results/tab_ood_cifar10_{args.type}_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_ood, f)

with open(f'results/tab_cal_cifar10_{args.type}_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_cal, f)
