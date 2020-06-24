import torch
from models.models import LeNetMadry
from models.hendrycks import MNIST_ConvNet
from laplace import llla, kfla, dla, llla_binary
import laplace.util as lutil
from util.calibration import temp_scaling_binary
from util.evaluation import *
from util.tables import *
import util.dataloaders as dl
from util.misc import *
from math import *
import numpy as np
import argparse
import pickle
import os
from util.plotting import plot_histogram
from tqdm import tqdm, trange
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
from tqdm import tqdm, trange
import torch.utils.data as data_utils


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=123)
parser.add_argument('--generate_histograms', action='store_true', default=False)
parser.add_argument('--generate_figures', action='store_true', default=False)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

train_loader = dl.binary_MNIST(3, 8, train=True, augm_flag=False)
val_loader, test_loader = dl.binary_MNIST(3, 8, train=False, val_size=1000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

test_loader_EMNIST, _ = dl.EMNIST(train=False, val_size=len(test_loader.dataset))
test_loader_FMNIST, _ = dl.FMNIST(train=False, val_size=len(test_loader.dataset))

ood_loader = dl.UniformNoise('MNIST', train=False, size=1000)
noise_loader = dl.UniformNoise('MNIST', size=2000)


def load_model():
    model = LeNetMadry(binary=True)
    model = model.cuda()
    model.load_state_dict(torch.load(f'./pretrained_models/binary_MNIST.pt'))
    model.eval()

    return model


tab_ood = {'MNIST - MNIST': [],
           'MNIST - EMNIST': [],
           'MNIST - FMNIST': [],
           'MNIST - FarAway': []}

tab_cal = {'MAP': ([], []),
           'Temp': ([], []),
           'LLLA': ([], []),
           'DLA': ([], []),
           'KFLA': ([], [])}


deltas = 100


"""
=============================================================================================
MAP
=============================================================================================
"""

print()

model = load_model()

# In-distribution
py_in, time_pred = timing(lambda: predict_binary(test_loader, model).cpu().numpy())
acc_in = np.mean((py_in >= 0.5) == targets)
conf_in = get_confidence(py_in, binary=True)
mmc = conf_in.mean()
# ece, mce = get_calib(py_in, targets)
ece, mce = 0, 0
save_res_ood(tab_ood['MNIST - MNIST'], mmc)
# save_res_cal(tab_cal['MAP'], ece, mce)
print(f'[In, MAP] Time: NA/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution - EMNIST
py_out = predict_binary(test_loader_EMNIST, model).cpu().numpy()
conf_emnist = get_confidence(py_out, binary=True)
mmc = conf_emnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - EMNIST'], mmc, auroc)
print(f'[Out-EMNIST, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FMNIST
py_out = predict_binary(test_loader_FMNIST, model).cpu().numpy()
conf_fmnist = get_confidence(py_out, binary=True)
mmc = conf_fmnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FMNIST'], mmc, auroc)
print(f'[Out-FMNIST, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = predict_binary(noise_loader, model, delta=deltas).cpu().numpy()
conf_farway = get_confidence(py_out, binary=True)
mmc = conf_farway.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAway'], mmc, auroc)
print(f'[Out-FarAway, MAP] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_fmnist, conf_farway], ['In - MNIST', 'Out - FMNIST', 'Out - FarAway'], 'hist_confs_mnist_binary_map')


"""
=============================================================================================
Temperature scaling
=============================================================================================
"""

model = load_model()
logits = predict_logit(val_loader, model).squeeze().cpu().numpy()
T, time = timing(lambda: temp_scaling_binary(logits, targets_val))

print(f'T = {T:.3f}. Done in {time:.1f}s')

# In-distribution
py_in, time_pred = timing(lambda: predict_binary(test_loader, model, T=T).cpu().numpy())
acc_in = np.mean((py_in >= 0.5) == targets)
conf_in = get_confidence(py_in, binary=True)
mmc = conf_in.mean()
# ece, mce = get_calib(py_in, targets)
ece, mce = 0, 0
save_res_ood(tab_ood['MNIST - MNIST'], mmc)
# save_res_cal(tab_cal['MAP'], ece, mce)
print(f'[In, Temp.] Time: NA/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution - EMNIST
py_out = predict_binary(test_loader_EMNIST, model, T=T).cpu().numpy()
conf_emnist = get_confidence(py_out, binary=True)
mmc = conf_emnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - EMNIST'], mmc, auroc)
print(f'[Out-EMNIST, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')


# Out-distribution - FMNIST
py_out = predict_binary(test_loader_FMNIST, model, T=T).cpu().numpy()
conf_fmnist = get_confidence(py_out, binary=True)
mmc = conf_fmnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FMNIST'], mmc, auroc)
print(f'[Out-FMNIST, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')


# Out-distribution - FarAway
py_out = predict_binary(noise_loader, model, T=T, delta=deltas).cpu().numpy()
conf_farway = get_confidence(py_out, binary=True)
mmc = conf_farway.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAway'], mmc, auroc)
print(f'[Out-FarAway, Temp.] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_fmnist, conf_farway], ['In - MNIST', 'Out - FMNIST', 'Out - FarAway'], 'hist_confs_mnist_binary_temp')


"""
=============================================================================================
LLLA
=============================================================================================
"""

model = load_model()
hessians, time_inf = timing(lambda: llla_binary.get_hessian(model, train_loader, mnist=False))

# interval = torch.tensor(np.linspace(1, 1000, 100)).cuda()
# var0 = llla_binary.gridsearch_var0(model, hessians, val_loader, ood_loader, interval, lam=0.5)
var0 = 939.4545  # optimal

print(var0)

mu, S = llla_binary.estimate_variance(var0, hessians)

# In-distribution
py_in, time_pred = timing(lambda: llla_binary.predict(test_loader, model, mu, S).cpu().numpy())
acc_in = np.mean((py_in >= 0.5) == targets)
conf_in = get_confidence(py_in, binary=True)
mmc = conf_in.mean()
# ece, mce = get_calib(py_in, targets)
ece, mce = 0, 0
save_res_ood(tab_ood['MNIST - MNIST'], mmc)
# save_res_cal(tab_cal['MAP'], ece, mce)
print(f'[In, LLLA] Time: {time_inf:.1f}/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution - EMNIST
py_out = llla_binary.predict(test_loader_EMNIST, model, mu, S).cpu().numpy()
conf_emnist = get_confidence(py_out, binary=True)
mmc_emnist = conf_emnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - EMNIST'], mmc_emnist, auroc)
print(f'[Out-EMNIST, LLLA] MMC: {mmc_emnist:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FMNIST
py_out = llla_binary.predict(test_loader_FMNIST, model, mu, S).cpu().numpy()
conf_fmnist = get_confidence(py_out, binary=True)
mmc_fmnist = conf_fmnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FMNIST'], mmc_fmnist, auroc)
print(f'[Out-FMNIST, LLLA] MMC: {mmc_fmnist:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = llla_binary.predict(noise_loader, model, mu, S, delta=deltas).cpu().numpy()
conf_farway = get_confidence(py_out, binary=True)
mmc = conf_farway.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAway'], mmc, auroc)
print(f'[Out-FarAway, LLLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_fmnist, conf_farway], ['In - MNIST', 'Out - FMNIST', 'Out - FarAway'], 'hist_confs_mnist_binary_llla')


if args.generate_figures:
    @torch.no_grad()
    def get_conf(dataloader, delta, model, mu, S, T=1, laplace=True):
        if laplace:
            py = llla_binary.predict(dataloader, model, mu, S, apply_sigm=True, delta=delta)
        else:
            py = predict_binary(dataloader, model, delta=delta, apply_sigm=True, T=T)

        # abs_z = torch.abs(z)
        conf = get_confidence(py.cpu().numpy(), binary=True)

        return conf


    max_x = 20
    deltas = np.arange(0, max_x+.1, 1)

    confs_map = np.transpose([get_conf(test_loader, delta, model, mu, S, laplace=False) for delta in deltas])
    confs_temp = np.transpose([get_conf(test_loader, delta, model, mu, S, T=T, laplace=False) for delta in deltas])
    confs_laplace = np.transpose([get_conf(test_loader, delta, model, mu, S, laplace=True) for delta in deltas])

    print(confs_map.shape, confs_laplace.shape)

    plt.clf()
    cmap = sns.color_palette()


    def plot_conf(confs, name='figs/conf_map_cifar10_binary', ylabel='Conf. (MAP)'):
        mean = confs.mean(0)
        std = confs.std(0)

        plt.fill_between(deltas, np.maximum(0.5, mean-3*std), mean+3*std, alpha=0.15)
        plt.plot(deltas, mean, lw=3)

        plt.axhline(0.5, lw=3, ls='--', c='k')

        # plt.xticks(range(0, 6))
        plt.xlim(0, max_x)
        plt.ylim(0, 1.05)
        plt.xlabel(r'$\delta$')
        plt.ylabel(ylabel)

        # plt.savefig(f'{name}.pdf', bbox_inches='tight')
        tikzplotlib.save(f'{name}.tex')

        # plt.show()
        plt.clf()


    plot_conf(confs_map, 'figs/conf_map_mnist_binary', 'Conf. (MAP)')
    plot_conf(confs_temp, 'figs/conf_temp_mnist_binary', 'Conf. (Temp.)')
    plot_conf(confs_laplace, 'figs/conf_laplace_mnist_binary ', 'Conf. (LLLA)')



"""
=============================================================================================
DiagLaplace
=============================================================================================
"""

model = load_model()
model_dla = dla.DiagLaplace(model)
_, time_inf = timing(lambda: model_dla.get_hessian(train_loader, binary=True))

interval = torch.linspace(1e-4, 0.05, 100)
var0 = model_dla.gridsearch_var0(val_loader, ood_loader, interval, n_classes=2, lam=0.5)
# var0 = torch.tensor(0.0037).cuda()  # optimal
print(var0)

model_dla.estimate_variance(var0)

#In-distribution
py_in, time_pred = timing(lambda: lutil.predict_binary(test_loader, model_dla).cpu().numpy())
acc_in = np.mean((py_in >= 0.5) == targets)
conf_in = get_confidence(py_in, binary=True)
mmc = conf_in.mean()
# ece, mce = get_calib(py_in, targets)
ece, mce = 0, 0
save_res_ood(tab_ood['MNIST - MNIST'], mmc)
# save_res_cal(tab_cal['MAP'], ece, mce)
print(f'[In, DLA] Time: {time_inf:.1f}/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution EMNIST
py_out = lutil.predict_binary(test_loader_EMNIST, model_dla).cpu().numpy()
conf_emnist = get_confidence(py_out, binary=True)
mmc_emnist = conf_emnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - EMNIST'], mmc_emnist, auroc)
print(f'[Out-EMNIST, DLA] MMC: {mmc_emnist:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FMNIST
py_out = lutil.predict_binary(test_loader_FMNIST, model_dla).cpu().numpy()
conf_fmnist = get_confidence(py_out, binary=True)
mmc_fmnist = conf_fmnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FMNIST'], mmc_fmnist, auroc)
print(f'[Out-FMNIST, DLA] MMC: {mmc_fmnist:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = lutil.predict_binary(loader_faraway, model_dla).cpu().numpy()
conf_farway = get_confidence(py_out, binary=True)
mmc = conf_farway.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAway'], mmc, auroc)
print(f'[Out-FarAway, DLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_fmnist, conf_farway], ['In - MNIST', 'Out - FMNIST', 'Out - FarAway'], 'hist_confs_mnist_binary_dla')


"""
=============================================================================================
KFLA
=============================================================================================
"""

model = load_model()
model_kfla = kfla.KFLA(model)
_, time_inf = timing(lambda: model_kfla.get_hessian(train_loader, binary=True))

interval = torch.linspace(1e-4, 1e-1, 100)
var0 = model_kfla.gridsearch_var0(val_loader, ood_loader, interval, n_classes=2, lam=0.5)
# var0 = torch.tensor(0.0082).cuda()  # optimal
print(var0)

model_kfla.estimate_variance(var0)

#In-distribution
py_in, time_pred = timing(lambda: lutil.predict_binary(test_loader, model_kfla).cpu().numpy())
acc_in = np.mean((py_in >= 0.5) == targets)
conf_in = get_confidence(py_in, binary=True)
mmc = conf_in.mean()
# ece, mce = get_calib(py_in, targets)
ece, mce = 0, 0
save_res_ood(tab_ood['MNIST - MNIST'], mmc)
# save_res_cal(tab_cal['MAP'], ece, mce)
print(f'[In, KFLA] Time: {time_inf:.1f}/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution EMNIST
py_out = lutil.predict_binary(test_loader_EMNIST, model_kfla).cpu().numpy()
conf_emnist = get_confidence(py_out, binary=True)
mmc_emnist = conf_emnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - EMNIST'], mmc_emnist, auroc)
print(f'[Out-EMNIST, KFLA] MMC: {mmc_emnist:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FMNIST
py_out = lutil.predict_binary(test_loader_FMNIST, model_kfla).cpu().numpy()
conf_fmnist = get_confidence(py_out, binary=True)
mmc_fmnist = conf_fmnist.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FMNIST'], mmc_fmnist, auroc)
print(f'[Out-FMNIST, KFLA] MMC: {mmc_fmnist:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = lutil.predict_binary(loader_faraway, model_kfla).cpu().numpy()
conf_farway = get_confidence(py_out, binary=True)
mmc = conf_farway.mean()
auroc = get_auroc_binary(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAway'], mmc, auroc)
print(f'[Out-FarAway, KFLA] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_fmnist, conf_farway], ['In - MNIST', 'Out - FMNIST', 'Out - FarAway'], 'hist_confs_mnist_binary_kfla')


"""
===========================================================================================
"""

# Save dict
if not os.path.exists('results/binary'):
    os.makedirs('results/binary')

with open(f'results/binary/tab_ood_mnist_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_ood, f)

with open(f'results/binary/tab_cal_mnist_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_cal, f)
