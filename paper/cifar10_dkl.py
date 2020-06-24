import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
from models import resnet_orig as resnet
from models import hendrycks as resnet_oe
from models import dkl
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
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()


train_loader = dl.CIFAR10(train=True, augm_flag=False)
val_loader, test_loader = dl.CIFAR10(train=False, val_size=2000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

test_loader_SVHN = dl.SVHN(train=False)
test_loader_LSUN = dl.LSUN_CR(train=False)


tab_ood = {'CIFAR10 - CIFAR10': [],
           'CIFAR10 - SVHN': [],
           'CIFAR10 - LSUN': [],
           'CIFAR10 - FarAway': [],
           'CIFAR10 - Adversarial': [],
           'CIFAR10 - FarAwayAdv': []}

tab_cal = {'DKL': ([], [])}


delta = 2000


np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model_dkl, likelihood = dkl.get_dkl_model(dataset='CIFAR10')

model_dkl.cuda()
likelihood.cuda()

state = torch.load(f'./pretrained_models/CIFAR10_dkl.pt')
model_dkl.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])
model_dkl.eval()
likelihood.eval()

# In-distribution
py_in, time_pred = timing(lambda: predict_dkl(test_loader, model_dkl, likelihood).cpu().numpy())
acc_in = np.mean(np.argmax(py_in, 1) == targets)
conf_in = get_confidence(py_in)
mmc = conf_in.mean()
ece, mce = get_calib(py_in, targets)
save_res_ood(tab_ood['CIFAR10 - CIFAR10'], mmc)
save_res_cal(tab_cal['DKL'], ece, mce)
print(f'[In, DKL] Time: NA/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution - SVHN
py_out = predict_dkl(test_loader_SVHN, model_dkl, likelihood).cpu().numpy()
conf_emnist = get_confidence(py_out)
mmc = conf_emnist.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - SVHN'], mmc, auroc)
print(f'[Out-SVHN, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - LSUN
py_out = predict_dkl(test_loader_LSUN, model_dkl, likelihood).cpu().numpy()
conf_fmnist = get_confidence(py_out)
mmc = conf_fmnist.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - LSUN'], mmc, auroc)
print(f'[Out-LSUN, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = predict_dkl(noise_loader, model_dkl, likelihood, delta=delta).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAway'], mmc, auroc)
print(f'[Out-FarAway, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = predict_dkl(test_loader_adv_nonasymp, model_dkl, likelihood).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAwayAdversarial
py_out = predict_dkl(test_loader_adv_asymp, model_dkl, likelihood).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['CIFAR10 - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()
print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_emnist, conf_farway], ['In - CIFAR10', 'Out - SVHN', 'Out - FarAway'], 'hist_confs_cifar10_dkl')

with open(f'results/tab_ood_cifar10_dkl_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_ood, f)

with open(f'results/tab_cal_cifar10_dkl_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_cal, f)
