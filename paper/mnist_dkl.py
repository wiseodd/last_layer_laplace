import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
import numpy as np
from models.models import LeNetMadry
from models.hendrycks import MNIST_ConvNet
from models import dkl
from laplace import llla, kfla, dla
import laplace.util as lutil
from pycalib.calibration_methods import TemperatureScaling
from util.evaluation import *
from util.tables import *
import util.dataloaders as dl
from util.misc import *
import util.adversarial as adv
from math import *
from tqdm import tqdm, trange
import argparse
import pickle
import os, sys
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib
from tqdm import tqdm, trange
import torch.utils.data as data_utils
from util.plotting import plot_histogram


parser = argparse.ArgumentParser()
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

train_loader = dl.MNIST(train=True, augm_flag=False)
val_loader, test_loader = dl.MNIST(train=False, val_size=2000)
targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

test_loader_EMNIST = dl.EMNIST(train=False)
test_loader_FMNIST = dl.FMNIST(train=False)


tab_ood = {'MNIST - MNIST': [],
           'MNIST - EMNIST': [],
           'MNIST - FMNIST': [],
           'MNIST - FarAway': [],
           'MNIST - Adversarial': [],
           'MNIST - FarAwayAdv': []}

tab_cal = {'DKL': ([], [])}


delta = 2000


np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


model_dkl, likelihood = dkl.get_dkl_model(dataset='MNIST')

model_dkl.cuda()
likelihood.cuda()

state = torch.load(f'./pretrained_models/MNIST_dkl.pt')
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
save_res_ood(tab_ood['MNIST - MNIST'], mmc)
save_res_cal(tab_cal['DKL'], ece, mce)
print(f'[In, DKL] Time: NA/{time_pred:.1f}s; Accuracy: {acc_in:.3f}; ECE: {ece:.3f}; MCE: {mce:.3f}; MMC: {mmc:.3f}')

# Out-distribution - EMNIST
py_out = predict_dkl(test_loader_EMNIST, model_dkl, likelihood).cpu().numpy()
conf_emnist = get_confidence(py_out)
mmc = conf_emnist.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['MNIST - EMNIST'], mmc, auroc)
print(f'[Out-EMNIST, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FMNIST
py_out = predict_dkl(test_loader_FMNIST, model_dkl, likelihood).cpu().numpy()
conf_fmnist = get_confidence(py_out)
mmc = conf_fmnist.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['MNIST - FMNIST'], mmc, auroc)
print(f'[Out-FMNIST, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAway
py_out = predict_dkl(noise_loader, model_dkl, likelihood, delta=delta).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAway'], mmc, auroc)
print(f'[Out-FarAway, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - Adversarial
py_out = predict_dkl(test_loader_adv_nonasymp, model_dkl, likelihood).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['MNIST - Adversarial'], mmc, auroc)
print(f'[Out-Adversarial, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

# Out-distribution - FarAwayAdversarial
py_out = predict_dkl(test_loader_adv_asymp, model_dkl, likelihood).cpu().numpy()
conf_farway = get_confidence(py_out)
mmc = conf_farway.mean()
auroc = get_auroc(py_in, py_out)
save_res_ood(tab_ood['MNIST - FarAwayAdv'], mmc, auroc)
print(f'[Out-FarAwayAdv, DKL] MMC: {mmc:.3f}; AUROC: {auroc:.3f}')

print()
print()

if args.generate_histograms:
    plot_histogram([conf_in, conf_emnist, conf_farway], ['In - MNIST', 'Out - EMNIST', 'Out - FarAway'], 'hist_confs_mnist_dkl')

with open(f'results/tab_ood_mnist_dkl_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_ood, f)

with open(f'results/tab_cal_mnist_dkl_{args.randseed}.pkl', 'wb') as f:
    pickle.dump(tab_cal, f)
