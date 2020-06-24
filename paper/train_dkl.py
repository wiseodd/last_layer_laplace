import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import torch
from models.models import LeNetMadry
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
from tqdm import tqdm, trange
import numpy as np
import argparse
import pickle
import os
import math
import gpytorch


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='MNIST')
parser.add_argument('--randseed', type=int, default=123)
args = parser.parse_args()

np.random.seed(args.randseed)
torch.manual_seed(args.randseed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if args.dataset == 'MNIST':
    train_loader = dl.MNIST(train=True)
    val_loader, test_loader = dl.MNIST(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR10':
    train_loader = dl.CIFAR10(train=True)
    val_loader, test_loader = dl.CIFAR10(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'SVHN':
    train_loader = dl.SVHN(train=True)
    val_loader, test_loader = dl.SVHN(train=False, augm_flag=False, val_size=2000)
elif args.dataset == 'CIFAR100':
    train_loader = dl.CIFAR100(train=True)
    val_loader, test_loader = dl.CIFAR100(train=False, augm_flag=False, val_size=2000)

targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
targets_val = torch.cat([y for x, y in val_loader], dim=0).numpy()

print(len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

model, likelihood = dkl.get_dkl_model(dataset=args.dataset)

model = model.cuda()
likelihood = likelihood.cuda()


if args.dataset == 'MNIST':
    lr = 1e-3
    opt = torch.optim.Adam([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 5e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, weight_decay=0)
else:
    lr = 0.1
    opt = torch.optim.SGD([
        {'params': model.feature_extractor.parameters(), 'weight_decay': 5e-4},
        {'params': model.gp_layer.hyperparameters(), 'lr': lr * 0.01},
        {'params': model.gp_layer.variational_parameters()},
        {'params': likelihood.parameters()},
    ], lr=lr, momentum=0.9, weight_decay=0)
    
mll = gpytorch.mlls.VariationalELBO(likelihood, model.gp_layer, num_data=len(train_loader.dataset))

pbar = trange(100)

for epoch in pbar:
    if epoch+1 in [50,75,90]:
        for group in opt.param_groups:
            group['lr'] *= .1

    train_loss = 0
    n = 0
    
    model.train()
    likelihood.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.long().cuda()
        
        opt.zero_grad()
        output = model(data)
        loss = -mll(output, target)
        loss.backward()
        opt.step()

        train_loss += loss.item()
        n += 1

    train_loss /= n
    
    # Validation accuracy
    # -------------------
    model.eval()
    likelihood.eval()
    
    with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
        pred_val = []
        
        for data, target in val_loader:
            if torch.cuda.is_available():
                data, target = data.cuda(), target.cuda()
            output = likelihood(model(data))  # This gives us 20 samples from the predictive distribution
            pred_val_ = output.probs.mean(0).cpu().numpy()  # Taking the mean over all of the sample we've drawn
            pred_val.append(pred_val_)
            
        pred_val = np.concatenate(pred_val, 0)
        acc_val = np.mean(np.argmax(pred_val, 1) == targets_val)*100

    pbar.set_description(f'[Epoch: {epoch+1}; train_loss: {train_loss:.4f}; val_acc: {acc_val:.1f}]')

torch.save({'model': model.state_dict(), 'likelihood': likelihood.state_dict()}, f'./pretrained_models/{args.dataset}_dkl.pt')

state = torch.load(f'./pretrained_models/{args.dataset}_dkl.pt')
model.load_state_dict(state['model'])
likelihood.load_state_dict(state['likelihood'])
model.eval()
likelihood.eval()


print()

# Test (in-distribution)
# ----------------------
model.eval()
likelihood.eval()

with torch.no_grad(), gpytorch.settings.num_likelihood_samples(20):
    py_in = []
    
    for data, target in test_loader:
        if torch.cuda.is_available():
            data, target = data.cuda(), target.cuda()
        output = likelihood(model(data))  # This gives us 20 samples from the predictive distribution
        py_in_ = output.probs.mean(0).cpu().numpy()  # Taking the mean over all of the sample we've drawn
        py_in.append(py_in_)

    py_in = np.concatenate(py_in, 0)

acc_in = np.mean(np.argmax(py_in, 1) == targets)*100
mmc = np.maximum(py_in, 1-py_in).mean()*100
print(f'[In, MAP] Accuracy: {acc_in:.3f}; MMC: {mmc:.3f}')
