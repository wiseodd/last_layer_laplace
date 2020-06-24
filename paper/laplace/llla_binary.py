import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from models import resnet_orig as resnet
from models import hendrycks as resnet_oe
from util.evaluation import get_auroc
from util import hessian
from backpack import backpack, extend
from backpack.extensions import KFAC
from math import *
from tqdm import tqdm, trange
import numpy as np


def get_hessian(model, train_loader, mnist=False):
    w = list(model.parameters())[-2].squeeze()
    b = list(model.parameters())[-1]
    mu = torch.cat([b, w])

    H = 0
    rho = 0.95

    for i, (x, y) in tqdm(enumerate(train_loader)):
        x, y = x.cuda(), y.float().cuda()
        m = len(x)

        model.zero_grad()
        phi = torch.cat([torch.ones(m, 1, device='cuda'), model.features(x)], dim=1)
        loss = F.binary_cross_entropy_with_logits(phi @ mu, y, reduction='sum')
        H_ = hessian.exact_hessian(loss, [mu]).detach()
        H = rho*H + (1-rho)*H_

    n_data = len(train_loader.dataset)
    # H = n_data*H

    return [mu, H.cuda()]


# @torch.no_grad()
def estimate_variance(var0, hessians, invert=True):
    if not invert:
        return hessians

    tau = 1/var0
    mu, H = hessians
    n = mu.shape[0]

    # Covariances for Laplace
    S = torch.inverse(H + tau*torch.eye(n, device='cuda'))

    return [mu, S]


def gridsearch_var0(model, hessians, val_loader, ood_loader, interval, lam=1):
    targets = torch.cat([y for x, y in val_loader], dim=0).float().cuda()
    targets_out = torch.ones_like(targets)*0.5
    vals, var0s = [], []
    pbar = tqdm(interval)

    for var0 in pbar:
        mu, S = estimate_variance(var0, hessians)

        preds = predict(val_loader, model, mu, S)
        preds_out = predict(ood_loader, model, mu, S)

        loss_in = F.binary_cross_entropy(preds, targets).detach().item()
        loss_out = F.binary_cross_entropy(preds_out, targets_out).detach().item()
        loss = loss_in + lam * loss_out

        vals.append(loss)
        var0s.append(var0)

        pbar.set_description(f'var0: {var0:.5f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}')

    best_var0 = var0s[np.argmin(vals)]

    return best_var0


@torch.no_grad()
def predict(dataloader, model, mu, S, apply_sigm=True, delta=1):
    py = []

    for x, y in dataloader:
        x, y = delta*x.cuda(), y.cuda()
        m = len(x)
        phi = torch.cat([torch.ones(m, 1, device='cuda'), model.features(x)], dim=1)

        mu_pred = phi @ mu
        var_pred = torch.diag(phi @ S @ phi.t())
        z = (1+pi*var_pred/8)**(-1/2)*mu_pred
        py_ = torch.sigmoid(z) if apply_sigm else z

        py.append(py_)

    return torch.cat(py, dim=0)
