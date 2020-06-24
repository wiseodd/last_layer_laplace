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
    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape
    lossfunc = nn.CrossEntropyLoss()

    extend(lossfunc, debug=False)
    extend(model.linear if not mnist else model.fc2, debug=False)

    with backpack(KFAC()):
        U, V = torch.zeros(m, m, device='cuda'), torch.zeros(n, n, device='cuda')
        B = torch.zeros(m, m, device='cuda')

        for i, (x, y) in tqdm(enumerate(train_loader)):
        # for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                U_, V_ = W.kfac
                B_ = b.kfac[0]

                rho = min(1-1/(i+1), 0.95)

                U = rho*U + (1-rho)*U_
                V = rho*V + (1-rho)*V_
                B = rho*B + (1-rho)*B_

    n_data = len(train_loader.dataset)

    M_W = W.t()
    M_b = b
    U = sqrt(n_data)*U
    V = sqrt(n_data)*V
    B = n_data*B

    return [M_W, M_b, U, V, B]


# @torch.no_grad()
def estimate_variance(var0, hessians, invert=True):
    if not invert:
        return hessians

    tau = 1/var0

    with torch.no_grad():
        M_W, M_b, U, V, B = hessians

    m, n = U.shape[0], V.shape[0]

    # Add priors
    U_ = U + torch.sqrt(tau)*torch.eye(m, device='cuda')
    V_ = V + torch.sqrt(tau)*torch.eye(n, device='cuda')
    B_ = B + tau*torch.eye(m, device='cuda')

    # Covariances for Laplace
    U_inv = torch.inverse(V_)  # Interchanged since W is transposed
    V_inv = torch.inverse(U_)
    B_inv = torch.inverse(B_)

    return [M_W, M_b, U_inv, V_inv, B_inv]


def gridsearch_var0(model, hessians, val_loader, ood_loader, interval, n_classes=10, lam=1):
    targets = torch.cat([y for x, y in val_loader], dim=0).cuda()
    vals, var0s = [], []
    pbar = tqdm(interval)

    for var0 in pbar:
        M_W, M_b, U, V, B = estimate_variance(var0, hessians)

        preds = predict(val_loader, model, M_W, M_b, U, V, B, 10)
        preds_out = predict(ood_loader, model, M_W, M_b, U, V, B, 10)

        loss_in = F.nll_loss(torch.log(preds + 1e-8), targets)
        # loss_out = torch.mean(torch.sum(-1/n_classes * torch.log(preds_out + 1e-8), 1))
        loss_out = -torch.log(preds_out + 1e-8).mean()
        # loss_out = -(preds_out*torch.log(preds_out)).sum(1).mean()
        loss = loss_in + lam*loss_out

        vals.append(loss)
        var0s.append(var0)

        pbar.set_description(f'var0: {var0:.5f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}')

    best_var0 = var0s[np.argmin(vals)]

    return best_var0


@torch.no_grad()
def predict(dataloader, model, M_W, M_b, U, V, B, n_samples=100, delta=1, apply_softmax=True):
    py = []

    for x, y in dataloader:
        x, y = delta*x.cuda(), y.cuda()
        phi = model.features(x)

        mu_pred = phi @ M_W + M_b
        Cov_pred = torch.diag(phi @ U @ phi.t()).view(-1, 1, 1) * V.unsqueeze(0) + B.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1) if apply_softmax else f_s

        py_ /= n_samples

        py.append(py_)

    return torch.cat(py, dim=0)
