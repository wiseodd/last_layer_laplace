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
from backpack import backpack, extend
from backpack.extensions import KFAC
from math import *
from tqdm import tqdm, trange
import numpy as np


def get_posterior(model, train_loader, var0, mnist=False, batch_size=128):
    W = list(model.parameters())[-2]
    b = list(model.parameters())[-1]
    m, n = W.shape
    lossfunc = torch.nn.CrossEntropyLoss()

    tau = 1/var0

    extend(lossfunc, debug=False)
    extend(model.linear if not mnist else model.fc2, debug=False)

    with backpack(KFAC()):
        U, V = torch.zeros(m, m, device='cuda'), torch.zeros(n, n, device='cuda')
        B = torch.zeros(m, m, device='cuda')

        # for i, (x, y) in tqdm(enumerate(train_loader)):
        for i, (x, y) in enumerate(train_loader):
            x, y = x.cuda(), y.cuda()

            model.zero_grad()
            lossfunc(model(x), y).backward()

            with torch.no_grad():
                # Hessian of weight
                U_, V_ = W.kfac
                B_ = b.kfac[0]

                # U_ = sqrt(batch_size)*U_ + sqrt(tau)*torch.eye(m, device='cuda')
                # V_ = sqrt(batch_size)*V_ + sqrt(tau)*torch.eye(n, device='cuda')
                # B_ = batch_size*B_ + tau*torch.eye(m, device='cuda')

                rho = min(1-1/(i+1), 0.95)

                U = rho*U + (1-rho)*U_
                V = rho*V + (1-rho)*V_
                B = rho*B + (1-rho)*B_


    # Predictive distribution
    with torch.no_grad():
        M_W_post = W.t()
        M_b_post = b

        # Add priors
        n_data = len(train_loader.dataset)
        U = sqrt(n_data)*U + sqrt(tau)*torch.eye(m, device='cuda')
        V = sqrt(n_data)*V + sqrt(tau)*torch.eye(n, device='cuda')
        B = n_data*B + tau*torch.eye(m, device='cuda')

        # Covariances for Laplace
        U_post = torch.inverse(V)  # Interchanged since W is transposed
        V_post = torch.inverse(U)
        B_post = torch.inverse(B)

    return M_W_post, M_b_post, U_post, V_post, B_post


@torch.no_grad()
def predict(dataloader, model, M_W, M_b, U, V, B, n_samples=100):
    py = []

    for x, y in dataloader:
        x, y = x.cuda(), y.cuda()
        phi = model.features(x)

        mu_pred = phi @ M_W + M_b
        Cov_pred = torch.diag(phi @ U @ phi.t()).view(-1, 1, 1) * V.unsqueeze(0) + B.unsqueeze(0)

        post_pred = MultivariateNormal(mu_pred, Cov_pred)

        # MC-integral
        py_ = 0

        for _ in range(n_samples):
            f_s = post_pred.rsample()
            py_ += torch.softmax(f_s, 1)

        py_ /= n_samples

        py.append(py_)

    return torch.cat(py, dim=0)
