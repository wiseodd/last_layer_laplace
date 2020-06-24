##########################################################################
#
#  Taken with modifications from
#  https://github.com/wjmaddox/swa_gaussian/
#
##########################################################################


import torch
from torch import nn, optim, autograd
from torch.nn import functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from util.evaluation import get_auroc
from util.kfac import KFAC
from math import *
from tqdm import tqdm, trange
import numpy as np


class KFLA(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py
    """

    def __init__(self, base_model):
        super().__init__()

        self.net = type(base_model)()
        self.net.load_state_dict(base_model.state_dict())
        self.net.eval()
        self.params = []
        self.net.apply(lambda module: laplace_parameters(module, self.params))

    def forward(self, x):
        return self.net.forward(x)

    def forward_sample(self, x):
        self.sample()
        return self.net.forward(x)

    def sample(self, scale=1, require_grad=False):
        for module, name in self.params:
            mod_class = module.__class__.__name__
            if mod_class not in ['Linear', 'Conv2d']:
                continue

            if name == 'bias':
                w = module.__getattr__(f'{name}_mean')
            else:
                M = module.__getattr__(f'{name}_mean')
                U_half = module.__getattr__(f'{name}_U_half')
                V_half = module.__getattr__(f'{name}_V_half')

                if len(M.shape) == 1:
                    M_ = M.unsqueeze(1)
                elif len(M.shape) > 2:
                    M_ = M.reshape(M.shape[0], np.prod(M.shape[1:]))
                else:
                    M_ = M

                E = torch.randn(*M_.shape)
                w = M_ + scale * U_half @ E @ V_half
                w = w.reshape(*M.shape)

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, torch.nn.Parameter(w))
            getattr(module, name)
        else:
            for module, name in self.params:
                mod_class = module.__class__.__name__
                if mod_class not in ['Linear', 'Conv2d']:
                    continue

                M = module.__getattr__(f'{name}_mean')
                U = module.__getattr__(f'{name}_U_half')
                V = module.__getattr__(f'{name}_V_half')

    def estimate_variance(self, train_loader, var0):
        tau = 1/var0
        criterion = nn.CrossEntropyLoss()
        opt = KFAC(self.net)
        U_halfs = {}
        V_halfs = {}

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        # for x, y in tqdm(train_loader):
        for x, y in train_loader:
            x = x.cuda(non_blocking=True)

            output = self(x)
            distribution = torch.distributions.Categorical(logits=output)
            y = distribution.sample()

            loss = criterion(output, y)
            loss.backward()
            opt.step()

        with torch.no_grad():
            for group in opt.param_groups:
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None

                module = group['mod']
                state = opt.state[module]

                U = state['ggt']
                V = state['xxt']

                m, n = int(U.shape[0]), int(V.shape[0])

                # Add priors
                n_data = len(train_loader.dataset)
                U = sqrt(n_data)*U + sqrt(tau)*torch.eye(m, device='cuda')
                V = sqrt(n_data)*V + sqrt(tau)*torch.eye(n, device='cuda')

                U_half = torch.cholesky(torch.inverse(U), upper=False)
                V_half = torch.cholesky(torch.inverse(V), upper=True)

                U_halfs[(module, 'weight')] = U_half
                V_halfs[(module, 'weight')] = V_half


        for module, name in self.params:
            mod_class = module.__class__.__name__
            if mod_class not in ['Linear', 'Conv2d']:
                continue

            if name == 'bias':
                continue

            U_half = U_halfs[(module, name)]
            V_half = V_halfs[(module, name)]

            module.__getattr__(f'{name}_U_half').copy_(U_half)
            module.__getattr__(f'{name}_V_half').copy_(V_half)

    def estimate_variance_batch(self, X, criterion, dist, var0):
        tau = 1/var0
        opt = KFAC(self.net)
        U_halfs = {}
        V_halfs = {}

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        output = self(X)
        distribution = dist(logits=output)
        y = distribution.sample()
        loss = criterion(output, y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            for group in opt.param_groups:
                if len(group['params']) == 2:
                    weight, bias = group['params']
                else:
                    weight = group['params'][0]
                    bias = None

                module = group['mod']
                state = opt.state[module]

                U = state['ggt']
                V = state['xxt']

                m, n = int(U.shape[0]), int(V.shape[0])

                # Add priors
                n_data = len(X)
                U = sqrt(n_data)*U + sqrt(tau)*torch.eye(m)
                V = sqrt(n_data)*V + sqrt(tau)*torch.eye(n)

                U_half = torch.cholesky(torch.inverse(U), upper=False)
                V_half = torch.cholesky(torch.inverse(V), upper=True)

                U_halfs[(module, 'weight')] = U_half
                V_halfs[(module, 'weight')] = V_half


        for module, name in self.params:
            mod_class = module.__class__.__name__
            if mod_class not in ['Linear', 'Conv2d']:
                continue

            if name == 'bias':
                continue

            U_half = U_halfs[(module, name)]
            V_half = V_halfs[(module, name)]

            module.__getattr__(f'{name}_U_half').copy_(U_half)
            module.__getattr__(f'{name}_V_half').copy_(V_half)


def laplace_parameters(module, params):
    mod_class = module.__class__.__name__
    if mod_class not in ['Linear', 'Conv2d']:
        return

    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            # print(module, name)
            continue

        data = module._parameters[name].data
        m, n = int(data.shape[0]), int(np.prod(data.shape[1:]))
        module._parameters.pop(name)
        module.register_buffer(f'{name}_mean', data)
        module.register_buffer(f'{name}_U_half', torch.zeros([m, m]))
        module.register_buffer(f'{name}_V_half', torch.zeros([n, n]))
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))


@torch.no_grad()
def predict(test_loader, model, n_samples=100):
    py = []

    # for x, y in tqdm(test_loader):
    for x, y in test_loader:
        x, y = x.cuda(), y.cuda()

        # MC-integral
        py_ = 0
        for _ in range(n_samples):
            out = model.forward_sample(x)
            py_ += torch.softmax(out, 1)

        py_ /= n_samples
        py.append(py_)

    return torch.cat(py, dim=0)
