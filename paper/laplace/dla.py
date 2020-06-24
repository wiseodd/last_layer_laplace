import torch
from torch import nn
import torch.nn.functional as F
from tqdm import tqdm, trange
import numpy as np
from math import *
import laplace.util as lutil
from util.evaluation import get_calib


class DiagLaplace(nn.Module):
    """
    Taken, with modification, from:
    https://github.com/wjmaddox/swa_gaussian/blob/master/swag/posteriors/diag_laplace.py
    """

    def __init__(self, base_model):
        super().__init__()

        self.net = base_model
        self.params = []
        self.net.apply(lambda module: dla_parameters(module, self.params))
        self.hessians = None
        self.n_params = sum(p.numel() for p in base_model.parameters())

    def forward(self, x):
        return self.net.forward(x)

    def forward_sample(self, x):
        self.sample()
        return self.net.forward(x)

    def sample(self, scale=1, require_grad=False):
        for module, name in self.params:
            mean = module.__getattr__(f'{name}_mean')
            var = module.__getattr__(f'{name}_var')

            eps = torch.randn(*mean.shape, device='cuda')
            w = mean + scale * torch.sqrt(var) * eps

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, w)
            getattr(module, name)
        else:
            for module, name in self.params:
                mean = module.__getattr__(f'{name}_mean')
                var = module.__getattr__(f'{name}_var')

    def sample_raw(self, var0, scale=1, require_grad=False):
        tau = 1/var0

        for module, name in self.params:
            mean = module.__getattr__(f'{name}_mean')
            var = module.__getattr__(f'{name}_var')

            eps = torch.randn(*mean.shape, device='cuda')
            w = mean + scale * torch.sqrt(1/(tau + var)) * eps

            if require_grad:
                w.requires_grad_()

            module.__setattr__(name, w)

    def estimate_variance(self, var0, invert=True):
        tau = 1/var0

        for module, name in self.params:
            h = self.hessians[(module, name)].clone()
            var = (1 / (h + tau)) if invert else h
            module.__getattr__(f'{name}_var').copy_(var)

    def get_hessian(self, train_loader, binary=False):
        criterion = nn.BCEWithLogitsLoss(reduction='mean') if binary else nn.CrossEntropyLoss(reduction='mean')
        diag_hess = dict()

        for module, name in self.params:
            var = module.__getattr__(f'{name}_var')
            diag_hess[(module, name)] = torch.zeros_like(var)

        # Populate parameters with the means
        self.sample(scale=0, require_grad=True)

        for x, y in tqdm(train_loader):
            x = x.cuda()

            self.net.zero_grad()
            out = self(x).squeeze()

            if binary:
                distribution = torch.distributions.Binomial(logits=out)
            else:
                distribution = torch.distributions.Categorical(logits=out)

            y = distribution.sample()
            loss = criterion(out, y)
            loss.backward()

            for module, name in self.params:
                grad = module.__getattr__(name).grad
                diag_hess[(module, name)] += grad**2

        n_data = len(train_loader.dataset)

        self.hessians = diag_hess

        return diag_hess

    def gridsearch_var0(self, val_loader, ood_loader, interval, n_classes=10, lam=1):
        vals, var0s = [], []
        pbar = tqdm(interval)

        for var0 in pbar:
            self.estimate_variance(var0)

            if n_classes == 2:
                preds_in, y_in = lutil.predict_binary(val_loader, self, 10, return_targets=True)
                preds_out = lutil.predict_binary(ood_loader, self, 10)

                loss_in = F.binary_cross_entropy(preds_in.squeeze(), y_in.float())
                loss_out = F.binary_cross_entropy(preds_out.squeeze(), torch.ones_like(y_in)*0.5)
            else:
                preds_in, y_in = lutil.predict(val_loader, self, n_samples=5, return_targets=True)
                preds_out = lutil.predict(ood_loader, self, n_samples=5)

                loss_in = F.nll_loss(torch.log(preds_in + 1e-8), y_in)
                loss_out = -torch.log(preds_out + 1e-8).mean()

            loss = loss_in + lam * loss_out

            vals.append(loss)
            var0s.append(var0)

            pbar.set_description(f'var0: {var0:.5f}, Loss-in: {loss_in:.3f}, Loss-out: {loss_out:.3f}, Loss: {loss:.3f}')

        best_var0 = var0s[np.argmin(vals)]

        return best_var0


def dla_parameters(module, params):
    for name in list(module._parameters.keys()):
        if module._parameters[name] is None:
            # print(module, name)
            continue

        data = module._parameters[name].data
        module._parameters.pop(name)
        module.register_buffer(f'{name}_mean', data)
        module.register_buffer(f'{name}_var', data.new(data.size()).zero_())
        module.register_buffer(name, data.new(data.size()).zero_())

        params.append((module, name))


