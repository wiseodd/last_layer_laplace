import numpy as np
import torch
from torch import nn, optim, autograd
import torch.nn.functional as F
import torch.utils.data as data_utils
from tqdm import tqdm
import os


def generate_adv_samples(model, x0, epsilon, lr=0.01, iters=40, p='inf', clamp=True):
    """
    Generate adversarial samples from initial samples x0
    via a confidence maximization in an epsilon-ball around x0
    """
    model.eval()

    x = x0.clone()
    x.requires_grad = True

    for _ in range(iters):
        y = F.softmax(model(x), 1)
        loss = torch.sum(y.max(1)[0])
        grad = autograd.grad(loss, x)[0]

        if p == 'inf':
            x = x + lr*torch.sign(grad)

            # Project onto epsilon-ball (wrt l_inf norm) around x0
            x = torch.max(torch.min(x, x0+epsilon), x0-epsilon)
            x = torch.clamp(x, 0, 1) if clamp else x
        else:
            x = x + lr*grad

            delta = x - x0
            norm = torch.norm(delta, p=p, dim=1)
            delta = torch.min(epsilon*torch.ones_like(norm), epsilon/norm)[:, None]*d
            x = x + delta
            x = torch.clamp(x, 0, 1) if clamp else x

    return x


def create_adv_loader(model, dataloader, fname, delta=1, epsilon=0.5, n_classes=10, load=False, p='inf'):
    if load:
        x_adv = torch.from_numpy(np.load(f'./cache/{fname}_{delta}.npy')).float()
    else:
        clamp = (delta == 1)  # Clamp to the unit box in the nonasymptotic regime
        x_adv = []
        for x, y in tqdm(dataloader):
            x = x.cuda()
            x_adv.append(generate_adv_samples(model, delta*x, epsilon=epsilon, clamp=clamp).detach().cpu())
        x_adv = torch.cat(x_adv, 0)

        if not os.path.exists('./cache/'):
            os.makedirs('./cache/')
        np.save(f'./cache/{fname}_{delta}.npy', x_adv.cpu().detach().numpy())

    adv_noise_set = data_utils.TensorDataset(x_adv, torch.zeros(len(x_adv), n_classes))
    return data_utils.DataLoader(adv_noise_set, batch_size=128, shuffle=False)
