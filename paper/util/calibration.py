import torch
import torch.nn.functional as F
import numpy as np


def temp_scaling_binary(logits, targets, init_T=1):
    logits = torch.from_numpy(logits)
    targets = torch.from_numpy(targets).float()

    T = torch.tensor(1).float()
    T.requires_grad = True

    opt = torch.optim.LBFGS([T], lr=1)

    def eval():
        opt.zero_grad()
        loss = F.binary_cross_entropy_with_logits(logits/T, targets)
        loss.backward()
        return loss

    opt.step(eval)

    return T.detach().item()
