##########################################################################
#
#  Taken from https://github.com/AlexMeinke/certified-certain-uncertainty
#
##########################################################################


import torch
from torchvision import datasets, transforms

import numpy as np
import scipy.ndimage.filters as filters


class Transpose(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return data.transpose(-1,-2)


class Gray(object):
    def __init__(self):
        pass
    def __call__(self, data):
        return data.mean(-3, keepdim=True)


class PermutationNoise(object):
    def __init__(self):
        pass
    def __call__(self, data):
        shape = data.shape
        new_data = 0*data
        idx = [torch.tensor(np.random.permutation(np.prod(shape[-2:])))]
        for i, x in enumerate(data):
            new_data[i] = (x.view(np.prod(shape[-2:]))[idx]).view(shape[-2:])
        return new_data


class GaussianFilter(object):
    def __init__(self):
        pass
    def __call__(self, data):
        sigma = 1.+1.5*torch.rand(1).item()
        return torch.tensor(filters.gaussian_filter(data, sigma, mode='reflect'))


class ContrastRescaling(object):
    def __init__(self):
        pass
    def __call__(self, data):
        gamma = 5+ 25.*torch.rand(1).item()
        return torch.sigmoid(gamma*(data-.5))


class AdversarialNoise(object):
    def __init__(self, model, device, epsilon=0.3):
        self.model = model
        self.pretransform = dl.noise_transform
        self.device = device
        self.epsilon = epsilon

    def __call__(self, data):
        perturbed = tt.generate_adv_noise(self.model, self.epsilon,
                                       device=self.device, batch_size=1,
                                       norm=20, num_of_it=40,
                                       alpha=0.01, seed_images=data.unsqueeze(0))
        return perturbed.squeeze(0)
