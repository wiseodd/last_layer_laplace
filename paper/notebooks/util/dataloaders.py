##########################################################################
#
#  Taken from https://github.com/AlexMeinke/certified-certain-uncertainty
#
##########################################################################


import torch
from torchvision import datasets, transforms
import torch.utils.data as data_utils

import numpy as np
import scipy.ndimage.filters as filters
import util.preproc as pre

from bisect import bisect_left


train_batch_size = 128
test_batch_size = 128

path = '../../Datasets'


def MNIST(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.MNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader


def EMNIST(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor(), pre.Transpose()] #EMNIST is rotated 90 degrees from MNIST
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.EMNIST(path, split='letters',
                              train=train, transform=transform, download=True)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def FMNIST(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=2),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def GrayCIFAR10(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.Compose([
                            transforms.Resize(28),
                            transforms.ToTensor(),
                            pre.Gray()
                       ])]
    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(28, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR10(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def Noise(dataset, train=True, batch_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform = transforms.Compose([
                    transforms.ToTensor(),
                    pre.PermutationNoise(),
                    pre.GaussianFilter(),
                    pre.ContrastRescaling()
                    ])
    if dataset=='MNIST':
        dataset = datasets.MNIST(path, train=train, transform=transform)
    elif dataset=='FMNIST':
        dataset = datasets.FashionMNIST(path, train=train, transform=transform)
    elif dataset=='SVHN':
        dataset = datasets.SVHN(path, split='train' if train else 'test', transform=transform)
    elif dataset=='CIFAR10':
        dataset = datasets.CIFAR10(path, train=train, transform=transform)
    elif dataset=='CIFAR100':
        dataset = datasets.CIFAR100(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    loader = PrecomputeLoader(loader, batch_size=batch_size, shuffle=True)
    return loader


def UniformNoise(dataset, train=False, batch_size=None):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    import torch.utils.data as data_utils

    if dataset in ['MNIST', 'FMNIST']:
        shape = (1, 28, 28)
    elif dataset in ['SVHN', 'CIFAR10', 'CIFAR100']:
        shape = (3, 32, 32)

    data = torch.rand((100*batch_size,) + shape)
    train = data_utils.TensorDataset(data, torch.zeros_like(data))
    loader = torch.utils.data.DataLoader(train, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader


def CIFAR10(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR10(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader


def CIFAR100(train=False, batch_size=None, augm_flag=False):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size
    transform_base = [transforms.ToTensor()]

    transform_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
        ] + transform_base)

    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.CIFAR100(path, train=train, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=1)
    return loader


def SVHN(train=True, batch_size=None, augm_flag=True):
    if batch_size==None:
        if train:
            batch_size=train_batch_size
        else:
            batch_size=test_batch_size

    if train:
        split = 'train'
    else:
        split = 'test'

    transform_base = [transforms.ToTensor()]
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4, padding_mode='edge'),
    ] + transform_base)
    transform_test = transforms.Compose(transform_base)

    transform_train = transforms.RandomChoice([transform_train, transform_test])

    transform = transform_train if (augm_flag and train) else transform_test

    dataset = datasets.SVHN(path, split=split, transform=transform, download=False)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=train, num_workers=4)
    return loader


# LSUN classroom
def LSUN_CR(train=False, batch_size=None, augm_flag=False):
    if train:
        print('Warning: Training set for LSUN not available')
    if batch_size is None:
        batch_size=test_batch_size

    transform_base = [transforms.ToTensor()]
    transform = transforms.Compose([
            transforms.Resize(size=(32, 32))
        ] + transform_base)
    data_dir = path
    dataset = datasets.LSUN(data_dir, classes=['classroom_val'], transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=4)
    return loader


def ImageNetMinusCifar10(train=False, batch_size=None, augm_flag=False):
    if train:
        print('Warning: Training set for ImageNet not available')
    if batch_size is None:
        batch_size=test_batch_size
    dir_imagenet = path + '/imagenet/val/'
    n_test_imagenet = 30000

    transform = transforms.ToTensor()

    dataset = torch.utils.data.Subset(datasets.ImageFolder(dir_imagenet, transform=transform),
                                            np.random.permutation(range(n_test_imagenet))[:10000])
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=False, num_workers=1)
    return loader


def PrecomputeLoader(loader, batch_size=100, shuffle=True):
    X = []
    L = []
    for x,l in loader:
        X.append(x)
        L.append(l)
    X = torch.cat(X, 0)
    L = torch.cat(L, 0)

    train = data_utils.TensorDataset(X, L)
    return data_utils.DataLoader(train, batch_size=batch_size, shuffle=shuffle)


def TinyImages(dataset, batch_size=None, shuffle=False, train=True, offset=0):
    if batch_size is None:
        batch_size = train_batch_size


    dataset_out = TinyImagesDataset(dataset, offset=offset)

    if train:
        loader = torch.utils.data.DataLoader(dataset_out, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=4)
    else:
        sampler = TinyImagesTestSampler(dataset_out)
        loader = torch.utils.data.DataLoader(dataset_out, batch_size=batch_size,
                                         shuffle=shuffle, num_workers=4, sampler=sampler)

    return loader

# Code from https://github.com/hendrycks/outlier-exposure
class TinyImagesDataset(torch.utils.data.Dataset):

    def __init__(self, dataset, offset=0):
        if dataset in ['CIFAR10', 'CIFAR100']:
            exclude_cifar = True
        else:
            exclude_cifar = False

        data_file = open('/home/alexm/scratch/80M_tiny_images/tiny_images.bin', "rb")

        def load_image(idx):
            data_file.seek(idx * 3072)
            data = data_file.read(3072)
            return np.fromstring(data, dtype='uint8').reshape(32, 32, 3, order="F")

        self.load_image = load_image
        self.offset = offset     # offset index


        transform_base = [transforms.ToTensor()]
        if dataset in ['MNIST', 'FMNIST']:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize(size=(28,28)),
                transforms.Lambda(lambda x: x.convert('L', (0.2989, 0.5870, 0.1140, 0))),
                ] + transform_base)
        else:
            transform = transforms.Compose([
                transforms.ToPILImage(),
                ] + transform_base)

        self.transform = transform
        self.exclude_cifar = exclude_cifar

        if exclude_cifar:
            self.cifar_idxs = []
            with open('./utils/80mn_cifar_idxs.txt', 'r') as idxs:
                for idx in idxs:
                    # indices in file take the 80mn database to start at 1, hence "- 1"
                    self.cifar_idxs.append(int(idx) - 1)

            # hash table option
            self.cifar_idxs = set(self.cifar_idxs)
            self.in_cifar = lambda x: x in self.cifar_idxs

    def __getitem__(self, index):
        index = (index + self.offset) % 79302016

        if self.exclude_cifar:
            while self.in_cifar(index):
                index = np.random.randint(79302017)

        img = self.load_image(index)
        if self.transform is not None:
            img = self.transform(img)
            #img = transforms.ToTensor()(img)

        return img, 0  # 0 is the class

    def __len__(self):
        return 79302017


# We want to make sure that at test time we randomly sample from images we haven't seen during training
class TinyImagesTestSampler(torch.utils.data.Sampler):
    def __init__(self, data_source):
        self.data_source = data_source
        self.min_index = 20000000
        self.max_index = 79302017

    def __iter__(self):
        return iter(iter((torch.randperm(self.max_index-self.min_index) + self.min_index ).tolist()))

    def __len__(self):
        return self.max_index - self.min_index


datasets_dict = {'MNIST':          MNIST,
                 'FMNIST':         FMNIST,
                 'cifar10_gray':   GrayCIFAR10,
                 'emnist':         EMNIST,
                 'CIFAR10':        CIFAR10,
                 'CIFAR100':       CIFAR100,
                 'SVHN':           SVHN,
                 'lsun_classroom': LSUN_CR,
                 'imagenet_minus_cifar10':  ImageNetMinusCifar10,
                 'noise': Noise,
                 'tiny': TinyImages,
                 }
