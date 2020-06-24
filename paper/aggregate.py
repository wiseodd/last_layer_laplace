import numpy as np
import pickle
import os, sys, argparse
from util.tables import *


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', help='Pick one \\{"MNIST", "CIFAR10", "SVHN", "CIFAR100"\\}', default='MNIST')
parser.add_argument('--type', help='Pick one \\{"plain", "ACET", "OE"\\}', default='plain')
parser.add_argument('--show', help='Pick one \\{"all", "mean", "std"\\}', default='all')
args = parser.parse_args()


path = './results/'
dicts_ood, dicts_cal = [], []
_, _, filenames = next(os.walk(path))

for fname in [fname for fname in filenames if '_ood_' in fname]:
    if f'_{args.dataset.lower()}_' in fname.lower() and args.type.lower() in fname.lower():
        print(fname)
        with open(f'{path}/{fname}', 'rb') as f:
            # Preprocess None
            d = pickle.load(f)
            for k in d.keys():
                d[k] = [-1 if x is None else x for x in d[k]]
            dicts_ood.append(d)

print()

for fname in [fname for fname in filenames if '_cal_' in fname]:
    if f'_{args.dataset.lower()}_' in fname.lower() and args.type.lower() in fname.lower():
        print(fname)
        with open(f'{path}/{fname}', 'rb') as f:
            d = pickle.load(f)
            dicts_cal.append(d)

print(len(dicts_ood), len(dicts_cal))

if args.dataset == 'MNIST':
    tab_ood = {'MNIST - MNIST': {'mean': [], 'std': []},
               'MNIST - EMNIST': {'mean': [], 'std': []},
               'MNIST - FMNIST': {'mean': [], 'std': []},
               'MNIST - FarAway': {'mean': [], 'std': []},
               'MNIST - Adversarial': {'mean': [], 'std': []},
               'MNIST - FarAwayAdv': {'mean': [], 'std': []}}
elif args.dataset == 'CIFAR10':
    tab_ood = {'CIFAR10 - CIFAR10': {'mean': [], 'std': []},
               'CIFAR10 - SVHN': {'mean': [], 'std': []},
               'CIFAR10 - LSUN': {'mean': [], 'std': []},
               'CIFAR10 - FarAway': {'mean': [], 'std': []},
               'CIFAR10 - Adversarial': {'mean': [], 'std': []},
               'CIFAR10 - FarAwayAdv': {'mean': [], 'std': []}}
elif args.dataset == 'SVHN':
    tab_ood = {'SVHN - SVHN': {'mean': [], 'std': []},
               'SVHN - CIFAR10': {'mean': [], 'std': []},
               'SVHN - LSUN': {'mean': [], 'std': []},
               'SVHN - FarAway': {'mean': [], 'std': []},
               'SVHN - Adversarial': {'mean': [], 'std': []},
               'SVHN - FarAwayAdv': {'mean': [], 'std': []}}
elif args.dataset == 'CIFAR100':
    tab_ood = {'CIFAR100 - CIFAR100': {'mean': [], 'std': []},
               'CIFAR100 - SVHN': {'mean': [], 'std': []},
               'CIFAR100 - LSUN': {'mean': [], 'std': []},
               'CIFAR100 - FarAway': {'mean': [], 'std': []},
               'CIFAR100 - Adversarial': {'mean': [], 'std': []},
               'CIFAR100 - FarAwayAdv': {'mean': [], 'std': []}}

for k in tab_ood.keys():
    # kk = k.replace(' ', '') if args.dataset not in ['MNIST', 'CIFAR10'] else k
    kk = k

    # Hotfix for CIFAR100
    if args.dataset == 'CIFAR100':
        if k == 'CIFAR100 - CIFAR100':
            for d in dicts_ood:
                d[kk][3] = -1
    # End hotfix for CIFAR100

    tab_ood[k]['mean'] = list(np.mean([x[kk] for x in dicts_ood], axis=0) * 100)
    tab_ood[k]['std'] = list(np.std([x[kk] for x in dicts_ood], axis=0) * 100)


print()
if args.show == 'all':
    print_latex_ood_entries_aggregate(tab_ood)
elif args.show == 'mean':
    print_latex_ood_entries_mean(tab_ood)
elif args.show == 'std':
    print_latex_ood_entries_std(tab_ood)
print()

print()
for k in dicts_cal[0].keys():
    mean_ece = np.mean([d[k][0] for d in dicts_cal])
    std_ece = np.std([d[k][0] for d in dicts_cal])
    mean_mce = np.mean([d[k][1] for d in dicts_cal])
    std_mce = np.std([d[k][1] for d in dicts_cal])

    print(f'[{k}, ECE]: {mean_ece:.1f}$\\pm${std_ece:.1f}')
    print(f'[{k}, MCE]: {mean_mce:.1f}$\\pm${std_mce:.1f}')
print()
