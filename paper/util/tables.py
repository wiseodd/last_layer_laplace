import numpy as np


def save_res_ood(tab, mmc, auroc=None):
    tab.append(mmc)
    tab.append(auroc)
    return tab


def print_latex_ood_entries(d: dict):
    for k in d.keys():
        entries = [f'{v:.1f}' if v is not None else '-' for v in d[k]]
        print(k + ' & ' + ' & '.join(entries) + ' \\\\')


def print_latex_ood_entries_mean(d: dict):
    for k in d.keys():
        entries = [f'{v:.1f}' if v != -100 else '-' for v in d[k]['mean']]
        print(k + ' & ' + ' & '.join(entries) + ' \\\\')


def print_latex_ood_entries_std(d: dict):
    for k in d.keys():
        entries = [f'{s:.1f}' if m != -100 else '-' for m, s in zip(d[k]['mean'], d[k]['std'])]
        print(k + ' & ' + ' & '.join(entries) + ' \\\\')


def print_latex_ood_entries_aggregate(d: dict):
    for k in d.keys():
        if len(d[k]['mean']) == 0:
            continue

        in_dset, out_dset = k.split(' - ')
        compute_max = in_dset != out_dset

        if compute_max:
            # Get min of MMC (even and 0th elements) and max of AUR (odd elements)
            lst = np.array(d[k]['mean'].copy()).round(1)
            lst[list(range(1, len(lst), 2))] = np.inf
            min_mmc_idxs = np.argwhere(lst == np.min(lst)).flatten()

            lst = np.array(d[k]['mean'].copy()).round(1)
            lst[list(range(0, len(lst), 2))] = -np.inf
            max_aur_idxs = np.argwhere(lst == np.max(lst)).flatten()

        entries = []
        for i, (m, s) in enumerate(zip(d[k]['mean'], d[k]['std'])):
            if m != -100:
                if compute_max and (i in min_mmc_idxs or i in max_aur_idxs):
                    str = f'\\textbf{{{m:.1f}$\\pm${s:.1f}}}'
                else:
                    str = f'{m:.1f}$\\pm${s:.1f}'
            else:
                str = '-'
            entries.append(str)
        print(k + ' & ' + ' & '.join(entries) + ' \\\\')


def save_res_cal(tabs, ece, mce):
    tabs[0].append(ece)
    tabs[1].append(mce)
    return tabs
