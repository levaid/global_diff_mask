import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import KDEpy
import torch.nn
from custom_modules import MaskedLinear, MaskedConv2d


sns.set_style('whitegrid')
matplotlib.use('Agg')


def plot_mask_and_weight(net: torch.nn.Module, current_iteration: int):

    currentiter = f'{current_iteration:05}'
    plt.figure(facecolor='w')
    for name, param in net.named_modules():
        if type(param) in [MaskedConv2d, MaskedLinear]:
            data = param.mask.detach().cpu().numpy().flatten()
            # x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
            # plt.plot(x, y, label=name)
            plt.hist(data, label=name, histtype='step', bins=100, cumulative=False, density=True)

    plt.title(f'Masks\' distribution in iter {currentiter}')
    plt.legend()
    # plt.ylim((0, 2))
    # plt.xlim((-5, 6))
    plt.savefig(f'run_details/iter_mask_{currentiter}.png', dpi=150)

    plt.figure(facecolor='w')
    for name, param in net.named_modules():
        if type(param) in [MaskedConv2d, MaskedLinear]:
            data = param.weight.detach().cpu().numpy().flatten()
            x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
            plt.plot(x, y, label=name)
            # plt.hist(data, label=name)

    plt.title(f'Weights in epoch {currentiter}')
    plt.legend()
    # plt.xlim((-6, 6))
    plt.savefig(f'run_details/iter_weight_{currentiter}.png', dpi=150)
    plt.close('all')


def plot_pruner(pruner: torch.nn.Module, current_iteration: int):

    currentiter = f'{current_iteration:05}'
    plt.figure(facecolor='w')
    for name, param in pruner.named_modules():
        if type(param) in [torch.nn.Linear]:
            data = param.weight.detach().cpu().numpy().flatten()
            x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
            plt.plot(x, y, label=name)

    plt.title(f'Pruner\'s weight distribution in iter {currentiter}')
    plt.legend()
    plt.xlim((-6, 6))
    plt.savefig(f'run_details/iter_pruner_{currentiter}.png', dpi=150)
