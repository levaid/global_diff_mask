import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import KDEpy
import torch.nn
from custom_modules import MaskedLinear, MaskedConv2d

def plot_mask_and_weight(net: torch.nn.Module, current_iteration: int):

    currentiter = f'{current_iteration:05}'
    plt.figure(facecolor='w')
    for name, param in net.named_modules():
        if type(param) in [MaskedConv2d, MaskedLinear]:
            data = param.mask.detach().cpu().numpy().flatten()
            x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
            plt.plot(x, y, label=name)

    plt.title(f'Masks\' distribution in iter {currentiter}')
    plt.legend()
    plt.ylim((0, 2))
    plt.xlim((-4, 6))
    plt.savefig(f'run_details/iter_mask_{currentiter}.png', dpi=150)

    plt.figure(facecolor='w')
    for name, param in net.named_modules():
        if type(param) in [MaskedConv2d, MaskedLinear]:
            data = param.weight.detach().cpu().numpy().flatten()
            x, y = KDEpy.FFTKDE(bw=0.1).fit(data).evaluate()
            plt.plot(x, y, label=name)

    plt.title(f'Weights in epoch {currentiter}')
    plt.legend()
    plt.xlim((-4, 6))
    plt.savefig(f'run_details/iter_weight_{currentiter}.png', dpi=150)
    plt.close('all')