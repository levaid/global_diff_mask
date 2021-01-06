import subprocess
from sklearn.model_selection import ParameterGrid
import random
from multiprocessing.dummy import Pool

params = {'num_training_epochs': [1, 2], 'initial_mode': ['weight', 'mask'], 'num_pruning_epochs': [0, 1], 'discretize': [False],
          'lr_net': [0.0005], 'lr_mask': [0.001], 'lr_pruner': [0.0005], 'sigmoid': [True, False], 'neptune': [False]}

param_grid = list(ParameterGrid(params))
random.shuffle(param_grid)
cmdline_form_arguments = [[arg for pair in [(f'--{k}', str(v)) for k, v in param_conf.items()] for arg in pair] for param_conf in param_grid]


def run_with_args(cmdline_args):
    subprocess.run(['python', 'main.py'] + cmdline_args)


p = Pool(2)

for output, error in p.imap(run_with_args, cmdline_form_arguments):
    pass
