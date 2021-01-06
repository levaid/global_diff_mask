import subprocess
from sklearn.model_selection import ParameterGrid
import random
from multiprocessing.dummy import Pool

params_1 = {'num_first_stage': [0, 1, 2, 5, 10, 20, 50], 'initial_mode': ['weight', 'mask'], 'num_second_stage': [50],
            'discretize': [True], 'discretization_quantile': [0.1, 0.3, 0.5, 0.7, 0.9], 'discretization_method': ['from_weight'], 'sigmoid': [False], 'neptune': [True]}

params_2 = {'num_first_stage': [0, 1, 2, 5, 10, 20, 50], 'initial_mode': ['weight', 'mask'], 'num_second_stage': [50],
            'discretize': [True], 'discretization_quantile': [0.1, 0.3, 0.5, 0.7, 0.9], 'discretization_method': ['from_mask'], 'sigmoid': [False, True], 'neptune': [True]}

params_3 = {'num_first_stage': [0, 1, 2, 5, 10, 20, 50], 'initial_mode': ['weight', 'mask'], 'num_second_stage': [50],
            'discretize': [False], 'sigmoid': [False, True], 'neptune': [True]}

param_grid = list(ParameterGrid(params_1)) + list(ParameterGrid(params_2))
random.shuffle(param_grid)
cmdline_form_arguments = [[arg for pair in [(f'--{k}', str(v)) for k, v in param_conf.items()] for arg in pair] for param_conf in param_grid]


def run_with_args(cmdline_args):
    subprocess.run(['python', 'main.py'] + cmdline_args)


p = Pool(4)

for output, error in p.imap(run_with_args, cmdline_form_arguments):
    pass
