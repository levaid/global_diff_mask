import subprocess
from sklearn.model_selection import ParameterGrid
import random
import multiprocessing
from multiprocessing.dummy import Pool
import time
import threading
import networks
import os

# correct ordering of cards
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def init(lock):
    global starting
    starting = lock


params_1 = {'num_first_stage': [0, 1, 2, 5, 10, 20, 50], 'initial_mode': ['weight'], 'num_second_stage': [50],
            'discretize': [True], 'discretization_quantile': [0.1, 0.3, 0.5, 0.7, 0.9], 'discretization_method': ['from_weight'], 'sigmoid': [False],
            'neptune': [True]}

params_2 = {'num_first_stage': [0, 1, 2, 5, 10, 20, 50], 'initial_mode': ['mask'], 'num_second_stage': [50],
            'discretize': [True, False], 'discretization_quantile': [0.1, 0.3, 0.5, 0.7, 0.9], 'discretization_method': ['from_mask'], 'sigmoid': [False, True],
            'neptune': [True]}


params_3 = {'num_first_stage': [50], 'initial_mode': ['weight'], 'num_second_stage': [50], 'discretization_quantile': [0.1, 0.3, 0.5, 0.7, 0.9],
            'discretization_method': ['from_weight'], 'neptune': [True]}

param_grid = []
for params in [params_1, params_2, params_3]:
    param_grid += list(ParameterGrid(params))

random.shuffle(param_grid)
cmdline_form_arguments = [[arg for pair in [(f'--{k}', str(v)) for k, v in param_conf.items()] for arg in pair] for param_conf in param_grid]


def run_with_args(cmdline_args):
    starting.acquire()  # no other process can get it until it is released
    threading.Timer(10, starting.release).start()
    freest_gpu = networks.get_freer_gpu()
    subprocess.run(['python', 'main.py'] + cmdline_args + ['--gpu', str(freest_gpu)])


p = Pool(processes=3,
         initializer=init, initargs=[multiprocessing.Lock()])

for _ in p.imap(run_with_args, cmdline_form_arguments):
    time.sleep(3)
    pass
