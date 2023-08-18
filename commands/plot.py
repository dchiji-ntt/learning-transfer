
import copy
import torch
import numpy as np
import datetime
from utils.output_manager import OutputManager

import importlib

def plot(exp_name, cfg, gpu_id, prefix=''):
    outman = OutputManager(cfg['output_dir'], exp_name, cfg['output_prefix_hashing'])

    plot_fn_name = exp_name
    plot_fn_name = plot_fn_name.replace('.', '_')
    plot_fn_name = plot_fn_name.replace('+', '_')
    plot_fn_name = plot_fn_name.replace('-', '_')

    plot_mod = importlib.import_module(f'plots.{plot_fn_name}')
    plot_fn = getattr(plot_mod, plot_fn_name)

    plot_fn(cfg, outman, prefix, gpu_id)

