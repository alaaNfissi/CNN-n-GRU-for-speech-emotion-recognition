#!/usr/bin/env python
# coding: utf-8

"""
Common imports used across the CNN-n-GRU project
"""

# System and OS
import os
import sys
from functools import partial
import random
import gc
import inspect

# PyTorch and related
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
from torch import cuda
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sn
import IPython.display as ipd

# ML tools
from tqdm import tqdm
from tqdm.notebook import tqdm as tqdm_notebook
import math
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

# Ray Tune for hyperparameter optimization
try:
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Set seeds for reproducibility
torch.manual_seed(0)
random.seed(1234)
np.random.seed(42) 