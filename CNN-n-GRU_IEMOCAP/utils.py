#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import IPython.display as ipd
from tqdm.notebook import tqdm
import math
from sklearn.metrics import confusion_matrix
import seaborn as sn
torch.manual_seed(0)
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
import random

from torch import cuda
import gc
import inspect
random.seed(1234)


iemocap_folder = 'IEMOCAP DATA FOLDER LOCATION'
iemocap_experiments_folder = 'EXPERIMENTS FOLDER LOCATION'


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, paths, labels):
        self.files = paths
        self.labels = labels
        
    def __getitem__(self, item):
        file = self.files[item]
        label = self.labels[item]
        file, sampling_rate = torchaudio.load(file)
        return file, sampling_rate, label
    
    def __len__(self):
        return len(self.files)


def get_dataset_partitions_pd(df, train_split=0.8, val_split=0.1, test_split=0.1, target_variable=None):
    assert (train_split + test_split + val_split) == 1
    
    # Only allows for equal validation and test splits
    assert val_split == test_split 

    # Shuffle
    df_sample = df.sample(frac=1, random_state=42)

    # Specify seed to always have the same split distribution between runs
    # If target variable is provided, generate stratified sets
    if target_variable is not None:
        grouped_df = df_sample.groupby(target_variable)
        arr_list = [np.split(g, [int(train_split * len(g)), int((1 - val_split) * len(g))]) for i, g in grouped_df]

        train_ds = pd.concat([t[0] for t in arr_list])
        val_ds = pd.concat([t[1] for t in arr_list])
        test_ds = pd.concat([v[2] for v in arr_list])

    else:
        indices_or_sections = [int(train_split * len(df)), int((1 - val_split) * len(df))]
        train_ds, val_ds, test_ds = np.split(df_sample, indices_or_sections)
    
    return train_ds.reset_index(drop=True), val_ds.reset_index(drop=True), test_ds.reset_index(drop=True)


def transform_data(data):
    data_dir = os.path.abspath(iemocap_folder+'/IEMOCAP')
    path_lst = []
    for i in data['path']:
        path = data_dir+ '/' + '/'.join(i.split('/')[9:])
        waveform, sampling_rate = torchaudio.load(i)
        waveform = waveform if waveform.shape[0] == 1 else waveform[0].unsqueeze(0)
        sample_rate = 16000
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torchaudio.save(path, waveform, sample_rate)
        path_lst.append(path)
    data = pd.DataFrame({'label':data.label, 'path':path_lst})
    return data

def load_data(path):
    data = pd.read_csv(path)
    data = data[data['source'] == 'IEMOCAP'].reset_index()
    del data['source']
    data.rename(columns={'labels':'label'}, inplace=True)
    data['label'] = data['label'].replace('exc', 'hap')
    data = data[data['label'].isin(['ang', 'hap', 'neu', 'sad'])].reset_index()
    #data['label'] = [i.split('_')[1] for i in data['label']]
    return transform_data(data)

def init_data_sets(data):
    train_ds, test_ds, val_ds  = get_dataset_partitions_pd(data,train_split=0.8, val_split=0.1, test_split=0.1, target_variable='label')
    train_set = MyDataset(train_ds['path'], train_ds['label'])
    validation_set = MyDataset(val_ds['path'], val_ds['label'])
    test_set = MyDataset(test_ds['path'], test_ds['label'])
    return train_set, validation_set, test_set




def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)




def get_less_used_gpu(gpus=None, debug=False):
    """Inspect cached/reserved and allocated memory on specified gpus and return the id of the less used device"""
    if gpus is None:
        warn = 'Falling back to default: all gpus'
        gpus = range(cuda.device_count())
    elif isinstance(gpus, str):
        gpus = [int(el) for el in gpus.split(',')]

    # check gpus arg VS available gpus
    sys_gpus = list(range(cuda.device_count()))
    if len(gpus) > len(sys_gpus):
        gpus = sys_gpus
        warn = f'WARNING: Specified {len(gpus)} gpus, but only {cuda.device_count()} available. Falling back to default: all gpus.\nIDs:\t{list(gpus)}'
    elif set(gpus).difference(sys_gpus):
        # take correctly specified and add as much bad specifications as unused system gpus
        available_gpus = set(gpus).intersection(sys_gpus)
        unavailable_gpus = set(gpus).difference(sys_gpus)
        unused_gpus = set(sys_gpus).difference(gpus)
        gpus = list(available_gpus) + list(unused_gpus)[:len(unavailable_gpus)]
        warn = f'GPU ids {unavailable_gpus} not available. Falling back to {len(gpus)} device(s).\nIDs:\t{list(gpus)}'

    cur_allocated_mem = {}
    cur_cached_mem = {}
    max_allocated_mem = {}
    max_cached_mem = {}
    for i in gpus:
        cur_allocated_mem[i] = cuda.memory_allocated(i)
        cur_cached_mem[i] = cuda.memory_reserved(i)
        max_allocated_mem[i] = cuda.max_memory_allocated(i)
        max_cached_mem[i] = cuda.max_memory_reserved(i)
    min_allocated = min(cur_allocated_mem, key=cur_allocated_mem.get)
    if debug:
        print(warn)
        print('Current allocated memory:', {f'cuda:{k}': v for k, v in cur_allocated_mem.items()})
        print('Current reserved memory:', {f'cuda:{k}': v for k, v in cur_cached_mem.items()})
        print('Maximum allocated memory:', {f'cuda:{k}': v for k, v in max_allocated_mem.items()})
        print('Maximum reserved memory:', {f'cuda:{k}': v for k, v in max_cached_mem.items()})
        print('Suggested GPU:', min_allocated)
    return min_allocated


def free_memory(to_delete: list, debug=False):
    calling_namespace = inspect.currentframe().f_back
    if debug:
        print('Before:')
        get_less_used_gpu(debug=True)

    for _var in to_delete:
        calling_namespace.f_locals.pop(_var, None)
        gc.collect()
        cuda.empty_cache()
    if debug:
        print('After:')
        get_less_used_gpu(debug=True)


def nr_of_right(pred, target):
    # count nr of right predictions
    return pred.squeeze().eq(target).sum().item()


def get_probable_idx(tensor):
    # find most probable wordclass index for each element in the batch
    return tensor.argmax(dim=-1)