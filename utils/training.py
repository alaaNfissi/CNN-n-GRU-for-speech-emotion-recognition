#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

# Import common modules from centralized location
from utils.common_imports import (
    os, partial, sys, torch, nn, F, optim, torchaudio,
    pd, np, plt, ipd, tqdm_notebook, math, confusion_matrix, sn,
    tune, CLIReporter, ASHAScheduler, random, 
    cuda, gc, inspect, Dataset, pad_sequence
)

# Handle imports for both package usage and direct script execution
try:
    # First try relative imports (when running as script directly)
    import config
except ImportError:
    # Fall back to absolute imports (when running as a module)
    import CNN_n_GRU.config as config


class MyDataset(Dataset):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        
    def __getitem__(self, idx):
        waveform = self.X[idx]
        emotion = self.Y[idx]
        return waveform, emotion
    
    def __len__(self):
        return len(self.Y)


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


def transform_data_tess(data):
    # Simply return the data with original paths and labels
    # Skip audio processing to avoid errors with file paths
    return data


def transform_data_iemocap(data):
    # Simply return the data with original paths and labels
    # Skip audio processing to avoid errors with file paths
    return data


def transform_data_ravdess(data):
    # Simply return the data with original paths and labels
    # Skip audio processing to avoid errors with file paths
    return data


def transform_data(data):
    """Route to the appropriate transform function based on dataset"""
    if config.DATASET.lower() == "tess":
        return transform_data_tess(data)
    elif config.DATASET.lower() == "iemocap":
        return transform_data_iemocap(data)
    elif config.DATASET.lower() == "ravdess":
        return transform_data_ravdess(data)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")


def load_data_tess(path):
    data = pd.read_csv(path)
    data = data[data['source'] == 'TESS'].reset_index()
    del data['source']
    data.rename(columns={'labels':'label'}, inplace=True)
    # The label is already the emotion name in the format "disgust", "happy", etc.
    # No need to split by underscore
    return transform_data(data)


def load_data_iemocap(path):
    data = pd.read_csv(path)
    data = data[data['source'] == 'IEMOCAP'].reset_index()
    del data['source']
    data.rename(columns={'labels':'label'}, inplace=True)
    data['label'] = data['label'].replace('exc', 'hap')
    data = data[data['label'].isin(['ang', 'hap', 'neu', 'sad'])].reset_index()
    return transform_data(data)


def load_data_ravdess(path):
    data = pd.read_csv(path)
    data = data[data['source'] == 'RAVDESS'].reset_index()
    del data['source']
    data.rename(columns={'labels':'label'}, inplace=True)
    return transform_data(data)


def load_data(path):
    """Route to the appropriate load function based on dataset"""
    if config.DATASET.lower() == "tess":
        return load_data_tess(path)
    elif config.DATASET.lower() == "iemocap":
        return load_data_iemocap(path)
    elif config.DATASET.lower() == "ravdess":
        return load_data_ravdess(path)
    else:
        raise ValueError(f"Unknown dataset: {config.DATASET}")


def init_data_sets(data):
    train_ds, test_ds, val_ds = get_dataset_partitions_pd(data, train_split=0.8, val_split=0.1, test_split=0.1, target_variable='label')
    train_set = MyDataset(train_ds['path'], train_ds['label'])
    validation_set = MyDataset(val_ds['path'], val_ds['label'])
    test_set = MyDataset(test_ds['path'], test_ds['label'])
    return train_set, validation_set, test_set


def pad_sequence(batch):
    # Make all tensor in a batch the same length
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch)
    batch = batch.permute(1, 0, 2)
    return batch


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