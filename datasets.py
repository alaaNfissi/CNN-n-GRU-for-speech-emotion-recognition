#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

"""
Dataset handling for Speech Emotion Recognition models.

This module provides functions for loading and processing speech emotion datasets.
It requires CSV files with the following structure:
- 'path': column containing the full path to the audio file
- 'label': column containing the emotion label (e.g., 'angry', 'happy', etc.)
- 'source': column indicating the dataset source ('TESS', 'IEMOCAP', or 'RAVDESS')

Users need to create these CSV files for their datasets with the above structure.
Example CSV format:
path,label,source
/path/to/audio1.wav,happy,TESS
/path/to/audio2.wav,angry,TESS
...

For TESS dataset: labels should be one of ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
For IEMOCAP dataset: labels should be one of ['ang', 'hap', 'neu', 'sad']
For RAVDESS dataset: labels should be one of ['neutral', 'calm', 'happy', 'sad', 'angry', 'fearful', 'disgust', 'surprised']
"""

# Import common modules from centralized location
from utils.common_imports import (
    os, torch, torchaudio, random, np, 
    DataLoader, Dataset, pad_sequence,
    train_test_split, pd
)
# Import modules not in common_imports directly
from torch.utils.data import random_split
import re

# Set random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Handle imports for both package usage and direct script execution
try:
    # First try relative imports (when running as script directly)
    import config
    from utils.training import MyDataset, transform_data, load_data, init_data_sets
except ImportError:
    # Fall back to absolute imports (when running as a module)
    import CNN_n_GRU.config as config
    from CNN_n_GRU.utils.training import MyDataset, transform_data, load_data, init_data_sets


def wordclass_to_index(word, dataset=None):
    """Convert emotion label to index based on the dataset"""
    dataset = dataset or config.DATASET.lower()
    
    if dataset == "tess":
        return torch.tensor(config.TESS_EMOTION_TO_IDX[word])
    elif dataset == "iemocap":
        return torch.tensor(config.IEMOCAP_EMOTION_TO_IDX[word])
    elif dataset == "ravdess":
        return torch.tensor(config.RAVDESS_EMOTION_TO_IDX[word])
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def index_to_wordclass(index, dataset=None):
    """Convert index to emotion label based on the dataset"""
    dataset = dataset or config.DATASET.lower()
    
    if dataset == "tess":
        return config.TESS_IDX_TO_EMOTION[index]
    elif dataset == "iemocap":
        return config.IEMOCAP_IDX_TO_EMOTION[index]
    elif dataset == "ravdess":
        return config.RAVDESS_IDX_TO_EMOTION[index]
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_emotion_classes():
    """Get emotion classes for the current dataset"""
    dataset = config.DATASET.lower()
    
    if dataset == "tess":
        return config.TESS_EMOTIONS
    elif dataset == "iemocap":
        return config.IEMOCAP_EMOTIONS
    elif dataset == "ravdess":
        return config.RAVDESS_EMOTIONS
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_num_classes():
    """Get the number of emotion classes for the current dataset"""
    return len(get_emotion_classes())


def collate_fn(batch):
    """
    Collate function for batching in DataLoader
    
    A data tuple now has the format:
    waveform, wordclass
    """
    tensors, targets = [], []
    
    # Gather in lists, and encode wordclasses as indices
    for path, wordclass in batch:
        # Load the audio file
        try:
            waveform, sample_rate = torchaudio.load(path)
            # Make sure waveform is the right shape
            waveform = waveform if waveform.shape[0] == 1 else waveform[0].unsqueeze(0)
            tensors.append(waveform)
            targets.append(wordclass_to_index(wordclass))
        except Exception as e:
            print(f"Error loading audio file {path}: {e}")
            continue
    
    if not tensors:
        raise ValueError("No valid audio files were loaded in this batch")
    
    # Use pad_sequence instead of stack to handle tensors of different lengths
    # First convert list of 2D tensors to a list of 1D tensors by permuting dimensions
    tensors_permuted = [tensor.permute(1, 0) for tensor in tensors]
    # Pad the sequence to make all tensors the same length
    padded = torch.nn.utils.rnn.pad_sequence(tensors_permuted)
    # Permute back to get the right dimensions [batch, channel, sequence]
    tensors_padded = padded.permute(1, 2, 0)
    
    targets = torch.stack(targets)
    
    return tensors_padded, targets


def get_data_path():
    """Get the appropriate data path based on dataset"""
    dataset = config.DATASET.lower()
    
    if dataset == "tess":
        return 'TESS_dataset.csv'
    elif dataset == "iemocap":
        return 'IEMOCAP_dataset.csv'
    elif dataset == "ravdess":
        return 'RAVDESS_dataset.csv'
    else:
        raise ValueError(f"Unknown dataset: {dataset}")


def get_dataloader(batch_size=32, shuffle_train=True, drop_last=True, num_workers=8):
    """
    Get dataloaders for train, validation, and test sets based on config.DATASET
    
    Parameters:
        batch_size (int): Batch size for the dataloaders
        shuffle_train (bool): Whether to shuffle the training data
        drop_last (bool): Whether to drop the last incomplete batch
        num_workers (int): Number of worker processes for data loading
        
    Returns:
        train_loader, validation_loader, test_loader: DataLoader objects
        emotion_classes: List of emotion class names
    """
    # Determine device and pin_memory setting
    pin_memory = torch.cuda.is_available()
    
    # Get data path based on dataset
    data_path = get_data_path()
    
    # Initialize datasets
    train_set, validation_set, test_set = init_data_sets(load_data(data_path))
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=shuffle_train,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    validation_loader = DataLoader(
        validation_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    
    # Get emotion classes
    emotion_classes = get_emotion_classes()
    
    return train_loader, validation_loader, test_loader, emotion_classes 