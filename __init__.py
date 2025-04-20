#!/usr/bin/env python
# coding: utf-8

"""
CNN-n-GRU: Deep learning models for Speech Emotion Recognition.

This package provides a collection of convolutional and recurrent neural network
models for speech emotion recognition using raw waveform data directly. The models
combine an n-layer convolutional neural network for extracting hierarchical acoustic
features with gated recurrent units to model temporal dependencies in speech.

Models available:
- CNN3GRU: 3-layer CNN with GRU for speech emotion recognition
- CNN5GRU: 5-layer CNN with GRU for speech emotion recognition
- CNN11GRU: 11-layer CNN with GRU for speech emotion recognition
- CNN18GRU: 18-layer CNN with GRU for speech emotion recognition
"""

from .models import CNN3GRU, CNN5GRU, CNN11GRU, CNN18GRU
from .config import *

__version__ = "1.0.0"
__author__ = "Alaa Nfissi"
__email__ = "alaa.nfissi@mail.concordia.ca"

__all__ = [
    "CNN3GRU", "CNN5GRU", "CNN11GRU", "CNN18GRU",
] 