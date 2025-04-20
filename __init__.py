#!/usr/bin/env python
# coding: utf-8

"""
CNN-n-GRU for Speech Emotion Recognition
=========================================

A package containing CNN-GRU hybrid models for speech emotion recognition.
Supports TESS, IEMOCAP, and RAVDESS datasets.
"""

__version__ = '1.0.0'

from .models import CNN3GRU, CNN5GRU, CNN11GRU, CNN18GRU

__all__ = ['CNN3GRU', 'CNN5GRU', 'CNN11GRU', 'CNN18GRU'] 