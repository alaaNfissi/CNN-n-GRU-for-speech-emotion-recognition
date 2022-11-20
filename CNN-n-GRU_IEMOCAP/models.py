#!/usr/bin/env python
# coding: utf-8

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

class CNN3GRU(nn.Module):
    def __init__(self, n_input, hidden_dim, n_layers, n_output=None, stride=4, n_channel=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.conv1 = nn.Conv1d(n_input, 2*n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(2*n_channel)
        self.relu1 = nn.LeakyReLU()
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(2*n_channel, 2*n_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(2*n_channel)
        self.relu2 = nn.LeakyReLU()
        self.pool2 = nn.MaxPool1d(4)
        
        self.fc1 = nn.Linear(2*n_channel, n_channel)
        self.relu3 = nn.LeakyReLU()
        
        self.gru1 = nn.GRU(n_channel, hidden_dim, n_layers, batch_first=True, bidirectional=False, dropout=0)
        self.fc2 = nn.Linear(hidden_dim, n_output)
        self.relu4 = nn.LeakyReLU()
       

    def forward(self, x, h):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = self.pool2(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(self.relu3(x))
        
        x, h = self.gru1(x, h)
        x = self.fc2(self.relu4(x[:,-1]))
        
        #x = self.rnn(x)
        
        return F.log_softmax(x, dim=1), h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class CNN5GRU(nn.Module):
    def __init__(self, n_input, hidden_dim, n_layers, n_output=None, stride=4, n_channel=5):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.relu1 = nn.LeakyReLU()
        
        self.pool1 = nn.MaxPool1d(4)
        
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3,stride=1,padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.relu2 = nn.LeakyReLU()
        
        self.pool2 = nn.MaxPool1d(4)
        
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3,stride=1,padding=1)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.relu3 = nn.LeakyReLU()
        
        self.pool3 = nn.MaxPool1d(4)
        
        self.conv4 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3,stride=1,padding=1)
        self.bn4 = nn.BatchNorm1d(4 * n_channel)
        self.relu4 = nn.LeakyReLU()
        
        self.pool4 = nn.MaxPool1d(4)
        
        self.fc1 = nn.Linear(4 * n_channel, 2*n_channel)
        self.relu5 = nn.LeakyReLU()
        
        self.gru1 = nn.GRU(2*n_channel, hidden_dim, n_layers, batch_first=True, bidirectional=False, dropout=0)
        self.fc2 = nn.Linear(hidden_dim, n_output)
        self.relu6 = nn.LeakyReLU()

    def forward(self, x, h):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.relu3(self.bn3(x))
        x = self.pool3(x)
        
        x = self.conv4(x)
        x = self.relu4(self.bn4(x))
        x = self.pool4(x)
        
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(self.relu5(x))
        
        x, h = self.gru1(x, h)
        x = self.fc2(self.relu6(x[:,-1]))
        
        return F.log_softmax(x, dim=1), h

    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class CNN11GRU(nn.Module):
    def __init__(self, n_input, hidden_dim, n_layers, n_output=None, stride=4, n_channel=11):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.relu1 = nn.LeakyReLU()
        
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.relu2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.relu3 = nn.LeakyReLU()
        
        self.pool2 = nn.MaxPool1d(4)

        self.conv4 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.relu4 = nn.LeakyReLU()
        
        self.conv5 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm1d(2 * n_channel)
        self.relu5 = nn.LeakyReLU()
        
        self.pool3 = nn.MaxPool1d(4)
 
        self.conv6 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm1d(4 * n_channel)
        self.relu6 = nn.LeakyReLU()
        
        self.conv7 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm1d(4 * n_channel)
        self.relu7 = nn.LeakyReLU()
        
        self.conv8 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm1d(4 * n_channel)
        self.relu8 = nn.LeakyReLU()
        
        self.pool4 = nn.MaxPool1d(4)

        self.conv9 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn9 = nn.BatchNorm1d(8 * n_channel)
        self.relu9 = nn.LeakyReLU()
        
        self.conv10 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn10 = nn.BatchNorm1d(8 * n_channel)
        self.relu10 = nn.LeakyReLU()

        self.fc1 = nn.Linear(8 * n_channel, 4 * n_channel)
        self.relu11 = nn.LeakyReLU()
        
        self.gru1 = nn.GRU(4*n_channel, hidden_dim, n_layers, batch_first=True, bidirectional=False, dropout=0)
        self.fc2 = nn.Linear(hidden_dim, n_output)
        self.relu12 = nn.LeakyReLU()

    def forward(self, x, h):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        
        x = self.conv3(x)
        x = self.relu3(self.bn1(x))
        
        x = self.pool2(x)

        x = self.conv4(x)
        x = self.relu4(self.bn4(x))
        
        x = self.conv5(x)
        x = self.relu5(self.bn5(x))
        
        x = self.pool3(x)

        x = self.conv6(x)
        x = self.relu6(self.bn6(x))
        
        x = self.conv7(x)
        x = self.relu7(self.bn7(x))
        
        x = self.conv8(x)
        x = self.relu8(self.bn8(x))
        
        x = self.pool4(x)

        x = self.conv9(x)
        x = self.relu9(self.bn9(x))
        
        x = self.conv10(x)
        x = self.relu10(self.bn10(x))

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(self.relu11(x))
        
        x, h = self.gru1(x, h)
        x = self.fc2(self.relu12(x[:,-1]))
        
        return F.log_softmax(x, dim=1), h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden


class CNN18GRU(nn.Module):
    def __init__(self, n_input, hidden_dim, n_layers, n_output=None, stride=4, n_channel=18):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=160, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.relu1 = nn.LeakyReLU()
        
        self.pool1 = nn.MaxPool1d(4, stride=None)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.relu2 = nn.LeakyReLU()
        
        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.relu3 = nn.LeakyReLU()
        
        self.conv4 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.relu4 = nn.LeakyReLU()
        
        self.conv5 = nn.Conv1d(n_channel, n_channel, kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.relu5 = nn.LeakyReLU()
        
        self.pool2 = nn.MaxPool1d(4, stride=None)

        self.conv6 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm1d(2 * n_channel)
        self.relu6 = nn.LeakyReLU()
        
        self.conv7 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm1d(2 * n_channel)
        self.relu7 = nn.LeakyReLU()
        
        self.conv8 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm1d(2 * n_channel)
        self.relu8 = nn.LeakyReLU()
        
        self.conv9 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3,padding=1)
        self.bn9 = nn.BatchNorm1d(2 * n_channel)
        self.relu9 = nn.LeakyReLU()
        
        self.pool3 = nn.MaxPool1d(4, stride=None)
 
        self.conv10 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn10 = nn.BatchNorm1d(4 * n_channel)
        self.relu10 = nn.LeakyReLU()
        
        self.conv11 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn11 = nn.BatchNorm1d(4 * n_channel)
        self.relu11 = nn.LeakyReLU()
        
        self.conv12 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn12 = nn.BatchNorm1d(4 * n_channel)
        self.relu12 = nn.LeakyReLU()
        
        self.conv13 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3,padding=1)
        self.bn13 = nn.BatchNorm1d(4 * n_channel)
        self.relu13 = nn.LeakyReLU()
        
        self.pool4 = nn.MaxPool1d(4, stride=None)

        self.conv14 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn14 = nn.BatchNorm1d(8 * n_channel)
        self.relu14 = nn.LeakyReLU()
        
        self.conv15 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn15 = nn.BatchNorm1d(8 * n_channel)
        self.relu15 = nn.LeakyReLU()
        
        self.conv16 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn16 = nn.BatchNorm1d(8 * n_channel)
        self.relu16 = nn.LeakyReLU()
        
        self.conv17 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3,padding=1)
        self.bn17 = nn.BatchNorm1d(8 * n_channel)
        self.relu17 = nn.LeakyReLU()

        self.fc1 = nn.Linear(8 * n_channel, 4 * n_channel)
        self.relu18 = nn.LeakyReLU()
        
        self.gru1 = nn.GRU(4 * n_channel, hidden_dim, n_layers, batch_first=True, bidirectional=False, dropout=0)
        self.fc2 = nn.Linear(hidden_dim, n_output)
        self.relu19 = nn.LeakyReLU()

    def forward(self, x, h):
        x = self.conv1(x)
        x = self.relu1(self.bn1(x))
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu2(self.bn2(x))
        
        x = self.conv3(x)
        x = self.relu3(self.bn3(x))
        
        x = self.conv4(x)
        x = self.relu4(self.bn4(x))
        
        x = self.conv5(x)
        x = self.relu5(self.bn5(x))
        
        x = self.pool2(x)

        x = self.conv6(x)
        x = self.relu6(self.bn6(x))
        
        x = self.conv7(x)
        x = self.relu7(self.bn7(x))
        
        x = self.conv8(x)
        x = self.relu8(self.bn8(x))
        
        x = self.conv9(x)
        x = self.relu9(self.bn9(x))
        
        x = self.pool3(x)

        x = self.conv10(x)
        x = self.relu10(self.bn10(x))
        
        x = self.conv11(x)
        x = self.relu11(self.bn11(x))
        
        x = self.conv12(x)
        x = self.relu12(self.bn12(x))
        
        x = self.conv13(x)
        x = self.relu13(self.bn13(x))
        
        x = self.pool4(x)

        x = self.conv14(x)
        x = self.relu14(self.bn14(x))
        
        x = self.conv15(x)
        x = self.relu15(self.bn15(x))
        
        x = self.conv16(x)
        x = self.relu16(self.bn16(x))
        
        x = self.conv17(x)
        x = self.relu17(self.bn17(x))

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(self.relu18(x))
        
        x, h = self.gru1(x, h)
        x = self.fc2(self.relu19(x[:,-1]))
        
        return F.log_softmax(x, dim=1), h
    
    def init_hidden(self, batch_size, device):
        weight = next(self.parameters()).data
        hidden = weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().to(device)
        return hidden
