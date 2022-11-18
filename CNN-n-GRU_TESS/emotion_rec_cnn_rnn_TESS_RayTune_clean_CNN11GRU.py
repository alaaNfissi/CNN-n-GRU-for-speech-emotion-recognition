#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

random.seed(1234)


# In[2]:

tess_folder = '/home/alaa/Downloads/Concordia/TÉLUQ/ser/Explo_1/Ph.D/CNN-n-GRU-for-speech-emotion-recognition'
tess_experiments_folder = '/home/alaa/Downloads/Concordia/TÉLUQ/ser/Explo_1/Ph.D/CNN-n-GRU-for-speech-emotion-recognition/CNN-n-GRU_TESS/tess_experiments'

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


# In[3]:


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


# In[4]:


def transform_data(data):
    data_dir = os.path.abspath(tess_folder+'/TESS')
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


# In[5]:


def load_data(path):
    data = pd.read_csv(path)
    data = data[data['source'] == 'TESS'].reset_index()
    del data['source']
    data.rename(columns={'labels':'label'}, inplace=True)
    data['label'] = [i.split('_')[1] for i in data['label']]
    return transform_data(data)

def init_data_sets(data):
    train_ds, test_ds, val_ds  = get_dataset_partitions_pd(data,train_split=0.8, val_split=0.1, test_split=0.1, target_variable='label')
    train_set = MyDataset(train_ds['path'], train_ds['label'])
    validation_set = MyDataset(val_ds['path'], val_ds['label'])
    test_set = MyDataset(test_ds['path'], test_ds['label'])
    return train_set, validation_set, test_set


# In[6]:


data = load_data('/home/alaa/Downloads/Concordia/TÉLUQ/ser/Explo_1/Ph.D/CNN-n-GRU-for-speech-emotion-recognition/Data_exploration/TESS_dataset.csv')


# In[7]:

wordclasses = sorted(list(data.label.unique()))


# In[10]:


class CNN11GRU(nn.Module):
    def __init__(self, n_input, hidden_dim, n_layers, n_output=len(wordclasses), stride=4, n_channel=11):
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
    

# In[11]:


def index_to_wordclass(index):
    # Return word based on index in wordclasses
    return wordclasses[index]

def wordclass_to_index(word):
    # Return the index of the word in wordclasses
    return torch.tensor(wordclasses.index(word))


# In[12]:


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.t() for item in batch]
    
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    return batch.permute(0, 2, 1)


# In[13]:


def collate_fn(batch):
    # A data tuple has the format:
    # waveform, sample_rate, wordclass, speaker_id, utterance_nr

    tensors, targets = [], []

    # Gather in lists, and encode wordclasses as indices
    for waveform, _, wordclass, *_ in batch:
        tensors += [waveform]
        targets += [wordclass_to_index(wordclass)]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    # stack - Concatenates a sequence of tensors along a new dimension
    targets = torch.stack(targets)

    return tensors, targets


# In[14]:


from torch import cuda
import gc
import inspect

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


# In[15]:


waveform_train, sample_rate = torchaudio.load(data['path'][0])
new_sr = 16000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)
transformed = transform(waveform_train)


# In[21]:


def train_CNN11GRU(config, checkpoint_dir=None, data_path=None, max_num_epochs=None):
    epoch_count = max_num_epochs
    log_interval = 20
    
    model_CNN11GRU = CNN11GRU(n_input=config["n_input"], hidden_dim=config["hidden_dim"], n_layers=config["n_layers"] , n_output=config["n_output"])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        pin_memory = True
        if torch.cuda.device_count() > 1:
            model_CNN11GRU = nn.DataParallel(model_CNN11GRU)
    model_CNN11GRU.to(device) 
    
    optimizer = optim.Adam(model_CNN11GRU.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_CNN11GRU.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    train_set, validation_set, _ = init_data_sets(load_data(data_path))
        
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=int(config["batch_size"]),
        shuffle=True,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=pin_memory,
    )
    
    validation_loader = torch.utils.data.DataLoader(
        validation_set,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=pin_memory,
    )
    
    losses_train = []
    losses_validation = []
    accuracy_train = []
    accuracy_validation = []
    
    for epoch in range(1, epoch_count + 1):
        
        model_CNN11GRU.train()
        right = 0
        h = model_CNN11GRU.init_hidden(int(config["batch_size"]), device)
        
        running_loss = 0.0
        epoch_steps = 0
        
        for batch_index, (data, target) in enumerate(train_loader):
        
            data = data.to(device)
            target = target.to(device)
        
            h = h.data
        
            #data = transform(data)
            output, h = model_CNN11GRU(data, h)

            pred = get_probable_idx(output)
            right += nr_of_right(pred, target)

            loss = F.nll_loss(output.squeeze(), target)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
            
            if batch_index % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_index * len(data)}/{len(train_loader.dataset)} ({100. * batch_index / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\tAccuracy: {right}/{len(train_loader.dataset)} ({100. * right / len(train_loader.dataset):.0f}%)")
            
                print("[%d, %5d] loss: %.3f" % (epoch + 1, batch_index + 1,
                                                running_loss / epoch_steps))
                running_loss = 0.0
                
            #pbar.update(pbar_update)
        
            losses_train.append(loss.item())
        
            free_memory([data, target, output, h])
        
        model_CNN11GRU.eval()
        right = 0
        
        val_loss = 0.0
        val_steps = 0
        h = model_CNN11GRU.init_hidden(int(config["batch_size"]), device)
        
        for data, target in validation_loader:
            with torch.no_grad():
                data = data.to(device)
                target = target.to(device)
        
                h = h.data
        
                #data = transform(data)
                output, h = model_CNN11GRU(data, h)

                pred = get_probable_idx(output)
                right += nr_of_right(pred, target)

                loss = F.nll_loss(output.squeeze(), target).cpu().numpy()
                
                val_loss += loss.item()
                val_steps += 1
                
                #pbar.update(pbar_update)
        
                free_memory([data, target, output, h])
            
            
        print(f"\nValidation Epoch: {epoch} \tLoss: {loss.item():.6f}\tAccuracy: {right}/{len(validation_loader.dataset)} ({100. * right / len(validation_loader.dataset):.0f}%)\n")

        acc = 100. * right / len(validation_loader.dataset)
        accuracy_validation.append(acc)
        
        losses_validation.append(loss.item())
        losses_validation = losses_validation
        
        lr_scheduler.step()
        
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save((model_CNN11GRU.state_dict(), optimizer.state_dict()), path)
            
        tune.report(loss=(val_loss / val_steps), accuracy=right / len(validation_loader.dataset))
    print("Finished Training !")


# In[22]:


def nr_of_right(pred, target):
    # count nr of right predictions
    return pred.squeeze().eq(target).sum().item()


# In[23]:


def get_probable_idx(tensor):
    # find most probable wordclass index for each element in the batch
    return tensor.argmax(dim=-1)


# In[24]:


def test(model, batch_size, data_path):
    model.eval()
    right = 0
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        pin_memory = True
    
    _, _, test_set = init_data_sets(load_data(data_path))
    
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=pin_memory,
        )
    
    h = model.init_hidden(batch_size, device)
    
    y_true = []
    y_pred = []
    with torch.no_grad():
        for data, target in test_loader:
        
            data = data.to(device)
            target = target.to(device)
        
            targets = target.data.cpu().numpy()
            y_true.extend(targets)
        
            h = h.data
        
            #data = transform(data)
            output, h = model(data, h)
        
        
            pred = get_probable_idx(output)
            #.cpu().numpy()
            right += nr_of_right(pred, target)
        
            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output)
        
            #free_memory([data, target, output, h])

    print(f"\nTest set accuracy: {right}/{len(test_loader.dataset)} ({100. * right / len(test_loader.dataset):.0f}%)\n")

    return (100. * right / len(test_loader.dataset)), y_pred, y_true


# In[25]:


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    data_path = os.path.abspath('../Data_exploration/TESS_dataset.csv')
    train_set, validation_set, _ = init_data_sets(load_data(data_path))
    #checkpoint_path = './Tune_CNN_3_GRU_TESS_checkpoint_dir/'
    #init_data_sets(load_data(data_path))
    config = {
        "n_input": tune.choice([transformed.shape[0]]),
        "hidden_dim": tune.choice([16]),
        "n_layers": tune.choice([11]),
        "n_output": tune.choice([len(wordclasses)]),
        "weight_decay": tune.loguniform(1e-6, 1e-2),
        "lr": tune.loguniform(1e-4, 1e-1),
        "batch_size": tune.grid_search([i for i in [2, 4, 8, 16, 32, 64] if i <= len(validation_set)])
    }
    
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    
    reporter = CLIReporter(
        # parameter_columns=["l1", "l2", "lr", "batch_size"],
        metric_columns=["loss", "accuracy", "training_iteration"])
    
    result = tune.run(
        tune.with_parameters(train_CNN11GRU, data_path=data_path, max_num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 32, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.abspath(tess_experiments_folder+"/TESS_CNN_11_GRU"),
        log_to_file=(os.path.abspath(tess_experiments_folder+"/TESS_CNN_11_GRU_stdout.log"), os.path.abspath(tess_experiments_folder+"/TESS_CNN_11_GRU_stderr.log")),
        name="TESS_CNN_11_GRU",
        resume='AUTO')
    
    
    best_trial = result.get_best_trial("loss", "min", "last")
    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    best_trained_model = CNN11GRU(n_input=best_trial.config["n_input"], hidden_dim=best_trial.config["hidden_dim"], n_layers=best_trial.config["n_layers"] , n_output=best_trial.config["n_output"])
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    CNN11GRU_test_acc_result, y_pred, y_true = test(best_trained_model, best_trial.config["batch_size"], data_path)
    #test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(CNN11GRU_test_acc_result))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=100, gpus_per_trial=1)

