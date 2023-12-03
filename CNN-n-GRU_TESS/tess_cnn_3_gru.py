#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

from utils import *
from models import CNN3GRU


data = load_data('TESS_dataset.csv')

waveform_train, sample_rate = torchaudio.load(data['path'][0])
new_sr = 16000
transform = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=new_sr)
transformed = transform(waveform_train)

wordclasses = sorted(list(data.label.unique()))


def index_to_wordclass(index):
    # Return word based on index in wordclasses
    return wordclasses[index]

def wordclass_to_index(word):
    # Return the index of the word in wordclasses
    return torch.tensor(wordclasses.index(word))

def collate_fn(batch):
    # A data tuple has the format:
    # waveform, sample_rate, wordclass

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


def train_CNN3GRU(config, checkpoint_dir=None, data_path=None, max_num_epochs=None):
    epoch_count = max_num_epochs
    log_interval = 20
    
    model_CNN3GRU = CNN3GRU(n_input=config["n_input"], hidden_dim=config["hidden_dim"], n_layers=config["n_layers"] , n_output=config["n_output"])
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        pin_memory = True
        if torch.cuda.device_count() > 1:
            model_CNN3GRU = nn.DataParallel(model_CNN3GRU)
    model_CNN3GRU.to(device) 
    
    optimizer = optim.Adam(model_CNN3GRU.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        model_CNN3GRU.load_state_dict(model_state)
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
        
        model_CNN3GRU.train()
        right = 0
        h = model_CNN3GRU.init_hidden(int(config["batch_size"]), device)
        
        running_loss = 0.0
        epoch_steps = 0
        
        for batch_index, (data, target) in enumerate(train_loader):
        
            data = data.to(device)
            target = target.to(device)
        
            h = h.data
        
            #data = transform(data)
            output, h = model_CNN3GRU(data, h)

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
        
        model_CNN3GRU.eval()
        right = 0
        
        val_loss = 0.0
        val_steps = 0
        h = model_CNN3GRU.init_hidden(int(config["batch_size"]), device)
        
        for data, target in validation_loader:
            with torch.no_grad():
                data = data.to(device)
                target = target.to(device)
        
                h = h.data
        
                #data = transform(data)
                output, h = model_CNN3GRU(data, h)

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
            torch.save((model_CNN3GRU.state_dict(), optimizer.state_dict()), path)
            
        tune.report(loss=(val_loss / val_steps), accuracy=right / len(validation_loader.dataset))
    print("Finished Training !")


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


def main(num_samples=10, max_num_epochs=10, gpus_per_trial=1):
    data_path = os.path.abspath('../Data_exploration/TESS_dataset.csv')
    train_set, validation_set, _ = init_data_sets(load_data(data_path))
    #checkpoint_path = './Tune_CNN_3_GRU_TESS_checkpoint_dir/'
    #init_data_sets(load_data(data_path))
    config = {
        "n_input": tune.choice([transformed.shape[0]]),
        "hidden_dim": tune.choice([16]),
        "n_layers": tune.choice([3]),
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
        tune.with_parameters(train_CNN3GRU, data_path=data_path, max_num_epochs=max_num_epochs),
        resources_per_trial={"cpu": 32, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=os.path.abspath(tess_experiments_folder+"/TESS_CNN_3_GRU"),
        log_to_file=(os.path.abspath(tess_experiments_folder+"/TESS_CNN_3_GRU_stdout.log"), os.path.abspath(tess_experiments_folder+"/TESS_CNN_3_GRU_stderr.log")),
        name="TESS_CNN_3_GRU",
        resume='AUTO')
    
    
    best_trial = result.get_best_trial("loss", "min", "last")
    
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial final validation accuracy: {}".format(
        best_trial.last_result["accuracy"]))
    
    best_trained_model = CNN3GRU(n_input=best_trial.config["n_input"], hidden_dim=best_trial.config["hidden_dim"], n_layers=best_trial.config["n_layers"] , n_output=best_trial.config["n_output"])
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

    CNN3GRU_test_acc_result, y_pred, y_true = test(best_trained_model, best_trial.config["batch_size"], data_path)
    #test_acc = test_accuracy(best_trained_model, device)
    print("Best trial test set accuracy: {}".format(CNN3GRU_test_acc_result))


if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=10, max_num_epochs=100, gpus_per_trial=1)

