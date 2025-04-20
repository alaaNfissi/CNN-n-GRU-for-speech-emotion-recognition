#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

import os
import sys
import argparse

# Add the parent directory to sys.path to make package imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Import common modules from centralized location
from utils.common_imports import (
    torch, nn, F, optim, np, plt, sn, tqdm, sys,
    tune, CLIReporter, ASHAScheduler, RAY_AVAILABLE
)

# Add DataLoader import
from torch.utils.data import DataLoader

# Handle imports for both package usage and direct script execution
try:
    # First try relative imports (when running as script directly)
    import config
    from models import CNN3GRU, CNN5GRU, CNN11GRU, CNN18GRU
    from datasets import get_dataloader, get_num_classes, get_emotion_classes
    from utils.metrics import nr_of_right, get_probable_idx, plot_confusion_matrix
    from utils.training import free_memory
except ImportError:
    # Fall back to absolute imports (when running as a module)
    import CNN_n_GRU.config as config
    from CNN_n_GRU.models import CNN3GRU, CNN5GRU, CNN11GRU, CNN18GRU
    from CNN_n_GRU.datasets import get_dataloader, get_num_classes, get_emotion_classes
    from CNN_n_GRU.utils.metrics import nr_of_right, get_probable_idx, plot_confusion_matrix
    from CNN_n_GRU.utils.training import free_memory

# Check if Ray is available, which is required for grid search
if not RAY_AVAILABLE:
    print("WARNING: Ray is not installed. Grid search functionality will not be available.")
    print("To install, run: pip install ray[tune]")


def get_model(model_name):
    """
    Get the model class based on the model name
    
    Parameters:
        model_name (str): Name of the model (cnn3gru, cnn5gru, cnn11gru, or cnn18gru)
        
    Returns:
        model_class: The PyTorch model class
    """
    model_name = model_name.lower()
    
    if model_name == 'cnn3gru':
        return CNN3GRU
    elif model_name == 'cnn5gru':
        return CNN5GRU
    elif model_name == 'cnn11gru':
        return CNN11GRU
    elif model_name == 'cnn18gru':
        return CNN18GRU
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train(model, train_loader, val_loader, device, num_epochs=50, lr=0.001, weight_decay=1e-5, save_path=None):
    """
    Train a model
    
    Parameters:
        model: The PyTorch model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        device: Device to run the model on
        num_epochs: Number of epochs to train for
        lr: Learning rate
        weight_decay: Weight decay for optimizer
        save_path: Path to save the model checkpoint
        
    Returns:
        model: The trained model
        history: Dictionary containing training history
    """
    # Create the experiments directory if it doesn't exist
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Set up optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    
    # Initialize tracking variables
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }
    
    log_interval = 10
    batch_size = train_loader.batch_size
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Dataset: {config.DATASET}")
    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {device}")
    
    best_val_loss = float('inf')
    
    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            h = model.init_hidden(data.size(0), device)
            h = h.data
            
            optimizer.zero_grad()
            
            output, h = model(data, h)
            loss = F.nll_loss(output, target)
            
            loss.backward()
            optimizer.step()
            
            pred = get_probable_idx(output)
            correct += nr_of_right(pred, target)
            total += target.size(0)
            
            running_loss += loss.item()
            
            if batch_idx % log_interval == 0:
                print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} "
                      f"({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}\t"
                      f"Accuracy: {100. * correct / total:.2f}%")
            
            free_memory([data, target, output, h])
        
        # Compute average training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                
                h = model.init_hidden(data.size(0), device)
                h = h.data
                
                output, h = model(data, h)
                loss = F.nll_loss(output, target)
                
                val_loss += loss.item()
                
                pred = get_probable_idx(output)
                correct += nr_of_right(pred, target)
                total += target.size(0)
                
                free_memory([data, target, output, h])
        
        # Compute average validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        print(f"Epoch {epoch} Summary: "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_loss < best_val_loss and save_path:
            best_val_loss = val_loss
            print(f"Saving best model with val_loss: {val_loss:.4f}")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc
            }, save_path)
        
        # Step the learning rate scheduler
        lr_scheduler.step()
    
    print("Training completed!")
    
    # Plot and save training curves
    if save_path:
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Accuracy')
        plt.plot(history['val_acc'], label='Val Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.title('Training and Validation Accuracy')
        
        plt.tight_layout()
        
        plots_dir = os.path.join(config.EXPERIMENTS_FOLDER, config.DATASET)
        os.makedirs(plots_dir, exist_ok=True)
        plt.savefig(os.path.join(plots_dir, f"{model.__class__.__name__}_training_curves.png"))
    
    return model, history


def test(model, test_loader, device):
    """
    Test a model on the test set
    
    Parameters:
        model: The PyTorch model to test
        test_loader: DataLoader for test data
        device: Device to run the model on
        
    Returns:
        test_acc: Test accuracy
        cm: Confusion matrix
    """
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="Testing"):
            data = data.to(device)
            target = target.to(device)
            
            h = model.init_hidden(data.size(0), device)
            h = h.data
            
            output, h = model(data, h)
            
            pred = get_probable_idx(output)
            correct += nr_of_right(pred, target)
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            free_memory([data, target, output, h])
    
    test_acc = 100. * correct / total
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Compute and plot confusion matrix
    emotion_classes = get_emotion_classes()
    cm = plot_confusion_matrix(all_targets, all_preds, emotion_classes)
    
    # Save confusion matrix
    plots_dir = os.path.join(config.EXPERIMENTS_FOLDER, config.DATASET)
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, f"{model.__class__.__name__}_confusion_matrix.png"))
    
    return test_acc, cm


def train_tune(config, model_class, checkpoint_dir=None, data_loaders=None):
    """
    Training function for ray.tune
    
    Parameters:
        config: Configuration dictionary provided by ray.tune
        model_class: PyTorch model class
        checkpoint_dir: Directory for checkpoints
        data_loaders: Tuple of (train_loader, val_loader)
    
    Returns:
        None
    """
    train_loader, val_loader = data_loaders
    
    # Always use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Custom batch size requires recreating dataloaders
    if config["batch_size"] != train_loader.batch_size:
        # Get dataset from loader
        train_dataset = train_loader.dataset
        val_dataset = val_loader.dataset
        
        # Recreate dataloaders with new batch size
        train_loader = DataLoader(
            train_dataset,
            batch_size=config["batch_size"],
            shuffle=True,
            drop_last=True,
            collate_fn=train_loader.collate_fn,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=config["batch_size"],
            shuffle=False,
            drop_last=True,
            collate_fn=val_loader.collate_fn,
            num_workers=8,
            pin_memory=torch.cuda.is_available(),
        )
    
    # Instantiate model
    model = model_class(
        n_input=1,
        hidden_dim=config["hidden_dim"],
        n_layers=config["num_layers"],
        n_output=get_num_classes(),
        dropout=config.get("dropout", 0.0)  # Add dropout parameter
    )
    model.to(device)
    
    # Set up optimizer based on config
    if config["optimizer"].lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"].lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), 
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    elif config["optimizer"].lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(), 
            lr=config["lr"],
            weight_decay=config["weight_decay"],
            momentum=0.9
        )
    else:
        # Default to Adam
        optimizer = optim.Adam(
            model.parameters(), 
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        )
    
    # Set up learning rate scheduler
    if config["scheduler"].lower() == "step":
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)
    elif config["scheduler"].lower() == "cosine":
        lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])
    else:
        # No scheduler
        lr_scheduler = None
    
    # Load checkpoint if provided
    if checkpoint_dir:
        checkpoint = os.path.join(checkpoint_dir, "checkpoint")
        model_state, optimizer_state = torch.load(checkpoint)
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
    
    # Training loop
    for epoch in range(config["epochs"]):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data = data.to(device)
            target = target.to(device)
            
            h = model.init_hidden(data.size(0), device)
            h = h.data
            
            optimizer.zero_grad()
            
            output, h = model(data, h)
            loss = F.nll_loss(output, target)
            
            loss.backward()
            optimizer.step()
            
            pred = get_probable_idx(output)
            correct += nr_of_right(pred, target)
            total += target.size(0)
            
            running_loss += loss.item()
            
            free_memory([data, target, output, h])
        
        # Compute average training metrics
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data = data.to(device)
                target = target.to(device)
                
                h = model.init_hidden(data.size(0), device)
                h = h.data
                
                output, h = model(data, h)
                loss = F.nll_loss(output, target)
                
                val_loss += loss.item()
                
                pred = get_probable_idx(output)
                correct += nr_of_right(pred, target)
                total += target.size(0)
                
                free_memory([data, target, output, h])
        
        # Compute average validation metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        # Report metrics to ray.tune
        tune.report(
            train_loss=train_loss, 
            train_accuracy=train_acc, 
            val_loss=val_loss, 
            val_accuracy=val_acc
        )
        
        # Step the learning rate scheduler if it exists
        if lr_scheduler is not None:
            lr_scheduler.step()
        
        # Save checkpoint
        with tune.checkpoint_dir(epoch) as checkpoint_dir:
            path = os.path.join(checkpoint_dir, "checkpoint")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )


def grid_search(model_class, train_loader, val_loader, save_dir="experiments"):
    """
    Perform grid search to find the best hyperparameters
    
    Parameters:
        model_class: PyTorch model class
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        save_dir: Directory to save results
        
    Returns:
        best_config: Dictionary containing the best hyperparameters
    """
    if not RAY_AVAILABLE:
        print("ERROR: Ray is not installed. Cannot perform grid search.")
        print("To install, run: pip install ray[tune]")
        return None
    
    import ray
    from ray import tune
    from ray.tune import CLIReporter
    from ray.tune.schedulers import ASHAScheduler
    
    print("Starting grid search...")
    
    # Initialize Ray if not already initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True)
    
    # Create the experiments directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Define hyperparameter search space (exhaustive grid search)
    epochs_for_trials = 30  # Reduced epochs for each trial
    
    # Define the full grid for hyperparameter search
    config = {
        "model_name": model_class.__name__,
        "lr": tune.grid_search([0.00001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05]),
        "weight_decay": tune.grid_search([0, 1e-6, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3]),
        "batch_size": tune.grid_search([16, 32, 64, 128]),
        "hidden_dim": tune.grid_search([32, 64, 128, 256, 512]),
        "num_layers": tune.grid_search([1, 2, 3, 4]),
        "dropout": tune.grid_search([0.0, 0.1, 0.2, 0.3, 0.5]),
        "optimizer": tune.grid_search(["adam", "adamw", "sgd"]),
        "scheduler": tune.grid_search(["step", "cosine", "none"]),
        "epochs": epochs_for_trials,
    }
    
    # Define scheduler - ASHA scheduler for early stopping of bad trials
    scheduler = ASHAScheduler(
        max_t=epochs_for_trials,
        grace_period=10,
        reduction_factor=2
    )
    
    # Define reporter for printing results
    reporter = CLIReporter(
        parameter_columns=["model_name", "lr", "weight_decay", "hidden_dim", "num_layers", "batch_size", "dropout", "optimizer", "scheduler"],
        metric_columns=["train_loss", "train_accuracy", "val_loss", "val_accuracy"],
        max_report_frequency=30
    )
    
    # Define the search algorithm
    search_alg = None  # For exhaustive grid search, no special algorithm needed
    
    # Determine available GPU resources
    num_gpus = torch.cuda.device_count()
    gpu_per_trial = 1 if num_gpus > 0 else 0
    
    print(f"Found {num_gpus} GPUs available for grid search")
    if num_gpus == 0:
        print("WARNING: No GPUs found! Training will be slow. Consider running on a machine with GPUs.")
    
    # Run the tuning
    result = tune.run(
        tune.with_parameters(
            train_tune,
            model_class=model_class,
            data_loaders=(train_loader, val_loader)
        ),
        resources_per_trial={"cpu": 2, "gpu": gpu_per_trial},
        config=config,
        metric="val_accuracy",
        mode="max",
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir=save_dir,
        name=f"grid_search_{model_class.__name__}",
        num_samples=1,  # Only needed for random search
        verbose=1,
        search_alg=search_alg
    )
    
    # Get best trial
    best_trial = result.get_best_trial("val_accuracy", "max", "last")
    best_config = best_trial.config
    best_checkpoint_dir = best_trial.checkpoint.value
    
    print(f"Best trial config: {best_config}")
    print(f"Best trial final validation accuracy: {best_trial.last_result['val_accuracy']}")
    
    # Load the best model
    best_checkpoint_path = os.path.join(best_checkpoint_dir, "checkpoint")
    
    # Print the location of the best checkpoint
    print(f"Best checkpoint saved at: {best_checkpoint_path}")
    
    return best_config


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CNN-n-GRU for Speech Emotion Recognition')
    
    parser.add_argument('--model', type=str, default='cnn18gru',
                      choices=['cnn3gru', 'cnn5gru', 'cnn11gru', 'cnn18gru'],
                      help='Model architecture to use (default: cnn18gru)')
    parser.add_argument('--epochs', type=int, default=config.EPOCHS,
                      help=f'Number of epochs to train (default: {config.EPOCHS})')
    parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE,
                      help=f'Batch size for training (default: {config.BATCH_SIZE})')
    parser.add_argument('--lr', type=float, default=config.LEARNING_RATE,
                      help=f'Learning rate (default: {config.LEARNING_RATE})')
    parser.add_argument('--wd', type=float, default=config.WEIGHT_DECAY,
                      help=f'Weight decay (default: {config.WEIGHT_DECAY})')
    parser.add_argument('--hidden_dim', type=int, default=config.HIDDEN_DIM,
                      help=f'Hidden dimension for GRU (default: {config.HIDDEN_DIM})')
    parser.add_argument('--num_layers', type=int, default=config.GRU_LAYERS,
                      help=f'Number of GRU layers (default: {config.GRU_LAYERS})')
    parser.add_argument('--dropout', type=float, default=0.0,
                      help='Dropout rate (default: 0.0)')
    parser.add_argument('--train', action='store_true',
                      help='Train the model')
    parser.add_argument('--test', action='store_true',
                      help='Test the model')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA if available (deprecated, GPU is used by default if available)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    parser.add_argument('--grid_search', action='store_true',
                      help='Perform grid search to find the best hyperparameters')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Always use GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get model class and instantiate model
    model_class = get_model(args.model)
    model = model_class(
        n_input=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        n_output=get_num_classes(),
        dropout=args.dropout
    )
    model.to(device)
    
    # Print model summary
    print(f"Model: {model.__class__.__name__}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Get dataloaders
    train_loader, val_loader, test_loader, emotion_classes = get_dataloader(
        batch_size=args.batch_size,
        shuffle_train=True,
        drop_last=True,
        num_workers=8
    )
    
    print(f"Dataset: {config.DATASET}")
    print(f"Emotion classes: {emotion_classes}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create save path
    save_dir = os.path.join(config.EXPERIMENTS_FOLDER, config.DATASET, args.model.lower())
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'best_model.pth')
    
    # Perform grid search if requested
    if args.grid_search:
        best_config = grid_search(
            model_class=model_class,
            train_loader=train_loader,
            val_loader=val_loader,
            save_dir=save_dir
        )
        
        # Update model with best hyperparameters
        model = model_class(
            n_input=1,
            hidden_dim=best_config["hidden_dim"],
            n_layers=best_config["num_layers"],
            n_output=get_num_classes(),
            dropout=best_config.get("dropout", 0.0)
        )
        model.to(device)
        
        # Update args with best hyperparameters
        args.lr = best_config["lr"]
        args.wd = best_config["weight_decay"]
        args.hidden_dim = best_config["hidden_dim"]
        args.num_layers = best_config["num_layers"]
        args.dropout = best_config.get("dropout", 0.0)
        
        print(f"Updated model with best hyperparameters:")
        print(f"  Learning rate: {args.lr}")
        print(f"  Weight decay: {args.wd}")
        print(f"  Hidden dimension: {args.hidden_dim}")
        print(f"  Number of GRU layers: {args.num_layers}")
        print(f"  Dropout: {args.dropout}")
    
    # Train
    if args.train:
        model, history = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            num_epochs=args.epochs,
            lr=args.lr,
            weight_decay=args.wd,
            save_path=save_path
        )
    
    # If model exists and we're not training, load it
    elif os.path.exists(save_path):
        print(f"Loading model from {save_path}")
        checkpoint = torch.load(save_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test
    if args.test or not args.train:
        test_acc, cm = test(
            model=model,
            test_loader=test_loader,
            device=device
        )


if __name__ == '__main__':
    main() 