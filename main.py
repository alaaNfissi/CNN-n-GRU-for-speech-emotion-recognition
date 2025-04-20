#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
from tqdm import tqdm
import sys

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
    import config
    from models import CNN3GRU, CNN5GRU, CNN11GRU, CNN18GRU
    from datasets import get_dataloader, get_num_classes, get_emotion_classes
    from utils.metrics import nr_of_right, get_probable_idx, plot_confusion_matrix
    from utils.training import free_memory


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
    parser.add_argument('--train', action='store_true',
                      help='Train the model')
    parser.add_argument('--test', action='store_true',
                      help='Test the model')
    parser.add_argument('--cuda', action='store_true',
                      help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Get model class and instantiate model
    model_class = get_model(args.model)
    model = model_class(
        n_input=1,
        hidden_dim=args.hidden_dim,
        n_layers=args.num_layers,
        n_output=get_num_classes()
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