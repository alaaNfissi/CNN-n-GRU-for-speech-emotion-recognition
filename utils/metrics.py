#!/usr/bin/env python
# coding: utf-8

# author: Alaa Nfissi

# Import common modules from centralized location
from utils.common_imports import (
    torch, np, confusion_matrix, plt, sn, sys
)

def nr_of_right(pred, target):
    """Count the number of correct predictions"""
    return pred.squeeze().eq(target).sum().item()


def get_probable_idx(tensor):
    """Find most probable class index for each element in the batch"""
    return tensor.argmax(dim=-1)


def accuracy(output, target):
    """Calculate accuracy between predictions and targets"""
    pred = get_probable_idx(output)
    correct = nr_of_right(pred, target)
    return correct / target.size(0)


def plot_confusion_matrix(y_true, y_pred, classes, normalize=True, title=None, cmap=plt.cm.Blues):
    """
    Plot confusion matrix with heatmap visualization
    
    Parameters:
        y_true (array): True labels
        y_pred (array): Predicted labels
        classes (list): List of class names
        normalize (bool): Whether to normalize the confusion matrix
        title (str): Title for the plot
        cmap: Colormap to use
        
    Returns:
        fig: The matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    
    # Set labels, title, and ticks
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           ylabel='True label',
           xlabel='Predicted label')
    
    if title:
        ax.set_title(title)
    
    # Rotate the tick labels and set alignment
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Loop over data dimensions and create text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    return fig


def evaluate_model(model, test_loader, device, class_mapping=None):
    """
    Evaluate a model on test data and report metrics
    
    Parameters:
        model: The PyTorch model to evaluate
        test_loader: DataLoader with test data
        device: Device to run the model on
        class_mapping: Dictionary mapping class indices to class names
        
    Returns:
        metrics: Dictionary containing evaluation metrics
    """
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0
    correct = 0
    total = 0
    
    criterion = torch.nn.NLLLoss()
    
    with torch.no_grad():
        for batch_idx, (data, sampling_rates, target_emotion) in enumerate(test_loader):
            data = data.to(device)
            target = torch.tensor([class_mapping[e] for e in target_emotion]).to(device)
            
            hidden = model.init_hidden(data.size(0), device)
            
            output, _ = model(data, hidden)
            loss = criterion(output, target)
            
            test_loss += loss.item()
            
            pred = get_probable_idx(output)
            correct += nr_of_right(pred, target)
            total += target.size(0)
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / total
    
    # Calculate class-wise accuracies
    cm = confusion_matrix(all_targets, all_preds)
    normalized_cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    class_accuracies = {i: normalized_cm[i, i] * 100 for i in range(len(normalized_cm))}
    
    metrics = {
        'loss': test_loss,
        'accuracy': test_accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm,
        'normalized_confusion_matrix': normalized_cm,
        'predictions': all_preds,
        'targets': all_targets
    }
    
    return metrics 