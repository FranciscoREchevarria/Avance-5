"""
Metrics visualization functions for ring detection model evaluation.
Each function corresponds to a metrics cell from the notebook.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import torch
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import seaborn as sns

def visualize_attention_maps(model, val_dataset, ring_types, n_samples=4, device='cuda'):
    """Visualiza mapas de atención para muestras de cada clase."""
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    model.model.to(device)

    hook_handle = model.get_attention()
    model.model.eval()

    fig, axs = plt.subplots(len(ring_types), n_samples, figsize=(4*n_samples, 4*len(ring_types)))

    for i, cls in enumerate(ring_types):
        # Encontrar índices de esta clase
        indices = [idx for idx, label in enumerate(val_dataset.labels)
                   if val_dataset.idx_to_class[label] == cls]
        if not indices:
            continue
        sample_indices = np.random.choice(indices, min(n_samples, len(indices)), replace=False)

        for j, idx in enumerate(sample_indices):
            image, label = val_dataset[idx]
            img_tensor = image.unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_tensor)

            # Extraer mapa de atención
            if 'att_layer' in model.activation:
                feature_map = model.activation['att_layer'][0].cpu()
                attention_map = feature_map.mean(dim=0, keepdim=True)
                attention_up = F.interpolate(
                    attention_map.unsqueeze(0),
                    size=img_tensor.shape[2:],
                    mode='bilinear', align_corners=False
                ).squeeze().cpu().numpy()

                # Denormalizar imagen
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img_np = image.cpu().numpy().transpose(1, 2, 0)
                img_np = std * img_np + mean
                img_np = np.clip(img_np, 0, 1)

                ax = axs[i, j] if len(ring_types) > 1 else axs[j]
                ax.imshow(img_np)
                ax.imshow(attention_up, cmap='jet', alpha=0.4)
                pred_cls = ring_types[torch.argmax(outputs[0] if isinstance(outputs, tuple) else outputs, 1).item()]
                ax.set_title(f"Real: {cls}\nPred: {pred_cls}", fontsize=9)
                ax.axis('off')

    hook_handle.remove()
    plt.suptitle("Mapas de atención del modelo", fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_training_metrics(trainer):
    """
    Plot training and validation loss and accuracy from metrics CSV file.
    
    Args:
        trainer: PyTorch Lightning trainer object with logger
    """
    # Read metrics from CSV file
    log_dir = Path(trainer.logger.log_dir)
    metrics_file = log_dir / 'metrics.csv'

    if metrics_file.exists():
        # Read CSV file
        df = pd.read_csv(metrics_file)
        print(metrics_file)
        
        # Extract specific columns and group by epoch
        metrics_df = df[['epoch', 'finetuning/train_acc_epoch', 
                         'finetuning/train_loss_epoch', 'finetuning/val_acc', 'finetuning/val_loss']].copy()
        
        # Drop rows where all metric values are NaN
        metrics_df = metrics_df.dropna(subset=['finetuning/train_loss_epoch',
                                                'finetuning/val_loss',
                                                'finetuning/train_acc_epoch',
                                                'finetuning/val_acc'], 
                                        how='all')
        
        # Group by epoch and take the first non-null value for each metric
        metrics_grouped = metrics_df.groupby('epoch').first()
        
        # Remove rows with NaN values in any column
        metrics_grouped = metrics_grouped.dropna()
        
        print("Metrics data:")
        print(metrics_grouped)
        
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        epochs = metrics_grouped.index
        
        # Plot Loss
        ax1.plot(epochs, metrics_grouped['finetuning/train_loss_epoch'], 'b-', 
                 label='Train Loss', marker='o', linewidth=2)
        ax1.plot(epochs, metrics_grouped['finetuning/val_loss'], 'r-', 
                 label='Val Loss', marker='s', linewidth=2)
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss', fontsize=12)
        ax1.set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot Accuracy
        ax2.plot(epochs, metrics_grouped['finetuning/train_acc_epoch'], 'b-', 
                 label='Train Accuracy', marker='o', linewidth=2)
        ax2.plot(epochs, metrics_grouped['finetuning/val_acc'], 'r-', 
                 label='Val Accuracy', marker='s', linewidth=2)
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.set_title('Training and Validation Accuracy', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    else:
        print(f"Metrics file not found at {metrics_file}")
        print(f"Log directory: {log_dir}")
        print("Available files:")
        for f in log_dir.glob('*'):
            print(f"  - {f.name}")


def plot_roc_curves(finetuned_model, datamodule, label_names=None):
    """
    Plot ROC curves for multilabel classification.
    
    Args:
        finetuned_model: Trained model with predict_proba method
        datamodule: Data module with test_dataloader method
        label_names: List of label names (default: ['Inner Ring', 'Outer Ring'])
        
    Returns:
        tuple: (probs_all, labels_all) - predictions and labels arrays
    """
    if label_names is None:
        label_names = ['Inner Ring', 'Outer Ring']
    
    # Get test predictions
    finetuned_model.eval()
    test_loader = datamodule.test_dataloader()

    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in test_loader:
            x, y = finetuned_model.batch_to_supervised_tuple(batch)
            x = x.to(finetuned_model.device)
            y = y.to(finetuned_model.device)
            
            # Get probabilities
            probs = finetuned_model.predict_proba(x)
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y.cpu().numpy())

    # Concatenate all batches
    probs_all = np.concatenate(all_probs, axis=0)  # Shape: (n_samples, 2)
    labels_all = np.concatenate(all_labels, axis=0)  # Shape: (n_samples, 2)

    print(f"Predictions shape: {probs_all.shape}")
    print(f"Labels shape: {labels_all.shape}")

    # Compute ROC curves for each label
    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#FF6B6B', '#4ECDC4']
    fprs = []
    tprs = []
    aucs = []

    # Plot ROC curves for each label
    for label_idx, label_name in enumerate(label_names):
        fpr, tpr, _ = roc_curve(labels_all[:, label_idx], probs_all[:, label_idx])
        roc_auc = auc(fpr, tpr)
        
        fprs.append(fpr)
        tprs.append(tpr)
        aucs.append(roc_auc)
        
        ax.plot(fpr, tpr, color=colors[label_idx], lw=2.5, 
                label=f'{label_name} (AUC = {roc_auc:.3f})')

    # Plot combined micro-average ROC curve
    fpr_micro, tpr_micro, _ = roc_curve(labels_all.ravel(), probs_all.ravel())
    roc_auc_micro = auc(fpr_micro, tpr_micro)

    ax.plot(fpr_micro, tpr_micro, color='#1E90FF', lw=2.5, linestyle='--',
            label=f'Micro-average (AUC = {roc_auc_micro:.3f})')

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle=':', label='Random Classifier')

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=13, fontweight='bold')
    ax.set_title('Multilabel ROC Curves - Ring Detection', fontsize=14, fontweight='bold')
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print AUC scores
    print("\n" + "="*50)
    print("ROC AUC Scores")
    print("="*50)
    for i, label_name in enumerate(label_names):
        print(f"{label_name:20s}: {aucs[i]:.4f}")
    print(f"{'Micro-average':20s}: {roc_auc_micro:.4f}")
    print("="*50)
    
    return probs_all, labels_all


def plot_confusion_matrices(probs_all, labels_all, label_names=None, threshold=0.5):
    """
    Plot confusion matrices for multilabel classification.
    
    Args:
        probs_all: Array of predicted probabilities, shape (n_samples, n_labels)
        labels_all: Array of true labels, shape (n_samples, n_labels)
        label_names: List of label names (default: ['Inner Ring', 'Outer Ring'])
        threshold: Classification threshold (default: 0.5)
    """
    if label_names is None:
        label_names = ['Inner Ring', 'Outer Ring']
    
    # Generate binary predictions (threshold at 0.5)
    preds_binary = (probs_all > threshold).astype(int)  # Shape: (n_samples, 2)
    labels_binary = labels_all.astype(int)  # Shape: (n_samples, 2)

    print(f"Binary predictions shape: {preds_binary.shape}")
    print(f"Binary labels shape: {labels_binary.shape}")

    # Create confusion matrices for each label
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for label_idx, label_name in enumerate(label_names):
        cm = confusion_matrix(labels_binary[:, label_idx], preds_binary[:, label_idx])
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Negative', 'Positive'])
        disp.plot(ax=axes[label_idx], cmap='Blues', values_format='d')
        
        axes[label_idx].set_title(f'{label_name} Confusion Matrix', fontsize=12, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Create combined confusion matrix (4 classes: [0,0], [1,0], [0,1], [1,1])
    # Map each sample to its label combination
    label_combinations = np.array([f"{labels_binary[i, 0]}{labels_binary[i, 1]}" for i in range(len(labels_binary))]).astype(int)
    pred_combinations = np.array([f"{preds_binary[i, 0]}{preds_binary[i, 1]}" for i in range(len(preds_binary))]).astype(int)

    # Create mapping: 00 -> 0, 10 -> 1, 01 -> 2, 11 -> 3
    combination_map = {0: 0, 10: 1, 1: 2, 11: 3}
    label_combo_mapped = np.array([combination_map[lc] for lc in label_combinations])
    pred_combo_mapped = np.array([combination_map[pc] for pc in pred_combinations])

    cm_combined = confusion_matrix(label_combo_mapped, pred_combo_mapped, labels=[0, 1, 2, 3])

    # Plot combined confusion matrix
    fig, ax = plt.subplots(figsize=(10, 8))
    combo_labels = ['No Rings\n[0,0]', 'Inner Ring\n[1,0]', 'Outer Ring\n[0,1]', 'Both Rings\n[1,1]']

    sns.heatmap(cm_combined, annot=True, fmt='d', cmap='Blues', ax=ax, 
                xticklabels=combo_labels, yticklabels=combo_labels,
                cbar_kws={'label': 'Count'}, annot_kws={'size': 14})

    ax.set_xlabel('Predicted Label', fontsize=13, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=13, fontweight='bold')
    ax.set_title('Combined Multilabel Confusion Matrix - Ring Detection', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()

    # Print detailed statistics
    print("\n" + "="*60)
    print("Per-Label Confusion Matrix Statistics")
    print("="*60)

    for label_idx, label_name in enumerate(label_names):
        cm = confusion_matrix(labels_binary[:, label_idx], preds_binary[:, label_idx])
        tn, fp, fn, tp = cm.ravel()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n{label_name}:")
        print(f"  True Negatives:  {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives:  {tp}")
        print(f"  Accuracy:  {accuracy:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  Recall:    {recall:.4f}")
        print(f"  F1-Score:  {f1:.4f}")

    print("="*60)
