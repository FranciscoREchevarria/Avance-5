"""
RingDetectionZoobot: Custom PyTorch Lightning module for multilabel ring classification.

This module provides a fine-tunable model for detecting inner and outer rings in galaxy images
using a pretrained Zoobot encoder.
"""

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import Accuracy, F1Score, Precision, Recall


class RingDetectionZoobot(pl.LightningModule):
    def __init__(
        self,
        encoder,
        encoder_dim: int = 512,
        hidden_dim: int = 256,
        dropout_rate: float = 0.4,
        encoder_lr: float = 1e-5,
        head_lr: float = 1e-4,
        weight_decay: float = 1e-4,
        pos_weight=None,
        **kwargs,
    ):
        """
        Custom Zoobot model for multilabel ring classification.
        
        Args:
            encoder: Pretrained Zoobot encoder (already instantiated).
            encoder_dim: Dimension of encoder feature output.
            hidden_dim: Hidden layer dimension in the classification head.
            dropout_rate: Dropout probability in the head.
            encoder_lr: Learning rate for encoder parameters during fine-tuning.
            head_lr: Learning rate for the classification head.
            weight_decay: Weight decay for the optimizer (AdamW).
            pos_weight: Optional tensor/array of shape [2] for BCEWithLogitsLoss
                        to handle class imbalance between inner/outer rings.
        """
        super().__init__()

        self.inner_ring_threshold = 0.7
        self.outer_ring_threshold = 0.5

        # Save lightweight hyperparameters for reproducibility / checkpoints
        self.save_hyperparameters(ignore=["encoder"])

        # Store encoder without calling parent init
        self.encoder = encoder

        # Define classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_dim, 2)  # Binary multilabel output
        )

        # Loss and metrics
        if pos_weight is not None:
            pos_weight_tensor = torch.as_tensor(pos_weight, dtype=torch.float32)
            self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        else:
            self.loss_fn = nn.BCEWithLogitsLoss()

        # Core metrics
        self.train_accuracy = Accuracy(task='multilabel', num_labels=2)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=2)

        # More informative multilabel metrics (macro over labels)
        self.train_f1_macro = F1Score(task='multilabel', num_labels=2, average='macro')
        self.val_f1_macro = F1Score(task='multilabel', num_labels=2, average='macro')
        self.train_precision_macro = Precision(task='multilabel', num_labels=2, average='macro')
        self.val_precision_macro = Precision(task='multilabel', num_labels=2, average='macro')
        self.train_recall_macro = Recall(task='multilabel', num_labels=2, average='macro')
        self.val_recall_macro = Recall(task='multilabel', num_labels=2, average='macro')

        # Optimizer hyperparameters
        self.encoder_lr = encoder_lr
        self.head_lr = head_lr
        self.weight_decay = weight_decay

        # Track freeze state
        self.encoder_frozen = True
        self.freeze_encoder()
    
    def freeze_encoder(self):
        """Freeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.encoder_frozen = True
        print("✓ Encoder frozen")
    
    def unfreeze_encoder(self):
        """Unfreeze encoder parameters."""
        for param in self.encoder.parameters():
            param.requires_grad = True
        self.encoder_frozen = False
        print("✓ Encoder unfrozen")
    
    def forward(self, x):
        features = self.encoder(x)
        logits = self.head(features)
        return logits
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        #prediction threshold is performed by class, allowing for different thresholds for inner vs outer ring detection if desired
        threshold = torch.tensor([self.inner_ring_threshold, self.outer_ring_threshold], device=logits.device)

        preds = (logits.sigmoid() > threshold).float()
        acc = self.train_accuracy(preds, y)
        f1 = self.train_f1_macro(preds, y)
        prec = self.train_precision_macro(preds, y)
        rec = self.train_recall_macro(preds, y)

        self.log('finetuning/train_loss', loss, on_epoch=True)
        self.log('finetuning/train_acc', acc, on_epoch=True)
        self.log('finetuning/train_f1_macro', f1, on_epoch=True)
        self.log('finetuning/train_precision_macro', prec, on_epoch=True)
        self.log('finetuning/train_recall_macro', rec, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        #prediction threshold is performed by class, allowing for different thresholds for inner vs outer ring detection if desired
        threshold = torch.tensor([self.inner_ring_threshold, self.outer_ring_threshold], device=logits.device)

        preds = (logits.sigmoid() > threshold).float()
        acc = self.val_accuracy(preds, y)
        f1 = self.val_f1_macro(preds, y)
        prec = self.val_precision_macro(preds, y)
        rec = self.val_recall_macro(preds, y)

        self.log('finetuning/val_loss', loss, on_epoch=True)
        self.log('finetuning/val_acc', acc, on_epoch=True)
        self.log('finetuning/val_f1_macro', f1, on_epoch=True)
        self.log('finetuning/val_precision_macro', prec, on_epoch=True)
        self.log('finetuning/val_recall_macro', rec, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with separate parameter groups for head/encoder."""
        # Head parameters are always trainable
        head_params = [p for p in self.head.parameters() if p.requires_grad]

        # Encoder parameters may be frozen during stage 1
        encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]

        if not head_params and not encoder_params:
            raise ValueError("No trainable parameters found!")

        param_groups = []

        if head_params:
            param_groups.append(
                {"params": head_params, "lr": self.head_lr}
            )

        if encoder_params:
            param_groups.append(
                {"params": encoder_params, "lr": self.encoder_lr}
            )

        optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.weight_decay,
        )

        return optimizer
    
    def predict(self, x, threshold=0.5):
        probs = self.forward(x).sigmoid()
        return (probs > threshold).float()

    def predict_proba(self, x):
        """
        Get probability predictions for binary multilabel classification.
        
        Args:
            x: Input image tensor
            
        Returns:
            Probability tensor of shape (batch_size, 2) with values in [0, 1]
        """
        logits = self.forward(x)
        probabilities = logits.sigmoid()
        return probabilities 

    def batch_to_supervised_tuple(self, batch):
        """Convert batch dictionary to (x, y) tuple for training."""
        # Your dataset returns {'image': tensor, 'ring_class': tensor}
        x = batch['image']
        y = batch['ring_class']
        return x, y
