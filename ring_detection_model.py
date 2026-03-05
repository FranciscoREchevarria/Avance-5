"""
RingDetectionZoobot: Custom PyTorch Lightning module for multilabel ring classification.

This module provides a fine-tunable model for detecting inner and outer rings in galaxy images
using a pretrained Zoobot encoder.
"""

import torch
from torch import nn
import lightning.pytorch as pl
from torchmetrics import Accuracy


class RingDetectionZoobot(pl.LightningModule):
    def __init__(self, encoder, hidden_dim=256, dropout_rate=0.4, **kwargs):
        """
        Custom Zoobot model for multilabel ring classification.
        
        Args:
            encoder: Pretrained Zoobot encoder (already instantiated)
            hidden_dim: Hidden layer dimension
            dropout_rate: Dropout probability
        """
        super().__init__()
        
        # Store encoder without calling parent init
        self.encoder = encoder

        #get encoder output dimension (assuming it's 512 for ConvNeXt Pico)
        encoder_dim = 512
        
        # Define classification head
        self.head = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(encoder_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate / 2),
            nn.Linear(hidden_dim, 2)  # Binary multilabel output
        )
        
        # Loss and metrics
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.train_accuracy = Accuracy(task='multilabel', num_labels=2)
        self.val_accuracy = Accuracy(task='multilabel', num_labels=2)
        
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
        
        preds = (logits.sigmoid() > 0.5).float()
        acc = self.train_accuracy(preds, y)
        
        self.log('finetuning/train_loss', loss, on_epoch=True)
        self.log('finetuning/train_acc', acc, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['ring_class']
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        
        preds = (logits.sigmoid() > 0.5).float()
        acc = self.val_accuracy(preds, y)
        
        self.log('finetuning/val_loss', loss, on_epoch=True)
        self.log('finetuning/val_acc', acc, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer with proper parameter groups."""
        # Collect trainable parameters
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        
        if not trainable_params:
            raise ValueError("No trainable parameters found!")
        
        return torch.optim.Adam(trainable_params, lr=1e-4)
    
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
