# src/models/survival_model.py
"""
Survival prediction model for startup data
"""

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from typing import Dict, Any, Optional
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score

# Import from your actual transformer files
try:
    # Import the main Transformer from transformer.py
    from transformer.transformer import Transformer
except ImportError:
    try:
        # Fallback if it's in modules.py
        from transformer.modules import Transformer
    except ImportError:
        raise ImportError("Could not find Transformer class in transformer module")

# Since you don't have cls_model.py, let's check modules.py for classification components
try:
    from transformer.modules import CLSModel
except ImportError:
    # Create our own simple classifier since CLSModel doesn't exist
    class CLSModel(nn.Module):
        """Simple classification head for survival prediction"""
        def __init__(self, d_model: int, num_classes: int = 2, dropout: float = 0.1):
            super().__init__()
            self.dropout = nn.Dropout(dropout)
            self.classifier = nn.Linear(d_model, num_classes)
        
        def forward(self, x):
            # x should be [batch_size, seq_len, d_model]
            # Use CLS token (first token) for classification
            if len(x.shape) == 3:  # [batch_size, seq_len, d_model]
                cls_output = x[:, 0, :]  # Take first token
            else:  # [batch_size, d_model]
                cls_output = x
            
            cls_output = self.dropout(cls_output)
            return self.classifier(cls_output)

log = logging.getLogger(__name__)

class StartupSurvivalModel(pl.LightningModule):
    """
    Model for predicting startup survival at different time horizons
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        max_length: int = 512,
        num_classes: int = 2,  # survived/died
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        max_steps: int = 10000,
        dropout: float = 0.1,
        prediction_windows: list = None,
        class_weights: Optional[torch.Tensor] = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        if prediction_windows is None:
            prediction_windows = [1, 2, 3, 4]
        self.prediction_windows = prediction_windows
        
        # Base transformer model
        self.transformer = Transformer(
            vocab_size=vocab_size,
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_length=max_length,
            dropout=dropout,
            **kwargs
        )
        
        # Classification head for survival prediction
        self.classifier = CLSModel(
            d_model=d_model,
            num_classes=num_classes,
            dropout=dropout
        )
        
        # Store class weights for loss calculation
        self.register_buffer('class_weights', class_weights)
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def forward(self, input_ids, padding_mask=None, **kwargs):
        """Forward pass through transformer and classifier"""
        # Get transformer embeddings
        transformer_output = self.transformer(
            input_ids=input_ids,
            padding_mask=padding_mask,
            **kwargs
        )
        
        # Get survival predictions
        survival_logits = self.classifier(transformer_output)
        
        return {
            'survival_logits': survival_logits,
            'transformer_output': transformer_output
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with survival loss"""
        output = self.forward(
            input_ids=batch['input_ids'],
            padding_mask=batch['padding_mask']
        )
        
        # Calculate survival loss
        loss = F.cross_entropy(
            output['survival_logits'], 
            batch['survival_label'].squeeze(),
            weight=self.class_weights
        )
        
        # Calculate accuracy
        preds = torch.argmax(output['survival_logits'], dim=1)
        acc = accuracy_score(
            batch['survival_label'].cpu().numpy().flatten(),
            preds.cpu().numpy()
        )
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        output = self.forward(
            input_ids=batch['input_ids'],
            padding_mask=batch['padding_mask']
        )
        
        # Calculate loss
        loss = F.cross_entropy(
            output['survival_logits'], 
            batch['survival_label'].squeeze(),
            weight=self.class_weights
        )
        
        # Store outputs for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'logits': output['survival_logits'],
            'labels': batch['survival_label'],
            'prediction_window': batch['prediction_window']
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at epoch end"""
        if not self.validation_step_outputs:
            return
        
        # Aggregate outputs
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        all_windows = torch.cat([x['prediction_window'] for x in self.validation_step_outputs])
        
        # Overall metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        preds = torch.argmax(all_logits, dim=1)
        probs = F.softmax(all_logits, dim=1)[:, 1]  # Probability of survival
        
        # Calculate metrics
        labels_np = all_labels.cpu().numpy().flatten()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        acc = accuracy_score(labels_np, preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average='weighted', zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.0  # In case of single class
        
        # Log overall metrics
        self.log('val_loss', avg_loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        self.log('val_precision', precision)
        self.log('val_recall', recall)
        self.log('val_f1', f1)
        self.log('val_auc', auc)
        
        # Metrics by prediction window
        for window in self.prediction_windows:
            window_mask = (all_windows == window).cpu().numpy()
            if window_mask.sum() > 0:
                window_labels = labels_np[window_mask]
                window_preds = preds_np[window_mask]
                
                if len(np.unique(window_labels)) > 1:  # Check for multiple classes
                    window_acc = accuracy_score(window_labels, window_preds)
                    self.log(f'val_acc_window_{window}y', window_acc)
        
        # Clear outputs
        self.validation_step_outputs.clear()
    
    def test_step(self, batch, batch_idx):
        """Test step"""
        output = self.forward(
            input_ids=batch['input_ids'],
            padding_mask=batch['padding_mask']
        )
        
        # Store outputs for final metrics
        self.test_step_outputs.append({
            'logits': output['survival_logits'],
            'labels': batch['survival_label'],
            'prediction_window': batch['prediction_window']
        })
        
        return output
    
    def on_test_epoch_end(self):
        """Calculate test metrics"""
        if not self.test_step_outputs:
            return
        
        # Similar to validation metrics but for test
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        all_windows = torch.cat([x['prediction_window'] for x in self.test_step_outputs])
        
        preds = torch.argmax(all_logits, dim=1)
        probs = F.softmax(all_logits, dim=1)[:, 1]
        
        labels_np = all_labels.cpu().numpy().flatten()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        # Calculate and log test metrics
        acc = accuracy_score(labels_np, preds_np)
        precision, recall, f1, _ = precision_recall_fscore_support(
            labels_np, preds_np, average='weighted', zero_division=0
        )
        
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.0
        
        self.log('test_acc', acc)
        self.log('test_precision', precision)
        self.log('test_recall', recall)
        self.log('test_f1', f1)
        self.log('test_auc', auc)
        
        # Test metrics by window
        for window in self.prediction_windows:
            window_mask = (all_windows == window).cpu().numpy()
            if window_mask.sum() > 0:
                window_labels = labels_np[window_mask]
                window_preds = preds_np[window_mask]
                
                if len(np.unique(window_labels)) > 1:
                    window_acc = accuracy_score(window_labels, window_preds)
                    self.log(f'test_acc_window_{window}y', window_acc)
        
        self.test_step_outputs.clear()
    
    def configure_optimizers(self):
        """Configure optimizer and scheduler"""
        optimizer = AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=self.hparams.max_steps,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
    
    @classmethod
    def load_pretrained(cls, checkpoint_path: str, **kwargs):
        """Load model from pretrained transformer checkpoint"""
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract model state dict
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Create new model instance
        model = cls(**kwargs)
        
        # Load transformer weights (partial loading)
        transformer_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('transformer.'):
                # Remove 'transformer.' prefix if present
                new_key = key.replace('transformer.', '', 1)
                transformer_state_dict[new_key] = value
            elif 'transformer' not in key and 'classifier' not in key:
                # Assume it's a transformer weight
                transformer_state_dict[key] = value
        
        # Load transformer weights
        missing_keys, unexpected_keys = model.transformer.load_state_dict(
            transformer_state_dict, strict=False
        )
        
        log.info(f"Loaded pretrained transformer. Missing keys: {len(missing_keys)}, "
                f"Unexpected keys: {len(unexpected_keys)}")
        
        return model