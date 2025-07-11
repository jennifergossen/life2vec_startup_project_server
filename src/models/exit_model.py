# src/models/exit_model.py
"""
Exit prediction model that loads pretrained startup2vec
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

# Import your existing transformer
from transformer.transformer import Transformer

log = logging.getLogger(__name__)

class StartupExitModel(pl.LightningModule):
    """
    Model for predicting startup exit using pretrained startup2vec
    """
    
    def __init__(
        self,
        pretrained_model_path: str,
        vocab_size: int = 13305,  # From your vocabulary
        num_classes: int = 2,
        freeze_encoder: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
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
        
        # Load pretrained transformer
        self.transformer = self._load_pretrained_transformer(pretrained_model_path)
        
        # Get hidden size from transformer
        self.hidden_size = self.transformer.hparams.hidden_size
        
        # Freeze encoder if requested
        if freeze_encoder:
            for param in self.transformer.parameters():
                param.requires_grad = False
            log.info("Frozen pretrained transformer encoder")
        else:
            log.info("Unfrozen pretrained transformer encoder")
        
        # Classification head for exit prediction
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(self.hidden_size // 2, num_classes)
        )
        
        # Store class weights for loss calculation
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights, dtype=torch.float))
        else:
            self.register_buffer('class_weights', None)
        
        # Metrics storage
        self.validation_step_outputs = []
        self.test_step_outputs = []
    
    def _load_pretrained_transformer(self, checkpoint_path: str) -> Transformer:
        """Load pretrained transformer from checkpoint"""
        log.info(f"Loading pretrained transformer from {checkpoint_path}")
        
        # Load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Extract state dict and hparams
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Extract hparams if available
        if 'hparams' in checkpoint:
            hparams_dict = checkpoint['hparams']
            log.info("Using hparams from checkpoint")
            
            # Convert dict to object with attributes
            class HParams:
                def __init__(self, hparams_dict):
                    # Set all attributes from the dictionary
                    for key, value in hparams_dict.items():
                        setattr(self, key, value)
                    
                    # Ensure required attributes exist with defaults
                    if not hasattr(self, 'hidden_size'):
                        self.hidden_size = getattr(self, 'd_model', 512)
                    if not hasattr(self, 'vocab_size'):
                        self.vocab_size = 13305
                    if not hasattr(self, 'max_length'):
                        self.max_length = 512
                    if not hasattr(self, 'attention_type'):
                        self.attention_type = "linear"
                    if not hasattr(self, 'emb_dropout'):
                        self.emb_dropout = 0.1
                    if not hasattr(self, 'ff_dropout'):
                        self.ff_dropout = 0.1
                    if not hasattr(self, 'dc_dropout'):
                        self.dc_dropout = 0.1
                    if not hasattr(self, 'epsilon'):
                        self.epsilon = 1e-6
                    if not hasattr(self, 'parametrize_emb'):
                        self.parametrize_emb = False
                    if not hasattr(self, 'norm_output_emb'):
                        self.norm_output_emb = False
                    if not hasattr(self, 'weight_tying'):
                        self.weight_tying = None
                    if not hasattr(self, 'feature_redraw_interval'):
                        self.feature_redraw_interval = 1000
                    if not hasattr(self, 'performer_attention'):
                        self.performer_attention = True
                    if not hasattr(self, 'nb_features'):
                        self.nb_features = None
                    if not hasattr(self, 'generalized_attention'):
                        self.generalized_attention = False
                    if not hasattr(self, 'kernel_transformation'):
                        self.kernel_transformation = False
                    if not hasattr(self, 'use_rotary_position_emb'):
                        self.use_rotary_position_emb = False
                    if not hasattr(self, 'ff_hidden_size'):
                        self.ff_hidden_size = self.hidden_size * 4
            
            hparams = HParams(hparams_dict)
        else:
            # Create minimal hparams based on your model
            log.warning("No hparams in checkpoint, using defaults")
            class HParams:
                def __init__(self):
                    self.vocab_size = 13305
                    self.hidden_size = 512
                    self.n_encoders = 6
                    self.n_heads = 8
                    self.max_length = 512
                    self.attention_type = "linear"
                    self.emb_dropout = 0.1
                    self.ff_dropout = 0.1
                    self.dc_dropout = 0.1
                    self.epsilon = 1e-6
                    self.parametrize_emb = False
                    self.norm_output_emb = False
                    self.weight_tying = None
                    self.feature_redraw_interval = 1000
                    self.performer_attention = True
                    self.nb_features = None
                    self.generalized_attention = False
                    self.kernel_transformation = False
                    self.use_rotary_position_emb = False
                    self.ff_hidden_size = 2048
            
            hparams = HParams()
        
        # Create transformer with the hparams
        transformer = Transformer(hparams=hparams)
        
        # Filter state dict to only transformer weights
        transformer_state_dict = {}
        for key, value in state_dict.items():
            # Remove any module prefixes and only keep transformer weights
            if key.startswith('transformer.'):
                new_key = key.replace('transformer.', '')
                transformer_state_dict[new_key] = value
            elif not any(prefix in key for prefix in ['classifier', 'mlm', 'sop', 'decoder']):
                # Include weights that don't belong to classification heads
                transformer_state_dict[key] = value
        
        # Load the weights
        missing_keys, unexpected_keys = transformer.load_state_dict(
            transformer_state_dict, strict=False
        )
        
        log.info(f"Loaded pretrained weights:")
        log.info(f"  Missing keys: {len(missing_keys)}")
        log.info(f"  Unexpected keys: {len(unexpected_keys)}")
        if missing_keys:
            log.info(f"  Missing: {missing_keys[:5]}...")  # Show first 5
        
        return transformer
    
    def forward(self, input_ids, padding_mask=None, **kwargs):
        """Forward pass through transformer and classifier"""
        # Use the finetuning forward method
        transformer_output = self.transformer.forward_finetuning(
            x=input_ids,
            padding_mask=padding_mask
        )
        
        # Use CLS token (first token) for classification
        cls_output = transformer_output[:, 0, :]  # [batch_size, hidden_size]
        
        # Get exit predictions
        exit_logits = self.classifier(cls_output)
        
        return {
            'exit_logits': exit_logits,
            'transformer_output': transformer_output
        }
    
    def training_step(self, batch, batch_idx):
        """Training step with exit loss"""
        output = self.forward(
            input_ids=batch['input_ids'],
            padding_mask=batch['padding_mask']
        )
        
        # Calculate exit loss
        loss = F.cross_entropy(
            output['exit_logits'], 
            batch['exit_label'].squeeze(),
            weight=self.class_weights
        )
        
        # Calculate accuracy
        preds = torch.argmax(output['exit_logits'], dim=1)
        acc = accuracy_score(
            batch['exit_label'].cpu().numpy().flatten(),
            preds.cpu().numpy()
        )
        
        # Log metrics
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        output = self.forward(
            input_ids=batch['input_ids'],
            padding_mask=batch['padding_mask']
        )
        
        # Calculate loss
        loss = F.cross_entropy(
            output['exit_logits'], 
            batch['exit_label'].squeeze(),
            weight=self.class_weights
        )
        
        # Store outputs for epoch-end metrics
        self.validation_step_outputs.append({
            'loss': loss,
            'logits': output['exit_logits'],
            'labels': batch['exit_label']
        })
        
        return loss
    
    def on_validation_epoch_end(self):
        """Calculate validation metrics at epoch end"""
        if not self.validation_step_outputs:
            return
        
        # Aggregate outputs
        all_logits = torch.cat([x['logits'] for x in self.validation_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.validation_step_outputs])
        
        # Overall metrics
        avg_loss = torch.stack([x['loss'] for x in self.validation_step_outputs]).mean()
        preds = torch.argmax(all_logits, dim=1)
        probs = F.softmax(all_logits, dim=1)[:, 1]  # Probability of exit
        
        # Calculate metrics
        labels_np = all_labels.cpu().numpy().flatten()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        acc = accuracy_score(labels_np, preds_np)
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, preds_np, average='weighted', zero_division=0
            )
        except:
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.0  # In case of single class
        
        # Log overall metrics
        self.log('val/loss', avg_loss, prog_bar=True)
        self.log('val/acc', acc, prog_bar=True)
        self.log('val/precision', precision)
        self.log('val/recall', recall)
        self.log('val/f1', f1)
        self.log('val/auc', auc)
        
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
            'logits': output['exit_logits'],
            'labels': batch['exit_label']
        })
        
        return output
    
    def on_test_epoch_end(self):
        """Calculate test metrics"""
        if not self.test_step_outputs:
            return
        
        all_logits = torch.cat([x['logits'] for x in self.test_step_outputs])
        all_labels = torch.cat([x['labels'] for x in self.test_step_outputs])
        
        preds = torch.argmax(all_logits, dim=1)
        probs = F.softmax(all_logits, dim=1)[:, 1]
        
        labels_np = all_labels.cpu().numpy().flatten()
        preds_np = preds.cpu().numpy()
        probs_np = probs.cpu().numpy()
        
        # Calculate and log test metrics
        acc = accuracy_score(labels_np, preds_np)
        
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                labels_np, preds_np, average='weighted', zero_division=0
            )
        except:
            precision, recall, f1 = 0.0, 0.0, 0.0
        
        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.0
        
        self.log('test/acc', acc)
        self.log('test/precision', precision)
        self.log('test/recall', recall)
        self.log('test/f1', f1)
        self.log('test/auc', auc)
        
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
            T_max=10000,
            eta_min=1e-6
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
