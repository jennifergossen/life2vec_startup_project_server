# src/models/survival_model.py
"""
Startup Survival Prediction Model
Following life2vec's cls_model.py pattern for binary classification
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from typing import Dict, Any
import logging

# Import our CDW Loss implementation
from .cdw_loss import CDW_CELoss

from ..transformer.transformer import Transformer

log = logging.getLogger(__name__)

class StartupSurvivalModel(pl.LightningModule):
    """
    Startup survival prediction model using pretrained life2vec encoder
    
    Architecture:
    1. Load pretrained startup2vec encoder (frozen/unfrozen)
    2. Add classification head for binary survival prediction
    3. Use CDW Cross-Entropy loss for class imbalance
    """
    
    def __init__(
        self,
        pretrained_model_path: str,
        num_classes: int = 2,
        freeze_encoder: bool = False,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        # CDW Loss parameters (life2vec values)
        cdw_alpha: float = 2.0,
        cdw_delta: float = 3.0,
        cdw_transform: str = "log",
        # Class weights for fallback
        class_weights: list = None,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.pretrained_model_path = pretrained_model_path
        self.num_classes = num_classes
        self.freeze_encoder = freeze_encoder
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        
        # Load pretrained model
        self._load_pretrained_encoder()
        
        # Classification head
        self.classification_head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(self.encoder.hparams.hidden_size, self.encoder.hparams.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.encoder.hparams.hidden_size // 2, num_classes)
        )
        
        # Loss function - CDW Cross-Entropy (life2vec methodology)
        self.loss_fn = CDW_CELoss(
            num_classes=num_classes,
            alpha=cdw_alpha,
            delta=cdw_delta,
            reduction="mean",
            transform=cdw_transform,
            eps=1e-8
        )
        log.info(f"Using CDW Cross-Entropy Loss (alpha={cdw_alpha}, delta={cdw_delta}, transform={cdw_transform})")
        
        # Metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.val_acc = torchmetrics.Accuracy(task="binary") 
        self.test_acc = torchmetrics.Accuracy(task="binary")
        
        self.train_auc = torchmetrics.AUROC(task="binary")
        self.val_auc = torchmetrics.AUROC(task="binary")
        self.test_auc = torchmetrics.AUROC(task="binary")
        
        self.train_f1 = torchmetrics.F1Score(task="binary")
        self.val_f1 = torchmetrics.F1Score(task="binary")
        self.test_f1 = torchmetrics.F1Score(task="binary")
        
        # Precision/Recall for both classes
        self.val_precision = torchmetrics.Precision(task="binary", num_classes=2, average=None)
        self.val_recall = torchmetrics.Recall(task="binary", num_classes=2, average=None)
    
    def _load_pretrained_encoder(self):
        """Load pretrained startup2vec encoder"""
        log.info(f"Loading pretrained model from {self.pretrained_model_path}")
        
        # Load pretrained model checkpoint
        checkpoint = torch.load(self.pretrained_model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
            hparams = checkpoint['hparams']
        else:
            # Direct state dict
            state_dict = checkpoint
            # You'll need to provide hparams manually or extract from model
            raise ValueError("Need hparams from pretrained model")
        
        # Create encoder model
        from ..models.pretrain import TransformerEncoder
        pretrained_model = TransformerEncoder(hparams=hparams)
        pretrained_model.load_state_dict(state_dict)
        
        # Extract just the transformer encoder (not MLM/SOP heads)
        self.encoder = pretrained_model.transformer
        
        # Freeze encoder if requested
        if self.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            log.info("Encoder frozen - only training classification head")
        else:
            log.info("Encoder unfrozen - training end-to-end")
    
    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass
        
        Args:
            batch: Dictionary containing:
                - input_ids: [batch, 4, seq_len] - life2vec 4D format
                - padding_mask: [batch, seq_len]
                
        Returns:
            logits: [batch, num_classes]
        """
        input_ids = batch["input_ids"]  # [batch, 4, seq_len]
        padding_mask = batch["padding_mask"]  # [batch, seq_len]
        
        # Run through encoder
        encoded = self.encoder(input_ids, padding_mask)  # [batch, seq_len, hidden_size]
        
        # Use [CLS] token representation (first token)
        cls_representation = encoded[:, 0, :]  # [batch, hidden_size]
        
        # Classification head
        logits = self.classification_head(cls_representation)  # [batch, num_classes]
        
        return logits
    
    def _shared_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Shared step for train/val/test"""
        # Filter out invalid labels (-1)
        valid_mask = (batch["survival_label"] >= 0).flatten()
        
        if not valid_mask.any():
            # No valid samples in batch
            return None, None, None
        
        # Filter batch to valid samples
        filtered_batch = {}
        for key, value in batch.items():
            if key == "survival_label":
                filtered_batch[key] = value[valid_mask]
            elif len(value.shape) > 1:
                filtered_batch[key] = value[valid_mask]
            else:
                filtered_batch[key] = value[valid_mask]
        
        labels = filtered_batch["survival_label"].long()
        
        # Forward pass
        logits = self.forward(filtered_batch)
        
        # Loss
        loss = self.loss_fn(logits, labels)
        
        # Predictions
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)
        
        return loss, preds, labels
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Training step"""
        result = self._shared_step(batch, batch_idx)
        if result[0] is None:
            return None
        
        loss, preds, labels = result
        
        # Metrics
        self.train_acc(preds, labels)
        self.train_auc(preds, labels)
        self.train_f1(preds, labels)
        
        # Logging
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True)
        self.log("train/auc", self.train_auc, on_step=False, on_epoch=True)
        self.log("train/f1", self.train_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Validation step"""
        result = self._shared_step(batch, batch_idx)
        if result[0] is None:
            return None
        
        loss, preds, labels = result
        
        # Metrics
        self.val_acc(preds, labels)
        self.val_auc(preds, labels)
        self.val_f1(preds, labels)
        self.val_precision(preds, labels)
        self.val_recall(preds, labels)
        
        # Logging
        self.log("val/loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True)
        self.log("val/auc", self.val_auc, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/f1", self.val_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int):
        """Test step"""
        result = self._shared_step(batch, batch_idx)
        if result[0] is None:
            return None
        
        loss, preds, labels = result
        
        # Metrics
        self.test_acc(preds, labels)
        self.test_auc(preds, labels)
        self.test_f1(preds, labels)
        
        # Logging
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True)
        self.log("test/auc", self.test_auc, on_step=False, on_epoch=True)
        self.log("test/f1", self.test_f1, on_step=False, on_epoch=True)
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizers - NO SCHEDULER (like pretraining)"""
        # You're right - the scheduler caused issues in pretraining, so let's not use it
        
        if self.freeze_encoder:
            # Only optimize classification head
            optimizer = torch.optim.Adam(
                self.classification_head.parameters(),
                lr=self.learning_rate,
                weight_decay=self.weight_decay
            )
            log.info("Optimizing classification head only (encoder frozen)")
        else:
            # Different learning rates for encoder vs head
            param_groups = [
                {
                    "params": self.encoder.parameters(),
                    "lr": self.learning_rate * 0.1,  # Lower LR for pretrained encoder
                    "weight_decay": self.weight_decay
                },
                {
                    "params": self.classification_head.parameters(),
                    "lr": self.learning_rate,  # Higher LR for new head
                    "weight_decay": self.weight_decay
                }
            ]
            
            optimizer = torch.optim.Adam(param_groups)
            log.info(f"Optimizing end-to-end: encoder LR={self.learning_rate * 0.1}, head LR={self.learning_rate}")
        
        # NO SCHEDULER - just return optimizer
        return optimizer
    
    def on_validation_epoch_end(self):
        """Log additional metrics at epoch end"""
        if hasattr(self, 'val_precision') and hasattr(self, 'val_recall'):
            precision = self.val_precision.compute()
            recall = self.val_recall.compute()
            
            if len(precision) >= 2 and len(recall) >= 2:
                # Log class-specific metrics
                self.log("val/precision_died", precision[0])
                self.log("val/precision_survived", precision[1])
                self.log("val/recall_died", recall[0])
                self.log("val/recall_survived", recall[1])
                
                # Reset metrics
                self.val_precision.reset()
                self.val_recall.reset()
