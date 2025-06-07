# src/models/cdw_loss.py
"""
Class Distance Weighted Cross-Entropy Loss
Implementation based on life2vec methodology
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class CDW_CELoss(nn.Module):
    """
    Class Distance Weighted Cross-Entropy Loss
    
    Following life2vec paper implementation for handling class imbalance
    """
    
    def __init__(
        self,
        num_classes: int,
        alpha: float = 2.0,
        delta: float = 3.0,
        reduction: str = "mean",
        transform: str = "log",
        eps: float = 1e-8
    ):
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.delta = delta
        self.reduction = reduction
        self.transform = transform
        self.eps = eps
        
        # Create distance matrix for ordinal classes
        self.register_buffer('distance_matrix', self._create_distance_matrix())
    
    def _create_distance_matrix(self):
        """Create distance matrix between classes"""
        distance_matrix = torch.zeros(self.num_classes, self.num_classes)
        for i in range(self.num_classes):
            for j in range(self.num_classes):
                distance_matrix[i, j] = abs(i - j)
        return distance_matrix
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor):
        """
        Forward pass
        
        Args:
            logits: [batch_size, num_classes] - model predictions
            targets: [batch_size] - true class labels
            
        Returns:
            loss: scalar loss value
        """
        batch_size = logits.size(0)
        
        # Get probabilities
        probs = F.softmax(logits, dim=1)
        
        # Get predicted classes
        pred_classes = torch.argmax(logits, dim=1)
        
        # Calculate distances between predicted and true classes
        distances = self.distance_matrix[pred_classes, targets]
        
        # Apply transform to distances
        if self.transform == "log":
            weights = torch.log(1 + self.alpha * distances + self.eps)
        elif self.transform == "power":
            weights = torch.pow(1 + distances, self.alpha)
        else:
            weights = 1 + self.alpha * distances
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(logits, targets, reduction='none')
        
        # Apply distance weights
        weighted_loss = weights * ce_loss
        
        # Apply reduction
        if self.reduction == "mean":
            return weighted_loss.mean()
        elif self.reduction == "sum":
            return weighted_loss.sum()
        else:
            return weighted_loss
