from typing import Optional, List

import torch
import torch.nn as nn

from ..training_metrics import LossComponent


class PearsonCorrelationLoss(LossComponent):
    """
    Pearson correlation coefficient loss between predictions and labels.
    Loss = 1 - r where r is the Pearson correlation coefficient.
    """
    def __init__(
        self, 
        weight: float = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names)
        self.epsilon = epsilon
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        # Shape: (batch, frames, channels, *spatial_dims)
        # Compute correlation per (batch, frame, channel) across spatial points
        
        # Get leading dimensions
        batch_size = predictions.shape[0]
        num_frames = predictions.shape[1]
        num_channels = predictions.shape[2]
        
        # Flatten spatial dimensions: (batch, frames, channels, -1)
        pred_flat = predictions.reshape(batch_size, num_frames, num_channels, -1)
        label_flat = labels.reshape(batch_size, num_frames, num_channels, -1)
        
        # Center across spatial dimension (dim=-1)
        pred_mean = pred_flat.mean(dim=-1, keepdim=True)
        label_mean = label_flat.mean(dim=-1, keepdim=True)
        
        pred_centered = pred_flat - pred_mean
        label_centered = label_flat - label_mean
        
        # Calculate Pearson correlation per (batch, frame, channel)
        covariance = (pred_centered * label_centered).mean(dim=-1)
        pred_std = torch.sqrt((pred_centered ** 2).mean(dim=-1) + self.epsilon)
        label_std = torch.sqrt((label_centered ** 2).mean(dim=-1) + self.epsilon)
        
        correlation = covariance / (pred_std * label_std + self.epsilon)
        
        # Loss = 1 - r (maximize positive correlation)
        # r ranges from -1 to 1, so loss ranges from 0 (perfect) to 2 (worst)
        loss = (1.0 - correlation).mean()
        weighted_loss = self.weight * loss
        
        return weighted_loss