from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training_metrics import LossComponent


class L1Loss(LossComponent):
    """
    L1 (Mean Absolute Error) loss between predictions and labels.
    """
    def __init__(
        self, 
        weight: float = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        reduction: str = 'mean'
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names)
        self.reduction = reduction
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.l1_loss(predictions, labels, reduction=self.reduction)
        weighted_loss = self.weight * loss
        
        return weighted_loss