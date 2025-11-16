from typing import Optional, List, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..training_metrics import LossComponent


class L2Loss(LossComponent):
    """
    L2 (Mean Squared Error) loss between predictions and labels.
    """
    def __init__(
        self, 
        weight: float = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        norm_stats: Dict[str, Dict[str, float]] = None,
        reduction: str = 'mean',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names, norm_stats=norm_stats)
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        loss = F.mse_loss(predictions, labels, reduction=self.reduction)
        weighted_loss = self.weight * loss
        return weighted_loss