from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..training_metrics import LossComponent, WeightSchedule


class L2Loss(LossComponent):
    """
    L2 (Mean Squared Error) loss between predictions and labels.
    Supports per-timestep and per-channel weighting.
    """
    def __init__(
        self, 
        weight: Union[float, WeightSchedule] = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        norm_stats: Dict[str, Dict[str, float]] = None,
        reduction: str = 'mean',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, 
                         field_names=field_names, norm_stats=norm_stats)
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Compute element-wise squared error
        unweighted = (predictions - labels) ** 2
        
        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_weight(unweighted.shape).to(predictions.device)
        
        # Apply weights element-wise
        weighted = unweighted * weight_tensor
        
        # Apply reduction
        if self.reduction == 'mean':
            total_loss = weighted.mean()
        elif self.reduction == 'sum':
            total_loss = weighted.sum()
        elif self.reduction == 'none':
            total_loss = weighted
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")
        
        if not return_detailed:
            return total_loss
        
        # Build detailed breakdown
        detailed = {}
        
        # Per-timestep: average over batch, channels, and spatial dims
        # Assumes shape: (batch, timesteps, channels, ...)
        if len(weighted.shape) >= 2:
            dims_to_reduce = [0] + list(range(2, len(weighted.shape)))
            detailed['per_timestep'] = weighted.mean(dim=dims_to_reduce).detach()
        
        # Per-channel: average over batch, timesteps, and spatial dims
        if len(weighted.shape) >= 3:
            dims_to_reduce = [0, 1] + list(range(3, len(weighted.shape)))
            detailed['per_channel'] = weighted.mean(dim=dims_to_reduce).detach()
        
        return total_loss, detailed