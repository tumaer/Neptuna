from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..training_metrics import LossComponent, WeightSchedule


class L1Loss(LossComponent):
    """
    L1 (Mean Absolute Error) loss between predictions and labels.
    Supports per-timestep and per-channel weighting.
    """
    def __init__(
        self, 
        weight: Union[float, WeightSchedule] = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        norm_stats: Dict[str, Dict[str, float]] = None,
        reduction: str = 'mean'
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, 
                         field_names=field_names, norm_stats=norm_stats)
        self.reduction = reduction
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Compute element-wise absolute error
        unweighted = torch.abs(predictions - labels)
        
        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_weight(unweighted.shape)
        
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
        
        # Per-timestep
        if len(weighted.shape) >= 2:
            dims_to_reduce = [0] + list(range(2, len(weighted.shape)))
            detailed['per_timestep'] = weighted.mean(dim=dims_to_reduce).detach()
        
        # Per-channel
        if len(weighted.shape) >= 3:
            dims_to_reduce = [0, 1] + list(range(3, len(weighted.shape)))
            detailed['per_channel'] = weighted.mean(dim=dims_to_reduce).detach()
        
        return total_loss, detailed