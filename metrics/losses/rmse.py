from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..training_metrics import LossComponent, WeightSchedule


class RMSE(LossComponent):
    """
    Root Mean Squared Error (RMSE) loss between predictions and labels.
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
        super().__init__(
            weight=weight,
            name=name,
            data_dim=data_dim,
            field_names=field_names,
            norm_stats=norm_stats
        )
        self.reduction = reduction
        self.epsilon = epsilon
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Element-wise squared error
        sq_error = (predictions - labels) ** 2

        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_weight(sq_error.shape).to(predictions.device)

        # Apply weights element-wise
        weighted_sq = sq_error * weight_tensor

        # First compute (weighted) MSE with the requested reduction
        if self.reduction == 'mean':
            mse = weighted_sq.mean()
        elif self.reduction == 'sum':
            mse = weighted_sq.sum()
        elif self.reduction == 'none':
            mse = weighted_sq
        else:
            raise ValueError(f"Unknown reduction: {self.reduction}")

        # Then take sqrt to get RMSE
        rmse = torch.sqrt(mse + self.epsilon)

        if not return_detailed:
            return rmse

        detailed: Dict[str, torch.Tensor] = {}

        # For detailed stats, we also report per-timestep and per-channel RMSE
        # Per-timestep: average squared error over batch, channels, and spatial dims
        # Assumes shape: (batch, timesteps, channels, ...)
        if len(weighted_sq.shape) >= 2:
            dims_to_reduce = [0] + list(range(2, len(weighted_sq.shape)))
            per_timestep_mse = weighted_sq.mean(dim=dims_to_reduce)
            detailed['per_timestep'] = torch.sqrt(per_timestep_mse + self.epsilon).detach()

        # Per-channel: average squared error over batch, timesteps, and spatial dims
        if len(weighted_sq.shape) >= 3:
            dims_to_reduce = [0, 1] + list(range(3, len(weighted_sq.shape)))
            per_channel_mse = weighted_sq.mean(dim=dims_to_reduce)
            detailed['per_channel'] = torch.sqrt(per_channel_mse + self.epsilon).detach()

        return rmse, detailed