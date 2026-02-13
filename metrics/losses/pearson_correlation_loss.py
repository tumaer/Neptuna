from typing import Literal, Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


class PearsonCorrelationLoss(LossComponent):
    """
    Pearson correlation coefficient loss between predictions and labels.
    Loss = 1 - r where r is the Pearson correlation coefficient.
    """
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        normalization: Literal['none', 'range', 'variance', 'std', 'norm', 'root_norm'] = 'none',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, 
                         field_names=field_names, norm_helper=norm_helper)
        self.epsilon = epsilon
        self.normalization = normalization
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False,
        keep_bc_dims: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
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

        unweighted = 1.0 - correlation

        weight_tensor = self.weight_schedule.get_loss_weight(unweighted.shape).to(predictions.device)
        weighted = unweighted * weight_tensor
        
        # Reduce to scalar or per-batch vector
        if keep_bc_dims:
            total_loss = weighted.mean(dim=1)
        else:
            total_loss = weighted.mean()

        if not return_detailed:
            return total_loss
        
        # Build detailed breakdown
        detailed = {}
        
        # Per-timestep: average over batch and channels
        # Shape: (frames,)
        detailed['per_timestep'] = weighted.mean(dim=(0, 2)).detach()
        
        # Per-channel: average over batch and frames
        # Shape: (channels,)
        detailed['per_channel'] = weighted.mean(dim=(0, 1)).detach()
        
        return total_loss, detailed