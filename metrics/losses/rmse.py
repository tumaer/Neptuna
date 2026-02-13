from typing import Optional, List, Dict, Union, Tuple, Literal

import torch
import torch.nn as nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


class RMSE(LossComponent):
    """
    Root Mean Squared Error (RMSE) loss between predictions and labels.

    Features:
      * Optional scalar, per-timestep, and per-channel weighting through
        `WeightSchedule`.
      * Optional per-timestep and per-channel breakdowns for analysis.

    Design notes
    ------------
        * Weighting:
                - Uses a broadcasted weight tensor from
                    `WeightSchedule.get_loss_weight` to scale the element-wise error.
                - This supports scalar, per-timestep, and per-channel weights via
                    the configured schedule.
        * Reduction:
                - Default is a full reduction over all dimensions.
                - When `keep_bc_dims=True`, batch and channel dimensions are
                    preserved and the loss is reduced over time + spatial axes only.
        * Detailed metrics:
                - Only computed when `return_detailed=True`.
                - `per_timestep` reduces over batch, channel, and spatial axes.
                - `per_channel` reduces over batch, time, and spatial axes.
                - These diagnostics are computed from the weighted error tensor and
                    are independent of `keep_bc_dims`.
    """

    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        reduction: str = 'mean',
        normalization: Literal['none', 'range', 'variance', 'std', 'norm', 'root_norm'] = 'none',
        epsilon: float = 1e-8
    ):
        super().__init__(
            weight=weight,
            name=name,
            data_dim=data_dim,
            field_names=field_names,
            norm_helper=norm_helper
        )
        if reduction not in ('mean', 'sum'):
            raise ValueError(f"Unsupported reduction: {reduction}")
        self.reduction = reduction
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

        # Compute element-wise squared error
        sq_error = (predictions - labels) ** 2

        # Normalize error if specified
        norm_error = self.norm_helper.normalize_error(
                sq_error,
                labels,
                self.data_dim,
                self.normalization,
                self.epsilon
            )

        # Get broadcastable weight tensor
        weight_tensor = self.weight_schedule.get_loss_weight(norm_error.shape).to(predictions.device)
        
        # Apply weights
        weighted_sq = norm_error * weight_tensor

        # Keep batch and channel dims (for rollout metrics)
        if keep_bc_dims:
            reduce_dims = [1] + list(range(3, weighted_sq.ndim))
        else:
            reduce_dims = list(range(1, weighted_sq.ndim))

        # Reduce and take square root
        per_sample_mse = self._reduce(weighted_sq, reduce_dims)
        per_sample_rmse = torch.sqrt(per_sample_mse + self.epsilon)

        # Aggregate over samples
        total_loss = per_sample_rmse if keep_bc_dims else self._reduce(per_sample_rmse)

        # Return single scalar (or tensor) loss
        if not return_detailed:
            return total_loss

        # ========================================================
        # Build detailed breakdown (for logging/loss weighting)
        # ========================================================
        detailed: Dict[str, torch.Tensor] = {}

        # Per-timestep
        dims_to_reduce = [0] + list(range(2, weighted_sq.ndim))
        per_timestep_mse = self._reduce(weighted_sq, dims_to_reduce)
        detailed['per_timestep'] = torch.sqrt(per_timestep_mse + self.epsilon).detach()

        # Per-channel
        dims_to_reduce = [0, 1] + list(range(3, weighted_sq.ndim))
        per_channel_mse = self._reduce(weighted_sq, dims_to_reduce)
        detailed['per_channel'] = torch.sqrt(per_channel_mse + self.epsilon).detach()

        # Return scalar loss and detailed breakdown dict
        return total_loss, detailed
    
    def _reduce(self, x: torch.Tensor, dims: Optional[List[int]] = None) -> torch.Tensor:
        # Reduction helper (mean or sum)
        if self.reduction == 'mean':
            return x.mean(dim=dims) if dims is not None else x.mean()
        return x.sum(dim=dims) if dims is not None else x.sum()