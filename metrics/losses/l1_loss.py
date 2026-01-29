from typing import Literal, Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


class L1Loss(LossComponent):
    """
    L1 (mean absolute error) loss between predictions and labels.

    Features:
      * Optional scalar, per-timestep, and per-channel weighting through
        `WeightSchedule`.
      * Optional per-timestep and per-channel breakdowns for analysis.

    Design notes
    ------------
    * Fast path:
        - When `weight_schedule.is_scalar_only()` is True, this reduces
          to a plain `abs(pred - label)` followed by `.mean()` and a
          single scalar multiply.
    * General path:
        - Uses a broadcasted weight tensor from `WeightSchedule.get_weight`.
        - Adds one elementwise multiply over the loss tensor.
    * Detailed metrics:
        - Only computed when `return_detailed=True`.
        - Training loops should disable them in the hot path.
    """
    def __init__(
        self,
        norm_helper: NormalizationHelper, 
        weight: Union[float, WeightSchedule] = 1.0, 
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        reduction: str = 'mean',
        normalization: Literal['none', 'range', 'variance'] = 'none',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, 
                         field_names=field_names, norm_helper=norm_helper)
        self.reduction = reduction
        self.normalization = normalization
        self.epsilon = epsilon
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False,
        keep_bc_dims: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # ------------------------------------------------------------------
        # Fast path: scalar-only schedule (no timestep/channel/component)
        # ------------------------------------------------------------------
        if self.weight_schedule.is_scalar_only():
            base = float(self.weight_schedule.base_weight)

            # Clean L1
            abs_err = torch.abs(predictions - labels)

            abs_err = self.norm_helper.normalize_error(
                abs_err,
                labels,
                self.data_dim,
                self.normalization,
                self.epsilon
            )

            if keep_bc_dims:
                # Keep batch and channel dims; reduce over time + spatial
                reduce_dims = [1] + list(range(3, abs_err.ndim))
                total_loss = self._reduce(abs_err, reduce_dims)
            else:
                total_loss = self._reduce(abs_err)

            if base != 1.0:
                total_loss = total_loss * base

            if not return_detailed:
                return total_loss

            detailed: Dict[str, torch.Tensor] = {}

            # Per-timestep: average over batch, channels, spatial dims
            dims_to_reduce = [0] + list(range(2, abs_err.ndim))
            per_timestep = self._reduce(abs_err, dims_to_reduce)
            if base != 1.0:
                per_timestep = per_timestep * base
            detailed['per_timestep'] = per_timestep.detach()

            # Per-channel: average over batch, timesteps, spatial dims
            dims_to_reduce = [0, 1] + list(range(3, abs_err.ndim))
            per_channel = self._reduce(abs_err, dims_to_reduce)
            if base != 1.0:
                per_channel = per_channel * base
            detailed['per_channel'] = per_channel.detach()

            return total_loss, detailed

        # ------------------------------------------------------------------
        # General path: some schedule active (timestep and/or channel)
        # ------------------------------------------------------------------
        # Compute element-wise absolute error
        unweighted = torch.abs(predictions - labels)

        unweighted = self.norm_helper.normalize_error(
                unweighted,
                labels,
                self.data_dim,
                self.normalization,
                self.epsilon
            )
        
        # Get broadcastable weight tensor on correct device
        weight_tensor = self.weight_schedule.get_loss_weight(unweighted.shape).to(predictions.device)
        
        weighted = unweighted * weight_tensor
        
        if keep_bc_dims:
            # Keep batch and channel dims; reduce over time + spatial
            reduce_dims = [1] + list(range(3, weighted.ndim))
            total_loss = self._reduce(weighted, reduce_dims)
        else:
            total_loss = self._reduce(weighted)
        
        if not return_detailed:
            return total_loss
        
        # Build detailed breakdown (aggregated diagnostics)
        detailed: Dict[str, torch.Tensor] = {}
        
        # Per-timestep
        dims_to_reduce = [0] + list(range(2, weighted.ndim))
        detailed['per_timestep'] = self._reduce(weighted, dims_to_reduce).detach()
        
        # Per-channel
        dims_to_reduce = [0, 1] + list(range(3, weighted.ndim))
        detailed['per_channel'] = self._reduce(weighted, dims_to_reduce).detach()
        
        return total_loss, detailed
    
    def _reduce(self, x: torch.Tensor, dims: Optional[List[int]] = None) -> torch.Tensor:
        if self.reduction == 'mean':
            return x.mean(dim=dims) if dims is not None else x.mean()
        return x.sum(dim=dims) if dims is not None else x.sum()