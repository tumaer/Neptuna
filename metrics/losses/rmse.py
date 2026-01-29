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
    * Fast path:
        - When `weight_schedule.is_scalar_only()` is True, this reduces
          to a plain `(pred - label)**2` followed by `.mean()`, `sqrt()`,
          and a single scalar multiply.
    * General path:
        - Uses a broadcasted weight tensor from `WeightSchedule.get_weight`.
        - Adds one elementwise multiply over the squared error tensor.
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

    def _reduce(self, x: torch.Tensor, dims: Optional[List[int]] = None) -> torch.Tensor:
        if self.reduction == 'mean':
            return x.mean(dim=dims) if dims is not None else x.mean()
        return x.sum(dim=dims) if dims is not None else x.sum()
    
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

            # Clean RMSE (per-sample sqrt)
            sq_error = (predictions - labels) ** 2
            norm_error = self.norm_helper.normalize_error(
                sq_error,
                labels,
                self.data_dim,
                self.normalization,
                self.epsilon
            )

            if keep_bc_dims:
                # Keep batch and channel; reduce over time + spatial
                reduce_dims = [1] + list(range(3, norm_error.ndim))
            else:
                reduce_dims = list(range(1, norm_error.ndim))
            per_sample_mse = self._reduce(norm_error, reduce_dims)
            per_sample_rmse = torch.sqrt(per_sample_mse + self.epsilon)

            total_loss = per_sample_rmse if keep_bc_dims else self._reduce(per_sample_rmse)

            if base != 1.0:
                total_loss = total_loss * base

            if not return_detailed:
                return total_loss

            detailed: Dict[str, torch.Tensor] = {}

            # Per-timestep: average over batch, channels, spatial dims
            dims_to_reduce = [0] + list(range(2, sq_error.ndim))
            per_timestep_mse = self._reduce(sq_error, dims_to_reduce)
            per_timestep = torch.sqrt(per_timestep_mse + self.epsilon)
            if base != 1.0:
                per_timestep = per_timestep * base
            detailed['per_timestep'] = per_timestep.detach()

            # Per-channel: average over batch, timesteps, spatial dims
            dims_to_reduce = [0, 1] + list(range(3, sq_error.ndim))
            per_channel_mse = self._reduce(sq_error, dims_to_reduce)
            per_channel = torch.sqrt(per_channel_mse + self.epsilon)
            if base != 1.0:
                per_channel = per_channel * base
            detailed['per_channel'] = per_channel.detach()

            return total_loss, detailed

        # ------------------------------------------------------------------
        # General path: some schedule active (timestep and/or channel)
        # ------------------------------------------------------------------
        sq_error = (predictions - labels) ** 2

        norm_error = self.norm_helper.normalize_error(
                sq_error,
                labels,
                self.data_dim,
                self.normalization,
                self.epsilon
            )

        # Broadcastable weights (at most (1, T, C, 1, ...)), on correct device
        weight_tensor = self.weight_schedule.get_loss_weight(norm_error.shape).to(predictions.device)
        weighted_sq = norm_error * weight_tensor

        if keep_bc_dims:
            # Keep batch and channel; reduce over time + spatial
            reduce_dims = [1] + list(range(3, weighted_sq.ndim))
        else:
            reduce_dims = list(range(1, weighted_sq.ndim))
        per_sample_mse = self._reduce(weighted_sq, reduce_dims)
        per_sample_rmse = torch.sqrt(per_sample_mse + self.epsilon)

        total_loss = per_sample_rmse if keep_bc_dims else self._reduce(per_sample_rmse)

        if not return_detailed:
            return total_loss

        detailed: Dict[str, torch.Tensor] = {}

        # Aggregated diagnostics (reductions over the weighted squared error)
        dims_to_reduce = [0] + list(range(2, weighted_sq.ndim))
        per_timestep_mse = self._reduce(weighted_sq, dims_to_reduce)
        detailed['per_timestep'] = torch.sqrt(per_timestep_mse + self.epsilon).detach()

        dims_to_reduce = [0, 1] + list(range(3, weighted_sq.ndim))
        per_channel_mse = self._reduce(weighted_sq, dims_to_reduce)
        detailed['per_channel'] = torch.sqrt(per_channel_mse + self.epsilon).detach()

        return total_loss, detailed