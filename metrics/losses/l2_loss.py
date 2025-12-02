from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..training_metrics import LossComponent, WeightSchedule


class L2Loss(LossComponent):
    """
    L2 (mean squared error) loss between predictions and labels.

    Features:
      * Optional scalar, per-timestep, and per-channel weighting through
        `WeightSchedule`.
      * Optional per-timestep and per-channel breakdowns for analysis.

    Design notes
    ------------
    * Fast path:
        - When `weight_schedule.is_scalar_only()` is True, this reduces
          to a plain `(pred - label)**2` followed by `.mean()` and a
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

        # ------------------------------------------------------------------
        # Fast path: scalar-only schedule (no timestep/channel/component)
        # ------------------------------------------------------------------
        if isinstance(self.weight_schedule, WeightSchedule) and self.weight_schedule.is_scalar_only():
            base = float(self.weight_schedule.base_weight)

            # Clean L2
            diff2 = (predictions - labels) ** 2
            total_loss = diff2.mean()

            if base != 1.0:
                total_loss = total_loss * base

            if not return_detailed:
                return total_loss

            detailed: Dict[str, torch.Tensor] = {}

            # Per-timestep: average over batch, channels, spatial dims
            if diff2.ndim >= 2:
                dims_to_reduce = [0] + list(range(2, diff2.ndim))
                per_timestep = diff2.mean(dim=dims_to_reduce)
                if base != 1.0:
                    per_timestep = per_timestep * base
                detailed['per_timestep'] = per_timestep.detach()

            # Per-channel: average over batch, timesteps, spatial dims
            if diff2.ndim >= 3:
                dims_to_reduce = [0, 1] + list(range(3, diff2.ndim))
                per_channel = diff2.mean(dim=dims_to_reduce)
                if base != 1.0:
                    per_channel = per_channel * base
                detailed['per_channel'] = per_channel.detach()

            return total_loss, detailed

        # ------------------------------------------------------------------
        # General path: some schedule active (timestep and/or channel)
        # ------------------------------------------------------------------
        unweighted = (predictions - labels) ** 2

        # Broadcastable weights (at most (1, T, C, 1, ...)), on correct device
        weight_tensor = self.weight_schedule.get_weight(unweighted.shape).to(predictions.device)
        weighted = unweighted * weight_tensor

        total_loss = weighted.mean()

        if not return_detailed:
            return total_loss

        detailed: Dict[str, torch.Tensor] = {}

        # Aggregated diagnostics (reductions over the weighted loss)
        if weighted.ndim >= 2:
            dims_to_reduce = [0] + list(range(2, weighted.ndim))
            detailed['per_timestep'] = weighted.mean(dim=dims_to_reduce).detach()

        if weighted.ndim >= 3:
            dims_to_reduce = [0, 1] + list(range(3, weighted.ndim))
            detailed['per_channel'] = weighted.mean(dim=dims_to_reduce).detach()

        return total_loss, detailed