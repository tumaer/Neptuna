from typing import Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..training_metrics import LossComponent, WeightSchedule


class VRMSE(LossComponent):
    """
    Variance-normalized RMSE:
        sqrt( <|u - v|^2> / (<|u - u_bar|^2> + eps) )
    where <> is an average over the same dimensions as the loss reduction.

    Features:
      * Optional scalar, per-timestep, and per-channel weighting through
        `WeightSchedule`.
      * Optional per-timestep and per-channel breakdowns for analysis.

    Design notes
    ------------
    * Fast path:
        - When `weight_schedule.is_scalar_only()` is True, skips weight
          tensor construction and applies scalar multiplication at the end.
    * General path:
        - Uses a broadcasted weight tensor from `WeightSchedule.get_weight`.
        - Adds one elementwise multiply over the error tensors.
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

        # ------------------------------------------------------------------
        # Fast path: scalar-only schedule (no timestep/channel/component)
        # ------------------------------------------------------------------
        if isinstance(self.weight_schedule, WeightSchedule) and self.weight_schedule.is_scalar_only():
            base = float(self.weight_schedule.base_weight)

            # Clean VRMSE computation
            sq_error = (predictions - labels) ** 2
            
            # Compute numerator: <|u - v|^2>
            num = sq_error.mean()

            # Compute denominator: <|u - u_bar|^2>
            # Mean over batch and spatial dims (keep time/channel structure for u_bar)
            if labels.ndim >= 2:
                dims_for_mean = [0] + list(range(2, labels.ndim))
            else:
                dims_for_mean = [0]
            
            u_bar = labels.mean(dim=dims_for_mean, keepdim=True)
            sq_dev = (labels - u_bar) ** 2
            denom = sq_dev.mean()

            total_loss = torch.sqrt(num / (denom + self.epsilon))

            if base != 1.0:
                total_loss = total_loss * base

            if not return_detailed:
                return total_loss

            detailed: Dict[str, torch.Tensor] = {}

            # Per-timestep: average over batch, channels, spatial dims
            if sq_error.ndim >= 2:
                dims_to_reduce = [0] + list(range(2, sq_error.ndim))
                num_t = sq_error.mean(dim=dims_to_reduce)
                
                u_bar_t = labels.mean(dim=dims_to_reduce, keepdim=True)
                sq_dev_t = (labels - u_bar_t) ** 2
                denom_t = sq_dev_t.mean(dim=dims_to_reduce)
                
                per_timestep = torch.sqrt(num_t / (denom_t + self.epsilon))
                if base != 1.0:
                    per_timestep = per_timestep * base
                detailed['per_timestep'] = per_timestep.detach()

            # Per-channel: average over batch, timesteps, spatial dims
            if sq_error.ndim >= 3:
                dims_to_reduce = [0, 1] + list(range(3, sq_error.ndim))
                num_c = sq_error.mean(dim=dims_to_reduce)
                
                u_bar_c = labels.mean(dim=dims_to_reduce, keepdim=True)
                sq_dev_c = (labels - u_bar_c) ** 2
                denom_c = sq_dev_c.mean(dim=dims_to_reduce)
                
                per_channel = torch.sqrt(num_c / (denom_c + self.epsilon))
                if base != 1.0:
                    per_channel = per_channel * base
                detailed['per_channel'] = per_channel.detach()

            return total_loss, detailed

        # ------------------------------------------------------------------
        # General path: some schedule active (timestep and/or channel)
        # ------------------------------------------------------------------
        sq_error = (predictions - labels) ** 2

        # Broadcastable weights (at most (1, T, C, 1, ...)), on correct device
        weight_tensor = self.weight_schedule.get_weight(sq_error.shape).to(predictions.device)
        weighted_sq_error = sq_error * weight_tensor

        # Compute numerator: <|u - v|^2>
        num = weighted_sq_error.mean()

        # Compute denominator: <|u - u_bar|^2> with same weighting
        if labels.ndim >= 2:
            dims_for_mean = [0] + list(range(2, labels.ndim))
        else:
            dims_for_mean = [0]

        u_bar = labels.mean(dim=dims_for_mean, keepdim=True)
        sq_dev = (labels - u_bar) ** 2
        weighted_sq_dev = sq_dev * weight_tensor
        denom = weighted_sq_dev.mean()

        total_loss = torch.sqrt(num / (denom + self.epsilon))

        if not return_detailed:
            return total_loss

        detailed: Dict[str, torch.Tensor] = {}

        # Aggregated diagnostics (reductions over the weighted errors)
        if labels.ndim >= 2:
            dims_to_reduce = [0] + list(range(2, labels.ndim))
            num_t = weighted_sq_error.mean(dim=dims_to_reduce)

            u_bar_t = labels.mean(dim=dims_to_reduce, keepdim=True)
            sq_dev_t = (labels - u_bar_t) ** 2
            denom_t = (sq_dev_t * weight_tensor).mean(dim=dims_to_reduce)

            detailed['per_timestep'] = torch.sqrt(
                num_t / (denom_t + self.epsilon)
            ).detach()

        if labels.ndim >= 3:
            dims_to_reduce = [0, 1] + list(range(3, labels.ndim))
            num_c = weighted_sq_error.mean(dim=dims_to_reduce)

            u_bar_c = labels.mean(dim=dims_to_reduce, keepdim=True)
            sq_dev_c = (labels - u_bar_c) ** 2
            denom_c = (sq_dev_c * weight_tensor).mean(dim=dims_to_reduce)

            detailed['per_channel'] = torch.sqrt(
                num_c / (denom_c + self.epsilon)
            ).detach()

        return total_loss, detailed