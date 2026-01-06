from typing import Literal, Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..loss_framework import LossComponent, WeightSchedule, apply_batch_wise_normalization, NormalizationHelper


class NegativityLoss(LossComponent):
    """
    Negativity index: sum of all negative values in predictions.

    This metric computes the sum of all negative values, providing a measure
    of how much "negative mass" exists in the predictions. Useful for monitoring
    physical validity in CFD simulations where certain quantities (e.g., density,
    pressure) should be non-negative.

    Features:
      * Optional scalar, per-timestep, and per-channel weighting through
        `WeightSchedule`.
      * Optional per-timestep and per-channel breakdowns for analysis.
      * Optional batch-wise normalization.
      * NOT differentiable (uses boolean masking).

    Design notes
    ------------
    * This is primarily a diagnostic/penalty metric.
    * The metric is non-differentiable due to the masking operation.
    * Returns the absolute value of the sum of negative values (positive number).
    * Fast path optimized for scalar-only weight schedules.
    * Detailed metrics only computed when `return_detailed=True`.
    """

    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        reduction: str = 'sum',
        normalization: Literal['none', 'magnitude', 'variance'] = 'none',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim,
                         field_names=field_names, norm_helper=norm_helper)
        self.reduction = reduction
        self.epsilon = epsilon
        self.normalization = normalization

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # Compute normalization factor from labels
        label_norm = torch.abs(labels).sum()
        label_norm = torch.clamp(label_norm, min=self.epsilon)

        # ------------------------------------------------------------------
        # Fast path: scalar-only schedule (no timestep/channel/component)
        # ------------------------------------------------------------------
        if isinstance(self.weight_schedule, WeightSchedule) and self.weight_schedule.is_scalar_only():
            base = float(self.weight_schedule.base_weight)

            # Compute negativity: sum of all negative values
            negative_mask = predictions < 0
            negative_values = torch.where(negative_mask, predictions, torch.zeros_like(predictions))
            negativity_sum = torch.abs(negative_values.sum())

            # Normalize by label magnitude
            total_loss = negativity_sum / label_norm

            if base != 1.0:
                total_loss = total_loss * base

            total_loss = apply_batch_wise_normalization(
                total_loss,
                labels,
                self.normalization,
                self.epsilon
            )

            if not return_detailed:
                return total_loss

            detailed: Dict[str, torch.Tensor] = {}

            # Per-timestep: sum over batch, channels, spatial dims
            if predictions.ndim >= 2:
                timesteps = predictions.shape[1]
                per_timestep = []
                
                for t in range(timesteps):
                    t_slice = predictions[:, t]
                    t_mask = t_slice < 0
                    t_negative = torch.where(t_mask, t_slice, torch.zeros_like(t_slice))
                    t_sum = torch.abs(t_negative.sum())
                    
                    # Normalize by corresponding timestep label magnitude
                    t_label_norm = torch.abs(labels[:, t]).sum()
                    t_label_norm = torch.clamp(t_label_norm, min=self.epsilon)
                    t_sum = t_sum / t_label_norm
                    
                    per_timestep.append(t_sum)
                
                per_timestep_tensor = torch.stack(per_timestep)
                if base != 1.0:
                    per_timestep_tensor = per_timestep_tensor * base
                detailed['per_timestep'] = per_timestep_tensor.detach()

            # Per-channel: sum over batch, timesteps, spatial dims
            if predictions.ndim >= 3:
                channels = predictions.shape[2]
                per_channel = []
                
                for c in range(channels):
                    c_slice = predictions[:, :, c]
                    c_mask = c_slice < 0
                    c_negative = torch.where(c_mask, c_slice, torch.zeros_like(c_slice))
                    c_sum = torch.abs(c_negative.sum())
                    
                    # Normalize by corresponding channel label magnitude
                    c_label_norm = torch.abs(labels[:, :, c]).sum()
                    c_label_norm = torch.clamp(c_label_norm, min=self.epsilon)
                    c_sum = c_sum / c_label_norm
                    
                    per_channel.append(c_sum)
                
                per_channel_tensor = torch.stack(per_channel)
                if base != 1.0:
                    per_channel_tensor = per_channel_tensor * base
                detailed['per_channel'] = per_channel_tensor.detach()

            return total_loss, detailed

        # ------------------------------------------------------------------
        # General path: some schedule active (timestep and/or channel)
        # ------------------------------------------------------------------
        # With weighted schedules, we weight the negative values before summing
        negative_mask = predictions < 0
        negative_values = torch.where(negative_mask, predictions, torch.zeros_like(predictions))
        
        # Make values positive for weighting
        unweighted = torch.abs(negative_values)

        # Broadcastable weights (at most (1, T, C, 1, ...)), on correct device
        weight_tensor = self.weight_schedule.get_loss_weight(unweighted.shape).to(predictions.device)
        weighted = unweighted * weight_tensor

        # Normalize by label magnitude
        total_loss = weighted.sum() / label_norm

        if not return_detailed:
            return total_loss

        detailed: Dict[str, torch.Tensor] = {}

        # Aggregated diagnostics (reductions over the weighted loss)
        if weighted.ndim >= 2:
            dims_to_reduce = [0] + list(range(2, weighted.ndim))
            per_timestep = weighted.sum(dim=dims_to_reduce)
            
            # Normalize each timestep by its label magnitude
            if labels.ndim >= 2:
                timesteps = labels.shape[1]
                for t in range(timesteps):
                    t_label_norm = torch.abs(labels[:, t]).sum()
                    t_label_norm = torch.clamp(t_label_norm, min=self.epsilon)
                    per_timestep[t] = per_timestep[t] / t_label_norm
            
            detailed['per_timestep'] = per_timestep.detach()

        if weighted.ndim >= 3:
            dims_to_reduce = [0, 1] + list(range(3, weighted.ndim))
            per_channel = weighted.sum(dim=dims_to_reduce)
            
            # Normalize each channel by its label magnitude
            if labels.ndim >= 3:
                channels = labels.shape[2]
                for c in range(channels):
                    c_label_norm = torch.abs(labels[:, :, c]).sum()
                    c_label_norm = torch.clamp(c_label_norm, min=self.epsilon)
                    per_channel[c] = per_channel[c] / c_label_norm
            
            detailed['per_channel'] = per_channel.detach()

        return total_loss, detailed