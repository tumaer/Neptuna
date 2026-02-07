from typing import Literal, Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


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
        normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
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
        input_frames: Optional[torch.Tensor],
        return_detailed: bool = False,
        keep_bc_dims: bool = False,
        preserve_component_grads: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        # Denormalize predictions/labels to physical space
        predictions = self.norm_helper.denormalize(predictions)
        labels = self.norm_helper.denormalize(labels)

        # Compute normalization factor from labels
        if keep_bc_dims:
            reduce_dims = [1] + list(range(3, labels.ndim))
            label_norm = torch.abs(labels).sum(dim=reduce_dims)
        else:
            label_norm = torch.abs(labels).sum()
        label_norm = torch.clamp(label_norm, min=self.epsilon)

        # With weighted schedules, we weight the negative values before summing
        negative_mask = predictions < 0
        negative_values = torch.where(negative_mask, predictions, torch.zeros_like(predictions))

        # Make values positive for weighting
        unweighted = torch.abs(negative_values)

        # Broadcastable weights (at most (1, T, C, 1, ...)), on correct device
        weight_tensor = self.weight_schedule.get_loss_weight(unweighted.shape).to(predictions.device)
        weighted = unweighted * weight_tensor

        # Normalize by label magnitude
        if keep_bc_dims:
            reduce_dims = [1] + list(range(3, weighted.ndim))
            total_loss = self._reduce(weighted, reduce_dims) / label_norm
        else:
            total_loss = self._reduce(weighted) / label_norm

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
            
            detailed['per_timestep'] = per_timestep if preserve_component_grads else per_timestep.detach()

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
            
            detailed['per_channel'] = per_channel if preserve_component_grads else per_channel.detach()

        return total_loss, detailed

    def _reduce(self, x: torch.Tensor, dims: Optional[List[int]] = None) -> torch.Tensor:
        if self.reduction == 'mean':
            return x.mean(dim=dims) if dims is not None else x.mean()
        return x.sum(dim=dims) if dims is not None else x.sum()