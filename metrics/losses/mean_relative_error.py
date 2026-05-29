from typing import Literal, Optional, List, Dict, Union, Tuple

import torch
import torch.nn as nn

from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper


class MeanRelativeError(LossComponent):
    """
    Average Relative Error loss: mean(|pred - label| / (|label| + epsilon))

    Computes pointwise relative error with respect to the label magnitude,
    then averages across all dimensions.

    Features:
      * Optional scalar, per-timestep, and per-channel weighting through
        `WeightSchedule`.
      * Optional per-timestep and per-channel breakdowns for analysis.
      * Configurable epsilon for numerical stability when labels approach zero.
      * Optional per-channel value thresholds to restrict computation to
        specific label magnitude ranges.
      * Optional visualization of threshold masking for debugging.

    Design notes
    ------------
    * General path:
        - Uses a broadcasted weight tensor from `WeightSchedule.get_weight`.
        - Adds one elementwise multiply over the loss tensor.
    * Detailed metrics:
        - Only computed when `return_detailed=True`.
        - Training loops should disable them in the hot path.
    * Thresholding:
        - When value_thresholds are specified for a channel, only pixels
          where min_threshold <= |label| <= max_threshold contribute to the loss.
        - The metric is NOT differentiable when thresholds are applied.
        - Thresholds are specified in physical (denormalized) units.
    """

    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        reduction: str = 'mean',
        normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
        epsilon: float = 1e-8,
        value_thresholds: Optional[Dict[str, Optional[List[float]]]] = None,
    ):
        """
        Args:
            value_thresholds:
                Optional dict mapping channel names to [min_threshold, max_threshold].
                Only pixels where min <= |label| <= max are included in the metric.
                Use None for a channel to include all values.
                Thresholds are in physical (denormalized) units.
                Example:
                    {'Density': [0.1, 10.0], 'Pressure': None}
            visualize: If True, creates visualization plots during forward pass (for debugging).
            viz_channel_idx: Channel index to visualize (default: 0).
            viz_save_dir: Directory to save visualization plots (default: None, just shows).
        """
        super().__init__(weight=weight, name=name, data_dim=data_dim,
                         field_names=field_names, norm_helper=norm_helper)
        self.reduction = reduction
        self.epsilon = epsilon
        self.normalization = normalization
        self.value_thresholds = value_thresholds or {}
        
        # Convert physical thresholds to normalized space if needed
        self._normalized_thresholds = self._prepare_thresholds()
    
    def _prepare_thresholds(self) -> Dict[int, Tuple[float, float]]:
        """
        Convert physical thresholds to normalized space and map to channel indices.
        
        Returns:
            Dict mapping channel index to (min_threshold, max_threshold) in normalized units.
        """
        if not self.value_thresholds or self.field_names is None:
            return {}
        
        normalized = {}
        for ch_name, threshold in self.value_thresholds.items():
            if threshold is None:
                continue
            
            if ch_name not in self.field_names:
                continue
            
            ch_idx = self.field_names.index(ch_name)
            
            # Use thresholds directly without normalization
            normalized[ch_idx] = (threshold[0], threshold[1])
        
        return normalized

    def _create_threshold_mask(
        self,
        labels: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """
        Create a binary mask indicating which pixels satisfy the threshold conditions.
        
        Args:
            labels: Label tensor, shape (B, T, C, *spatial)
        
        Returns:
            Boolean mask of same shape as labels, or None if no thresholds configured.
        """
        if not self._normalized_thresholds:
            return None
        
        # Start with all True
        mask = torch.ones_like(labels, dtype=torch.bool)
        
        # Apply per-channel thresholds
        for ch_idx, (min_thresh, max_thresh) in self._normalized_thresholds.items():
            # Get absolute values for this channel
            abs_channel = torch.abs(labels[:, :, ch_idx])
            
            # Apply threshold: min <= |label| <= max
            channel_mask = (abs_channel >= min_thresh) & (abs_channel <= max_thresh)
            
            # Update the full mask for this channel
            mask[:, :, ch_idx] = channel_mask
        
        return mask


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

        # Create threshold mask if needed
        threshold_mask = self._create_threshold_mask(labels)
        
        # Compute relative error (before any masking)
        abs_diff = torch.abs(predictions - labels)
        abs_labels = torch.abs(labels)
        rel_error = abs_diff / (abs_labels + self.epsilon)
        
        # Store unmasked relative error for detailed metrics
        rel_error_full = rel_error

        # ------------------------------------------------------------------
        # General path: some schedule active (timestep and/or channel)
        # ------------------------------------------------------------------
        unweighted = rel_error

        # Apply threshold mask if configured
        if threshold_mask is not None:
            mask_float = threshold_mask.float()
            if keep_bc_dims:
                reduce_dims = [1] + list(range(3, unweighted.ndim))
                mask_sum = mask_float.sum(dim=reduce_dims)
                masked_sum = (unweighted * mask_float).sum(dim=reduce_dims)
                total_loss = masked_sum / (mask_sum + self.epsilon)
                total_loss = torch.where(
                    mask_sum > self.epsilon,
                    total_loss,
                    torch.zeros_like(total_loss),
                )
            else:
                mask_sum = mask_float.sum()
                if mask_sum > self.epsilon:
                    total_loss = (unweighted * mask_float).sum() / mask_sum
                else:
                    total_loss = torch.tensor(0.0, device=predictions.device)
            
            if not return_detailed:
                return total_loss
            
            # Compute detailed metrics using full (unmasked) tensor
            detailed: Dict[str, torch.Tensor] = {}
            
            if predictions.ndim >= 2:
                dims_to_reduce = [0] + list(range(2, rel_error_full.ndim))
                per_timestep = rel_error_full.mean(dim=dims_to_reduce)
                detailed['per_timestep'] = per_timestep if preserve_component_grads else per_timestep.detach()
            
            if predictions.ndim >= 3:
                dims_to_reduce = [0, 1] + list(range(3, rel_error_full.ndim))
                per_channel = rel_error_full.mean(dim=dims_to_reduce)
                detailed['per_channel'] = per_channel if preserve_component_grads else per_channel.detach()
            
            mask_fraction = threshold_mask.float().mean()
            detailed['mask_fraction'] = mask_fraction if preserve_component_grads else mask_fraction.detach()
            
            return total_loss, detailed

        # Broadcastable weights (at most (1, T, C, 1, ...)), on correct device
        weight_tensor = self.weight_schedule.get_loss_weight(unweighted.shape).to(predictions.device)
        weighted = unweighted * weight_tensor

        if keep_bc_dims:
            reduce_dims = [1] + list(range(3, weighted.ndim))
            total_loss = weighted.mean(dim=reduce_dims)
        else:
            total_loss = weighted.mean()

        if not return_detailed:
            return total_loss

        detailed: Dict[str, torch.Tensor] = {}

        # Aggregated diagnostics (reductions over the weighted loss)
        if weighted.ndim >= 2:
            dims_to_reduce = [0] + list(range(2, weighted.ndim))
            per_timestep = weighted.mean(dim=dims_to_reduce)
            detailed['per_timestep'] = per_timestep if preserve_component_grads else per_timestep.detach()

        if weighted.ndim >= 3:
            dims_to_reduce = [0, 1] + list(range(3, weighted.ndim))
            per_channel = weighted.mean(dim=dims_to_reduce)
            detailed['per_channel'] = per_channel if preserve_component_grads else per_channel.detach()

        return total_loss, detailed