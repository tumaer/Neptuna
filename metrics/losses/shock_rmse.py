from typing import Dict, List, Optional, Tuple, Union, Literal
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper
from .h1_semi_norm import spatial_gradient, spatial_gradient3d


class ShockRMSE(LossComponent):
    """
    Shock RMSE loss: computes RMSE only in regions with high gradient magnitude.
    
    Useful for focusing on shock waves, discontinuities, and sharp features in CFD.
    Uses the H1 semi-norm (gradient magnitude) of the labels to identify shock regions.

    Args:
        norm_helper: Normalization helper for physical/normalized conversions.
        weight: Overall weight for this loss component.
        name: Optional name for the loss component.
        data_dim: Dimensionality of spatial data (1, 2, or 3).
        field_names: List of field names matching channel dimension.
        gradient_threshold: Threshold range [min, max] for gradient magnitude in physical units.
                          Only regions where min <= |∇label| <= max contribute to loss.
        threshold_softness: Smoothness of sigmoid transition (fraction of range width).
                           Smaller = sharper transition, larger = softer.
        blur_sigma: Standard deviation for Gaussian blur (in grid cells).
                   Set to 0 to disable blurring. Applied after soft masking.
        gradient_mode: Mode for gradient computation ('sobel' or 'diff').
        normalization: Type of batch-wise normalization ('none', 'magnitude', 'variance').
        epsilon: Small constant for numerical stability.
        per_channel_thresholds: Optional dict mapping channel names to individual thresholds.
                               If None, uses gradient_threshold for all channels.
        value_range: Optional dict mapping channel names to [min, max] value ranges.
                    Only pixels where min <= label <= max are considered for gradient masking.
                    Useful to exclude interface boundaries (e.g., air-water) from shock detection.
                    Values are in physical units.
        visualize: If True, creates visualization plots during forward pass (for debugging).
        viz_channel_idx: Channel index to visualize (default: 0).
        viz_save_dir: Directory to save visualization plots (default: None, just shows).
    """
    
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        gradient_threshold: Tuple[float, float] = (0.1, float('inf')),
        threshold_softness: float = 0.1,
        blur_sigma: float = 0.0,
        gradient_mode: Literal['sobel', 'diff'] = 'diff',
        normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
        epsilon: float = 1e-8,
        per_channel_thresholds: Optional[Dict[str, Tuple[float, float]]] = None,
        value_range: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        super().__init__(
            weight=weight,
            name=name or "shock_rmse",
            data_dim=data_dim,
            field_names=field_names,
            norm_helper=norm_helper,
        )
        
        self.gradient_threshold = gradient_threshold
        self.threshold_softness = threshold_softness
        self.blur_sigma = blur_sigma
        self.gradient_mode = gradient_mode
        self.normalization = normalization
        self.epsilon = epsilon
        self.per_channel_thresholds = per_channel_thresholds or {}
        self.value_range = value_range or {}
        
        # Validate gradient_threshold
        if len(self.gradient_threshold) != 2:
            raise ValueError(
                f"gradient_threshold must have exactly 2 values (min, max), got {len(self.gradient_threshold)}"
            )
        if self.gradient_threshold[0] < 0:
            raise ValueError(
                f"gradient_threshold[0] must be >= 0, got {self.gradient_threshold[0]}"
            )
        if self.gradient_threshold[0] >= self.gradient_threshold[1]:
            raise ValueError(
                f"gradient_threshold[0] must be < gradient_threshold[1], got {self.gradient_threshold}"
            )
        
        # Normalize thresholds to model space
        self._normalized_thresholds = self._prepare_thresholds()
        self._normalized_value_ranges = self._prepare_value_ranges()

    def _prepare_thresholds(self) -> Dict[int, Tuple[float, float]]:
        """
        Convert physical gradient thresholds to normalized space and map to channel indices.
        
        Returns:
            Dict mapping channel index to (min_threshold, max_threshold) in normalized units.
        """
        if self.field_names is None:
            # Use global threshold for all channels (index agnostic)
            return {-1: self.gradient_threshold}
        
        normalized = {}
        
        # Process per-channel thresholds
        for ch_name, threshold in self.per_channel_thresholds.items():
            if ch_name not in self.field_names:
                continue
            
            ch_idx = self.field_names.index(ch_name)
            # Use thresholds directly without normalization
            normalized[ch_idx] = (threshold[0], threshold[1])
        
        # Set default threshold for channels without specific thresholds
        for idx, ch_name in enumerate(self.field_names):
            if idx not in normalized:
                # Use thresholds directly without normalization
                normalized[idx] = (self.gradient_threshold[0], self.gradient_threshold[1])
        
        return normalized

    def _prepare_value_ranges(self) -> Dict[int, Tuple[float, float]]:
        """
        Convert physical value ranges to normalized space and map to channel indices.
        
        Returns:
            Dict mapping channel index to (min_value, max_value) in normalized units.
        """
        if not self.value_range or self.field_names is None:
            return {}
        
        normalized = {}
        
        for ch_name, value_range in self.value_range.items():
            if ch_name not in self.field_names:
                continue
            
            ch_idx = self.field_names.index(ch_name)
            
            # Convert physical value range to normalized space
            if self.norm_helper is not None:
                min_val_norm = self.norm_helper.normalize_scalar(value_range[0], ch_name)
                max_val_norm = self.norm_helper.normalize_scalar(value_range[1], ch_name)
            else:
                min_val_norm = value_range[0]
                max_val_norm = value_range[1]
            
            normalized[ch_idx] = (min_val_norm, max_val_norm)
        
        return normalized

    def _compute_gradient_magnitude(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Compute gradient magnitude for each channel.
        
        Args:
            tensor: Input tensor, shape (B, T, C, *spatial).
            
        Returns:
            Gradient magnitude, shape (B, T, C, *spatial).
        """
        # Infer spatial dimensionality
        ndim = self.data_dim if self.data_dim is not None else (tensor.ndim - 3)
        
        b, t, c = tensor.shape[:3]
        spatial_shape = tensor.shape[3:]
        
        if ndim == 1:
            # Reshape to (B*T*C, 1, 1, H) for 2D conv
            h = spatial_shape[0]
            x_2d = tensor.reshape(b * t * c, 1, 1, h)
            grads = spatial_gradient(x_2d, mode=self.gradient_mode, order=1, normalized=True)
            # grads: (B*T*C, 1, 2, 1, H) -> take only H gradient
            grads = grads[:, :, 0, :, :].reshape(b, t, c, h)
            grad_mag = torch.abs(grads)
            
        elif ndim == 2:
            # Reshape to (B*T, C, H, W)
            h, w = spatial_shape
            x_2d = tensor.reshape(b * t, c, h, w)
            grads = spatial_gradient(x_2d, mode=self.gradient_mode, order=1, normalized=True)
            # grads: (B*T, C, 2, H, W) -> compute magnitude
            grads = grads.reshape(b, t, c, 2, h, w)
            grad_mag = torch.sqrt((grads ** 2).sum(dim=3) + self.epsilon)
            
        elif ndim == 3:
            # Reshape to (B*T, C, D, H, W)
            d, h, w = spatial_shape
            x_3d = tensor.reshape(b * t, c, d, h, w)
            grads = spatial_gradient3d(x_3d, mode=self.gradient_mode, order=1)
            # grads: (B*T, C, 3, D, H, W) -> compute magnitude
            grads = grads.reshape(b, t, c, 3, d, h, w)
            grad_mag = torch.sqrt((grads ** 2).sum(dim=3) + self.epsilon)
            
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {ndim}. Expected 1, 2, or 3.")
        
        return grad_mag

    def _create_value_mask(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Create soft mask for value range filtering.
        
        Args:
            labels: Label tensor, shape (B, T, C, *spatial).
            
        Returns:
            Soft mask in [0, 1], same shape as labels. Fully differentiable.
        """
        b, t, c = labels.shape[:3]
        mask = torch.ones_like(labels)
        
        # Apply per-channel value ranges
        for ch_idx, (val_min_norm, val_max_norm) in self._normalized_value_ranges.items():
            range_width = val_max_norm - val_min_norm
            softness = self.threshold_softness * range_width
            
            # Get label values for this channel
            label_ch = labels[:, :, ch_idx]
            
            # Soft thresholding using sigmoids (fully differentiable)
            # Lower boundary: sigmoid((label - val_min) / softness)
            # Upper boundary: sigmoid((val_max - label) / softness)
            lower_mask = torch.sigmoid((label_ch - val_min_norm) / (softness + self.epsilon))
            upper_mask = torch.sigmoid((val_max_norm - label_ch) / (softness + self.epsilon))
            
            # Combine: mask ≈ 1 when val_min < label < val_max, smooth transitions outside
            mask[:, :, ch_idx] = lower_mask * upper_mask
        
        return mask

    def _create_shock_mask(self, gradient_magnitude: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Create soft mask for shock regions using sigmoid functions.
        
        Args:
            gradient_magnitude: Gradient magnitude tensor, shape (B, T, C, *spatial).
            labels: Label tensor, shape (B, T, C, *spatial).
            
        Returns:
            Soft mask in [0, 1], same shape as gradient_magnitude. Fully differentiable.
        """
        b, t, c = gradient_magnitude.shape[:3]
        mask = torch.ones_like(gradient_magnitude)
        
        # First, apply value range filtering if configured
        if self._normalized_value_ranges:
            value_mask = self._create_value_mask(labels)
            mask = mask * value_mask
        
        # Apply per-channel gradient thresholds
        for ch_idx in range(c):
            if ch_idx in self._normalized_thresholds:
                grad_min_norm, grad_max_norm = self._normalized_thresholds[ch_idx]
            elif -1 in self._normalized_thresholds:
                # Use global threshold
                grad_min_norm, grad_max_norm = self._normalized_thresholds[-1]
            else:
                continue
            
            range_width = grad_max_norm - grad_min_norm
            softness = self.threshold_softness * range_width
            
            # Get gradient magnitude for this channel
            grad_ch = gradient_magnitude[:, :, ch_idx]
            
            # Soft thresholding using sigmoids (fully differentiable)
            # Lower boundary: sigmoid((grad - grad_min) / softness)
            # Upper boundary: sigmoid((grad_max - grad) / softness)
            lower_mask = torch.sigmoid((grad_ch - grad_min_norm) / (softness + self.epsilon))
            upper_mask = torch.sigmoid((grad_max_norm - grad_ch) / (softness + self.epsilon))
            
            # Combine: mask ≈ 1 when grad_min < grad < grad_max, smooth transitions outside
            # Multiply with existing mask (which includes value filtering)
            mask[:, :, ch_idx] = mask[:, :, ch_idx] * lower_mask * upper_mask
        
        # Apply Gaussian blur if requested (adds additional spatial smoothing)
        if self.blur_sigma > 0:
            mask = self._apply_gaussian_blur(mask, self.blur_sigma)
        
        return mask

    def _apply_gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply Gaussian blur to mask using separable convolution.
        
        Args:
            mask: Soft mask, shape (B, T, C, *spatial).
            sigma: Standard deviation in grid cells.
            
        Returns:
            Blurred mask in [0, 1], same shape as input. Fully differentiable.
        """
        spatial_dims = mask.ndim - 3
        
        if spatial_dims not in [1, 2, 3]:
            raise ValueError(f"Expected 1D, 2D or 3D spatial data, got {spatial_dims}D")
        
        kernel_radius = int(3 * sigma)
        if kernel_radius == 0:
            return mask
        
        kernel_size = 2 * kernel_radius + 1
        x = torch.arange(-kernel_radius, kernel_radius + 1, dtype=mask.dtype, device=mask.device)
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        original_shape = mask.shape
        B, T, C = mask.shape[:3]
        
        # Reshape to (B*T, C, *spatial) for convolution
        mask_flat = mask.reshape(B * T, C, *mask.shape[3:])
        
        if spatial_dims == 1:
            # 1D blur: treat as 2D with height=1
            mask_flat = mask_flat.unsqueeze(-2)  # (B*T, C, 1, W)
            kernel_x = kernel_1d.view(1, 1, 1, kernel_size)
            mask_flat = F.conv2d(mask_flat, kernel_x.expand(C, 1, 1, kernel_size), 
                                groups=C, padding=(0, kernel_radius))
            mask_flat = mask_flat.squeeze(-2)
            
        elif spatial_dims == 2:
            kernel_x = kernel_1d.view(1, 1, kernel_size, 1)
            mask_flat = F.conv2d(mask_flat, kernel_x.expand(C, 1, kernel_size, 1),
                                groups=C, padding=(kernel_radius, 0))
            
            kernel_y = kernel_1d.view(1, 1, 1, kernel_size)
            mask_flat = F.conv2d(mask_flat, kernel_y.expand(C, 1, 1, kernel_size),
                                groups=C, padding=(0, kernel_radius))
            
        elif spatial_dims == 3:
            kernel_x = kernel_1d.view(1, 1, kernel_size, 1, 1)
            mask_flat = F.conv3d(mask_flat, kernel_x.expand(C, 1, kernel_size, 1, 1),
                                groups=C, padding=(kernel_radius, 0, 0))
            
            kernel_y = kernel_1d.view(1, 1, 1, kernel_size, 1)
            mask_flat = F.conv3d(mask_flat, kernel_y.expand(C, 1, 1, kernel_size, 1),
                                groups=C, padding=(0, kernel_radius, 0))
            
            kernel_z = kernel_1d.view(1, 1, 1, 1, kernel_size)
            mask_flat = F.conv3d(mask_flat, kernel_z.expand(C, 1, 1, 1, kernel_size),
                                groups=C, padding=(0, 0, kernel_radius))
        
        mask_blurred = mask_flat.reshape(original_shape)
        
        return mask_blurred


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
        """
        Compute shock RMSE loss.
        
        Args:
            model: The model being trained.
            predictions: Model outputs, shape (B, T, C, *spatial).
            labels: Ground truth, shape (B, T, C, *spatial).
            return_detailed: If True, return detailed breakdown.
            
        Returns:
            If return_detailed=False: scalar loss tensor.
            If return_detailed=True: (loss, detailed_dict) where detailed_dict contains:
                - 'mask_fraction': fraction of cells in shock region
                - 'per_timestep': per-timestep breakdown (if applicable)
                - 'per_channel': per-channel breakdown (if applicable)
        """
        # Compute gradient magnitude of labels (normalized space)
        label_grad_mag = self._compute_gradient_magnitude(labels)
        
        # Create soft mask for shock regions (includes value range filtering)
        mask = self._create_shock_mask(label_grad_mag, labels)
        
        # Compute squared error in normalized space
        squared_error = (predictions - labels) ** 2
        
        # Apply mask
        masked_squared_error = squared_error * mask
        
        # Compute mean over masked regions
        if keep_bc_dims:
            reduce_dims = [1] + list(range(3, masked_squared_error.ndim))
            mask_sum = mask.sum(dim=reduce_dims)
            mse_sum = masked_squared_error.sum(dim=reduce_dims)
            unweighted_rmse = torch.sqrt(mse_sum / (mask_sum + self.epsilon))
            unweighted_rmse = torch.where(
                mask_sum > self.epsilon,
                unweighted_rmse,
                torch.zeros_like(unweighted_rmse),
            )
        else:
            mask_sum = mask.sum()
            if mask_sum > self.epsilon:
                unweighted_rmse = torch.sqrt(masked_squared_error.sum() / mask_sum)
            else:
                # Fallback: no shock cells found, return zero loss
                unweighted_rmse = torch.zeros((), device=predictions.device, dtype=predictions.dtype)

        # Apply weight schedule
        weighted_rmse = self.weight_schedule.base_weight * unweighted_rmse
        
        if not return_detailed:
            return weighted_rmse
        
        detailed = {}
        
        # Conditionally detach based on preserve_component_grads
        squared_error_for_detailed = squared_error if preserve_component_grads else squared_error.detach()
        mask_for_detailed = mask if preserve_component_grads else mask.detach()
        
        # Add mask statistics
        total_elements = mask_for_detailed.numel()
        mask_sum_for_detailed = mask_for_detailed.sum()
        detailed['mask_fraction'] = mask_sum_for_detailed / total_elements
        
        # Per-timestep breakdown (if mask has sufficient elements)
        if (mask.sum() > self.epsilon) and predictions.ndim >= 2:
            timesteps = predictions.shape[1]
            per_timestep = []
            for t in range(timesteps):
                t_squared_error = squared_error_for_detailed[:, t]
                t_mask = mask_for_detailed[:, t]
                t_mask_sum = t_mask.sum()
                if t_mask_sum > self.epsilon:
                    t_rmse = torch.sqrt((t_squared_error * t_mask).sum() / t_mask_sum)
                else:
                    t_rmse = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
                per_timestep.append(t_rmse)
            detailed['per_timestep'] = torch.stack(per_timestep)
        
        # Per-channel breakdown (if mask has sufficient elements)
        if (mask.sum() > self.epsilon) and predictions.ndim >= 3:
            channels = predictions.shape[2]
            per_channel = []
            for c in range(channels):
                c_squared_error = squared_error_for_detailed[:, :, c]
                c_mask = mask_for_detailed[:, :, c]
                c_mask_sum = c_mask.sum()
                if c_mask_sum > self.epsilon:
                    c_rmse = torch.sqrt((c_squared_error * c_mask).sum() / c_mask_sum)
                else:
                    c_rmse = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
                per_channel.append(c_rmse)
            detailed['per_channel'] = torch.stack(per_channel)
        
        return weighted_rmse, detailed