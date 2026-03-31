from typing import Dict, List, Optional, Tuple, Union, Literal
import torch
from torch import nn
import torch.nn.functional as F
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
                  Applied to the selected `threshold_field_key` only.
        threshold_softness: Smoothness of sigmoid transition (fraction of range width).
                           Smaller = sharper transition, larger = softer.
        blur_sigma: Standard deviation for Gaussian blur (in grid cells).
                   Set to 0 to disable blurring. Applied after soft masking.
        gradient_mode: Mode for gradient computation ('sobel' or 'diff').
        normalization: Type of batch-wise normalization ('none', 'magnitude', 'variance').
        epsilon: Small constant for numerical stability.
        per_channel_thresholds: Optional dict mapping channel names to thresholds.
                       If provided, threshold for `threshold_field_key` overrides
                       `gradient_threshold`.
        value_range: Optional dict mapping channel names to [min, max] value ranges.
                    Only pixels where min <= label <= max are considered for gradient masking.
                    Useful to exclude interface boundaries (e.g., air-water) from shock detection.
                    Values are in physical units.
        threshold_field_key: Field name used to build the shock mask. The resulting
                    mask is broadcast and applied to all fields.
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
        threshold_field_key: Optional[str] = None,
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
        self.threshold_field_key = threshold_field_key
        
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
        
        # Resolve channel used to build mask and normalize thresholds/ranges.
        self._threshold_field_idx = self._resolve_threshold_field_idx()
        self._normalized_threshold = self._prepare_threshold()
        self._normalized_value_range = self._prepare_value_range()

    def _resolve_threshold_field_idx(self) -> int:
        """Resolve the channel index used to compute the shock mask."""
        if self.field_names is None:
            return 0

        if self.threshold_field_key is None:
            return 0

        if self.threshold_field_key in self.field_names:
            return self.field_names.index(self.threshold_field_key)

        raise ValueError(
            f"threshold_field_key '{self.threshold_field_key}' not found in field_names: {self.field_names}"
        )

    def _prepare_threshold(self) -> Tuple[float, float]:
        """
        Prepare gradient thresholds for the single mask field.
        
        Returns:
            (min_threshold, max_threshold)
        """
        # If per-channel thresholds are provided and contain
        # the mask field, use that threshold for mask generation.
        if self.field_names is not None and self.per_channel_thresholds:
            field_name = self.field_names[self._threshold_field_idx]
            if field_name in self.per_channel_thresholds:
                threshold = self.per_channel_thresholds[field_name]
                return (threshold[0], threshold[1])

            # Case-insensitive lookup.
            field_name_lower = field_name.lower()
            for ch_name, threshold in self.per_channel_thresholds.items():
                if ch_name.lower() == field_name_lower:
                    return (threshold[0], threshold[1])

        return (self.gradient_threshold[0], self.gradient_threshold[1])

    def _prepare_value_range(self) -> Optional[Tuple[float, float]]:
        """
        Convert value range (for the mask field) to normalized space.
        
        Returns:
            (min_value, max_value) in normalized units, or None.
        """
        if not self.value_range or self.field_names is None:
            return None

        field_name = self.field_names[self._threshold_field_idx]

        selected_range = None
        if field_name in self.value_range:
            selected_range = self.value_range[field_name]
        else:
            field_name_lower = field_name.lower()
            for ch_name, value_range in self.value_range.items():
                if ch_name.lower() == field_name_lower:
                    selected_range = value_range
                    break

        if selected_range is None:
            return None

        if self.norm_helper is not None:
            min_val_norm = self.norm_helper.normalize_scalar(selected_range[0], field_name)
            max_val_norm = self.norm_helper.normalize_scalar(selected_range[1], field_name)
        else:
            min_val_norm = selected_range[0]
            max_val_norm = selected_range[1]

        return (min_val_norm, max_val_norm)

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

    def _create_value_mask(self, label_field: torch.Tensor) -> torch.Tensor:
        """
        Create soft mask for value range filtering on the selected mask field.
        
        Args:
            label_field: Label tensor for the selected field, shape (B, T, *spatial).
            
        Returns:
            Soft mask in [0, 1], same shape as label_field. Fully differentiable.
        """
        if self._normalized_value_range is None:
            return torch.ones_like(label_field)

        val_min_norm, val_max_norm = self._normalized_value_range
        range_width = val_max_norm - val_min_norm
        softness = self.threshold_softness * range_width

        lower_mask = torch.sigmoid((label_field - val_min_norm) / (softness + self.epsilon))
        upper_mask = torch.sigmoid((val_max_norm - label_field) / (softness + self.epsilon))

        return lower_mask * upper_mask

    def _create_shock_mask(self, gradient_magnitude: torch.Tensor, label_field: torch.Tensor) -> torch.Tensor:
        """
        Create soft mask for shock regions using a single selected field.
        
        Args:
            gradient_magnitude: Gradient magnitude of selected field, shape (B, T, *spatial).
            label_field: Label values of selected field, shape (B, T, *spatial).
            
        Returns:
            Soft mask in [0, 1], same shape as selected field. Fully differentiable.
        """
        mask = torch.ones_like(gradient_magnitude)

        # Apply value-range filtering on selected field if configured.
        value_mask = self._create_value_mask(label_field)
        mask = mask * value_mask

        grad_min_norm, grad_max_norm = self._normalized_threshold

        range_width = grad_max_norm - grad_min_norm
        softness = self.threshold_softness * range_width

        lower_mask = torch.sigmoid((gradient_magnitude - grad_min_norm) / (softness + self.epsilon))
        upper_mask = torch.sigmoid((grad_max_norm - gradient_magnitude) / (softness + self.epsilon))

        mask = mask * lower_mask * upper_mask
        
        # Apply Gaussian blur if requested (adds additional spatial smoothing)
        if self.blur_sigma > 0:
            mask = self._apply_gaussian_blur(mask.unsqueeze(2), self.blur_sigma).squeeze(2)
        
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

    def _extract_threshold_field(self, tensor: torch.Tensor) -> torch.Tensor:
        """Extract the field used for mask generation, shape (B, T, *spatial)."""
        if tensor.ndim < 4:
            raise ValueError(
                f"Expected tensor with shape (B, T, C, ...), got {tensor.shape}"
            )

        if self._threshold_field_idx >= tensor.shape[2]:
            raise ValueError(
                f"threshold field index {self._threshold_field_idx} is out of bounds for C={tensor.shape[2]}"
            )

        return tensor.select(dim=2, index=self._threshold_field_idx)


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
        # Compute gradient magnitude on one selected field only.
        label_field = self._extract_threshold_field(labels)  # (B, T, *spatial)
        label_field_for_grad = label_field.unsqueeze(2)  # (B, T, 1, *spatial)
        label_grad_mag = self._compute_gradient_magnitude(label_field_for_grad).squeeze(2)

        # Create mask from selected field and broadcast to all channels.
        mask = self._create_shock_mask(label_grad_mag, label_field)  # (B, T, *spatial)
        mask_bc = mask.unsqueeze(2)  # (B, T, 1, *spatial)
        
        # Compute squared error in normalized space
        squared_error = (predictions - labels) ** 2
        
        # Apply mask
        masked_squared_error = squared_error * mask_bc
        
        # Compute mean over masked regions
        if keep_bc_dims:
            reduce_dims = [1] + list(range(3, masked_squared_error.ndim))
            mask_sum = mask_bc.sum(dim=reduce_dims)
            mse_sum = masked_squared_error.sum(dim=reduce_dims)
            unweighted_rmse = torch.sqrt(mse_sum / (mask_sum + self.epsilon))
            unweighted_rmse = torch.where(
                mask_sum > self.epsilon,
                unweighted_rmse,
                torch.zeros_like(unweighted_rmse),
            )
        else:
            mask_sum = mask_bc.sum() * predictions.shape[2]
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
        mask_for_detailed = mask_bc if preserve_component_grads else mask_bc.detach()
        mask_for_detailed = mask_for_detailed.expand_as(squared_error_for_detailed)
        weight_scalar = self.weight_schedule.base_weight
        
        # Add mask statistics
        total_elements = mask_for_detailed.numel()
        mask_sum_for_detailed = mask_for_detailed.sum()
        detailed['mask_fraction'] = mask_sum_for_detailed / total_elements
        
        # Per-timestep breakdown (if mask has sufficient elements)
        if (mask_for_detailed.sum() > self.epsilon) and predictions.ndim >= 2:
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
                per_timestep.append(weight_scalar * t_rmse)
            detailed['per_timestep'] = torch.stack(per_timestep)
        
        # Per-channel breakdown (if mask has sufficient elements)
        if (mask_for_detailed.sum() > self.epsilon) and predictions.ndim >= 3:
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
                per_channel.append(weight_scalar * c_rmse)
            detailed['per_channel'] = torch.stack(per_channel)
        
        return weighted_rmse, detailed