from typing import Dict, List, Optional, Tuple, Union
import torch
from torch import nn
from ..loss_framework import LossComponent, WeightSchedule, apply_batch_wise_normalization, NormalizationHelper


class InterfaceRMSE(LossComponent):
    """
    Interface RMSE loss: computes RMSE for density field only within a specified range.
    
    Useful for focusing on interface regions between fluids with sharp density jumps.

    Based on IRMSE concept from "Bubbleformer: Forecasting Boiling with Transformers"
    https://arxiv.org/abs/2507.21244

    We use a soft masking approach based on density thresholds to maintain differentiability;
    Original paper uses signed distance functions.
    """
    
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        density_key: str = "Density",
        density_range: Tuple[float, float] = (0.4, 0.6),
        range_softness: float = 0.05,
        blur_sigma: float = 0.0,
        normalization: str = 'none',
        eps: float = 1e-8,
    ):
        """
        Args:
            weight: Loss weight (scalar or WeightSchedule).
            name: Optional name for this loss component.
            data_dim: Spatial dimensionality (2D or 3D).
            field_names: List of field names matching channel dimension.
            norm_stats: Normalization statistics (mean/std) for denormalization.
            density_key: Name of density field in field_names.
            density_range: (min, max) physical density values defining interface region.
            range_softness: Smoothness of sigmoid transition (fraction of range width).
                           Smaller = sharper transition, larger = softer.
            blur_sigma: Standard deviation for Gaussian blur (in grid cells). 
                        Set to 0 to disable blurring. Applied after soft masking.
            normalization: Type of normalization ('none', 'magnitude', 'variance').
            eps: Small constant for numerical stability.
        """
        super().__init__(
            weight=weight,
            name=name or "interface_rmse",
            data_dim=data_dim,
            field_names=field_names,
            norm_helper=norm_helper,
        )
        
        self.density_key = density_key
        self.density_range = density_range
        self.range_softness = range_softness
        self.blur_sigma = blur_sigma
        self.normalization = normalization
        self.eps = eps
        
        # Validate density_range
        if len(self.density_range) != 2:
            raise ValueError(
                f"density_range must have exactly 2 values (min, max), got {len(self.density_range)}"
            )
        if self.density_range[0] >= self.density_range[1]:
            raise ValueError(
                f"density_range[0] must be < density_range[1], got {self.density_range}"
            )
        
        # Validate density_key exists
        if field_names is not None and density_key not in field_names:
            raise ValueError(
                f"density_key '{density_key}' not found in field_names: {field_names}"
            )
        
        # Normalize density range to match input data normalization
        self.density_range_norm = None

        self._compute_normalized_density_range()

    def _compute_normalized_density_range(self):
        """Convert physical density range to normalized space."""
        
        rho_min, rho_max = self.density_range
        rho_min_norm = self.norm_helper.normalize_scalar(rho_min, self.density_key)
        rho_max_norm = self.norm_helper.normalize_scalar(rho_max, self.density_key)
        
        self.density_range_norm = (rho_min_norm, rho_max_norm)

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        input_frames: Optional[torch.Tensor],
        return_detailed: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute interface RMSE loss.
        
        Args:
            model: The model being trained.
            predictions: Model outputs, shape (B, T, C, *spatial).
            labels: Ground truth, shape (B, T, C, *spatial).
            return_detailed: If True, return detailed breakdown.
            
        Returns:
            If return_detailed=False: scalar loss tensor.
            If return_detailed=True: (loss, detailed_dict) where detailed_dict contains:
                - 'mask_fraction': fraction of cells in interface region
                - 'unweighted_rmse': RMSE before weighting (normalized)
                - 'physical_rmse': RMSE in physical units
        """
        # Extract density field (normalized)
        pred_density_norm = self._extract_density(predictions)
        true_density_norm = self._extract_density(labels)
        
        # Create soft mask for interface region (using normalized range)
        mask = self._create_interface_mask(true_density_norm)
        
        # Compute masked squared error in normalized space
        squared_error_norm = (pred_density_norm - true_density_norm) ** 2
        masked_squared_error = squared_error_norm * mask
        
        # Compute mean over masked regions
        mask_sum = mask.sum()
        if mask_sum > self.eps:
            unweighted_rmse_norm = torch.sqrt(masked_squared_error.sum() / mask_sum)
        else:
            # Fallback: no interface cells found, return zero loss
            unweighted_rmse_norm = torch.zeros((), device=predictions.device, dtype=predictions.dtype)
        
        # Apply per-batch normalization if requested
        normalized_rmse = apply_batch_wise_normalization(
            unweighted_rmse_norm,
            labels,
            self.normalization,
            self.eps
        )
        
        # Apply weight schedule
        weighted_rmse = self.weight_schedule.base_weight * normalized_rmse
        
        if not return_detailed:
            return weighted_rmse
        
        detailed = {}
        
        return weighted_rmse, detailed
    
    def _extract_density(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Extract density field from multi-channel tensor (stays normalized).
        
        Args:
            tensor: Normalized data, shape (B, T, C, *spatial).
            
        Returns:
            Normalized density field, shape (B, T, *spatial).
        """
        if self.field_names is None:
            raise ValueError(
                f"{self.name} requires field_names to be set."
            )
        
        if tensor.ndim < 4:
            raise ValueError(
                f"Expected tensor with shape (B, T, C, ...), got {tensor.shape}"
            )
        
        try:
            density_idx = self.field_names.index(self.density_key)
        except ValueError:
            raise ValueError(
                f"Density key '{self.density_key}' not found in field_names: {self.field_names}"
            )
        
        density_norm = tensor.select(dim=2, index=density_idx)  # (B, T, *spatial)
        
        return density_norm


    def _create_interface_mask(self, density_norm: torch.Tensor) -> torch.Tensor:
        """
        Create soft mask for interface region using sigmoid functions.
        
        Args:
            density_norm: Normalized density field, shape (B, T, *spatial).
            
        Returns:
            Soft mask in [0, 1], same shape as density. Fully differentiable.
        """
        # Use normalized density range
        if self.density_range_norm is None:
            raise ValueError(
                f"{self.name} requires norm_stats to compute normalized density range."
            )
        
        rho_min_norm, rho_max_norm = self.density_range_norm
        range_width = rho_max_norm - rho_min_norm
        
        # Compute softness parameter in normalized space
        # range_softness is a fraction of the physical range, apply to normalized range
        softness = self.range_softness * range_width
        
        # Soft thresholding using sigmoids (fully differentiable)
        # Lower boundary: sigmoid((rho - rho_min) / softness)
        # Upper boundary: sigmoid((rho_max - rho) / softness)
        lower_mask = torch.sigmoid((density_norm - rho_min_norm) / (softness + self.eps))
        upper_mask = torch.sigmoid((rho_max_norm - density_norm) / (softness + self.eps))
        
        # Combine: mask ≈ 1 when rho_min < rho < rho_max, smooth transitions outside
        mask = lower_mask * upper_mask
        
        # Apply Gaussian blur if requested (adds additional spatial smoothing)
        if self.blur_sigma > 0:
            mask = self._apply_gaussian_blur(mask, self.blur_sigma)
        
        return mask
    
    def _apply_gaussian_blur(self, mask: torch.Tensor, sigma: float) -> torch.Tensor:
        """
        Apply Gaussian blur to mask using separable convolution.
        
        Args:
            mask: Soft mask, shape (B, T, *spatial).
            sigma: Standard deviation in grid cells.
            
        Returns:
            Blurred mask in [0, 1], same shape as input. Fully differentiable.
        """
        import torch.nn.functional as F
        
        spatial_dims = mask.ndim - 2
        
        if spatial_dims not in [2, 3]:
            raise ValueError(f"Expected 2D or 3D spatial data, got {spatial_dims}D")
        
        kernel_radius = int(3 * sigma)
        if kernel_radius == 0:
            return mask
        
        kernel_size = 2 * kernel_radius + 1
        x = torch.arange(-kernel_radius, kernel_radius + 1, dtype=mask.dtype, device=mask.device)
        kernel_1d = torch.exp(-0.5 * (x / sigma) ** 2)
        kernel_1d = kernel_1d / kernel_1d.sum()
        
        original_shape = mask.shape
        B, T = mask.shape[:2]
        mask_flat = mask.reshape(B * T, 1, *mask.shape[2:])
        
        if spatial_dims == 2:
            kernel_x = kernel_1d.view(1, 1, kernel_size, 1)
            mask_flat = F.conv2d(mask_flat, kernel_x, padding=(kernel_radius, 0))
            
            kernel_y = kernel_1d.view(1, 1, 1, kernel_size)
            mask_flat = F.conv2d(mask_flat, kernel_y, padding=(0, kernel_radius))
            
        elif spatial_dims == 3:
            kernel_x = kernel_1d.view(1, 1, kernel_size, 1, 1)
            mask_flat = F.conv3d(mask_flat, kernel_x, padding=(kernel_radius, 0, 0))
            
            kernel_y = kernel_1d.view(1, 1, 1, kernel_size, 1)
            mask_flat = F.conv3d(mask_flat, kernel_y, padding=(0, kernel_radius, 0))
            
            kernel_z = kernel_1d.view(1, 1, 1, 1, kernel_size)
            mask_flat = F.conv3d(mask_flat, kernel_z, padding=(0, 0, kernel_radius))
        
        mask_blurred = mask_flat.reshape(original_shape)
        
        return mask_blurred