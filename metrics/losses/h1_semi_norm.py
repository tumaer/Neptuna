from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal, Optional, List, Any, Union, Dict, Tuple
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper
import matplotlib.pyplot as plt

# Filter kernels adapted from Kornia
# https://github.com/kornia/kornia

class H1SemiNorm(LossComponent):
    """
    H1 semi-norm loss for 1D, 2D, and 3D data on regular grids.
    
    Args:
        weight: Overall weight for this loss component
        name: Optional name for the loss component
        data_dim: Dimensionality of spatial data (1, 2, or 3)
        field_names: Optional list of field names
        mode: Gradient computation mode, 'sobel' or 'diff'
        reduction: 'mean', 'sum', or 'none'
    """
    
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: float = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        mode: Literal['sobel', 'diff'] = 'diff',
        reduction: str = 'mean',
        normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names, norm_helper=norm_helper)
        self.mode = mode
        self.reduction = reduction
        self.normalization = normalization
        self.epsilon = epsilon
        
    def _compute_gradients(self, x: torch.Tensor) -> torch.Tensor:
        """Compute spatial gradients based on input dimensionality."""
        # Infer ndim from tensor shape if data_dim not set
        ndim = self.data_dim if self.data_dim is not None else (x.ndim - 3)
        
        if ndim == 1:
            # 1D: (B, F, C, H) -> reshape to (B*F*C, 1, 1, H) for 2D conv
            b, f, c, h = x.shape
            x_2d = x.reshape(b * f * c, 1, 1, h)
            grads = spatial_gradient(x_2d, mode=self.mode, order=1, normalized=True)
            # grads: (B*F*C, 1, 2, 1, H) -> take only H gradient
            grads = grads[:, :, 0, :, :].reshape(b, f, c, h)
            
        elif ndim == 2:
            # 2D: (B, F, C, H, W) -> reshape to (B*F, C, H, W)
            b, f, c, h, w = x.shape
            x_2d = x.reshape(b * f, c, h, w)
            grads = spatial_gradient(x_2d, mode=self.mode, order=1, normalized=True)
            # grads: (B*F, C, 2, H, W) -> reshape to (B, F, C, 2, H, W)
            grads = grads.reshape(b, f, c, 2, h, w)
            
        elif ndim == 3:
            # 3D: (B, F, C, D, H, W) -> reshape to (B*F, C, D, H, W)
            b, f, c, d, h, w = x.shape
            x_3d = x.reshape(b * f, c, d, h, w)
            grads = spatial_gradient3d(x_3d, mode=self.mode, order=1)
            # grads: (B*F, C, 3, D, H, W) -> reshape to (B, F, C, 3, D, H, W)
            grads = grads.reshape(b, f, c, 3, d, h, w)
            
        else:
            raise ValueError(f"Unsupported number of spatial dimensions: {ndim}. Expected 1, 2, or 3.")
            
        return grads
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False,
        keep_bc_dims: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute Sobolev loss.
        
        Args:
            model: Neural network model (unused, but required by interface)
            predictions: Predicted values (B, F, C, *spatial_dims)
            labels: Target values (B, F, C, *spatial_dims)
        
        Returns:
            Weighted scalar loss value
        """

        # Compute gradients
        pred_grads = self._compute_gradients(predictions)
        target_grads = self._compute_gradients(labels)
        
        # Compute squared gradient differences
        grad_diff = pred_grads - target_grads
        
        # Sum over gradient components (last dim for 1D, dim=-3 for 2D/3D)
        # Result shape: (B, F, C, *spatial_dims)
        ndim = self.data_dim if self.data_dim is not None else (predictions.ndim - 3)
        
        if ndim == 1:
            # grad_diff shape: (B, F, C, H) - already no gradient dimension
            unweighted = grad_diff ** 2
        elif ndim == 2:
            # grad_diff shape: (B, F, C, 2, H, W) - sum over 2 gradient directions
            unweighted = (grad_diff ** 2).sum(dim=3)
        elif ndim == 3:
            # grad_diff shape: (B, F, C, 3, D, H, W) - sum over 3 gradient directions
            unweighted = (grad_diff ** 2).sum(dim=3)

        norm_error = self.norm_helper.normalize_error(
                unweighted,
                labels,
                self.data_dim,
                self.normalization,
                self.epsilon
            )

        # Average over spatial dimensions to get (B, F, C)
        spatial_dims = tuple(range(3, norm_error.ndim))
        unweighted = norm_error.mean(dim=spatial_dims)
        
        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_loss_weight(unweighted.shape).to(predictions.device)
        
        # Apply weights element-wise
        weighted = unweighted * weight_tensor
        
        # Reduce to scalar or per-batch vector
        if keep_bc_dims:
            if self.reduction == 'sum':
                total_loss = weighted.sum(dim=1)
            else:
                total_loss = weighted.mean(dim=1)
        else:
            if self.reduction == 'mean':
                total_loss = weighted.mean()
            elif self.reduction == 'sum':
                total_loss = weighted.sum()
            else:
                total_loss = weighted.mean()
        
        if not return_detailed:
            return total_loss
        
        # Build detailed breakdown
        detailed = {}
        
        # Per-timestep: average over batch and channels
        detailed['per_timestep'] = weighted.mean(dim=(0, 2)).detach()
        
        # Per-channel: average over batch and frames
        detailed['per_channel'] = weighted.mean(dim=(0, 1)).detach()
        
        return total_loss, detailed

def normalize_kernel2d(input: torch.Tensor) -> torch.Tensor:
    """Normalize both derivative and smoothing kernel."""
    norm = input.abs().sum(dim=-1).sum(dim=-1)
    return input / (norm[..., None, None])

def get_sobel_kernel_3x3(device=None, dtype=None) -> torch.Tensor:
    """Return a sobel kernel of 3x3."""
    return torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], 
                       device=device, dtype=dtype)

def get_sobel_kernel_5x5_2nd_order(device=None, dtype=None) -> torch.Tensor:
    """Return a 2nd order sobel kernel of 5x5."""
    return torch.tensor(
        [
            [-1.0, 0.0, 2.0, 0.0, -1.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-6.0, 0.0, 12.0, 0.0, -6.0],
            [-4.0, 0.0, 8.0, 0.0, -4.0],
            [-1.0, 0.0, 2.0, 0.0, -1.0],
        ],
        device=device, dtype=dtype
    )

def _get_sobel_kernel_5x5_2nd_order_xy(device=None, dtype=None) -> torch.Tensor:
    """Return a 2nd order sobel kernel of 5x5 for cross derivatives."""
    return torch.tensor(
        [
            [-1.0, -2.0, 0.0, 2.0, 1.0],
            [-2.0, -4.0, 0.0, 4.0, 2.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
            [2.0, 4.0, 0.0, -4.0, -2.0],
            [1.0, 2.0, 0.0, -2.0, -1.0],
        ],
        device=device, dtype=dtype
    )

def get_diff_kernel_3x3(device=None, dtype=None) -> torch.Tensor:
    """Return a first order derivative kernel of 3x3."""
    return torch.tensor([[-0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [-0.0, 0.0, 0.0]], 
                       device=device, dtype=dtype)

def get_diff_kernel3d(device=None, dtype=None) -> torch.Tensor:
    """Return a first order derivative kernel of 3x3x3."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -0.5, 0.0], [0.0, 0.0, 0.0], [0.0, 0.5, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, -0.5, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device, dtype=dtype
    )
    return kernel[:, None, ...]

def get_diff_kernel3d_2nd_order(device=None, dtype=None) -> torch.Tensor:
    """Return a second order derivative kernel of 3x3x3."""
    kernel = torch.tensor(
        [
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, -2.0, 0.0], [0.0, 1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, -2.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[1.0, 0.0, -1.0], [0.0, 0.0, 0.0], [-1.0, 0.0, 1.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            ],
            [
                [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0], [0.0, -1.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, -1.0, 0.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            ],
            [
                [[0.0, 0.0, 0.0], [1.0, 0.0, -1.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0], [-1.0, 0.0, 1.0], [0.0, 0.0, 0.0]],
            ],
        ],
        device=device, dtype=dtype
    )
    return kernel[:, None, ...]

def get_sobel_kernel2d(device=None, dtype=None) -> torch.Tensor:
    """Return 1st order gradient for sobel operator."""
    kernel_x = get_sobel_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])

def get_diff_kernel2d(device=None, dtype=None) -> torch.Tensor:
    """Return 1st order gradient for diff operator."""
    kernel_x = get_diff_kernel_3x3(device=device, dtype=dtype)
    kernel_y = kernel_x.transpose(0, 1)
    return torch.stack([kernel_x, kernel_y])


def get_sobel_kernel2d_2nd_order(device=None, dtype=None) -> torch.Tensor:
    """Return 2nd order gradient for sobel operator."""
    gxx = get_sobel_kernel_5x5_2nd_order(device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = _get_sobel_kernel_5x5_2nd_order_xy(device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])

def get_diff_kernel2d_2nd_order(device=None, dtype=None) -> torch.Tensor:
    """Return 2nd order gradient for diff operator."""
    gxx = torch.tensor([[0.0, 0.0, 0.0], [1.0, -2.0, 1.0], [0.0, 0.0, 0.0]], 
                      device=device, dtype=dtype)
    gyy = gxx.transpose(0, 1)
    gxy = torch.tensor([[-1.0, 0.0, 1.0], [0.0, 0.0, 0.0], [1.0, 0.0, -1.0]], 
                      device=device, dtype=dtype)
    return torch.stack([gxx, gxy, gyy])

def get_spatial_gradient_kernel2d(mode: str, order: int, device=None, dtype=None) -> torch.Tensor:
    """Return kernel for 1st or 2nd order image gradients."""
    if mode not in ['sobel', 'diff']:
        raise ValueError(f"Mode should be 'sobel' or 'diff'. Got {mode}")
    if order not in [1, 2]:
        raise ValueError(f"Order should be 1 or 2. Got {order}")

    if mode == "sobel" and order == 1:
        kernel = get_sobel_kernel2d(device=device, dtype=dtype)
    elif mode == "sobel" and order == 2:
        kernel = get_sobel_kernel2d_2nd_order(device=device, dtype=dtype)
    elif mode == "diff" and order == 1:
        kernel = get_diff_kernel2d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel2d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented for order {order} on mode {mode}")

    return kernel

def get_spatial_gradient_kernel3d(mode: str, order: int, device=None, dtype=None) -> torch.Tensor:
    """Return kernel for 1st or 2nd order 3D gradients."""
    if mode not in ['sobel', 'diff']:
        raise ValueError(f"Mode should be 'sobel' or 'diff'. Got {mode}")
    if order not in [1, 2]:
        raise ValueError(f"Order should be 1 or 2. Got {order}")

    if mode == "diff" and order == 1:
        kernel = get_diff_kernel3d(device=device, dtype=dtype)
    elif mode == "diff" and order == 2:
        kernel = get_diff_kernel3d_2nd_order(device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"Not implemented 3d gradient kernel for order {order} on mode {mode}")

    return kernel

def spatial_gradient(input: torch.Tensor, mode: str = "sobel", order: int = 1, 
                     normalized: bool = True) -> torch.Tensor:
    """Compute spatial gradients in x and y directions.
    
    Args:
        input: input image tensor with shape (B, C, H, W)
        mode: derivatives modality, can be: 'sobel' or 'diff'
        order: the order of the derivatives
        normalized: whether the output is normalized
        
    Returns:
        derivatives with shape (B, C, 2, H, W) for 1st order or (B, C, 3, H, W) for 2nd order
    """
    if input.ndim != 4:
        raise ValueError(f"Expected 4D input, got {input.ndim}D")

    # Allocate kernel
    kernel = get_spatial_gradient_kernel2d(mode, order, device=input.device, dtype=input.dtype)
    if normalized:
        kernel = normalize_kernel2d(kernel)

    # Prepare kernel
    b, c, h, w = input.shape
    tmp_kernel = kernel[:, None, ...]

    # Pad with replicate for spatial dims
    spatial_pad = [kernel.size(1) // 2, kernel.size(1) // 2, 
                   kernel.size(2) // 2, kernel.size(2) // 2]
    out_channels = 3 if order == 2 else 2
    padded_inp = F.pad(input.reshape(b * c, 1, h, w), spatial_pad, "replicate")
    out = F.conv2d(padded_inp, tmp_kernel, groups=1, padding=0, stride=1)
    return out.reshape(b, c, out_channels, h, w)

def spatial_gradient3d(input: torch.Tensor, mode: str = "diff", order: int = 1) -> torch.Tensor:
    """Compute spatial gradients in 3D.
    
    Args:
        input: input tensor with shape (B, C, D, H, W)
        mode: derivatives modality, can be: 'sobel' or 'diff'
        order: the order of the derivatives
        
    Returns:
        spatial gradients with shape (B, C, 3, D, H, W) for 1st order or 
        (B, C, 6, D, H, W) for 2nd order
    """
    if input.ndim != 5:
        raise ValueError(f"Expected 5D input, got {input.ndim}D")

    b, c, d, h, w = input.shape
    dev = input.device
    dtype = input.dtype
    
    if (mode == "diff") and (order == 1):
        # Special case implementation for speed
        x = F.pad(input, 6 * [1], "replicate")
        center = slice(1, -1)
        left = slice(0, -2)
        right = slice(2, None)
        out = torch.empty(b, c, 3, d, h, w, device=dev, dtype=dtype)
        out[..., 0, :, :, :] = x[..., center, center, right] - x[..., center, center, left]
        out[..., 1, :, :, :] = x[..., center, right, center] - x[..., center, left, center]
        out[..., 2, :, :, :] = x[..., right, center, center] - x[..., left, center, center]
        out = 0.5 * out
    else:
        # Allocate kernel
        kernel = get_spatial_gradient_kernel3d(mode, order, device=dev, dtype=dtype)
        tmp_kernel = kernel.repeat(c, 1, 1, 1, 1)
        kernel_flip = tmp_kernel.flip(-3)

        # Pad with replicate for spatial dims
        spatial_pad = [
            kernel.size(2) // 2, kernel.size(2) // 2,
            kernel.size(3) // 2, kernel.size(3) // 2,
            kernel.size(4) // 2, kernel.size(4) // 2,
        ]
        out_ch = 6 if order == 2 else 3
        out = F.conv3d(F.pad(input, spatial_pad, "replicate"), kernel_flip, 
                      padding=0, groups=c).view(b, c, out_ch, d, h, w)
    return out

class SpatialGradient(nn.Module):
    """Module to compute first order image derivative in x and y.
    
    Args:
        mode: derivatives modality, can be: 'sobel' or 'diff'
        order: the order of the derivatives
        normalized: whether the output is normalized
    """

    def __init__(self, mode: str = "sobel", order: int = 1, normalized: bool = True) -> None:
        super().__init__()
        self.normalized = normalized
        self.order = order
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return spatial_gradient(input, self.mode, self.order, self.normalized)


class SpatialGradient3d(nn.Module):
    """Module to compute spatial gradients in 3D.
    
    Args:
        mode: derivatives modality, can be: 'sobel' or 'diff'
        order: the order of the derivatives
    """

    def __init__(self, mode: str = "diff", order: int = 1) -> None:
        super().__init__()
        self.order = order
        self.mode = mode

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return spatial_gradient3d(input, self.mode, self.order)