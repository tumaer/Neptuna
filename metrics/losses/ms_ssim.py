import math
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper
from typing import Literal, Optional, List, Dict, Union, Tuple

# Adapted from mssim.pytorch:
# https://github.com/lartpang/mssim.pytorch

class MSSSIM(LossComponent):
    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: float = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        window_size=11,
        sigma=1.5,
        *,
        K1=0.01,
        K2=0.03,
        L=1,
        keep_bc_dims=False,
        padding=None,
        ensemble_kernel=True,
        normalization: Literal['none', 'range', 'variance', 'std'] = 'none',
        epsilon: float = 1e-8
    ):
        """Calculate the mean SSIM (MSSIM) between two 4D tensors.

        Args:
            weight: Loss component weight. Defaults to 1.0.
            name: Name of the loss component. Defaults to None.
            data_dim: Spatial dimension (2 or 3). Defaults to None.
            field_names: List of field/channel names. Defaults to None.
            window_size: Gaussian filter window size. Defaults to 11.
            sigma: Gaussian filter sigma. Defaults to 1.5.
            K1: SSIM stability constant. Defaults to 0.01.
            K2: SSIM stability constant. Defaults to 0.03.
            L: Dynamic range of pixel values. Defaults to 1.
            keep_bc_dims: Whether to preserve batch dimension. Defaults to False.
            padding: Gaussian filter padding. If None, uses window_size//2. Defaults to None.
            ensemble_kernel: Whether to fuse cascaded 1D kernels into one kernel. Defaults to True.


        Reference:
        [1] Wang, Zhou et al. “Image quality assessment: from error visibility to structural similarity.” IEEE Transactions on Image Processing 13 (2004): 600-612.
        [2] Wang, Zhou et al. “Multi-scale structural similarity for image quality assessment.” (2003).
        """
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names, norm_helper=norm_helper)
        
        self.data_dim = data_dim
        self.window_size = window_size
        self.C1 = (K1 * L) ** 2  # equ 7 in ref1
        self.C2 = (K2 * L) ** 2  # equ 7 in ref1
        self.keep_bc_dims = keep_bc_dims
        self.normalization = normalization
        self.epsilon = epsilon

        if self.data_dim < 2:
            raise ValueError("msssim only supports data_dim>=2")

        self.gaussian_filter = GaussianFilter(
            data_dim=self.data_dim,
            window_size=window_size,
            in_channels=len(self.field_names) if self.field_names is not None else 1,
            sigma=sigma,
            padding=padding,
            ensemble_kernel=ensemble_kernel,
        )

    @torch.amp.autocast('cuda', enabled=False)
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
        """Calculate the mean SSIM (MSSIM) between two 3d/4d/5d tensors.

        Args:
            model (nn.Module): The model (unused in SSIM calculation)
            predictions (Tensor): 3d/4d/5d tensor
            labels (Tensor): 3d/4d/5d tensor

        Returns:
            Tensor: Weighted SSIM loss
        """
        original_shape = predictions.shape
        B, T, C = predictions.shape[:3]
        spatial_dims = predictions.shape[3:]
        
        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_loss_weight(original_shape).to(predictions.device)
        
        # Reshape to (B*T, C, spatial_dims...)
        predictions_weighted = predictions.reshape(B * T, C, *spatial_dims)
        labels_weighted = labels.reshape(B * T, C, *spatial_dims)
        
        # Ensure filter buffers match predictions device/dtype (avoid moving predictions to CPU)
        gaussian_window = self.gaussian_filter.gaussian_window
        if gaussian_window.device != predictions.device or gaussian_window.dtype != predictions.dtype:
            self.gaussian_filter = self.gaussian_filter.to(device=predictions.device, dtype=predictions.dtype)
            gaussian_window = self.gaussian_filter.gaussian_window

        msssim_bt_c = self.msssim(predictions_weighted, labels_weighted, keep_bc_dims=True)

        # Apply weights after MS-SSIM so weighting only affects aggregation
        reduce_dims = tuple(list(range(3, weight_tensor.ndim)))
        weight_tc = weight_tensor.mean(dim=reduce_dims)
        loss_bt_c = (1.0 - msssim_bt_c).view(B, T, C) * weight_tc

        if keep_bc_dims:
            # loss_bt_c is (B*T, C) -> reshape and aggregate over T to get (B, C)
            loss = loss_bt_c.mean(dim=1)
        else:
            loss = loss_bt_c.mean()
        
        if not return_detailed:
            return loss

        detailed: Dict[str, torch.Tensor] = {}

        per_channel_loss = loss_bt_c.mean(dim=(0, 1))  # (C,)
        detailed['per_channel'] = per_channel_loss if preserve_component_grads else per_channel_loss.detach()

        return loss, detailed

    def ssim(self, x, y, keep_bc_dims: bool = False):
        ssim, _ = self._ssim(x, y)

        if keep_bc_dims:
            return ssim.flatten(2).mean(-1)
        else:
            return ssim.mean()

    def _ssim(self, x, y):
        mu_x = self.gaussian_filter(x)  # equ 14
        mu_y = self.gaussian_filter(y)  # equ 14
        sigma2_x = self.gaussian_filter(x * x) - mu_x * mu_x  # equ 15
        sigma2_y = self.gaussian_filter(y * y) - mu_y * mu_y  # equ 15
        sigma_xy = self.gaussian_filter(x * y) - mu_x * mu_y  # equ 16

        A1 = 2 * mu_x * mu_y + self.C1
        A2 = 2 * sigma_xy + self.C2
        B1 = mu_x * mu_x + mu_y * mu_y + self.C1
        B2 = sigma2_x + sigma2_y + self.C2

        # equ 12, 13 in ref1
        l = A1 / B1
        cs = A2 / B2
        ssim = l * cs
        return ssim, cs
    
    def msssim(self, x, y, keep_bc_dims: bool = False):
        ms_components = []
        for i, w in enumerate((0.0448, 0.2856, 0.3001, 0.2363, 0.1333)):
            ssim, cs = self._ssim(x, y)

            if keep_bc_dims:
                ssim = ssim.flatten(2).mean(-1)
                cs = cs.flatten(2).mean(-1)
            else:
                ssim = ssim.mean()
                cs = cs.mean()

            if i == 4:
                ms_components.append(ssim**w)
            else:
                ms_components.append(cs**w)
                bs, *c, h, w = x.shape
                padding = [s % 2 for s in (h, w)]  # spatial padding
                if len(c) > 1:
                    # only pooling in the spatial domain
                    x = x.reshape(bs, -1, h, w)
                    y = y.reshape(bs, -1, h, w)
                x = F.avg_pool2d(x, kernel_size=2, stride=2, padding=padding)
                y = F.avg_pool2d(y, kernel_size=2, stride=2, padding=padding)
                if len(c) > 1:
                    x = x.reshape(bs, *c, h // 2, w // 2)
                    y = y.reshape(bs, *c, h // 2, w // 2)
        msssim = math.prod(ms_components)  # equ 7 in ref2
        return msssim


FILTER = {
    1: F.conv1d,
    2: F.conv2d,
    3: F.conv3d,
}


class GaussianFilter(nn.Module):
    def __init__(self, data_dim, window_size, in_channels, sigma, padding=None, ensemble_kernel=True):
        """Gaussian Filer for 1D, 2D or 3D data (3D/4D/5D tensor)

        Args:
            data_dim (int, optional): The dimension of the data.
            window_size (int or Tuple[int], optional): The window size of the gaussian filter.
            in_channels (int, optional): The number of channels of the 4d tensor.
            sigma (float or Tuple[float], optional): The sigma of the gaussian filter.
            padding (int or Tuple[int], optional): The padding of the gaussian filter. Defaults to None. If it is set to None, the filter will use window_size//2 as the padding. Another common setting is 0.
            ensemble_kernel (bool, optional): Whether to fuse the two cascaded 1d kernel into a 2d kernel. Defaults to True.
        """
        super().__init__()
        if data_dim not in [1, 2, 3]:
            raise ValueError(f"data_dim must be 1, 2 or 3, but got {data_dim}.")
        self.data_dim = data_dim
        self.filter = FILTER[self.data_dim]

        if isinstance(window_size, int):
            window_size = [window_size] * self.data_dim
        if not all([w % 2 == 1 for w in window_size]):
            raise ValueError(f"Window size must be odd, but got {window_size}.")
        self.window_size = window_size

        if padding is None:
            padding = [w // 2 for w in window_size]
        if isinstance(padding, int):
            padding = [padding] * self.data_dim
        self.padding = padding

        if isinstance(sigma, (float, int)):
            sigma = [sigma] * self.data_dim
        self.sigma2 = [s**2 for s in sigma]

        assert len(self.window_size) == len(self.padding) == len(self.sigma2) == self.data_dim
        kernels = [self._get_gaussian_window1d(w, s2) for w, s2 in zip(self.window_size, self.sigma2)]

        self.ensemble_kernel = ensemble_kernel
        if self.ensemble_kernel:
            kernels = self._get_gaussian_windowNd(kernels)
            kernels = kernels.reshape(1, 1, *self.window_size).repeat_interleave(repeats=in_channels, dim=0)
            self.register_buffer(name="gaussian_window", tensor=kernels)
        else:
            for dim_idx, kernel in enumerate(kernels, start=2):
                base_shape = [1, 1] + [1] * self.data_dim
                base_shape[dim_idx] = -1
                kernel = kernel.reshape(*base_shape).repeat_interleave(repeats=in_channels, dim=0)
                if dim_idx == 2:
                    name = "gaussian_window"
                else:
                    name = f"gaussian_window_{dim_idx}"
                self.register_buffer(name=name, tensor=kernel)

    @staticmethod
    def _get_gaussian_window1d(window_size, sigma2):
        x = torch.arange(-(window_size // 2), window_size // 2 + 1)
        w = torch.exp(-0.5 * x**2 / sigma2)
        w = w / w.sum()
        return w

    def _get_gaussian_windowNd(self, gaussian_windows_1d):
        for dim_idx, kernel in enumerate(gaussian_windows_1d, start=2):
            base_shape = [1, 1] + [1] * self.data_dim
            base_shape[dim_idx] = -1
            kernel = kernel.reshape(*base_shape)
            if dim_idx == 2:
                w = kernel
            else:
                w = w * kernel
        return w

    def __repr__(self):
        base_str = f"{self.__class__.__name__} with Kernel: {self.gaussian_window.shape}"
        if not self.ensemble_kernel:
            for dim_idx in range(3, self.data_dim + 2):
                kernel = self.get_buffer(f"gaussian_window_{dim_idx}")
                base_str += f", {kernel.shape}"
        return base_str

    def forward(self, x):
        if self.ensemble_kernel:
            # ensemble kernel: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/3add4532d3f633316cba235da1c69e90f0dfb952/pytorch_ssim/__init__.py#L11-L15
            x = self.filter(input=x, weight=self.gaussian_window, stride=1, padding=self.padding, groups=x.shape[1])
        else:
            # splitted kernel: https://github.com/VainF/pytorch-msssim/blob/2398f4db0abf44bcd3301cfadc1bf6c94788d416/pytorch_msssim/ssim.py#L48
            for i, d in enumerate(x.shape[2:], start=2):
                if d >= self.window_size[i - 2]:
                    w = self.get_buffer(target="gaussian_window" if i == 2 else f"gaussian_window_{i}")
                    x = self.filter(input=x, weight=w, stride=1, padding=self.padding, groups=x.shape[1])
                else:
                    warnings.warn(
                        f"Skipping Gaussian Smoothing at dimension {i} for x: {x.shape} and window size: {self.window_size}"
                    )
        return x


