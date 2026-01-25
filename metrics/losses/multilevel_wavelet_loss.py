import torch
import torch.nn as nn
import ptwt

from typing import Literal, Optional, List, Dict, Union, Tuple
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper

# Based on paper by L. Prandtl et al.,
# 'Wavelet-Based Loss for High-Frequency Interface Dynamics',
# https://arxiv.org/abs/2209.02316

class MultilevelWaveletLoss(LossComponent):
    """
    Wavelet-based loss from 'Wavelet-Based Loss for High-Frequency Interface Dynamics'.

    Expects predictions and targets with shape:
        (B, T, C, *spatial_dims)
    where len(spatial_dims) is 1, 2, or 3.
    """

    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: float = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        wavelet: str = "db2",
        alpha: float = 100.0,
        beta: float = 10.0,
        eps: float = 1e-6,
        spatial_level: Optional[int] = None,
        temporal_level: Optional[int] = None,
        mode_spatial: str = "reflect",
        mode_temporal: str = "reflect",
        reduction: str = "mean",
        normalization: Literal['none', 'magnitude', 'variance'] = 'none',
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names, norm_helper=norm_helper)
        assert reduction in ("mean", "sum")
        self.wavelet = wavelet
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.spatial_level = spatial_level
        self.temporal_level = temporal_level
        self.mode_spatial = mode_spatial
        self.mode_temporal = mode_temporal
        self.reduction = reduction
        self.normalization = normalization

    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = False,
        keep_batch_dim: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        predictions, labels: (B, T, C, *spatial)
        Returns weighted total loss.
        """
        base = float(self.weight_schedule.base_weight)

        if predictions.shape != labels.shape:
            raise ValueError(f"predictions and labels must have same shape, got {predictions.shape} vs {labels.shape}")

        original_shape = predictions.shape
        
        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_loss_weight(original_shape).to(predictions.device)
        
        # Apply weights to inputs (scale by sqrt to preserve wavelet properties)
        weight_sqrt = torch.sqrt(weight_tensor)
        predictions_weighted = predictions * weight_sqrt
        labels_weighted = labels * weight_sqrt

        if keep_batch_dim:
            # Compute per-sample loss
            per_batch = []
            for b in range(predictions_weighted.shape[0]):
                pred_b = predictions_weighted[b:b+1]
                label_b = labels_weighted[b:b+1]

                Lws = self._wavelet_loss_spatial(pred_b, label_b)
                Lwt = self._wavelet_loss_temporal(pred_b, label_b)
                total_b = (Lws + self.beta * Lwt) * base
                per_batch.append(total_b)

            total = torch.stack(per_batch, dim=0)
        else:
            # spatial wavelet loss (over spatial dimensions only)
            Lws = self._wavelet_loss_spatial(predictions_weighted, labels_weighted)

            # temporal wavelet loss (over time dimension only)
            Lwt = self._wavelet_loss_temporal(predictions_weighted, labels_weighted)

            # total (weighted)
            total = (Lws + self.beta * Lwt) * base

        total = self.norm_helper.normalize_loss(
            total,
            self.normalization,
            self.eps
        )
        
        if not return_detailed:
            return total
        
        # Wavelet loss doesn't support detailed breakdown due to non-linear aggregation
        return total, {}


    def _reduce(self, x: torch.Tensor) -> torch.Tensor:
        if self.reduction == "mean":
            return x.mean()
        else:
            return x.sum()

    def _wavelet_loss_spatial(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L_ws = sum_t |W_xyz(Y_t) - W_xyz(Ŷ_t)|
        using multi-level DWT across spatial dimensions.
        """
        pred_coeffs = self._wavelet_transform_spatial(pred)
        target_coeffs = self._wavelet_transform_spatial(target)

        if len(pred_coeffs) != len(target_coeffs):
            raise RuntimeError("Mismatch in number of spatial wavelet coefficient tensors.")

        loss = 0.0
        for pc, tc in zip(pred_coeffs, target_coeffs):
            loss = loss + self._reduce(torch.abs(pc - tc))
        return loss

    def _wavelet_transform_spatial(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Apply multi-level DWT over spatial dimensions only.
        Returns a list of log2(|HF_i| + eps) tensors for all detail subbands.
        """
        B, T, C, *spatial = x.shape
        D = len(spatial)
        if D not in (1, 2, 3):
            raise ValueError(f"Expected 1D, 2D or 3D spatial data, got {D}D.")

        # Flatten batch/time/channel into a single leading dimension
        x_flat = x.reshape(-1, *spatial)

        log_details: list[torch.Tensor] = []

        if D == 1:
            # shape: (N, L)
            coeffs = ptwt.wavedec(
                x_flat,
                self.wavelet,
                mode=self.mode_spatial,
                level=self.spatial_level,
                axis=-1,
            )
            # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD1]
            approx, *details = coeffs
            for d in details:
                log_details.append(torch.log2(d.abs() + self.eps))

        elif D == 2:
            # shape: (N, H, W)
            coeffs = ptwt.wavedec2(
                x_flat,
                self.wavelet,
                mode=self.mode_spatial,
                level=self.spatial_level,
                axes=(-2, -1),
            )
            # coeffs = (cA_n, T_n, ..., T1)
            # each T_l is a WaveletDetailTuple2d: (H, V, D)
            approx = coeffs[0]
            detail_tuples = coeffs[1:]
            for dt in detail_tuples:
                # dt.horizontal, dt.vertical, dt.diagonal
                for subband in dt:
                    log_details.append(torch.log2(subband.abs() + self.eps))

        elif D == 3:
            # shape: (N, D, H, W)
            coeffs = ptwt.wavedec3(
                x_flat,
                self.wavelet,
                mode=self.mode_spatial,
                level=self.spatial_level,
                axes=(-3, -2, -1),
            )
            # coeffs = (cA_n, D_n, ..., D1)
            # each D_l is a dict with keys like "aad", "ada", ..., "ddd"
            approx = coeffs[0]
            detail_dicts = coeffs[1:]
            for dct in detail_dicts:
                # iterate over all 3D detail subbands at this level
                for subband in dct.values():
                    log_details.append(torch.log2(subband.abs() + self.eps))

        return log_details

    def _wavelet_loss_temporal(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Compute L_wt = |W_t(Y) - W_t(Ŷ)|
        using DWT along the time axis (axis=1).
        """
        pred_coeffs = self._wavelet_transform_temporal(pred)
        target_coeffs = self._wavelet_transform_temporal(target)

        if len(pred_coeffs) != len(target_coeffs):
            raise RuntimeError("Mismatch in number of temporal wavelet coefficient tensors.")

        loss = 0.0
        for pc, tc in zip(pred_coeffs, target_coeffs):
            loss = loss + self._reduce(torch.abs(pc - tc))
        return loss

    def _wavelet_transform_temporal(self, x: torch.Tensor) -> list[torch.Tensor]:
        """
        Apply multi-level 1D DWT along the time dimension (axis=1),
        leaving batch, channel, and spatial dims as-is.

        Returns log2(|HF_i| + eps) detail coefficient tensors.
        """
        # x shape: (B, T, C, *spatial)
        coeffs = ptwt.wavedec(
            x,  # any dimensional tensor
            self.wavelet,
            mode=self.mode_temporal,
            level=self.temporal_level,
            axis=1,  # time dimension
        )

        approx, *details = coeffs
        log_details = [torch.log2(d.abs() + self.eps) for d in details]
        return log_details