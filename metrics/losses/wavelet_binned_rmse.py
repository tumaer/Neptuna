import math
import torch
import torch.nn as nn
import ptwt  # pip install ptwt
from ..loss_framework import LossComponent, WeightSchedule, NormalizationHelper
from typing import Literal, Optional, List, Sequence, Dict, Union, Tuple

# Inspired by fRMSE from the paper by Takamoto et al.,
# 'PDEBENCH: An Extensive Benchmark for Scientific Machine Learning'
# https://arxiv.org/abs/2210.07182
# Adapted to use wavelet transforms (rather than Fourier transforms) for
# frequency binning, to support non-periodic BCs and spatially localized features.

class WaveletBinnedRMSE(LossComponent):
    """
    Wavelet-binned RMSE loss for spatial data.

    Expects predictions and targets with shape:
        (B, T, C, *spatial_dims)
    where len(spatial_dims) is 1, 2, or 3.

    For each spatial wavelet level i, computes RMSE_i between
    pred and target wavelet detail coefficients at that level.
    Levels are ordered from highest spatial frequency (i=0) to
    lowest spatial frequency (i=n_levels-1).

    You can aggregate the per-level RMSEs with 'aggregate' or
    get them back (return_per_level=True).
    """

    def __init__(
        self,
        norm_helper: NormalizationHelper,
        weight: float = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        wavelet: str = "db2",
        spatial_level: Optional[int] = None,
        mode_spatial: str = "reflect",
        aggregate: str = "mean",   # 'mean', 'sum', or 'none'
        level_weights: Optional[Sequence[float]] = None,
        normalize_weights: bool = True,
        return_per_level: bool = False,
        normalization: Literal['none', 'range', 'variance', 'std', 'norm', 'root_norm'] = 'none',
        epsilon: float = 1e-8
    ):
        super().__init__(weight=weight, name=name, data_dim=data_dim, field_names=field_names, norm_helper=norm_helper)
        assert aggregate in ("mean", "sum", "weighted")
        self.wavelet = wavelet
        self.spatial_level = spatial_level
        self.mode_spatial = mode_spatial
        self.aggregate = aggregate
        self.normalize_weights = normalize_weights
        self.return_per_level = return_per_level
        self.normalization = normalization
        self.epsilon = epsilon

        self._raw_level_weights = (
            list(level_weights) if level_weights is not None else None
        )

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
        predictions, labels: (B, T, C, *spatial_dims)
        Returns weighted loss (and optionally per-level RMSE).
        """
        if predictions.shape != labels.shape:
            raise ValueError(f"predictions and labels must have same shape, got {predictions.shape} vs {labels.shape}")

        original_shape = predictions.shape
        B, T, C, *spatial = predictions.shape
        D = len(spatial)
        if D not in (1, 2, 3):
            raise ValueError(f"Expected 1D, 2D or 3D spatial data, got {D}D.")

        # Get weight tensor with proper broadcasting
        weight_tensor = self.weight_schedule.get_loss_weight(original_shape).to(predictions.device)
        
        # Flatten batch/time/channel into a single leading dimension
        pred_flat = predictions.permute(0, 2, 1, *range(3, 3 + D)).reshape(B * C, T, *spatial)
        target_flat = labels.permute(0, 2, 1, *range(3, 3 + D)).reshape(B * C, T, *spatial)

        if D == 1:
            per_level_rmse = self._binned_rmse_1d(pred_flat, target_flat, keep_bc_dims=True)
        elif D == 2:
            per_level_rmse = self._binned_rmse_2d(pred_flat, target_flat, keep_bc_dims=True)
        else:  # D == 3
            per_level_rmse = self._binned_rmse_3d(pred_flat, target_flat, keep_bc_dims=True)

        per_level_rmse = per_level_rmse.view(B, C, -1)

        # Apply weights after RMSE so weighting only affects aggregation
        reduce_dims = tuple(range(3, weight_tensor.ndim))
        weight_tc = weight_tensor.mean(dim=reduce_dims) if reduce_dims else weight_tensor
        weight_bc = weight_tc.mean(dim=1)  # (B, C)
        per_level_rmse = per_level_rmse * weight_bc[..., None] * self.weight

        # Aggregate across levels to get a loss
        if self.aggregate == "mean":
            loss_bc = per_level_rmse.mean(dim=-1)
        elif self.aggregate == "sum":
            loss_bc = per_level_rmse.sum(dim=-1)
        elif self.aggregate == "weighted":
            weights = self._get_level_weights(
                n_levels=per_level_rmse.shape[-1],
                device=per_level_rmse.device,
                dtype=per_level_rmse.dtype,
            )
            loss_bc = (per_level_rmse * weights).sum(dim=-1)
        else:
            raise RuntimeError(f"Unknown aggregate mode: {self.aggregate}")

        loss = loss_bc if keep_bc_dims else loss_bc.mean()

        if not return_detailed:
            return loss
        
        detailed: Dict[str, torch.Tensor] = {}

        per_channel = loss_bc.mean(dim=0)
        detailed['per_channel'] = per_channel if preserve_component_grads else per_channel.detach()

        return loss, detailed

    # ------------------------------------------------------------------
    # 1D case
    # ------------------------------------------------------------------
    def _binned_rmse_1d(
        self,
        pred_flat: torch.Tensor,
        target_flat: torch.Tensor,
        keep_bc_dims: bool = False
    ) -> torch.Tensor:
        """
        pred_flat, target_flat: (B, N_tc, L)
        Returns: rmse_per_level: (B, n_levels) if keep_bc_dims else (n_levels,)
        """
        # Multi-level DWT along last axis
        coeffs_pred = ptwt.wavedec(
            pred_flat,
            self.wavelet,
            mode=self.mode_spatial,
            level=self._resolve_level(pred_flat.shape[-1]),
            axis=-1,
        )
        coeffs_tgt = ptwt.wavedec(
            target_flat,
            self.wavelet,
            mode=self.mode_spatial,
            level=self._resolve_level(target_flat.shape[-1]),
            axis=-1,
        )

        # coeffs = [cA_n, cD_n, cD_{n-1}, ..., cD_1]
        approx_p, *details_p = coeffs_pred
        approx_t, *details_t = coeffs_tgt

        if len(details_p) != len(details_t):
            raise RuntimeError("Mismatch in number of wavelet levels between pred and target (1D).")

        # Reorder details so index 0 = highest frequency (D_1)
        details_p = list(reversed(details_p))
        details_t = list(reversed(details_t))

        rmse_levels = []
        for dp, dt in zip(details_p, details_t):
            diff = dp - dt
            if keep_bc_dims:
                reduce_dims = list(range(1, diff.ndim))
            else:
                reduce_dims = list(range(0, diff.ndim))
            mse = diff.pow(2).mean(dim=reduce_dims)
            rmse = torch.sqrt(mse)
            rmse_levels.append(rmse)

        return torch.stack(rmse_levels, dim=-1)  # (B, n_levels) or (n_levels,)

    # ------------------------------------------------------------------
    # 2D case
    # ------------------------------------------------------------------
    def _binned_rmse_2d(
        self,
        pred_flat: torch.Tensor,
        target_flat: torch.Tensor,
        keep_bc_dims: bool = False
    ) -> torch.Tensor:
        """
        pred_flat, target_flat: (B, N_tc, H, W)
        Returns: rmse_per_level: (B, n_levels) if keep_bc_dims else (n_levels,)
        """
        coeffs_pred = ptwt.wavedec2(
            pred_flat,
            self.wavelet,
            mode=self.mode_spatial,
            level=self._resolve_level(min(pred_flat.shape[-2:])),
            axes=(-2, -1),
        )
        coeffs_tgt = ptwt.wavedec2(
            target_flat,
            self.wavelet,
            mode=self.mode_spatial,
            level=self._resolve_level(min(target_flat.shape[-2:])),
            axes=(-2, -1),
        )

        # coeffs = (cA_n, (cH_n, cV_n, cD_n), ..., (cH_1, cV_1, cD_1))
        approx_p = coeffs_pred[0]
        approx_t = coeffs_tgt[0]
        details_p = coeffs_pred[1:]
        details_t = coeffs_tgt[1:]

        if len(details_p) != len(details_t):
            raise RuntimeError("Mismatch in number of wavelet levels between pred and target (2D).")

        # Reorder so index 0 = highest frequency (level 1)
        details_p = list(reversed(details_p))
        details_t = list(reversed(details_t))

        rmse_levels = []
        for (Hp, Vp, Dp), (Ht, Vt, Dt) in zip(details_p, details_t):
            # We want MSE over all three subbands combined
            sq_sum = 0.0
            count = 0
            for bp, bt in ((Hp, Ht), (Vp, Vt), (Dp, Dt)):
                diff = bp - bt
                if keep_bc_dims:
                    sq_sum = sq_sum + diff.pow(2).sum(dim=list(range(1, diff.ndim)))
                    count = count + diff[0].numel()
                else:
                    sq_sum = sq_sum + diff.pow(2).sum()
                    count = count + diff.numel()
            mse = sq_sum / count
            rmse = torch.sqrt(mse)
            rmse_levels.append(rmse)

        return torch.stack(rmse_levels, dim=-1)

    # ------------------------------------------------------------------
    # 3D case
    # ------------------------------------------------------------------
    def _binned_rmse_3d(
        self,
        pred_flat: torch.Tensor,
        target_flat: torch.Tensor,
        keep_bc_dims: bool = False
    ) -> torch.Tensor:
        """
        pred_flat, target_flat: (N, D, H, W)
        Returns: rmse_per_level: (n_levels,)
        """
        min_spatial = min(pred_flat.shape[-3:])
        coeffs_pred = ptwt.wavedec3(
            pred_flat,
            self.wavelet,
            mode=self.mode_spatial,
            level=self._resolve_level(min_spatial),
            axes=(-3, -2, -1),
        )
        coeffs_tgt = ptwt.wavedec3(
            target_flat,
            self.wavelet,
            mode=self.mode_spatial,
            level=self._resolve_level(min_spatial),
            axes=(-3, -2, -1),
        )

        # coeffs = (cA_n, D_n, ..., D_1)
        approx_p = coeffs_pred[0]
        approx_t = coeffs_tgt[0]
        detail_dicts_p = coeffs_pred[1:]
        detail_dicts_t = coeffs_tgt[1:]

        if len(detail_dicts_p) != len(detail_dicts_t):
            raise RuntimeError("Mismatch in number of wavelet levels between pred and target (3D).")

        # Reorder so index 0 = highest frequency (level 1)
        detail_dicts_p = list(reversed(detail_dicts_p))
        detail_dicts_t = list(reversed(detail_dicts_t))

        rmse_levels = []
        for dct_p, dct_t in zip(detail_dicts_p, detail_dicts_t):
            # keys like 'aad', 'ada', ..., 'ddd' (7 high-frequency subbands)
            keys = sorted(dct_p.keys())
            sq_sum = 0.0
            count = 0
            for k in keys:
                bp = dct_p[k]
                bt = dct_t[k]
                diff = bp - bt
                if keep_bc_dims:
                    sq_sum = sq_sum + diff.pow(2).sum(dim=list(range(1, diff.ndim)))
                    count = count + diff[0].numel()
                else:
                    sq_sum = sq_sum + diff.pow(2).sum()
                    count = count + diff.numel()
            mse = sq_sum / count
            rmse = torch.sqrt(mse)
            rmse_levels.append(rmse)

        return torch.stack(rmse_levels, dim=-1)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------
    def _resolve_level(self, min_len: int) -> Optional[int]:
        """
        If self.spatial_level is set, use that.
        Otherwise, approximate max sensible level as floor(log2(min_len)).
        """
        if self.spatial_level is not None:
            return self.spatial_level
        if min_len <= 1:
            return 0
        return int(math.log2(min_len))

    def _get_level_weights(self, n_levels: int, device, dtype):
        """
        Turn YAML-provided 'level_weights' into a tensor of length n_levels.
        We interpret index 0 as the highest-frequency bin (level 1).
        """
        if self._raw_level_weights is None:
            # Default: uniform weights across all levels
            return torch.full((n_levels,), 1.0 / n_levels, device=device, dtype=dtype)

        w = torch.tensor(self._raw_level_weights, dtype=dtype, device=device)
        if w.numel() != n_levels:
            raise ValueError(
                f"level_weights length ({w.numel()}) does not match number of levels ({n_levels}). "
                f"Either adjust 'spatial_level' or your 'level_weights' in the YAML config."
            )

        if self.normalize_weights:
            w_sum = w.sum()
            if w_sum > 0:
                w = w / w_sum
        return w