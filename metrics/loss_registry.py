# FILE: metrics/loss_registry.py
from metrics.losses import (
    MAE,
    MSE,
    SSIM,
    MSSSIM,
    PearsonCorrelationLoss,
    SinkhornDivergence,
    H1SemiNorm,
    H2SemiNorm,
    MultilevelWaveletLoss,
    WaveletBinnedRMSE,
    IntegralConservationRMSE,
    RMSE,
    InterfaceRMSE,
    MeanRelativeError,
    NegativityLoss,
    ShockRMSE,
    PDEResidualLoss,
)

# Loss registry with metadata:
#  - class: the loss class
#  - default_name: fallback name if not provided in YAML
#  - default_config: fallback config_file (relative path under config/loss_config)
LOSS_REGISTRY = {
    "MSE": {
        "class": MSE,
        "default_name": "MSE",
        "default_config": "loss_metrics/mse/mse_default",
        "channel_aggregation": "linear",
    },
    "MAE": {
        "class": MAE,
        "default_name": "MAE",
        "default_config": "loss_metrics/mae/mae_default",
        "channel_aggregation": "linear",
    },
    "SSIM": {
        "class": SSIM,
        "default_name": "SSIM",
        "default_config": "loss_metrics/SSIM/ssim_default",
        "channel_aggregation": "linear",
    },
    "MSSSIM": {
        "class": MSSSIM,
        "default_name": "MSSSIM",
        "default_config": "loss_metrics/MSSSIM/msssim_default",
        "channel_aggregation": "linear",
    },
    "PearsonCorrelationLoss": {
        "class": PearsonCorrelationLoss,
        "default_name": "PearsonCorrelationLoss",
        "default_config": "loss_metrics/PearsonCorrelationLoss/pearson_correlation_loss_default",
        "channel_aggregation": "linear",
    },
    "SinkhornDivergence": {
        "class": SinkhornDivergence,
        "default_name": "SinkhornDivergence",
        "default_config": "loss_metrics/SinkhornDivergence/sinkhorn_divergence_default",
        "channel_aggregation": "linear",
    },
    "H1SemiNorm": {
        "class": H1SemiNorm,
        "default_name": "H1SemiNorm",
        "default_config": "loss_metrics/H1SemiNorm/h1_semi_norm_default",
        "channel_aggregation": "linear",
    },
    "H2SemiNorm": {
        "class": H2SemiNorm,
        "default_name": "H2SemiNorm",
        "default_config": "loss_metrics/H2SemiNorm/h2_semi_norm_default",
        "channel_aggregation": "linear",
    },
    "MultilevelWaveletLoss": {
        "class": MultilevelWaveletLoss,
        "default_name": "MultilevelWaveletLoss",
        "default_config": "loss_metrics/MultilevelWaveletLoss/multilevel_wavelet_loss_default",
        "channel_aggregation": "linear",
    },
    "WaveletBinnedRMSE": {
        "class": WaveletBinnedRMSE,
        "default_name": "WaveletBinnedRMSE",
        "default_config": "loss_metrics/WaveletBinnedRMSE/wavelet_binned_rmse_default",
        "channel_aggregation": "linear",
    },
    "IntegralConservationRMSE": {
        "class": IntegralConservationRMSE,
        "default_name": "IntegralConservationRMSE",
        "default_config": "loss_metrics/IntegralConservationRMSE/integral_conservation_rmse_default",
        "channel_aggregation": "sqrt",
        "sub_components": True,  # Indicates this loss decomposes as sub-components rather than channels
    },
    "RMSE": {
        "class": RMSE,
        "default_name": "RMSE",
        "default_config": "loss_metrics/RMSE/rmse_default",
        "channel_aggregation": "sqrt",
    },
    "InterfaceRMSE": {
        "class": InterfaceRMSE,
        "default_name": "InterfaceRMSE",
        "default_config": "loss_metrics/InterfaceRMSE/interface_rmse_default",
        "channel_aggregation": "sqrt",
    },
    "MeanRelativeError": {
        "class": MeanRelativeError,
        "default_name": "MeanRelativeError",
        "default_config": "loss_metrics/MeanRelativeError/mean_relative_error_default",
    },
    "NegativityLoss": {
        "class": NegativityLoss,
        "default_name": "NegativityLoss",
        "default_config": "loss_metrics/NegativityLoss/negativity_loss_default",
    },
    "ShockRMSE": {
        "class": ShockRMSE,
        "default_name": "ShockRMSE",
        "default_config": "loss_metrics/ShockRMSE/shock_rmse_default",
    },
    "PDEResidualLoss": {
        "class": PDEResidualLoss,
        "default_name": "PDEResidualLoss",
        "default_config": "loss_metrics/PDEResidualLoss/pde_residual_loss_default",
    },
}

def get_loss_entry(loss_type: str):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return LOSS_REGISTRY[loss_type]