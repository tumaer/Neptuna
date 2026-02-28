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
        "config_path": "loss_metrics/MSE/",
        "default_config": "mse_default",
        "channel_aggregation": "linear",
    },
    "MAE": {
        "class": MAE,
        "default_name": "MAE",
        "config_path": "loss_metrics/MAE/",
        "default_config": "mae_default",
        "channel_aggregation": "linear",
    },
    "SSIM": {
        "class": SSIM,
        "default_name": "SSIM",
        "config_path": "loss_metrics/SSIM/",
        "default_config": "ssim_default",
        "channel_aggregation": "linear",
    },
    "MSSSIM": {
        "class": MSSSIM,
        "default_name": "MSSSIM",
        "config_path": "loss_metrics/MSSSIM/",
        "default_config": "msssim_default",
        "channel_aggregation": "linear",
    },
    "PearsonCorrelationLoss": {
        "class": PearsonCorrelationLoss,
        "default_name": "PearsonCorrelationLoss",
        "config_path": "loss_metrics/PearsonCorrelationLoss/",
        "default_config": "pearson_correlation_loss_default",
        "channel_aggregation": "linear",
    },
    "SinkhornDivergence": {
        "class": SinkhornDivergence,
        "default_name": "SinkhornDivergence",
        "config_path": "loss_metrics/SinkhornDivergence/",
        "default_config": "sinkhorn_divergence_default",
        "channel_aggregation": "linear",
    },
    "H1SemiNorm": {
        "class": H1SemiNorm,
        "default_name": "H1SemiNorm",
        "config_path": "loss_metrics/H1SemiNorm/",
        "default_config": "h1_semi_norm_default",
        "channel_aggregation": "linear",
    },
    "H2SemiNorm": {
        "class": H2SemiNorm,
        "default_name": "H2SemiNorm",
        "config_path": "loss_metrics/H2SemiNorm/",
        "default_config": "h2_semi_norm_default",
        "channel_aggregation": "linear",
    },
    "MultilevelWaveletLoss": {
        "class": MultilevelWaveletLoss,
        "default_name": "MultilevelWaveletLoss",
        "config_path": "loss_metrics/MultilevelWaveletLoss/",
        "default_config": "multilevel_wavelet_loss_default",
        "channel_aggregation": "linear",
    },
    "WaveletBinnedRMSE": {
        "class": WaveletBinnedRMSE,
        "default_name": "WaveletBinnedRMSE",
        "config_path": "loss_metrics/WaveletBinnedRMSE/",
        "default_config": "wavelet_binned_rmse_default",
        "channel_aggregation": "linear",
    },
    "IntegralConservationRMSE": {
        "class": IntegralConservationRMSE,
        "default_name": "IntegralConservationRMSE",
        "config_path": "loss_metrics/IntegralConservationRMSE/",
        "default_config": "integral_conservation_rmse_default",
        "channel_aggregation": "sqrt",
        "sub_components": True,  # Indicates this loss decomposes as sub-components rather than channels
    },
    "RMSE": {
        "class": RMSE,
        "default_name": "RMSE",
        "config_path": "loss_metrics/RMSE/",
        "default_config": "rmse_default",
        "channel_aggregation": "sqrt",
    },
    "InterfaceRMSE": {
        "class": InterfaceRMSE,
        "default_name": "InterfaceRMSE",
        "config_path": "loss_metrics/InterfaceRMSE/",
        "default_config": "interface_rmse_default",
        "channel_aggregation": "sqrt",
    },
    "MeanRelativeError": {
        "class": MeanRelativeError,
        "default_name": "MeanRelativeError",
        "config_path": "loss_metrics/MeanRelativeError/",
        "default_config": "mean_relative_error_default",
    },
    "NegativityLoss": {
        "class": NegativityLoss,
        "default_name": "NegativityLoss",
        "config_path": "loss_metrics/NegativityLoss/",
        "default_config": "negativity_loss_default",
    },
    "ShockRMSE": {
        "class": ShockRMSE,
        "default_name": "ShockRMSE",
        "config_path": "loss_metrics/ShockRMSE/",
        "default_config": "shock_rmse_default",
    },
    "PDEResidualLoss": {
        "class": PDEResidualLoss,
        "default_name": "PDEResidualLoss",
        "config_path": "loss_metrics/PDEResidualLoss/",
        "default_config": "pde_residual_loss_default",
    },
}

def get_loss_entry(loss_type: str):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return LOSS_REGISTRY[loss_type]