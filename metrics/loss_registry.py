# FILE: metrics/loss_registry.py
from metrics.losses import (
    L1Loss,
    L2Loss,
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
    VRMSE,
    NRMSE,
    InterfaceRMSE,
    MeanRelativeError,
    NegativityLoss,
    ShockRMSE,
)

# Loss registry with metadata:
#  - class: the loss class
#  - default_name: fallback name if not provided in YAML
#  - default_config: fallback config_file (relative path under config/loss_config)
LOSS_REGISTRY = {
    "L2Loss": {
        "class": L2Loss,
        "default_name": "L2Loss",
        "default_config": "loss_metrics/L2_Loss/l2_loss_default",
    },
    "L1Loss": {
        "class": L1Loss,
        "default_name": "L1Loss",
        "default_config": "loss_metrics/L1_Loss/l1_loss_default",
    },
    "SSIM": {
        "class": SSIM,
        "default_name": "SSIM",
        "default_config": "loss_metrics/SSIM/ssim_default",
    },
    "MSSSIM": {
        "class": MSSSIM,
        "default_name": "MSSSIM",
        "default_config": "loss_metrics/MSSSIM/msssim_default",
    },
    "PearsonCorrelationLoss": {
        "class": PearsonCorrelationLoss,
        "default_name": "PearsonCorrelationLoss",
        "default_config": "loss_metrics/PearsonCorrelationLoss/pearson_correlation_loss_default",
    },
    "SinkhornDivergence": {
        "class": SinkhornDivergence,
        "default_name": "SinkhornDivergence",
        "default_config": "loss_metrics/SinkhornDivergence/sinkhorn_divergence_default",
    },
    "H1SemiNorm": {
        "class": H1SemiNorm,
        "default_name": "H1SemiNorm",
        "default_config": "loss_metrics/H1SemiNorm/h1_semi_norm_default",
    },
    "H2SemiNorm": {
        "class": H2SemiNorm,
        "default_name": "H2SemiNorm",
        "default_config": "loss_metrics/H2SemiNorm/h2_semi_norm_default",
    },
    "MultilevelWaveletLoss": {
        "class": MultilevelWaveletLoss,
        "default_name": "MultilevelWaveletLoss",
        "default_config": "loss_metrics/MultilevelWaveletLoss/multilevel_wavelet_loss_default",
    },
    "WaveletBinnedRMSE": {
        "class": WaveletBinnedRMSE,
        "default_name": "WaveletBinnedRMSE",
        "default_config": "loss_metrics/WaveletBinnedRMSE/wavelet_binned_rmse_default",
    },
    "IntegralConservationRMSE": {
        "class": IntegralConservationRMSE,
        "default_name": "IntegralConservationRMSE",
        "default_config": "loss_metrics/IntegralConservationRMSE/integral_conservation_rmse_default",
    },
    "RMSE": {
        "class": RMSE,
        "default_name": "RMSE",
        "default_config": "loss_metrics/RMSE/rmse_default",
    },
    "VRMSE": {
        "class": VRMSE,
        "default_name": "VRMSE",
        "default_config": "loss_metrics/VRMSE/vrmse_default",
    },
    "NRMSE": {
        "class": NRMSE,
        "default_name": "NRMSE",
        "default_config": "loss_metrics/NRMSE/nrmse_default",
    },
    "InterfaceRMSE": {
        "class": InterfaceRMSE,
        "default_name": "InterfaceRMSE",
        "default_config": "loss_metrics/InterfaceRMSE/interface_rmse_default",
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
}

def get_loss_entry(loss_type: str):
    if loss_type not in LOSS_REGISTRY:
        raise ValueError(f"Unknown loss type: {loss_type}")
    return LOSS_REGISTRY[loss_type]