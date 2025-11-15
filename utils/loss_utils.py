from omegaconf import DictConfig, OmegaConf
from metrics.training_metrics import CompositeLoss
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
    WaveletBinnedRMSE
)

def fetch_loss_metric(cfg) -> CompositeLoss:
    """
    Creates a CompositeLoss instance from hydra config.
    
    Args:
        cfg: Hydra config object containing loss configuration
        
    Returns:
        CompositeLoss instance with configured components
    """
    loss_registry = {
        'L2Loss': L2Loss,
        'L1Loss': L1Loss,
        'SSIM': SSIM,
        'MSSSIM': MSSSIM,
        'PearsonCorrelationLoss': PearsonCorrelationLoss,
        'SinkhornDivergence': SinkhornDivergence,
        'H1SemiNorm': H1SemiNorm,
        'H2SemiNorm': H2SemiNorm,
        'MultilevelWaveletLoss': MultilevelWaveletLoss,
        'WaveletBinnedRMSE': WaveletBinnedRMSE,
    }
    
    loss_components = []

    data_dim = cfg.data_config.dimension
    field_names = cfg.data_config.filter_features.filter_out_channels
    
    for component_cfg in cfg.loss_config.loss.components:
        loss_type = component_cfg.type
        weight = component_cfg.get('weight', 1.0)
        name = component_cfg.get('name', None)
        
        if loss_type not in loss_registry:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Load metric-specific config if provided
        metric_params = {}
        if 'config_file' in component_cfg:
            config_path = f"config/loss_config/{component_cfg.config_file}.yaml"
            metric_config = OmegaConf.load(config_path)
            metric_params = OmegaConf.to_container(metric_config, resolve=True)
        
        loss_class = loss_registry[loss_type]
        loss_instance = loss_class(
            weight=weight, 
            name=name, 
            data_dim=data_dim, 
            field_names=field_names,
            **metric_params
        )
        loss_components.append(loss_instance)
    
    return CompositeLoss(loss_components=loss_components)