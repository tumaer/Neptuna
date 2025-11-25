from omegaconf import DictConfig, OmegaConf
import torch
from metrics.training_metrics import CompositeLoss, WeightSchedule
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
    IntegralConservationRMSE
)
from typing import Union

def create_weight_schedule(component_cfg) -> Union[float, WeightSchedule]:
    """
    Creates a weight or WeightSchedule from config.
    
    Args:
        component_cfg: Configuration for a single loss component
        
    Returns:
        Either a scalar weight or a WeightSchedule instance
    """
    base_weight = component_cfg.get('weight', 1.0)
    timestep_weights = component_cfg.get('timestep_weights', None)
    channel_weights = component_cfg.get('channel_weights', None)
    component_weights = component_cfg.get('component_weights', None)
    
    # If no schedule weights specified, return scalar (backward compatible)
    if timestep_weights is None and channel_weights is None and component_weights is None:
        return base_weight
    
    # Convert OmegaConf ListConfig to tensor for timestep_weights
    if timestep_weights is not None:
        if OmegaConf.is_list(timestep_weights):
            timestep_weights = OmegaConf.to_container(timestep_weights, resolve=True)
        
        if isinstance(timestep_weights, (list, tuple)):
            timestep_weights = torch.tensor(timestep_weights, dtype=torch.float32)
        elif not isinstance(timestep_weights, torch.Tensor):
            raise ValueError(f"timestep_weights must be list, tuple, or tensor, got {type(timestep_weights)}")
    
    # Convert OmegaConf ListConfig to tensor for channel_weights
    if channel_weights is not None:
        if OmegaConf.is_list(channel_weights):
            channel_weights = OmegaConf.to_container(channel_weights, resolve=True)
        
        if isinstance(channel_weights, (list, tuple)):
            channel_weights = torch.tensor(channel_weights, dtype=torch.float32)
        elif not isinstance(channel_weights, torch.Tensor):
            raise ValueError(f"channel_weights must be list, tuple, or tensor, got {type(channel_weights)}")
    
    # Convert OmegaConf DictConfig to regular dict for component_weights
    if component_weights is not None:
        if OmegaConf.is_dict(component_weights):
            component_weights = OmegaConf.to_container(component_weights, resolve=True)
        
        if not isinstance(component_weights, dict):
            raise ValueError(f"component_weights must be a dict, got {type(component_weights)}")
    
    return WeightSchedule(
        base_weight=base_weight,
        timestep_weights=timestep_weights,
        channel_weights=channel_weights,
        component_weights=component_weights
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
        'IntegralConservationRMSE': IntegralConservationRMSE
    }
    
    loss_components = []

    data_dim = cfg.data_config.dimension
    field_names = cfg.data_config.filter_features.filter_out_channels
    norm_stats = cfg.data_config.data_normalization_stats
    
    for component_cfg in cfg.loss_config.loss.components:
        loss_type = component_cfg.type
        name = component_cfg.get('name', None)
        
        if loss_type not in loss_registry:
            raise ValueError(f"Unknown loss type: {loss_type}")
        
        # Create weight or weight schedule from config
        weight = create_weight_schedule(component_cfg)
        
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
            norm_stats=norm_stats,
            **metric_params
        )
        loss_components.append(loss_instance)
    
    return CompositeLoss(loss_components=loss_components)

def fetch_eval_loss_config(cfg):
    #TODO: Make this more robust by adapting eval_loss_cfg to the dataset

    eval_loss_config_path = "./config/loss_config/infer_loss.yaml"
    eval_loss_cfg = OmegaConf.load(eval_loss_config_path)
    
    full_eval_cfg = OmegaConf.create({
        "loss_config": eval_loss_cfg,
        "data_config": cfg.data_config
    })

    return full_eval_cfg