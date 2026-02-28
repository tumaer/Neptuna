from omegaconf import DictConfig, OmegaConf
import torch
from metrics.loss_framework import CompositeLoss, NestedCompositeLoss, WeightSchedule, NormalizationHelper, LossComponent
from metrics.loss_registry import get_loss_entry
from typing import Union, Optional, List, Dict
from metrics.loss_weighting_strategies import LossWeightingStrategyBase
from metrics.loss_weighting_strategy_registry import get_loss_weighting_strategy_entry

def create_loss_weight_schedule(component_cfg) -> Union[float, WeightSchedule]:
    """
    Creates a loss weighting schedule (WeightSchedule) from config.
    
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

def fetch_loss_metric(data_config, loss_dict) -> CompositeLoss:
    """
    Creates a CompositeLoss instance from hydra config.
    Supports nested composite components.
    """
    loss_components = []

    data_dim = data_config.dimension
    field_names = data_config.filter_features.filter_out_channels
    norm_stats = data_config.data_normalization_stats
    norm_strategy = data_config.data_normalization_strategy
    is_residual = False
    
    for component_cfg in loss_dict.components:
        loss_component = _create_loss_component(
            component_cfg,
            data_dim,
            field_names,
            norm_stats,
            norm_strategy,
            is_residual
        )
        loss_components.append(loss_component)
    
    return CompositeLoss(loss_components=loss_components)


def _create_loss_component(
    component_cfg,
    data_dim: int,
    field_names: List[str],
    norm_stats: Dict,
    norm_strategy: str,
    is_residual: bool
) -> LossComponent:
    """
    Recursively create a loss component, handling nested composites.
    """
    loss_type = component_cfg.type
    
    # Handle nested composite
    if loss_type in ("CompositeLoss", "NestedCompositeLoss"):
        name = component_cfg.get("name", "NestedComposite")
        weight = create_loss_weight_schedule(component_cfg)
        
        # Recursively create sub-components
        sub_components = []
        if hasattr(component_cfg, "sub_components"):
            for sub_cfg in component_cfg.sub_components:
                sub_comp = _create_loss_component(
                    sub_cfg,
                    data_dim,
                    field_names,
                    norm_stats,
                    norm_strategy,
                    is_residual
                )
                sub_components.append(sub_comp)
        
        norm_helper = NormalizationHelper(
            norm_stats=norm_stats,
            norm_strategy=norm_strategy,
            channel_names=field_names,
            is_residual=is_residual,
        )
        
        return NestedCompositeLoss(
            sub_components=sub_components,
            weight=weight,
            name=name,
            norm_helper=norm_helper,
            data_dim=data_dim,
            field_names=field_names,
        )
    
    # Handle regular loss component (existing code)
    registry_entry = get_loss_entry(loss_type)
    loss_class = registry_entry["class"]
    default_name = registry_entry["default_name"]
    default_config = registry_entry["default_config"]
    config_base_path = registry_entry["config_path"]
    
    name = component_cfg.get("name", default_name)
    weight = create_loss_weight_schedule(component_cfg)
    
    norm_helper = NormalizationHelper(
        norm_stats=norm_stats,
        norm_strategy=norm_strategy,
        channel_names=field_names,
        is_residual=is_residual,
    )
    
    # Load metric-specific config
    metric_params = {}
    if hasattr(component_cfg, "metric_params"):
        metric_params = OmegaConf.to_container(
            component_cfg.metric_params, resolve=True
        )
    else: 
        config_file = component_cfg.get("config_file", default_config)
        if config_file is not None:
            config_path = f"config/train_strategy_config/{config_base_path}{config_file}.yaml"
            try:
                metric_config = OmegaConf.load(config_path)
                metric_params = OmegaConf.to_container(metric_config, resolve=True)
            except Exception as e:
                print(f"Warning: Could not load metric config from {config_path}: {e}")
    
    return loss_class(
        weight=weight,
        name=name,
        data_dim=data_dim,
        field_names=field_names,
        norm_helper=norm_helper,
        **metric_params,
    )


def create_loss_weighting_strategy(train_loss_dict) -> Optional[LossWeightingStrategyBase]:
    """
    Creates a LossWeightingStrategyBase instance from hydra config.
    
    Args:
        cfg: Hydra config containing loss_config.loss_weighting_strategy
        
    Returns:
        LossWeightingStrategyBase instance or None if not configured
    """

    if not hasattr(train_loss_dict, 'train_loss_weighting_strategy'):
        return None
    
    scheduler_cfg = train_loss_dict.train_loss_weighting_strategy
    
    if not scheduler_cfg.get('enabled', False):
        return None
    
    scheduler_type = scheduler_cfg.type
    
    # Pull metadata from registry
    registry_entry = get_loss_weighting_strategy_entry(scheduler_type)
    scheduler_class = registry_entry["class"]
    default_config = registry_entry["default_config"]
    use_gradients = registry_entry.get("use_gradients", False)
    
    # Load scheduler-specific config
    scheduler_params = {}
    if hasattr(scheduler_cfg, "scheduler_params"):
        # Already populated by prepare_config or checkpoint
        scheduler_params = OmegaConf.to_container(
            scheduler_cfg.scheduler_params, resolve=True
        )
    else:
        # Determine config_file: explicit in YAML or from registry default
        config_file = scheduler_cfg.get("config_file", default_config)
        if config_file is not None:
            config_path = f"config/train_strategy_config/{config_file}.yaml"
            try:
                scheduler_config = OmegaConf.load(config_path)
                scheduler_params = OmegaConf.to_container(scheduler_config, resolve=True)
            except Exception as e:
                print(f"Warning: Could not load scheduler config from {config_path}: {e}")
        else:
            # No config_file and no scheduler_params – use defaults
            scheduler_params = {}
    
    scheduler_params['use_gradients'] = use_gradients

    # Create scheduler instance
    scheduler_instance = scheduler_class(**scheduler_params)
    
    return scheduler_instance


def _override_loss_weights_recursively(component_cfg, num_channels: int):
    """
    Recursively override timestep and channel weights for a component config.
    Handles both regular components and nested composites.
    
    Args:
        component_cfg: OmegaConf component configuration
        num_channels: Number of channels for channel_weights
    """
    # Override weights at this level
    component_cfg.timestep_weights = [1.0]
    component_cfg.channel_weights = [1.0] * num_channels
    
    # If this is a composite, recursively process sub-components
    if hasattr(component_cfg, 'sub_components'):
        for sub_component in component_cfg.sub_components:
            _override_loss_weights_recursively(sub_component, num_channels)


def fetch_infer_loss_dict(cfg):
    """
    Loads infer loss config and overrides timestep_weights and channel_weights.
    Inference requires the loss object to be initialized with per-channel and timestep
    weighting in order to compute rollout metrics (compute_metrics_for_n_rollouts)
    """
    eval_loss_config_path = "./config/infer_config/infer_loss.yaml" #TODO: avoid hardcoding "infer_loss.yaml".
    eval_loss_cfg = OmegaConf.load(eval_loss_config_path)

    num_channels = len(cfg.data_config.filter_features.filter_out_channels)
    
    for component in eval_loss_cfg.loss.components:
        _override_loss_weights_recursively(component, num_channels)
    
    return eval_loss_cfg.loss

#TODO: remove this
# def fetch_train_loss_dict(cfg):
#     return cfg.loss_config.train_loss

# #TODO: remove this
# def fetch_eval_loss_dict(cfg):
#     return cfg.loss_config.validation_loss

    