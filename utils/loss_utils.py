from omegaconf import DictConfig, OmegaConf
import torch
from metrics.training_metrics import CompositeLoss, WeightSchedule
from metrics.loss_registry import get_loss_entry
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
    """
    loss_components = []

    data_dim = cfg.data_config.dimension
    field_names = cfg.data_config.filter_features.filter_out_channels
    norm_stats = cfg.data_config.data_normalization_stats
    
    for component_cfg in cfg.loss_config.loss.components:
        loss_type = component_cfg.type

        # Pull metadata from registry
        registry_entry = get_loss_entry(loss_type)
        loss_class = registry_entry["class"]
        default_name = registry_entry["default_name"]
        default_config = registry_entry["default_config"]

        # 1) Derive name if missing (use default_name, typically == type)
        name = component_cfg.get("name", default_name)

        # 2) Create weight or weight schedule
        weight = create_weight_schedule(component_cfg)

        # 3) Load metric-specific config
        metric_params = {}
        if hasattr(component_cfg, "metric_params"):
            # Already populated by prepare_config or checkpoint
            metric_params = OmegaConf.to_container(
                component_cfg.metric_params, resolve=True
            )
        else:
            # Determine config_file: explicit in YAML or from registry default
            config_file = component_cfg.get("config_file", default_config)
            if config_file is not None:
                config_path = f"config/loss_config/{config_file}.yaml"
                try:
                    metric_config = OmegaConf.load(config_path)
                    metric_params = OmegaConf.to_container(metric_config, resolve=True)
                except Exception as e:
                    print(f"Warning: Could not load metric config from {config_path}: {e}")
            else:
                # No config_file and no metric_params – OK if class uses only defaults
                metric_params = {}

        loss_instance = loss_class(
            weight=weight,
            name=name,
            data_dim=data_dim,
            field_names=field_names,
            norm_stats=norm_stats,
            **metric_params,
        )
        loss_components.append(loss_instance)
    
    return CompositeLoss(loss_components=loss_components)

def fetch_eval_loss_config(cfg):
    """
    Loads evaluation loss config and overrides timestep_weights and channel_weights
    to ensure uniform weighting across all timesteps and channels.
    """
    eval_loss_config_path = "./config/loss_config/infer_loss.yaml"
    eval_loss_cfg = OmegaConf.load(eval_loss_config_path)
    
    # Get number of channels from data config
    num_channels = len(cfg.data_config.filter_features.filter_out_channels)
    
    # Override weights for each component
    for component in eval_loss_cfg.loss.components:
        component.timestep_weights = [1.0]
        component.channel_weights = [1.0] * num_channels
    
    full_eval_cfg = OmegaConf.create({
        "loss_config": eval_loss_cfg,
        "data_config": cfg.data_config
    })

    return full_eval_cfg