from metrics.weight_schedulers import UncertaintyWeighting

# Weight scheduler registry with metadata:
#  - class: the scheduler class
#  - default_name: fallback name if not provided in YAML
#  - default_config: fallback config_file (relative path under config/weight_scheduler_config)
WEIGHT_SCHEDULER_REGISTRY = {
    "UncertaintyWeighting": {
        "class": UncertaintyWeighting,
        "default_name": "UncertaintyWeighting",
        "default_config": "test_loss_weight/uncertainty_weighting_default",
    },
}

def get_weight_scheduler_entry(scheduler_type: str):
    """
    Retrieve registry entry for a weight scheduler type.
    
    Args:
        scheduler_type: Name of the scheduler type
        
    Returns:
        Dictionary containing class, default_name, and default_config
        
    Raises:
        ValueError: If scheduler_type is not registered
    """
    if scheduler_type not in WEIGHT_SCHEDULER_REGISTRY:
        available = ", ".join(WEIGHT_SCHEDULER_REGISTRY.keys())
        raise ValueError(
            f"Unknown weight scheduler type: {scheduler_type}. "
            f"Available types: {available}"
        )
    return WEIGHT_SCHEDULER_REGISTRY[scheduler_type]