from metrics.weighting_strategies import (
    GaussianLikelihood,
)

# Weight scheduler registry with metadata:
#  - class: the scheduler class
#  - default_name: fallback name if not provided in YAML
#  - default_config: fallback config_file (relative path under config/loss_weighting_strategy_config)
WEIGHT_SCHEDULER_REGISTRY = {
    "GaussianLikelihood": {
        "class": GaussianLikelihood,
        "default_name": "GaussianLikelihood",
        "default_config": "weighting_strategies/GaussianLikelihood/gaussian_likelihood_default",
    },
}

def get_loss_weighting_strategy_entry(scheduler_type: str):
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