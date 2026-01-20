from metrics.weighting_strategies import (
    ReLoBRaLo,
    ResidualBasedAttention,
    SoftAdapt,
    BalancedResidualDecayRate,
    GradNorm,
    InverseDirichlet,
    LearningRateAnnealing,
)

# Weight scheduler registry with metadata:
#  - class: the scheduler class
#  - default_name: fallback name if not provided in YAML
#  - default_config: fallback config_file (relative path under config/loss_weighting_strategy_config)
WEIGHT_SCHEDULER_REGISTRY = {
    "ReLoBRaLo": {
        "class": ReLoBRaLo,
        "default_name": "ReLoBRaLo",
        "default_config": "weighting_strategies/ReLoBRaLo/relobralo_default",
    },
    "ResidualBasedAttention": {
        "class": ResidualBasedAttention,
        "default_name": "ResidualBasedAttention",
        "default_config": "weighting_strategies/ResidualBasedAttention/residual_based_attention_default",
    },
    "SoftAdapt": {
        "class": SoftAdapt,
        "default_name": "SoftAdapt",
        "default_config": "weighting_strategies/SoftAdapt/soft_adapt_default",
    },
    "BalancedResidualDecayRate": {
        "class": BalancedResidualDecayRate,
        "default_name": "BalancedResidualDecayRate",
        "default_config": "weighting_strategies/BalancedResidualDecayRate/balanced_residual_decay_rate_default",
    },
    "GradNorm": {
        "class": GradNorm,
        "default_name": "GradNorm",
        "default_config": "weighting_strategies/GradNorm/grad_norm_default",
        "grad_stats": ["norm"],
    },
    "InverseDirichlet": {
        "class": InverseDirichlet,
        "default_name": "InverseDirichlet",
        "default_config": "weighting_strategies/InverseDirichlet/inverse_dirichlet_default",
        "grad_stats": ["var"],
    },
    "LearningRateAnnealing": {
        "class": LearningRateAnnealing,
        "default_name": "LearningRateAnnealing",
        "default_config": "weighting_strategies/LearningRateAnnealing/learning_rate_annealing_default",
        "grad_stats": ["norm", "max"],
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