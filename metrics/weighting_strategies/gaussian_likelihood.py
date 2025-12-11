from typing import Dict, List
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch

class GaussianLikelihood(LossWeightingStrategyBase):
    """
    Weight components inversely proportional to their uncertainty (variance).
    
    Based on "Multi-Task Learning Using Uncertainty to Weigh Losses"
    (Kendall et al., 2018) - homoscedastic uncertainty approach.
    
    w_i = 1 / (2 * sigma_i^2)
    
    where sigma_i^2 is estimated from loss variance over the epoch.
    """
    
    def __init__(
        self,
        update_frequency: int = 1,
        momentum: float = 0.9,
        min_weight: float = 0.1,
        max_weight: float = 10.0
    ):
        super().__init__(update_frequency)
        self.momentum = momentum
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.running_variance: Dict[str, float] = {}
    
    def compute_statistics(self, losses: List[float]) -> Dict[str, float]:
        """
        Compute statistics from a list of loss values.
        
        Args:
            losses: List of loss values
            
        Returns:
            Dictionary with mean, std, min, max
        """
        if not losses:
            return None
        
        losses_tensor = torch.tensor(losses)
        return {
            'mean': float(losses_tensor.mean()),
            'std': float(losses_tensor.std()),
            'min': float(losses_tensor.min()),
            'max': float(losses_tensor.max()),
            'count': len(losses)
        }
    
    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        new_weights = {}
        
        # Only update components that have loss history
        for component_name in current_weights.keys():
            if component_name not in loss_history or not loss_history[component_name]:
                # Keep existing weight if no history available
                new_weights[component_name] = current_weights[component_name].copy()
                continue
            
            # Compute statistics from loss history
            stats = self.compute_statistics(loss_history[component_name])
            if stats is None:
                new_weights[component_name] = current_weights[component_name].copy()
                continue
            
            # Use std as measure of uncertainty
            variance = stats['std'] ** 2
            
            # Apply momentum smoothing
            if component_name in self.running_variance:
                variance = (
                    self.momentum * self.running_variance[component_name] +
                    (1 - self.momentum) * variance
                )
            self.running_variance[component_name] = variance
            
            # Compute new weight: inversely proportional to uncertainty
            new_weight = 1.0 / (2.0 * variance + 1e-8)
            
            # Clip to reasonable range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            # Preserve existing schedule structure, update only base_weight
            new_weights[component_name] = current_weights[component_name].copy()
            new_weights[component_name]['base_weight'] = float(new_weight)
        
        return new_weights