# metrics/loss_weighting_strategies.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, List
import torch

class LossWeightingStrategyBase(ABC):
    """
    Base class for adaptive loss weight scheduling strategies.
    
    Strategies update component weights based on training statistics
    collected over an epoch.
    """
    
    def __init__(self, update_frequency: int = 1):
        """
        Args:
            update_frequency: Update weights every N epochs
        """
        self.update_frequency = update_frequency
        self.current_epoch = 0
        self.history: Dict[str, List[List[float]]] = {}  # component -> [epoch_losses_list]
    
    @abstractmethod
    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Compute new weight schedules based on loss history.
        
        Args:
            loss_history: Dictionary mapping component names to their loss values:
                {
                  'component_name': [loss1, loss2, loss3, ...],
                }
            current_weights: Current weight schedules from CompositeLoss.get_weight_dict()
            
        Returns:
            Updated weight dictionary in same format as current_weights
        """
        pass
    
    def should_update(self, epoch: int) -> bool:
        """Check if weights should be updated at this epoch."""
        return epoch > 0 and epoch % self.update_frequency == 0
    
    def step(
        self,
        epoch: int,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict]
    ) -> Optional[Dict[str, Dict]]:
        """
        Step the scheduler. Returns new weights if update is due, else None.
        
        Args:
            epoch: Current epoch number
            loss_history: Dictionary of component names to lists of loss values
            current_weights: Current weight dictionary from CompositeLoss
        """
        self.current_epoch = epoch
        
        # Store history for analysis
        for component_name, losses in loss_history.items():
            if component_name not in self.history:
                self.history[component_name] = []
            self.history[component_name].append(losses.copy())
        
        if self.should_update(epoch):
            return self.compute_new_weights(loss_history, current_weights)
        return None


class UncertaintyWeighting(LossWeightingStrategyBase):
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