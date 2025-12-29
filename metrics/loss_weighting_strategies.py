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
        Compute new loss weight schedules based on loss history.
        
        Args:
            loss_history: Dictionary mapping component names to their loss values:
                {
                  'component_name': [loss1, loss2, loss3, ...],
                }
            current_weights: Current weight schedules from CompositeLoss.get_loss_weight_dict()
            
        Returns:
            Updated weight dictionary in same format as current_weights
        """
        pass
    
    def should_update(self, epoch: int) -> bool:
        """Check if loss weights should be updated at this epoch."""
        return epoch > 0 and epoch % self.update_frequency == 0
    
    def step(
        self,
        epoch: int,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict]
    ) -> Optional[Dict[str, Dict]]:
        """
        Step the scheduler. Returns new loss weights if update is due, else None.
        
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
