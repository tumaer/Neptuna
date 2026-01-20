# metrics/loss_weighting_strategies.py
from abc import ABC, abstractmethod
from typing import Dict, Optional, List, Union
import torch

class LossWeightingStrategyBase(ABC):
    """
    Base class for adaptive loss weight scheduling strategies.
    
    Strategies update component weights based on training statistics
    collected over an epoch.
    """
    
    def __init__(self, update_frequency: int = 1, use_gradients: bool = False):
        """
        Args:
            update_frequency: Update weights every N epochs
        """
        self.update_frequency = update_frequency
        self.use_gradients = use_gradients
        self.current_epoch = 0
        self.history: Dict[str, List[List[float]]] = {}  # component -> [epoch_losses_list]
        # Nested grad stats: component -> stat_name -> [epoch_stats_list]
        self.grad_stats_history: Dict[str, Dict[str, List[List[float]]]] = {}
    
    @abstractmethod
    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_stats_history: Optional[Dict[str, Dict[str, List[float]]]] = None
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
        current_weights: Dict[str, Dict],
        grad_stats_history: Optional[Dict[str, Dict[str, List[float]]]] = None
    ) -> Optional[Dict[str, Dict]]:
        """
        Step the scheduler. Returns new loss weights if update is due, else None.
        
        Args:
            epoch: Current epoch number
            loss_history: Dictionary of component names to lists of loss values
            current_weights: Current weight dictionary from CompositeLoss
        """
        self.current_epoch = epoch
        
        # Store loss history for analysis
        for component_name, losses in loss_history.items():
            if component_name not in self.history:
                self.history[component_name] = []
            self.history[component_name].append(losses.copy())
        
        # Store gradient stats history (nested dict)
        if grad_stats_history is not None:
            for component_name, stats_dict in grad_stats_history.items():
                if component_name not in self.grad_stats_history:
                    self.grad_stats_history[component_name] = {}
                if isinstance(stats_dict, dict):
                    for stat_name, stat_values in stats_dict.items():
                        self.grad_stats_history[component_name].setdefault(stat_name, []).append(stat_values.copy())

        if self.should_update(epoch):
            return self.compute_new_weights(loss_history, current_weights, grad_stats_history)
        return None

    def _parse_hierarchical_key(self, key: str) -> tuple[str, Optional[str], Optional[int]]:
        """
        Parse a hierarchical loss key into components.
        
        Args:
            key: Loss component key (e.g., 'L2Loss', 'L2Loss/channel_0', 'RMSE/domain/mass')
            
        Returns:
            Tuple of (base_name, sub_component_name, channel_idx)
            Examples:
                'L2Loss' -> ('L2Loss', None, None)
                'L2Loss/channel_0' -> ('L2Loss', None, 0)
                'RMSE/domain/mass' -> ('RMSE', 'domain/mass', None)
        """
        if '/' not in key:
            return key, None, None
        
        # Check if it's a channel key
        if '/channel_' in key:
            base_name, channel_part = key.split('/channel_')
            channel_idx = int(channel_part)
            return base_name, None, channel_idx
        
        # Otherwise it's a sub-component key
        parts = key.split('/')
        base_name = parts[0]
        sub_name = '/'.join(parts[1:])
        return base_name, sub_name, None
    
    def _group_hierarchical_losses(
        self, 
        loss_history: Dict[str, List[float]]
    ) -> Dict[str, Dict[str, Union[List[float], Dict]]]:
        """
        Group hierarchical loss history by base component.
        
        Args:
            loss_history: Flat dictionary of all losses
            
        Returns:
            Grouped dictionary:
            {
                'L2Loss': {
                    'base': [loss_values],
                    'channels': {0: [losses], 1: [losses], ...},
                },
                'RMSE': {
                    'base': [loss_values],
                    'components': {'domain/mass': [losses], ...}
                }
            }
        """
        grouped: Dict[str, Dict] = {}
        
        for key, losses in loss_history.items():
            base_name, sub_name, channel_idx = self._parse_hierarchical_key(key)
            
            if base_name not in grouped:
                grouped[base_name] = {
                    'base': None,
                    'channels': {},
                    'components': {}
                }
            
            if sub_name is None and channel_idx is None:
                # Base component
                grouped[base_name]['base'] = losses
            elif channel_idx is not None:
                # Per-channel
                grouped[base_name]['channels'][channel_idx] = losses
            else:
                # Per-component
                grouped[base_name]['components'][sub_name] = losses
        
        return grouped