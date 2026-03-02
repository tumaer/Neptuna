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
            key: Loss component key (e.g., 'MSE', 'MSE/channel_0', 'RMSE/domain/mass')
            
        Returns:
            Tuple of (base_name, sub_component_name, channel_idx)
            Examples:
                'MSE' -> ('MSE', None, None)
                'MSE/channel_0' -> ('MSE', None, 0)
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
                'MSE': {
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

    def _get_previous_weight(
        self,
        loss_key: str,
        current_weights: Dict[str, Dict],
        default: float = 1.0,
    ) -> float:
        """
        Extract the previous scalar weight for a flat loss_key from the hierarchical weight dict.

        Args:
            loss_key: Flat key (base, base/channel_i, or base/subcomponent)
            current_weights: Nested weight dict from CompositeLoss.get_loss_weight_dict()
            default: Fallback value if the key is missing
        """
        base_name, sub_name, channel_idx = self._parse_hierarchical_key(loss_key)

        if base_name not in current_weights:
            return float(default)

        config = current_weights[base_name]

        if sub_name is None and channel_idx is None:
            return float(config.get("base_weight", default))

        if channel_idx is not None:
            if "channel_weights" in config:
                channel_weights = config["channel_weights"]
                if channel_idx < len(channel_weights):
                    return float(channel_weights[channel_idx])
            return float(default)

        if sub_name is not None:
            if "component_weights" in config:
                return float(config["component_weights"].get(sub_name, default))
            return float(default)

        return float(default)

    def _reconstruct_weight_dict(
        self,
        new_weight_scalars: Dict[str, float],
        current_weights: Dict[str, Dict],
    ) -> Dict[str, Dict]:
        """
        Reconstruct hierarchical dict (base_weight / channel_weights / component_weights)
        from a flat mapping loss_key -> scalar.
        """
        new_weights: Dict[str, Dict] = {}

        for base_name, config in current_weights.items():
            new_config = config.copy()

            has_channel_updates = False
            if "channel_weights" in config:
                for ch_idx in range(len(config["channel_weights"])):
                    if f"{base_name}/channel_{ch_idx}" in new_weight_scalars:
                        has_channel_updates = True
                        break

            has_component_updates = False
            if "component_weights" in config:
                for sub_name in config["component_weights"].keys():
                    if f"{base_name}/{sub_name}" in new_weight_scalars:
                        has_component_updates = True
                        break

            if base_name in new_weight_scalars and not (has_channel_updates or has_component_updates):
                new_config["base_weight"] = float(new_weight_scalars[base_name])

            if "channel_weights" in config:
                channel_weights = config["channel_weights"]
                if torch.is_tensor(channel_weights):
                    channel_weights = channel_weights.clone()
                elif hasattr(channel_weights, "copy"):
                    channel_weights = channel_weights.copy()
                else:
                    channel_weights = list(channel_weights)

                for ch_idx in range(len(channel_weights)):
                    ch_key = f"{base_name}/channel_{ch_idx}"
                    if ch_key in new_weight_scalars:
                        channel_weights[ch_idx] = float(new_weight_scalars[ch_key])
                new_config["channel_weights"] = channel_weights

            if "component_weights" in config:
                component_weights = config["component_weights"].copy()
                for sub_name in list(component_weights.keys()):
                    comp_key = f"{base_name}/{sub_name}"
                    if comp_key in new_weight_scalars:
                        component_weights[sub_name] = float(new_weight_scalars[comp_key])
                new_config["component_weights"] = component_weights

            new_weights[base_name] = new_config

        return new_weights