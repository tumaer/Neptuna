from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Tuple

import torch
import torch.nn as nn

class WeightSchedule(nn.Module):
    """
    Handles per-timestep, per-channel, and per-component weighting for loss components.
    """
    def __init__(
        self,
        base_weight: float = 1.0,
        timestep_weights: Optional[torch.Tensor] = None,
        channel_weights: Optional[torch.Tensor] = None,
        component_weights: Optional[Dict[str, float]] = None,  # NEW
    ):
        super().__init__()
        self.base_weight = base_weight
        
        # Register as buffers so they move with the model
        if timestep_weights is not None:
            self.register_buffer('timestep_weights', timestep_weights)
        else:
            self.timestep_weights = None
            
        if channel_weights is not None:
            self.register_buffer('channel_weights', channel_weights)
        else:
            self.channel_weights = None
        
        # Component weights (for losses with multiple sub-components)
        self.component_weights = component_weights or {}
    
    def get_weight(self, shape: Optional[torch.Size] = None) -> torch.Tensor:
        """
        Returns the weight tensor, broadcasting appropriately.
        
        Args:
            shape: Expected shape (batch, timesteps, channels, ...) for broadcasting
        """
        weight = torch.tensor(self.base_weight, device=self.get_device())
        
        if self.timestep_weights is not None and shape is not None:
            t_weights = self.timestep_weights.view(1, -1, *([1] * (len(shape) - 2)))
            weight = weight * t_weights
            
        if self.channel_weights is not None and shape is not None:
            c_weights = self.channel_weights.view(1, 1, -1, *([1] * (len(shape) - 3)))
            weight = weight * c_weights
            
        return weight
    
    def get_component_weight(self, component_name: str) -> float:
        """Get weight for a specific sub-component."""
        return self.component_weights.get(component_name, 1.0)
    
    def get_device(self) -> torch.device:
        if self.timestep_weights is not None:
            return self.timestep_weights.device
        if self.channel_weights is not None:
            return self.channel_weights.device
        return torch.device('cpu')
    
    def to_dict(self) -> Dict[str, Union[float, torch.Tensor, Dict[str, float]]]:
        """Returns a dictionary representation of the weights."""
        result = {'base_weight': self.base_weight}
        if self.timestep_weights is not None:
            result['timestep_weights'] = self.timestep_weights
        if self.channel_weights is not None:
            result['channel_weights'] = self.channel_weights
        if self.component_weights:
            result['component_weights'] = self.component_weights
        return result

class LossComponent(nn.Module, ABC):
    """
    A single loss term. Must return a scalar loss tensor.
    Can optionally return detailed breakdown of loss components.
    """
    def __init__(
        self,
        weight: Union[float, WeightSchedule] = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
        norm_stats: Dict[str, Dict[str, float]] = None,
    ):
        super().__init__()
        
        # Handle both scalar weights and WeightSchedule
        if isinstance(weight, (int, float)):
            self.weight_schedule = WeightSchedule(base_weight=float(weight))
        else:
            self.weight_schedule = weight
            
        self.name = name or self.__class__.__name__
        self.data_dim = data_dim
        self.field_names = field_names
        self.norm_stats = norm_stats
    
    @property
    def weight(self) -> float:
        """Backward compatibility: returns base weight."""
        return self.weight_schedule.base_weight

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Compute the loss.
        
        Args:
            model: The model being trained
            predictions: Model predictions
            labels: Ground truth labels
            return_detailed: If True, return (total_loss, detailed_dict)
        
        Returns:
            If return_detailed is False: scalar total loss
            If return_detailed is True: (total_loss, detailed_dict) where detailed_dict contains:
                - 'per_timestep': loss per timestep (if applicable)
                - 'per_channel': loss per channel (if applicable)
                - 'unweighted': element-wise unweighted loss
                - 'weighted': element-wise weighted loss
        """
        ...


class CompositeLoss(LossComponent):
    """
    Combines multiple loss components into a single weighted loss.
    """
    def __init__(
        self, 
        loss_components: List[LossComponent],
        name: Optional[str] = None
    ):
        super().__init__(weight=1.0, name=name)
        self.loss_components = nn.ModuleList(loss_components)
    
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
        return_detailed: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]]:
        """
        Compute composite loss.
        
        Args:
            model: The model being trained
            predictions: Model predictions
            labels: Ground truth labels
            return_detailed: If True, return detailed breakdown of all components
        
        Returns:
            If return_detailed is False: scalar total loss
            If return_detailed is True: (total_loss, detailed_dict) where detailed_dict is:
                {component_name: scalar_loss} if component doesn't support detailed
                {component_name: detailed_breakdown_dict} if component supports detailed
        """
        total_loss = 0.0
        detailed_dict = {} if return_detailed else None
        
        for loss_component in self.loss_components:
            result = loss_component(model, predictions, labels, return_detailed=return_detailed)
            
            if return_detailed:
                component_loss, component_detailed = result
                total_loss = total_loss + component_loss
                detailed_dict[loss_component.name] = {
                    'total': component_loss.detach() if isinstance(component_loss, torch.Tensor) else component_loss,
                    **component_detailed  # Spread any additional detailed breakdowns
                }
            else:
                total_loss = total_loss + result
        
        if return_detailed:
            return total_loss, detailed_dict
        return total_loss
    
    def get_weight_dict(self) -> Dict[str, Dict[str, Union[float, torch.Tensor, Dict[str, float]]]]:
        """
        Returns a dictionary of all loss component weights.
        Includes nested component_weights for losses with sub-components.
        """
        weight_dict = {}
        for loss_component in self.loss_components:
            weight_dict[loss_component.name] = loss_component.weight_schedule.to_dict()
        return weight_dict
    
    def update_weights(self, weight_dict: Dict[str, Dict[str, Union[float, torch.Tensor, Dict[str, float]]]]):
        """
        Update weights from a dictionary.
        Supports nested component_weights for losses with sub-components.
        """
        for loss_component in self.loss_components:
            if loss_component.name in weight_dict:
                updates = weight_dict[loss_component.name]
                
                if 'base_weight' in updates:
                    loss_component.weight_schedule.base_weight = updates['base_weight']
                
                if 'timestep_weights' in updates:
                    loss_component.weight_schedule.register_buffer(
                        'timestep_weights', updates['timestep_weights']
                    )
                
                if 'channel_weights' in updates:
                    loss_component.weight_schedule.register_buffer(
                        'channel_weights', updates['channel_weights']
                    )
                
                if 'component_weights' in updates:
                    loss_component.weight_schedule.component_weights = updates['component_weights']