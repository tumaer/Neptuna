from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn


class LossComponent(nn.Module, ABC):
    """
    A single loss term. Must return a scalar loss tensor.
    """
    def __init__(
        self,
        weight: float = 1.0,
        name: Optional[str] = None,
        data_dim: int = None,
        field_names: List[str] = None,
    ):
        super().__init__()
        self.weight = weight
        self.name = name or self.__class__.__name__
        self.data_dim = data_dim
        self.field_names = field_names

    @abstractmethod
    def forward(
        self,
        model: nn.Module,
        predictions: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
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
    ) -> torch.Tensor:
        total_loss = 0.0
        
        for loss_component in self.loss_components:
            loss = loss_component(model, predictions, labels)
            total_loss = total_loss + loss
        
        return total_loss