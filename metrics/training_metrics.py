from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Union, Tuple

import torch
import torch.nn as nn

class WeightSchedule(nn.Module):
    """
    Configurable weighting for loss components.

    Supports:
      * A scalar base weight (always applied).
      * Optional per-timestep weights: shape (T,).
      * Optional per-channel weights: shape (C,).
      * Optional per-component scalar overrides (for composite losses).

    Lightweight implementation:
      * Stores only 1D buffers for timesteps and channels.
      * `get_weight()` returns a small broadcastable tensor
         of shape at most (1, T, C, 1, ..., 1).
      * Elementwise weighting is a single multiply over the loss
        tensor.
      * `is_scalar_only()` enables "fast paths" in loss functions when
        no per-timestep/channel/component weights are configured.
    """
    def __init__(
        self,
        base_weight: float = 1.0,
        timestep_weights: Optional[torch.Tensor] = None,
        channel_weights: Optional[torch.Tensor] = None,
        component_weights: Optional[Dict[str, float]] = None,
    ):
        super().__init__()
        self.base_weight = float(base_weight)
        
        # Optional per-timestep weights (shape: T)
        if timestep_weights is not None:
            self.register_buffer('timestep_weights', timestep_weights)
        else:
            self.timestep_weights = None
        
        # Optional per-channel weights (shape: C)
        if channel_weights is not None:
            self.register_buffer('channel_weights', channel_weights)
        else:
            self.channel_weights = None
        
        # Optional per-component scalar weights (used by composite losses)
        self.component_weights = component_weights or {}

        # Cached flag for fast-path checks in losses
        self._is_scalar_only = (
            self.timestep_weights is None and
            self.channel_weights is None and
            not self.component_weights
        )

    # ------- fast-path helpers -------
    def is_scalar_only(self) -> bool:
        """
        Returns True if this schedule reduces to a single scalar weight.

        Conditions:
          - No per-timestep weights.
          - No per-channel weights.
          - No component-specific weights.

        When True, loss implementations can skip calling `get_weight()`
        and behave like a plain scalar-weighted reduction.
        """
        return self._is_scalar_only

    def has_timestep_weights(self) -> bool:
        """Returns True if per-timestep weighting is configured."""
        return self.timestep_weights is not None

    def has_channel_weights(self) -> bool:
        """Returns True if per-channel weighting is configured."""
        return self.channel_weights is not None

    # ------- main API -------
    def get_weight(self, shape: Optional[torch.Size] = None) -> torch.Tensor:
        """
        Construct a broadcastable weight tensor for a given loss tensor shape.

        Args:
            shape:
                Expected loss tensor shape, typically
                (batch, timesteps, channels, ...).

        Returns:
            A tensor suitable for elementwise multiplication with the
            loss tensor. The returned tensor:
              * Lives on the same device as stored buffers.
              * Has shape at most (1, T, C, 1, ..., 1).
              * Does NOT materialize a full (B, T, C, ...) tensor.
        """
        device = self.get_device()
        base = float(self.base_weight)

        if shape is None:
            # Scalar tensor, used rarely
            return torch.tensor(base, device=device)

        dims = len(shape)

        # Start from scalar tensor and fold in optional 1D weights
        weight = torch.tensor(base, device=device)

        if self.timestep_weights is not None:
            # (T,) -> (1, T, 1, 1, ...)
            t = self.timestep_weights.to(device)
            t = t.view(1, -1, *([1] * (dims - 2)))
            weight = weight * t

        if self.channel_weights is not None:
            # (C,) -> (1, 1, C, 1, 1, ...)
            c = self.channel_weights.to(device)
            c = c.view(1, 1, -1, *([1] * (dims - 3)))
            weight = weight * c

        return weight
    
    def get_component_weight(self, component_name: str) -> float:
        """
        Return a scalar override for a named sub-component, if present.

        Used by composite losses that internally split into multiple
        named terms. Defaults to 1.0 for unknown names.
        """
        return self.component_weights.get(component_name, 1.0)
    
    def get_device(self) -> torch.device:
        """
        Infer the device where weights live.

        Preference order:
          1. timestep_weights device
          2. channel_weights device
          3. CPU (no weights registered)

        Loss functions typically call `.to(predictions.device)` on
        the result of `get_weight()`, so device mismatches are avoided.
        """
        if self.timestep_weights is not None:
            return self.timestep_weights.device
        if self.channel_weights is not None:
            return self.channel_weights.device
        return torch.device('cpu')
    
    def to_dict(self) -> Dict[str, Union[float, torch.Tensor, Dict[str, float]]]:
        """
        Serialize this schedule to a simple dictionary.

        Contains:
          - 'base_weight': float
          - optional 'timestep_weights': Tensor
          - optional 'channel_weights': Tensor
          - optional 'component_weights': Dict[str, float]

        Used for logging, checkpointing, or external schedule updates.
        """
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
    Base class for a single loss term.

    Responsibilities:
      * Store a `WeightSchedule` describing how this loss is weighted.
      * Implement a `forward` that returns a scalar loss tensor.
      * Optionally provide a "detailed" breakdown of per-timestep/channel stats.

    Design and performance notes
    ----------------------------
    * Each subclass should implement a fast path when
      `self.weight_schedule.is_scalar_only()` is True:
        - Compute the unweighted loss.
        - Apply a single scalar factor.
        - This is as close to native PyTorch MSE/MAE as possible.
    * Detailed breakdowns (per-timestep/per-channel) are optional
      and guarded by `return_detailed`. The training loop should
      typically set `return_detailed=False` for the hot path and
      enable it only for periodic logging / validation.
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
        
        # Normalize scalar weights to a WeightSchedule for a unified API
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
            model:
                The model being trained. Included for losses that
                may inspect parameters or intermediate states.
            predictions:
                Model outputs.
            labels:
                Ground truth targets.
            return_detailed:
                If False:
                    Return a single scalar loss tensor.
                If True:
                    Return (total_loss, detailed_dict), where
                    detailed_dict MAY contain keys such as:
                      * 'per_timestep': 1D tensor [T]
                      * 'per_channel': 1D tensor [C]
                    (Exact contents depend on the subclass.)

        Returns:
            If return_detailed is False:
                A scalar loss tensor suitable for backprop.
            If return_detailed is True:
                A tuple (loss, detailed_dict).

        Performance contract:
            * The main training loop should call with
              `return_detailed=False` for maximum throughput.
            * Implementations should keep the `return_detailed=False`
              path as lean as possible (ideally a small number of
              tensor ops and a single reduction).
        """
        ...


class CompositeLoss(LossComponent):
    """
    Combine multiple `LossComponent` instances into a single scalar loss.

    Behavior:
      * Forwards predictions/labels to each sub-component.
      * Sums their scalar losses into a single total loss.
      * Optionally aggregates each component's detailed breakdown.
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
    ) -> Union[
        torch.Tensor,
        Tuple[torch.Tensor, Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor]]]]
    ]:
        """
        Compute the composite loss over all configured components.

        Args:
            model:       The model being trained.
            predictions: Model outputs.
            labels:      Ground truth targets.
            return_detailed:
                If False:
                    Return a single scalar loss = sum of component losses.
                If True:
                    Return (total_loss, detailed_dict) where
                    detailed_dict[component_name] is a dict with:
                        - 'total': scalar loss for that component
                        - any additional keys returned by the component
                          (e.g. 'per_timestep', 'per_channel').

        Notes:
            * The hot training path should typically set
              `return_detailed=False`.
            * Detailed stats are forwarded from components as-is;
              this class does not perform extra reductions.
        """
        total_loss: Optional[torch.Tensor] = None
        detailed_dict = {} if return_detailed else None
        
        for loss_component in self.loss_components:
            if return_detailed:
                component_loss, component_detailed = loss_component(
                    model, predictions, labels, return_detailed=True
                )
            else:
                component_loss = loss_component(
                    model, predictions, labels, return_detailed=False
                )
                component_detailed = None  # type: ignore[assignment]

            # Initialize accumulator on first component to avoid float promotion
            if total_loss is None:
                total_loss = component_loss
            else:
                total_loss = total_loss + component_loss

            if return_detailed:
                detailed_dict[loss_component.name] = {
                    'total': component_loss.detach(),
                    **component_detailed
                }

        # If there were no components, total_loss will still be None
        if total_loss is None:
            # Fallback: zero scalar on the same device as predictions
            total_loss = predictions.new_tensor(0.0)

        if return_detailed:
            return total_loss, detailed_dict  # type: ignore[arg-type]
        return total_loss
    
    def get_weight_dict(self) -> Dict[str, Dict[str, Union[float, torch.Tensor, Dict[str, float]]]]:
        """
        Return a nested dictionary of weight schedules for all components.

        Structure:
            {
              component_name: {
                'base_weight': float,
                optional 'timestep_weights': Tensor,
                optional 'channel_weights': Tensor,
                optional 'component_weights': Dict[str, float],
              },
              ...
            }

        Intended for:
          * Inspecting or logging current weights.
          * Exporting schedules for external tuning.
        """
        weight_dict = {}
        for loss_component in self.loss_components:
            weight_dict[loss_component.name] = loss_component.weight_schedule.to_dict()
        return weight_dict
    
    def update_weights(self, weight_dict: Dict[str, Dict[str, Union[float, torch.Tensor, Dict[str, float]]]]):
        """
        Update weight schedules of sub-components from a nested dictionary.

        Args:
            weight_dict:
                Dictionary in the format produced by `get_weight_dict()`.

        Notes:
            * This can be used to dynamically adjust base/timestep/channel
              weights without reconstructing the loss objects.
            * If you modify timestep/channel/component weights here and
              you rely on `is_scalar_only()`, ensure that the schedule's
              internal flags stay consistent with your updates.
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