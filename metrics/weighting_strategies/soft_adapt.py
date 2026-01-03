from typing import Dict, List
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch

class SoftAdapt(LossWeightingStrategyBase):
    """
    SoftAdapt: Adaptive loss weighting based on rate of change.
    
    Based on "SoftAdapt: Techniques for Adaptive Loss Weighting of Neural 
    Networks with Multi-Part Loss Functions"
    https://arxiv.org/abs/1912.12355
    
    Assigns higher weights to loss components that are changing more slowly,
    allowing the network to focus on harder tasks.
    
    rate_i = (L_i(t) - L_i(t-window)) / window
    w_i = softmax(rate_i / temperature)
    """
    
    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        lookback_window: int = 5,
        temperature: float = 1.0,
        epsilon: float = 1e-8,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        normalize_weights: bool = True,
        use_exponential: bool = False
    ):
        """
        Args:
            update_frequency: Update weights every N epochs
            lookback_window: Number of epochs to look back for rate computation
            temperature: Temperature parameter for softmax (higher = more uniform)
            epsilon: Small constant for numerical stability
            min_weight: Minimum allowed weight
            max_weight: Maximum allowed weight
            normalize_weights: Whether to normalize weights to sum to num_components
            use_exponential: If True, use exponential weighting instead of softmax
        """
        super().__init__(update_frequency, use_gradients)
        self.lookback_window = lookback_window
        self.temperature = temperature
        self.epsilon = epsilon
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize_weights = normalize_weights
        self.use_exponential = use_exponential
        self.loss_history_buffer: Dict[str, List[float]] = {}
    
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
    
    def compute_average_loss(self, losses: List[float]) -> float:
        """
        Compute average loss for the current epoch.
        
        Args:
            losses: List of loss values from the epoch
            
        Returns:
            Average loss value
        """
        if not losses:
            return 1.0
        
        stats = self.compute_statistics(losses)
        return stats['mean']
    
    def compute_rate_of_change(
        self,
        component_name: str,
        current_loss: float
    ) -> float:
        """
        Compute rate of change over the lookback window.
        
        Args:
            component_name: Name of the loss component
            current_loss: Current epoch's average loss
            
        Returns:
            Rate of change (negative means decreasing loss)
        """
        if component_name not in self.loss_history_buffer:
            return 0.0
        
        history = self.loss_history_buffer[component_name]
        
        if len(history) < (self.lookback_window + 1):
            return 0.0
        
        past_loss = history[-(self.lookback_window + 1)]
        rate = (current_loss - past_loss) / float(self.lookback_window)
        
        return rate
    
    def compute_adaptive_weights(
        self,
        rates: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Compute adaptive weights using softmax over negative rates.
        
        Components with slower decrease (higher rate) get higher weight.
        
        Args:
            rates: Dictionary of component names to rate of change values
            
        Returns:
            Dictionary of component names to adaptive weights
        """
        if not rates:
            return {}
        
        component_names = list(rates.keys())
        rate_values = torch.tensor([rates[name] for name in component_names])
        
        beta = 1.0 / (self.temperature + self.epsilon)

        if self.use_exponential:
            shifted = beta * (rate_values - rate_values.max())
            weights = torch.exp(shifted)
            weights = weights / (weights.sum() + self.epsilon)
        else:
            weights = torch.softmax(beta * rate_values, dim=0)
        
        weights = weights * len(component_names)
        
        adaptive_weights = {
            name: float(weights[i])
            for i, name in enumerate(component_names)
        }
        
        return adaptive_weights
    
    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Compute new weights based on SoftAdapt algorithm.
        
        Args:
            loss_history: Dictionary of component names to loss value lists
            current_weights: Current weight dictionary
            
        Returns:
            Updated weight dictionary
        """
        new_weights = {}
        current_losses = {}
        rates = {}
        
        # Step 1: Compute current average losses and update history
        for component_name in current_weights.keys():
            if component_name not in loss_history or not loss_history[component_name]:
                current_losses[component_name] = 1.0
            else:
                avg_loss = self.compute_average_loss(loss_history[component_name])
                current_losses[component_name] = avg_loss
                
                if component_name not in self.loss_history_buffer:
                    self.loss_history_buffer[component_name] = []
                
                self.loss_history_buffer[component_name].append(avg_loss)
                
                max_buffer = self.lookback_window + 20
                if len(self.loss_history_buffer[component_name]) > max_buffer:
                    self.loss_history_buffer[component_name] = \
                        self.loss_history_buffer[component_name][-max_buffer:]
        
        # Step 2: Compute rates of change
        for component_name in current_weights.keys():
            if component_name in current_losses:
                rate = self.compute_rate_of_change(
                    component_name,
                    current_losses[component_name]
                )
                rates[component_name] = rate
            else:
                rates[component_name] = 0.0
        
        # Step 3: Compute adaptive weights
        adaptive_weights = self.compute_adaptive_weights(rates)
        
        # Step 4: Apply weights to each component
        for component_name in current_weights.keys():
            if component_name in adaptive_weights:
                new_weight = adaptive_weights[component_name]
            else:
                # Fallback to uniform weight
                new_weight = 1.0
            
            # Clip to reasonable range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            
            new_weights[component_name] = current_weights[component_name].copy()
            new_weights[component_name]['base_weight'] = float(new_weight)
        
        # Step 5: Optional normalization
        if self.normalize_weights and new_weights:
            total_weight = sum(w['base_weight'] for w in new_weights.values())
            num_components = len(new_weights)
            normalization_factor = num_components / max(total_weight, self.epsilon)
            
            for component_name in new_weights:
                new_weights[component_name]['base_weight'] *= normalization_factor
                # Re-apply clipping after normalization
                new_weights[component_name]['base_weight'] = max(
                    self.min_weight,
                    min(self.max_weight, new_weights[component_name]['base_weight'])
                )
        
        return new_weights