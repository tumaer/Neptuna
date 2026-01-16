from typing import Dict, List, Optional
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
    
    Supports hierarchical weight updates for base components, per-channel,
    and per-sub-component levels.
    
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
        # Buffer now tracks all hierarchical keys
        self.loss_history_buffer: Dict[str, List[float]] = {}
    
    def compute_statistics(self, losses: List[float]) -> Optional[Dict[str, float]]:
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
        loss_key: str,
        current_loss: float
    ) -> float:
        """
        Compute rate of change over the lookback window.
        
        Args:
            loss_key: Loss key (base, channel, or sub-component)
            current_loss: Current epoch's average loss
            
        Returns:
            Rate of change (negative means decreasing loss)
        """
        if loss_key not in self.loss_history_buffer:
            return 0.0
        
        history = self.loss_history_buffer[loss_key]
        
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
            rates: Dictionary of loss keys to rate of change values
            
        Returns:
            Dictionary of loss keys to adaptive weights
        """
        if not rates:
            return {}
        
        all_loss_keys = list(rates.keys())
        rate_values = torch.tensor([rates[key] for key in all_loss_keys])
        
        beta = 1.0 / (self.temperature + self.epsilon)

        if self.use_exponential:
            shifted = beta * (rate_values - rate_values.max())
            weights = torch.exp(shifted)
            weights = weights / (weights.sum() + self.epsilon)
        else:
            weights = torch.softmax(beta * rate_values, dim=0)
        
        weights = weights * len(all_loss_keys)
        
        adaptive_weights = {
            key: float(weights[i])
            for i, key in enumerate(all_loss_keys)
        }
        
        return adaptive_weights
    
    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict]:
        """
        Compute new weights based on SoftAdapt algorithm for all loss keys uniformly.
        
        Steps:
          1) Compute current average losses for all loss keys
          2) Update history buffer
          3) Compute rates of change
          4) Compute adaptive weights via softmax
          5) Reconstruct hierarchical weight dictionary
        
        Args:
            loss_history: Dictionary of loss keys to loss value lists
            current_weights: Current weight dictionary
            grad_norm_history: Optional (unused in SoftAdapt)
            
        Returns:
            Updated weight dictionary
        """
        # Get all loss keys from history
        all_loss_keys = list(loss_history.keys())
        if not all_loss_keys:
            return {k: v.copy() for k, v in current_weights.items()}
        
        current_losses = {}
        rates = {}
        
        # --- Step 1 & 2: Compute current average losses and update history ---
        for loss_key in all_loss_keys:
            hist = loss_history.get(loss_key, [])
            if not hist:
                current_losses[loss_key] = 1.0
            else:
                avg_loss = self.compute_average_loss(hist)
                current_losses[loss_key] = avg_loss
                
                if loss_key not in self.loss_history_buffer:
                    self.loss_history_buffer[loss_key] = []
                
                self.loss_history_buffer[loss_key].append(avg_loss)
                
                max_buffer = self.lookback_window + 20
                if len(self.loss_history_buffer[loss_key]) > max_buffer:
                    self.loss_history_buffer[loss_key] = \
                        self.loss_history_buffer[loss_key][-max_buffer:]
        
        # --- Step 3: Compute rates of change ---
        for loss_key in all_loss_keys:
            if loss_key in current_losses:
                rate = self.compute_rate_of_change(
                    loss_key,
                    current_losses[loss_key]
                )
                rates[loss_key] = rate
            else:
                rates[loss_key] = 0.0
        
        # --- Step 4: Compute adaptive weights ---
        adaptive_weights = self.compute_adaptive_weights(rates)
        
        # --- Step 5: Apply clipping ---
        new_weight_scalars: Dict[str, float] = {}
        for loss_key in all_loss_keys:
            if loss_key in adaptive_weights:
                new_weight = adaptive_weights[loss_key]
            else:
                new_weight = 1.0
            
            # Clip to reasonable range
            new_weight = max(self.min_weight, min(self.max_weight, new_weight))
            new_weight_scalars[loss_key] = float(new_weight)
        
        # --- Step 6: Optional normalization ---
        if self.normalize_weights:
            total_weight = sum(new_weight_scalars.values())
            num_keys = len(new_weight_scalars)
            normalization_factor = num_keys / max(total_weight, self.epsilon)
            
            for loss_key in all_loss_keys:
                new_weight_scalars[loss_key] *= normalization_factor
                # Re-apply clipping after normalization
                new_weight_scalars[loss_key] = max(
                    self.min_weight,
                    min(self.max_weight, new_weight_scalars[loss_key])
                )
        
        # --- Step 7: Reconstruct hierarchical weight dictionary ---
        new_weights = self._reconstruct_weight_dict(new_weight_scalars, current_weights)
        
        return new_weights

    def _get_previous_weight(self, loss_key: str, current_weights: Dict[str, Dict]) -> float:
        """Extract the previous weight for a given loss key."""
        base_name, sub_name, channel_idx = self._parse_hierarchical_key(loss_key)
        
        if base_name not in current_weights:
            return 1.0
        
        config = current_weights[base_name]
        
        # Base component weight
        if sub_name is None and channel_idx is None:
            return float(config.get('base_weight', 1.0))
        
        # Per-channel weight
        if channel_idx is not None:
            if 'channel_weights' in config:
                channel_weights = config['channel_weights']
                if channel_idx < len(channel_weights):
                    return float(channel_weights[channel_idx])
            return 1.0
        
        # Per-component weight
        if sub_name is not None:
            if 'component_weights' in config:
                component_weights = config['component_weights']
                return float(component_weights.get(sub_name, 1.0))
            return 1.0
        
        return 1.0

    def _reconstruct_weight_dict(
        self, 
        new_weight_scalars: Dict[str, float], 
        current_weights: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Reconstruct the hierarchical weight dictionary from flat scalar weights.
        
        Args:
            new_weight_scalars: Flat dict of loss_key -> weight
            current_weights: Current weight structure to preserve format
            
        Returns:
            Hierarchical weight dictionary matching current_weights format
        """
        new_weights: Dict[str, Dict] = {}
        
        for base_name, config in current_weights.items():
            new_config = config.copy()
            
            # Update base weight if present
            if base_name in new_weight_scalars:
                new_config['base_weight'] = new_weight_scalars[base_name]
            
            # Update channel weights if present
            if 'channel_weights' in config:
                channel_weights = config['channel_weights'].clone()
                num_channels = len(channel_weights)
                
                for ch_idx in range(num_channels):
                    ch_key = f"{base_name}/channel_{ch_idx}"
                    if ch_key in new_weight_scalars:
                        channel_weights[ch_idx] = new_weight_scalars[ch_key]
                
                new_config['channel_weights'] = channel_weights
            
            # Update component weights if present
            if 'component_weights' in config:
                component_weights = config['component_weights'].copy()
                
                for sub_name in component_weights.keys():
                    comp_key = f"{base_name}/{sub_name}"
                    if comp_key in new_weight_scalars:
                        component_weights[sub_name] = new_weight_scalars[comp_key]
                
                new_config['component_weights'] = component_weights
            
            new_weights[base_name] = new_config
        
        return new_weights