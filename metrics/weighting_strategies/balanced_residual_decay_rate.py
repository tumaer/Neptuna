from typing import Dict, List, Optional
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch


class BalancedResidualDecayRate(LossWeightingStrategyBase):
    """
    Self-adaptive weights based on balanced residual decay rate (RDR). 
    
    Based on Chen et al. (2024) "Self-adaptive weights based on balanced residual 
    decay rate for physics-informed neural networks and deep operator networks" 
    https://arxiv.org/abs/2407.01613
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        alpha: float = 0.9,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        temperature: float = 1.0,
        epsilon: float = 1e-8
    ):
        """
        Args:
            update_frequency: Update weights every N epochs
            alpha: Momentum factor (used as beta for EMA of both r4-hat and weights)
            min_weight: Clip lower bound
            max_weight: Clip upper bound
            temperature: Unused (kept for API compatibility)
            epsilon: Numerical stability
        """
        super().__init__(update_frequency, use_gradients)
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)

        # EMA of the 4th-moment proxy per component
        self.ema_r4: Dict[str, float] = {}

    def compute_statistics(self, losses: List[float]) -> Optional[Dict[str, float]]:
        """Compute mean/std/count for a list of scalar losses."""
        if not losses:
            return None
        t = torch.tensor(losses, dtype=torch.float32)
        return {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)) if t.numel() > 1 else 0.0,
            "count": int(t.numel())
        }

    def _clip(self, x: float) -> float:
        return max(self.min_weight, min(self.max_weight, x))

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict]:
        """
        Compute new weights using BRDR for all loss components uniformly.

        Steps:
          1) For each loss key (base, channel, or sub-component):
               Compute IRDR using mean loss and EMA of 4th moment
          2) Mean-normalize all IRDRs
          3) Apply momentum update to get new weights
          4) Parse hierarchical keys and reconstruct the weight dictionary
        """
        # --- Step 1: Compute IRDR for all loss keys ---
        irdr: Dict[str, float] = {}
        
        for loss_key, losses in loss_history.items():
            if not losses:
                continue

            stats = self.compute_statistics(losses)
            if stats is None:
                continue

            L = max(float(stats["mean"]), 0.0)
            r2 = L
            r4 = L * L

            prev_ema = self.ema_r4.get(loss_key, r4)
            ema = self.alpha * prev_ema + (1.0 - self.alpha) * r4
            self.ema_r4[loss_key] = float(ema)

            denom = float(torch.sqrt(torch.tensor(ema + self.epsilon)))
            irdr[loss_key] = float(r2 / denom) if denom > 0.0 else 0.0

        if not irdr:
            return {k: v.copy() for k, v in current_weights.items()}

        # --- Step 2: Mean-normalize IRDRs ---
        mean_irdr = sum(irdr.values()) / len(irdr)
        if mean_irdr <= self.epsilon:
            w_ref = {k: 1.0 for k in irdr.keys()}
        else:
            w_ref = {k: (v / mean_irdr) for k, v in irdr.items()}

        # --- Step 3: Apply momentum update and clip ---
        new_weight_scalars: Dict[str, float] = {}
        
        for loss_key, wref in w_ref.items():
            # Get previous weight for this key
            prev_w = self._get_previous_weight(loss_key, current_weights)
            
            # Momentum update
            w_new = self.alpha * prev_w + (1.0 - self.alpha) * wref
            new_weight_scalars[loss_key] = self._clip(w_new)

        # --- Step 4: Reconstruct hierarchical weight dictionary ---
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