from typing import Dict, List, Optional
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch


class ResidualBasedAttention(LossWeightingStrategyBase):
    """
    Residual-Based Attention (RBA) for adaptive loss weighting. 
    
    Based on "Residual-based Attention and Connection to Information 
    Bottleneck Theory in PINNs" 
    https://arxiv.org/abs/2307.00379
    
    Supports hierarchical weight updates for base components, per-channel,
    and per-sub-component levels.
    """

    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        gamma: float = 0.9,                # decay factor (paper's gamma)
        eta_star: float = 1.0,             # update step size (paper's eta*)
        residual_mode: str = "mean",       # 'mean', 'max', 'std'
        add_constant: float = 0.0,         # optional +c after update (paper mentions variants)
        eps: float = 1e-8,                 # numerical stability
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        normalize_weights: bool = True,    # normalize sum to num_components (optional, not in paper)
    ):
        super().__init__(update_frequency, use_gradients)

        if not (0.0 <= gamma < 1.0):
            raise ValueError("gamma should be in [0, 1).")

        self.gamma = float(gamma)
        self.eta_star = float(eta_star)
        self.residual_mode = residual_mode
        self.add_constant = float(add_constant)
        self.eps = float(eps)

        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.normalize_weights = bool(normalize_weights)

        # Running weights now track all hierarchical keys
        self.running_weights: Dict[str, float] = {}

    def compute_statistics(self, losses: List[float]) -> Optional[Dict[str, float]]:
        if not losses:
            return None
        t = torch.tensor(losses, dtype=torch.float32)
        return {
            "mean": float(t.mean()),
            "std": float(t.std(unbiased=False)),
            "min": float(t.min()),
            "max": float(t.max()),
            "count": int(t.numel()),
        }

    def compute_residual_proxy(self, losses: List[float]) -> float:
        """
        Per-component residual proxy R_c derived from loss history.

        This is a DESIGN CHOICE for component-level adaptation. Common choices:
        - mean: average loss over the epoch/steps
        - max: maximum observed loss
        - std: variability proxy (here: mean + std to keep scale > 0)
        """
        if not losses:
            return 1.0

        stats = self.compute_statistics(losses)
        if stats is None:
            return 1.0

        if self.residual_mode == "mean":
            return max(stats["mean"], 0.0)
        elif self.residual_mode == "max":
            return max(stats["max"], 0.0)
        elif self.residual_mode == "std":
            return max(stats["mean"] + stats["std"], 0.0)
        else:
            raise ValueError(f"Unknown residual_mode: {self.residual_mode}")

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Dict]:
        """
        Compute updated weights using paper-consistent RBA dynamics for all loss keys:

            w_c <- gamma * w_c + eta_star * (R_c / max_j R_j)

        where R_c is the per-component residual proxy.
        
        Steps:
          1) Compute residual proxies R_c for all loss keys (base, channel, sub-component)
          2) Normalize by max residual
          3) Apply RBA update with decay
          4) Reconstruct hierarchical weight dictionary
        """
        # Get all loss keys from history
        all_loss_keys = list(loss_history.keys())
        if not all_loss_keys:
            return {k: v.copy() for k, v in current_weights.items()}

        # --- Step 1: Compute residual proxies R_c for all loss keys ---
        residual_proxy: Dict[str, float] = {}
        for loss_key in all_loss_keys:
            hist = loss_history.get(loss_key, [])
            residual_proxy[loss_key] = float(self.compute_residual_proxy(hist))

        # --- Step 2: Normalize by max over all components ---
        max_r = max(residual_proxy.values()) if residual_proxy else 1.0
        max_r = max(max_r, self.eps)

        # --- Step 3: RBA-style update for all loss keys ---
        new_weight_scalars: Dict[str, float] = {}
        
        for loss_key in all_loss_keys:
            # Get previous weight for this key
            prev_w = self.running_weights.get(
                loss_key,
                self._get_previous_weight(loss_key, current_weights)
            )

            # Normalize residual and apply RBA update
            normalized_r = residual_proxy[loss_key] / max_r
            updated_w = self.gamma * prev_w + self.eta_star * normalized_r

            if self.add_constant != 0.0:
                updated_w = updated_w + self.add_constant

            # Clip to bounds
            updated_w = max(self.min_weight, min(self.max_weight, float(updated_w)))

            # Store in running weights
            self.running_weights[loss_key] = float(updated_w)
            new_weight_scalars[loss_key] = float(updated_w)

        # --- Step 4: Optional normalization ---
        if self.normalize_weights:
            total = sum(new_weight_scalars.values())
            n = len(new_weight_scalars)
            if total > self.eps:
                factor = n / total
                for loss_key in all_loss_keys:
                    w = new_weight_scalars[loss_key] * factor
                    w = max(self.min_weight, min(self.max_weight, float(w)))
                    new_weight_scalars[loss_key] = w
                    self.running_weights[loss_key] = w

        # --- Step 5: Reconstruct hierarchical weight dictionary ---
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