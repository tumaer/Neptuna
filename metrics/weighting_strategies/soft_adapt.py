from typing import Dict, List, Optional
from ..loss_weighting_strategies import LossWeightingStrategyBase
import torch

class SoftAdapt(LossWeightingStrategyBase):
    """
    SoftAdapt: Adaptive loss weighting based on rate of change.
    
    Based on "SoftAdapt: Techniques for Adaptive Loss Weighting of Neural 
    Networks with Multi-Part Loss Functions"
    https://arxiv.org/abs/1912.12355
    """
    
    def __init__(
        self,
        update_frequency: int = 1,
        use_gradients: bool = False,
        temperature_T: float = 1.0,
        epsilon: float = 1e-8,
        min_weight: Optional[float] = None,
        max_weight: Optional[float] = None,
        normalize_to_num_components: bool = False,
        keep_previous_until_ready: bool = True,
    ):
        """
        Args:
            update_frequency: Update weights every N epochs
            use_gradients: Unused here (kept for base-class signature compatibility)
            temperature_T: Paper's temperature T in exp(T * Δ). T=0 => uniform weights.
            epsilon: Numerical stability constant
            min_weight: Optional clipping lower bound (paper doesn't include; set None to disable)
            max_weight: Optional clipping upper bound (paper doesn't include; set None to disable)
            normalize_to_num_components:
                If False (default): weights sum to 1 (paper softmax).
                If True: weights are scaled to sum to k (often used to keep loss magnitude similar).
            keep_previous_until_ready:
                If True: until each key has a previous epoch value, return the current weights unchanged.
                If False: fall back to uniform weights when previous values are missing.
        """
        super().__init__(update_frequency, use_gradients)
        self.T = float(temperature_T)
        self.epsilon = float(epsilon)
        self.min_weight = min_weight
        self.max_weight = max_weight
        self.normalize_to_num_components = normalize_to_num_components
        self.keep_previous_until_ready = keep_previous_until_ready

        # Stores last epoch's mean loss for each key
        self.prev_epoch_mean: Dict[str, float] = {}

    def compute_statistics(self, losses: List[float]) -> Optional[Dict[str, float]]:
        if not losses:
            return None
        losses_tensor = torch.tensor(losses, dtype=torch.float32)
        return {
            "mean": float(losses_tensor.mean()),
            "std": float(losses_tensor.std(unbiased=False)),
            "min": float(losses_tensor.min()),
            "max": float(losses_tensor.max()),
            "count": len(losses),
        }

    def compute_average_loss(self, losses: List[float]) -> float:
        if not losses:
            return 1.0
        stats = self.compute_statistics(losses)
        return 1.0 if stats is None else float(stats["mean"])

    @staticmethod
    def _softmax_stable(logits: torch.Tensor, dim: int = 0) -> torch.Tensor:
        # Numerically stable softmax
        shifted = logits - logits.max(dim=dim, keepdim=True).values
        exp = torch.exp(shifted)
        return exp / (exp.sum(dim=dim, keepdim=True) + 1e-12)

    def _compute_weights_from_deltas(self, deltas: Dict[str, float]) -> Dict[str, float]:
        keys = list(deltas.keys())
        if not keys:
            return {}

        k = len(keys)

        # Paper special case: T == 0 => uniform weights
        if abs(self.T) < self.epsilon:
            w = torch.full((k,), 1.0 / k, dtype=torch.float32)
        else:
            delta_vec = torch.tensor([deltas[key] for key in keys], dtype=torch.float32)
            logits = self.T * delta_vec
            w = self._softmax_stable(logits, dim=0)

        # Optional convention: scale to sum to k
        if self.normalize_to_num_components:
            w = w * float(k)

        # Optional clipping (not in paper; disabled by default)
        if self.min_weight is not None or self.max_weight is not None:
            min_w = self.min_weight if self.min_weight is not None else float("-inf")
            max_w = self.max_weight if self.max_weight is not None else float("inf")
            w = torch.clamp(w, min=min_w, max=max_w)

            target_sum = float(k) if self.normalize_to_num_components else 1.0
            w_sum = float(w.sum().item())
            if w_sum > self.epsilon:
                w = w * (target_sum / w_sum)

        return {key: float(w[i].item()) for i, key in enumerate(keys)}

    def compute_new_weights(
        self,
        loss_history: Dict[str, List[float]],
        current_weights: Dict[str, Dict],
        grad_norm_history: Optional[Dict[str, List[float]]] = None,
    ) -> Dict[str, Dict]:
        """
        Paper-aligned SoftAdapt update (per epoch):
        1) Compute epoch mean loss for each key: L_i(t)
        2) Compute one-step delta: Δ_i(t) = L_i(t) - L_i(t-1)
        3) Compute weights: w_i(t) = softmax(T * Δ_i(t))
        4) Reconstruct hierarchical weight dict
        5) Store L_i(t) as previous for next epoch
        """
        all_loss_keys = list(loss_history.keys())
        if not all_loss_keys:
            return {k: v.copy() for k, v in current_weights.items()}

        # Step 1: per-epoch mean loss per key (detached scalar)
        cur_mean: Dict[str, float] = {
            key: self.compute_average_loss(loss_history.get(key, []))
            for key in all_loss_keys
        }

        # Step 2: compute deltas using previous epoch mean
        deltas: Dict[str, float] = {}
        missing_prev = False
        for key in all_loss_keys:
            if key not in self.prev_epoch_mean:
                missing_prev = True
                continue
            deltas[key] = float(cur_mean[key] - self.prev_epoch_mean[key])

        # If we don't have previous values yet, decide behavior
        if missing_prev:
            if self.keep_previous_until_ready:
                # Update prev cache but do not change weights yet
                for key in all_loss_keys:
                    self.prev_epoch_mean[key] = float(cur_mean[key])
                return {k: v.copy() for k, v in current_weights.items()}
            else:
                # Fall back to uniform (paper T=0 behavior) for all keys, then proceed
                deltas = {key: 0.0 for key in all_loss_keys}

        for key in all_loss_keys:
            if key not in deltas:
                deltas[key] = 0.0

        # Step 3: compute adaptive weights
        new_weight_scalars = self._compute_weights_from_deltas(deltas)

        # Step 4: reconstruct hierarchical dict
        new_weights = self._reconstruct_weight_dict(new_weight_scalars, current_weights)

        # Step 5: update previous epoch cache
        for key in all_loss_keys:
            self.prev_epoch_mean[key] = float(cur_mean[key])

        return new_weights

