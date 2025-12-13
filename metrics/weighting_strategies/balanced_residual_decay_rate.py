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
        alpha: float = 0.9,
        min_weight: float = 0.01,
        max_weight: float = 100.0,
        temperature: float = 1.0,  # kept for API compatibility
        epsilon: float = 1e-8
    ):
        """
        Args:
            update_frequency: Update weights every N epochs
            alpha: Momentum factor (used as beta for EMA of both r4-hat and weights)
            min_weight: Clip lower bound (safety; not essential to BRDR)
            max_weight: Clip upper bound (safety; not essential to BRDR)
            temperature: Unused (kept to preserve constructor compatibility)
            epsilon: Numerical stability
        """
        super().__init__(update_frequency)
        self.alpha = float(alpha)
        self.min_weight = float(min_weight)
        self.max_weight = float(max_weight)
        self.temperature = float(temperature)
        self.epsilon = float(epsilon)

        # EMA of the 4th-moment proxy per component: \hat{r4}
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
        current_weights: Dict[str, Dict]
    ) -> Dict[str, Dict]:
        """
        Compute new component weights using component-wise BRDR.

        Steps:
          1) For each component i:
               L_i := mean loss over current epoch
               r2_i := L_i
               r4_i := L_i^2
               ema_r4_i <- alpha * ema_r4_i + (1-alpha) * r4_i
               irdr_i := r2_i / sqrt(ema_r4_i + eps)

          2) Mean-normalize:
               w_ref_i := irdr_i / mean(irdr)

          3) Momentum update weights:
               w_i <- alpha * w_i_prev + (1-alpha) * w_ref_i

          4) (Optional safety) clip to [min_weight, max_weight]
        """
        new_weights: Dict[str, Dict] = {}
        component_names = list(current_weights.keys())

        # --- Step 1: compute irdr per component (from scalar losses) ---
        irdr: Dict[str, float] = {}
        for name in component_names:
            if name not in loss_history or not loss_history[name]:
                new_weights[name] = current_weights[name].copy()
                continue

            stats = self.compute_statistics(loss_history[name])
            if stats is None:
                new_weights[name] = current_weights[name].copy()
                continue

            L = float(stats["mean"])
            L = max(L, 0.0)

            r2 = L
            r4 = L * L

            prev_ema = self.ema_r4.get(name, r4)
            ema = self.alpha * prev_ema + (1.0 - self.alpha) * r4
            self.ema_r4[name] = float(ema)

            denom = float(torch.sqrt(torch.tensor(ema + self.epsilon)))
            irdr[name] = float(r2 / denom) if denom > 0.0 else 0.0

        if not irdr:
            return {k: v.copy() for k, v in current_weights.items()}

        # --- Step 2: mean-normalize to get reference weights (mean(w_ref)=1) ---
        mean_irdr = sum(irdr.values()) / max(len(irdr), 1)
        if mean_irdr <= self.epsilon:
            w_ref = {k: 1.0 for k in irdr.keys()}
        else:
            w_ref = {k: (v / mean_irdr) for k, v in irdr.items()}

        # --- Step 3: momentum update weights (EMA) ---
        for name, wref in w_ref.items():
            prev_w = float(current_weights[name].get("base_weight", 1.0))
            w_new = self.alpha * prev_w + (1.0 - self.alpha) * float(wref)
            w_new = self._clip(w_new)

            updated = current_weights[name].copy()
            updated["base_weight"] = w_new
            new_weights[name] = updated

        # --- Fill any untouched components with existing weights ---
        for name in component_names:
            if name not in new_weights:
                new_weights[name] = current_weights[name].copy()

        return new_weights